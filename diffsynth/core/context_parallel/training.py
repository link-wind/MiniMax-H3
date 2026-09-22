from __future__ import annotations

import random
from typing import Iterator, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler


def cp_process_layout(process_index: int, cp_world_size: int) -> tuple[int, int]:
    """Return (dp_rank, cp_rank) for a CP-contiguous process layout."""
    if cp_world_size < 1:
        raise ValueError(f"cp_world_size must be positive, got {cp_world_size}")
    return process_index // cp_world_size, process_index % cp_world_size


def validate_cp_topology(
    process_count: int,
    cp_world_size: int,
    dp_world_size: int | None = None,
) -> dict[str, int]:
    if process_count < 1:
        raise ValueError(f"process_count must be positive, got {process_count}")
    if cp_world_size < 1:
        raise ValueError(f"cp_world_size must be positive, got {cp_world_size}")
    if process_count % cp_world_size != 0:
        raise ValueError(
            f"process_count {process_count} is not divisible by cp_world_size "
            f"{cp_world_size}"
        )
    resolved_dp = process_count // cp_world_size
    if dp_world_size is not None and dp_world_size != resolved_dp:
        raise ValueError(
            f"dp_world_size {dp_world_size} != process_count / cp_world_size "
            f"({resolved_dp})"
        )
    return {
        "process_count": int(process_count),
        "cp_world_size": int(cp_world_size),
        "dp_world_size": int(resolved_dp),
    }


def validate_cp_setup(
    *,
    process_count: int,
    cp_world_size: int,
    dp_world_size: int | None = None,
    deepspeed_config: dict | None = None,
) -> dict[str, object]:
    """Return a CPU-safe validation report for the CP training launch shape."""
    topology = validate_cp_topology(
        process_count,
        cp_world_size,
        dp_world_size=dp_world_size,
    )
    if cp_world_size <= 1:
        parameter_reduction = "none"
    else:
        zero_stage = int((deepspeed_config or {}).get("zero_stage", 0))
        if zero_stage == 3:
            parameter_reduction = "deepspeed-zero3-global-group"
        elif zero_stage in (1, 2):
            parameter_reduction = (
                f"deepspeed-zero{zero_stage}-with-cp-reduction"
            )
        else:
            parameter_reduction = "global-group"
    return {
        **topology,
        "dataloader": "cp-aware" if cp_world_size > 1 else "plain",
        "prepare_dataloader": should_prepare_dataloader(cp_world_size),
        "loss_reduction": "cp-weighted-video-audio",
        "parameter_gradient_reduction": parameter_reduction,
    }


def build_cp_training_dataloader(
    dataset,
    *,
    process_index: int = 0,
    process_count: int = 1,
    cp_world_size: int = 1,
    num_workers: int = 0,
    cp_seed: int = 42,
):
    """Build a CP-aware dataloader for the H3 SFT training entrypoint."""
    if cp_world_size > 1:
        dp_rank, _ = cp_process_layout(process_index, cp_world_size)
        generator = torch.Generator()
        generator.manual_seed(cp_seed + dp_rank)
        sampler = CPAwareSampler(
            dataset,
            process_index=process_index,
            cp_world_size=cp_world_size,
            process_count=process_count,
            generator=generator,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=lambda x: x[0],
            num_workers=num_workers,
        )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x[0],
        num_workers=num_workers,
    )


def should_prepare_dataloader(cp_world_size: int) -> bool:
    """Return whether accelerator.prepare should own the dataloader."""
    return cp_world_size <= 1


def should_reduce_cp_parameter_gradients(
    cp_world_size: int,
    strategy: str = "global-group",
) -> bool:
    """Return whether training should add explicit CP parameter reduction."""
    if cp_world_size <= 1:
        return False
    if strategy not in {"global-group", "dp-only"}:
        raise ValueError(
            f"strategy must be 'global-group' or 'dp-only', got {strategy!r}"
        )
    return strategy == "dp-only"


class CPAwareSampler(Sampler[int]):
    """Yield sample indices replicated across ranks in the same CP group."""

    def __init__(
        self,
        data_source,
        process_index: int,
        cp_world_size: int,
        process_count: int | None = None,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ):
        if cp_world_size < 1:
            raise ValueError(f"cp_world_size must be positive, got {cp_world_size}")
        if process_count is None:
            process_count = cp_world_size
        if process_count % cp_world_size != 0:
            raise ValueError(
                f"process_count {process_count} is not divisible by cp_world_size "
                f"{cp_world_size}"
            )
        self.data_source = data_source
        self.process_index = int(process_index)
        self.cp_world_size = int(cp_world_size)
        self.dp_world_size = int(process_count) // self.cp_world_size
        self.dp_rank, self.cp_rank = cp_process_layout(
            self.process_index, self.cp_world_size
        )
        self.shuffle = bool(shuffle)
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            if self.generator is not None:
                order = torch.randperm(len(indices), generator=self.generator).tolist()
                indices = [indices[i] for i in order]
            else:
                random.shuffle(indices)
        num_samples = (len(indices) + self.dp_world_size - 1) // self.dp_world_size
        for i in range(num_samples):
            source_index = i * self.dp_world_size + self.dp_rank
            if source_index < len(indices):
                yield indices[source_index]
            else:
                yield indices[source_index % len(indices)]

    def __len__(self) -> int:
        return max(
            0,
            (len(self.data_source) + self.dp_world_size - 1) // self.dp_world_size,
        )


def broadcast_cp_tensor(
    tensor: torch.Tensor,
    cp_rank: int,
    cp_world_size: int,
    group=None,
) -> torch.Tensor:
    """Broadcast from CP leader. Without a real group this is a logical no-op."""
    if cp_world_size is None or cp_world_size <= 1 or group is None:
        return tensor
    dist.broadcast(tensor, group_src=0, group=group)
    return tensor


def create_cp_process_group(
    process_index: int,
    cp_world_size: int,
) -> object | None:
    """Create the contiguous CP process group for a CP-contiguous layout.

    Returns None when CP is disabled or no distributed runtime is active. The
    returned group is only meaningful inside an already-initialized
    torch.distributed runtime such as Accelerate/DeepSpeed.
    """
    if cp_world_size <= 1:
        return None
    if not dist.is_available() or not dist.is_initialized():
        return None
    dp_rank, _ = cp_process_layout(process_index, cp_world_size)
    ranks = list(range(dp_rank * cp_world_size, (dp_rank + 1) * cp_world_size))
    return dist.new_group(ranks=ranks)


def reduce_cp_parameter_gradients(model: nn.Module, group=None) -> None:
    """All-reduce all trainable parameter gradients inside a CP group.

    This is the integration point for a DP-only parameter state strategy. With
    the default DeepSpeed ZeRO-3 global-group strategy it should NOT be called,
    because DeepSpeed already reduces gradients across the global process group.
    """
    if group is None:
        return
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, group=group)


def mse_local_sum_count(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the local MSE numerator and valid element count."""
    diff = (pred.float() - target.float()).pow(2)
    local_sum = diff.sum() * weight
    local_count = torch.tensor(
        pred.numel(), dtype=torch.float64, device=pred.device
    )
    return local_sum, local_count


def reduce_cp_weighted_mean(
    local_sum: torch.Tensor,
    local_count: torch.Tensor,
    group=None,
) -> torch.Tensor:
    """Reduce sum/count and return the global weighted mean."""
    if group is not None:
        local_sum = _AllReduce.apply(local_sum.clone(), group)
        local_count = _AllReduce.apply(local_count.clone(), group)
    return local_sum / local_count


def h3_cp_loss(
    video_sum: torch.Tensor,
    video_count: torch.Tensor,
    audio_sum: torch.Tensor,
    audio_count: torch.Tensor,
    group=None,
) -> torch.Tensor:
    """Reduce H3 video/audio MSE and combine as video_mean + audio_mean."""
    video_mean = reduce_cp_weighted_mean(video_sum, video_count, group=group)
    audio_mean = reduce_cp_weighted_mean(audio_sum, audio_count, group=group)
    return video_mean + audio_mean


class _AllReduce(torch.autograd.Function):
    """All-reduce with an explicit autograd backward hook."""

    @staticmethod
    def forward(ctx, tensor, group):
        ctx.cp_group = group
        output = tensor.clone()
        dist.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.clone()
        dist.all_reduce(grad, group=ctx.cp_group)
        return grad, None
