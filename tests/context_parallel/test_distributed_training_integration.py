import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffsynth.core.context_parallel import (
    broadcast_cp_tensor,
    build_cp_training_dataloader,
    create_cp_process_group,
    h3_cp_loss,
    should_prepare_dataloader,
)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, init_method):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    cp_group = create_cp_process_group(rank, 2)
    try:
        assert should_prepare_dataloader(1) is True
        assert should_prepare_dataloader(2) is False

        dataset = list(range(100))
        dataloader = build_cp_training_dataloader(
            dataset,
            process_index=rank,
            process_count=world_size,
            cp_world_size=2,
            cp_seed=7,
        )
        iterator = iter(dataloader)
        samples = torch.tensor([next(iterator) for _ in range(3)])
        gathered_samples = [torch.empty_like(samples) for _ in range(2)]
        dist.all_gather(gathered_samples, samples, group=cp_group)
        torch.testing.assert_close(
            gathered_samples[0], gathered_samples[1]
        )

        torch.manual_seed(100)
        if rank == 0:
            random_tensor = torch.randn(4, dtype=torch.float64)
        else:
            random_tensor = torch.empty(4, dtype=torch.float64)
        random_tensor = broadcast_cp_tensor(
            random_tensor,
            cp_rank=rank,
            cp_world_size=2,
            group=cp_group,
        )
        gathered_random = [
            torch.empty_like(random_tensor) for _ in range(2)
        ]
        dist.all_gather(gathered_random, random_tensor, group=cp_group)
        torch.testing.assert_close(
            gathered_random[0], gathered_random[1]
        )

        video_sums = [1.0, 2.0]
        video_counts = [2.0, 3.0]
        audio_sums = [2.0, 4.0]
        audio_counts = [4.0, 6.0]
        video_sum = torch.tensor(video_sums[rank], dtype=torch.float64)
        video_count = torch.tensor(video_counts[rank], dtype=torch.float64)
        audio_sum = torch.tensor(audio_sums[rank], dtype=torch.float64)
        audio_count = torch.tensor(audio_counts[rank], dtype=torch.float64)
        loss = h3_cp_loss(
            video_sum,
            video_count,
            audio_sum,
            audio_count,
            group=cp_group,
        )
        expected = (
            sum(video_sums) / sum(video_counts)
            + sum(audio_sums) / sum(audio_counts)
        )
        torch.testing.assert_close(
            loss, torch.tensor(expected, dtype=torch.float64)
        )
    finally:
        dist.destroy_process_group()


def test_gloo_cp_training_helpers_are_distributed_safe():
    if not dist.is_available():
        return
    world_size = 2
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    mp.start_processes(
        _worker,
        args=(world_size, init_method),
        nprocs=world_size,
        join=True,
    )
