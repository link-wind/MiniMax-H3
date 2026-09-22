import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffsynth.core.context_parallel import (
    PackedAttentionMetadata,
    distributed_ring_varlen_attention,
    exact_varlen_attention,
    split_sequence_indices,
)


def _worker(rank, world_size, init_method):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.new_group(list(range(world_size)))
    try:
        torch.manual_seed(124)
        metadata = PackedAttentionMetadata.from_cu_seqlens([0, 8, 12])
        local_start, local_end = split_sequence_indices(12, world_size)[rank]

        q = torch.randn(
            12,
            4,
            16,
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )
        k = torch.randn(
            12,
            4,
            16,
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )
        v = torch.randn(
            12,
            4,
            16,
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )
        reference = exact_varlen_attention(
            q, k, v, metadata.cu_seqlens
        )

        q_local = (
            q[local_start:local_end]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        k_local = (
            k[local_start:local_end]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        v_local = (
            v[local_start:local_end]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        out = distributed_ring_varlen_attention(
            q_local,
            k_local,
            v_local,
            metadata,
            rank=rank,
            world_size=world_size,
            group=group,
        )
        torch.testing.assert_close(
            out,
            reference[local_start:local_end],
            atol=2e-2,
            rtol=2e-2,
        )

        out.sum().backward()
        ref_q = q.detach().clone().requires_grad_(True)
        ref_k = k.detach().clone().requires_grad_(True)
        ref_v = v.detach().clone().requires_grad_(True)
        ref_out = exact_varlen_attention(
            ref_q, ref_k, ref_v, metadata.cu_seqlens
        )
        ref_grads = torch.autograd.grad(
            ref_out.sum(),
            (ref_q, ref_k, ref_v),
        )
        for actual, expected in (
            (q_local.grad, ref_grads[0][local_start:local_end]),
            (k_local.grad, ref_grads[1][local_start:local_end]),
            (v_local.grad, ref_grads[2][local_start:local_end]),
        ):
            torch.testing.assert_close(
                actual, expected, atol=2e-2, rtol=2e-2
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least 2 CUDA devices",
)
def test_nccl_ring_attention_forward_and_gradients():
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = f"file://{tmpdir}/shared_init"
        mp.start_processes(
            _worker,
            args=(world_size, init_method),
            nprocs=world_size,
            join=True,
        )
