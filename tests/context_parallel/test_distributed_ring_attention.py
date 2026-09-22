import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffsynth.core.context_parallel import (
    PackedAttentionMetadata,
    distributed_ring_varlen_attention,
    exact_varlen_attention,
    split_sequence_indices,
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
    group = dist.new_group(list(range(world_size)))
    try:
        torch.manual_seed(123)
        metadata = PackedAttentionMetadata.from_cu_seqlens([0, 6, 13])
        spans = split_sequence_indices(metadata.seq_len, world_size)
        local_start, local_end = spans[rank]

        q = torch.randn(13, 2, 4, dtype=torch.float64)
        k = torch.randn(13, 2, 4, dtype=torch.float64)
        v = torch.randn(13, 2, 4, dtype=torch.float64)

        q_ref = q.detach().requires_grad_(True)
        k_ref = k.detach().requires_grad_(True)
        v_ref = v.detach().requires_grad_(True)
        ref_out = exact_varlen_attention(
            q_ref, k_ref, v_ref, metadata.cu_seqlens
        )
        ref_grads = torch.autograd.grad(
            ref_out.sum(),
            (q_ref, k_ref, v_ref),
            retain_graph=True,
        )

        q_local = q[local_start:local_end].detach().clone().requires_grad_(True)
        k_local = k[local_start:local_end].detach().clone().requires_grad_(True)
        v_local = v[local_start:local_end].detach().clone().requires_grad_(True)
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
            ref_out[local_start:local_end],
            atol=1e-6,
            rtol=1e-6,
        )

        out.sum().backward()
        torch.testing.assert_close(
            q_local.grad,
            ref_grads[0][local_start:local_end],
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            k_local.grad,
            ref_grads[1][local_start:local_end],
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            v_local.grad,
            ref_grads[2][local_start:local_end],
            atol=1e-6,
            rtol=1e-6,
        )
    finally:
        dist.destroy_process_group()


def test_gloo_ring_attention_forward_and_gradients_match_reference():
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


def _worker_boundary_crossing(rank, world_size, init_method):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.new_group(list(range(world_size)))
    try:
        torch.manual_seed(124)
        metadata = PackedAttentionMetadata.from_cu_seqlens([0, 8, 12])
        spans = split_sequence_indices(metadata.seq_len, world_size)
        local_start, local_end = spans[rank]

        q = torch.randn(
            12, 2, 4, dtype=torch.float64, requires_grad=True
        )
        k = torch.randn(
            12, 2, 4, dtype=torch.float64, requires_grad=True
        )
        v = torch.randn(
            12, 2, 4, dtype=torch.float64, requires_grad=True
        )

        ref = exact_varlen_attention(q, k, v, metadata.cu_seqlens)
        ref_grads = torch.autograd.grad(
            ref.sum(),
            (q, k, v),
            retain_graph=True,
        )
        q_local = q[local_start:local_end].detach().clone().requires_grad_(True)
        k_local = k[local_start:local_end].detach().clone().requires_grad_(True)
        v_local = v[local_start:local_end].detach().clone().requires_grad_(True)
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
            ref[local_start:local_end],
            atol=1e-6,
            rtol=1e-6,
        )
        out.sum().backward()
        for actual, expected in zip(
            (q_local.grad, k_local.grad, v_local.grad),
            (
                ref_grads[0][local_start:local_end],
                ref_grads[1][local_start:local_end],
                ref_grads[2][local_start:local_end],
            ),
        ):
            torch.testing.assert_close(
                actual, expected, atol=1e-6, rtol=1e-6
            )
    finally:
        dist.destroy_process_group()


def test_gloo_ring_attention_crosses_segment_boundary():
    if not dist.is_available():
        return
    world_size = 2
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    mp.start_processes(
        _worker_boundary_crossing,
        args=(world_size, init_method),
        nprocs=world_size,
        join=True,
    )


def _worker_uneven_shards(rank, world_size, init_method):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.new_group(list(range(world_size)))
    try:
        torch.manual_seed(125)
        metadata = PackedAttentionMetadata.from_cu_seqlens([0, 6, 13])
        spans = split_sequence_indices(metadata.seq_len, world_size)
        local_start, local_end = spans[rank]

        q = torch.randn(13, 3, 5, dtype=torch.float64)
        k = torch.randn(13, 3, 5, dtype=torch.float64)
        v = torch.randn(13, 3, 5, dtype=torch.float64)

        q_ref = q.detach().requires_grad_(True)
        k_ref = k.detach().requires_grad_(True)
        v_ref = v.detach().requires_grad_(True)
        ref_out = exact_varlen_attention(
            q_ref, k_ref, v_ref, metadata.cu_seqlens
        )
        ref_grads = torch.autograd.grad(
            ref_out.sum(),
            (q_ref, k_ref, v_ref),
            retain_graph=True,
        )

        q_local = q[local_start:local_end].detach().clone().requires_grad_(True)
        k_local = k[local_start:local_end].detach().clone().requires_grad_(True)
        v_local = v[local_start:local_end].detach().clone().requires_grad_(True)
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
            ref_out[local_start:local_end],
            atol=1e-6,
            rtol=1e-6,
        )
        out.sum().backward()
        for actual, expected in zip(
            (q_local.grad, k_local.grad, v_local.grad),
            (
                ref_grads[0][local_start:local_end],
                ref_grads[1][local_start:local_end],
                ref_grads[2][local_start:local_end],
            ),
        ):
            torch.testing.assert_close(
                actual, expected, atol=1e-6, rtol=1e-6
            )
    finally:
        dist.destroy_process_group()


def test_gloo_ring_attention_uneven_shards_use_reduce_scatter():
    if not dist.is_available():
        return
    world_size = 3
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    mp.start_processes(
        _worker_uneven_shards,
        args=(world_size, init_method),
        nprocs=world_size,
        join=True,
    )
