from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .metadata import (
    PackedAttentionMetadata,
    PackedShardMetadata,
    split_sequence_indices,
)

try:
    from flash_attn.flash_attn_interface import (
        _flash_attn_varlen_backward,
        _flash_attn_varlen_forward,
    )

    _HAS_FLASH_ATTN = True
except ImportError:  # pragma: no cover - optional GPU dependency
    _HAS_FLASH_ATTN = False


def _can_use_flash(q: torch.Tensor) -> bool:
    return (
        _HAS_FLASH_ATTN
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
    )


def _span_segments(
    metadata: PackedAttentionMetadata,
    start: int,
    end: int,
) -> list[tuple[int, int, int, int, int]]:
    """Split a global span into (segment_id, local_start, local_end, global_start, global_end)."""
    if end <= start:
        return []
    segments: list[tuple[int, int, int, int, int]] = []
    cursor = start
    for boundary in metadata.cu_seqlens[1:]:
        if boundary <= cursor:
            continue
        span_end = min(boundary, end)
        segment_id = metadata.segment_id_at(cursor)
        if span_end > cursor:
            segments.append(
                (
                    segment_id,
                    cursor - start,
                    span_end - start,
                    cursor,
                    span_end,
                )
            )
        cursor = span_end
        if cursor >= end:
            break
    if cursor < end:
        segments.append(
            (
                metadata.segment_id_at(cursor),
                cursor - start,
                end - start,
                cursor,
                end,
            )
        )
    return segments


def _one_segment_cu_seqlens(length: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([0, length], dtype=torch.int32, device=device)


def _flash_varlen_segment_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_len = q.shape[0]
    k_len = k.shape[0]
    block_out, block_lse, _, _ = _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q=_one_segment_cu_seqlens(q_len, q.device),
        cu_seqlens_k=_one_segment_cu_seqlens(k_len, k.device),
        max_seqlen_q=q_len,
        max_seqlen_k=k_len,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )
    return block_out, block_lse


def _flash_varlen_segment_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_len = q.shape[0]
    k_len = k.shape[0]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _flash_attn_varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q=_one_segment_cu_seqlens(q_len, q.device),
        cu_seqlens_k=_one_segment_cu_seqlens(k_len, k.device),
        max_seqlen_q=q_len,
        max_seqlen_k=k_len,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        rng_state=None,
        zero_tensors=False,
    )
    return dq, dk, dv


def _merge_flash_block(
    running_out: torch.Tensor,
    running_lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_lse = block_lse.transpose(0, 1).unsqueeze(-1).to(torch.float32)
    has_previous = torch.isfinite(running_lse)
    merge_weight = torch.sigmoid(block_lse - running_lse)
    # Update in-place to avoid materializing running_out - block_out, which is
    # the largest temporary in the online-softmax merge for 30s packed tokens.
    running_out.mul_(1.0 - merge_weight)
    running_out.addcmul_(block_out, merge_weight)
    merged_lse = running_lse + torch.nn.functional.softplus(block_lse - running_lse)
    running_lse.copy_(torch.where(has_previous, merged_lse, block_lse))
    return running_out, running_lse


def exact_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference full-sequence varlen attention for [S, H, D] tensors."""
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q/k/v must be [S, H, D]")
    head_dim = q.shape[-1]
    if scale is None:
        scale = head_dim**-0.5
    bounds = tuple(int(x) for x in cu_seqlens)
    out = torch.empty_like(q)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop == start:
            continue
        q_seg = q[start:stop].transpose(0, 1).unsqueeze(0)
        k_seg = k[start:stop].transpose(0, 1).unsqueeze(0)
        v_seg = v[start:stop].transpose(0, 1).unsqueeze(0)
        scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        out_seg = torch.matmul(weights, v_seg)
        out[start:stop] = out_seg.squeeze(0).transpose(0, 1)
    return out


def _same_segment_mask(
    metadata: PackedAttentionMetadata,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
    device: torch.device,
) -> torch.Tensor:
    q_segment_ids = torch.tensor(
        metadata.segment_ids_for_span(q_start, q_end),
        dtype=torch.long,
        device=device,
    )
    k_segment_ids = torch.tensor(
        metadata.segment_ids_for_span(k_start, k_end),
        dtype=torch.long,
        device=device,
    )
    return q_segment_ids[:, None, None] == k_segment_ids[None, None, :]


def logical_ring_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: PackedAttentionMetadata,
    rank: int,
    world_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute one logical CP shard without real collectives.

    The caller owns the full q/k/v tensors, so this is a CPU/CI-safe reference
    for the math and metadata. The actual distributed path will consume the
    same shard metadata and segment mask rules.
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q/k/v must be [S, H, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have identical shapes")
    if q.shape[0] != metadata.seq_len:
        raise ValueError(
            f"q seq_len {q.shape[0]} != metadata.seq_len {metadata.seq_len}"
        )
    if scale is None:
        scale = q.shape[-1] ** -0.5

    spans = split_sequence_indices(metadata.seq_len, world_size)
    local_start, local_end = spans[rank]
    q_local = q[local_start:local_end]
    if q_local.numel() == 0:
        return torch.empty_like(q_local)

    num_queries, num_heads, _ = q_local.shape
    min_logit = torch.finfo(q.dtype).min
    running_max = torch.full(
        (num_queries, num_heads, 1),
        min_logit,
        dtype=q.dtype,
        device=q.device,
    )
    denominator = torch.zeros(
        (num_queries, num_heads, 1), dtype=q.dtype, device=q.device
    )
    accumulator = torch.zeros_like(q_local)

    for chunk_rank in range(world_size):
        chunk_start, chunk_end = spans[chunk_rank]
        if chunk_end == chunk_start:
            continue
        k_chunk = k[chunk_start:chunk_end]
        v_chunk = v[chunk_start:chunk_end]
        scores = torch.einsum("qhd,khd->qhk", q_local, k_chunk) * scale
        mask = _same_segment_mask(
            metadata,
            local_start,
            local_end,
            chunk_start,
            chunk_end,
            q.device,
        )
        scores = scores.masked_fill(~mask, float("-inf"))
        chunk_max = scores.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, chunk_max)
        rescale = torch.exp(running_max - new_max)
        probs = torch.exp(scores - new_max)
        denominator = denominator * rescale + probs.sum(dim=-1, keepdim=True)
        accumulator = accumulator * rescale + torch.einsum(
            "qhk,khd->qhd", probs, v_chunk
        )
        running_max = new_max

    return accumulator / denominator


def logical_ring_varlen_attention_all(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: PackedAttentionMetadata,
    world_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Concatenate all logical shards back into full global token order."""
    shards = [
        logical_ring_varlen_attention(
            q, k, v, metadata, rank=rank, world_size=world_size, scale=scale
        )
        for rank in range(world_size)
    ]
    return torch.cat(shards, dim=0)


def _rotate_kv_chunk(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    group,
    receive_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    recv_k = torch.empty(
        (receive_len,) + tuple(k.shape[1:]),
        dtype=k.dtype,
        device=k.device,
    )
    recv_v = torch.empty_like(recv_k)
    ops = [
        dist.P2POp(dist.isend, k, group=group, group_peer=next_rank),
        dist.P2POp(dist.isend, v, group=group, group_peer=next_rank),
        dist.P2POp(dist.irecv, recv_k, group=group, group_peer=prev_rank),
        dist.P2POp(dist.irecv, recv_v, group=group, group_peer=prev_rank),
    ]
    for request in dist.batch_isend_irecv(ops):
        request.wait()
    return recv_k, recv_v


def _reduce_scatter_kv_grad_chunks(
    grad_k_chunks: list[torch.Tensor],
    grad_v_chunks: list[torch.Tensor],
    rank: int,
    world_size: int,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-rank K/V chunk gradients with one reduce-scatter each.

    Every rank computes partial gradients for every global K/V chunk. The old
    all-reduce-per-chunk path issued ``world_size`` collectives per K/V tensor;
    reduce-scatter moves exactly the same data once and leaves each rank with
    its own local chunk.
    """
    def reduce_with_allreduce(grad_chunks: list[torch.Tensor]) -> torch.Tensor:
        for chunk_rank in range(world_size):
            if grad_chunks[chunk_rank].numel() == 0:
                continue
            chunk = grad_chunks[chunk_rank].contiguous()
            dist.all_reduce(chunk, group=group)
            grad_chunks[chunk_rank] = chunk
        return grad_chunks[rank].contiguous()

    if world_size <= 1:
        return grad_k_chunks[0].contiguous(), grad_v_chunks[0].contiguous()

    # reduce_scatter_tensor uses group order for the output slot. The Ring CP
    # layout assumes group order equals logical CP rank, so only use it when
    # every rank agrees; otherwise fall back to the allreduce path.
    local_order = torch.tensor(
        1 if dist.get_rank(group) == int(rank) else 0,
        dtype=torch.long,
        device=grad_k_chunks[rank].device,
    )
    dist.all_reduce(local_order, op=dist.ReduceOp.MIN, group=group)
    if int(local_order.item()) != 1:
        return (
            reduce_with_allreduce(grad_k_chunks),
            reduce_with_allreduce(grad_v_chunks),
        )

    def reduce_with_reduce_scatter(
        grad_chunks: list[torch.Tensor],
    ) -> torch.Tensor:
        max_len = max(int(chunk.shape[0]) for chunk in grad_chunks)
        if max_len == 0:
            return grad_chunks[rank].contiguous()
        tail_shape = tuple(grad_chunks[rank].shape[1:])
        padded = torch.zeros(
            (world_size * max_len,) + tail_shape,
            dtype=grad_chunks[rank].dtype,
            device=grad_chunks[rank].device,
        )
        for chunk_rank, chunk in enumerate(grad_chunks):
            if chunk.numel() == 0:
                continue
            start = chunk_rank * max_len
            padded[start : start + chunk.shape[0]] = chunk
        output = torch.empty(
            (max_len,) + tail_shape,
            dtype=grad_chunks[rank].dtype,
            device=grad_chunks[rank].device,
        )
        dist.reduce_scatter_tensor(
            output,
            padded,
            op=dist.ReduceOp.SUM,
            group=group,
        )
        local_len = int(grad_chunks[rank].shape[0])
        return output[:local_len].contiguous()

    return (
        reduce_with_reduce_scatter(grad_k_chunks),
        reduce_with_reduce_scatter(grad_v_chunks),
    )

class _DistributedRingVarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        metadata,
        rank,
        world_size,
        group,
        scale,
        reduce_kv_gradients,
    ):
        if group is None:
            raise RuntimeError("distributed_ring_varlen_attention requires a CP group")
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError("q/k/v must be [S, H, D]")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q/k/v must have identical shapes")

        rank = int(rank)
        world_size = int(world_size)
        spans = split_sequence_indices(metadata.seq_len, world_size)
        local_start, local_end = spans[rank]
        if q.shape[0] != local_end - local_start:
            raise ValueError(
                f"local q length {q.shape[0]} != local span length "
                f"{local_end - local_start}"
            )
        if scale is None:
            scale = q.shape[-1] ** -0.5

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        num_queries, num_heads, head_dim = q.shape
        min_logit = torch.finfo(q.dtype).min
        running_max = torch.full(
            (num_queries, num_heads, 1),
            min_logit,
            dtype=q.dtype,
            device=q.device,
        )
        running_sum = torch.zeros_like(running_max)
        accumulator = torch.zeros_like(q)

        current_k = k
        current_v = v
        for step in range(world_size):
            chunk_rank = (rank - step) % world_size
            chunk_start, chunk_end = spans[chunk_rank]
            if chunk_end > chunk_start:
                scores = (
                    torch.einsum("qhd,khd->qhk", q, current_k) * scale
                )
                mask = _same_segment_mask(
                    metadata,
                    local_start,
                    local_end,
                    chunk_start,
                    chunk_end,
                    q.device,
                )
                scores = scores.masked_fill(~mask, float("-inf"))
                chunk_max = scores.max(dim=-1, keepdim=True).values
                new_max = torch.maximum(running_max, chunk_max)
                rescale = torch.exp(running_max - new_max)
                exp_scores = torch.exp(scores - new_max)
                running_sum = (
                    running_sum * rescale
                    + exp_scores.sum(dim=-1, keepdim=True)
                )
                accumulator = accumulator * rescale + torch.einsum(
                    "qhk,khd->qhd", exp_scores, current_v
                )
                running_max = new_max

            if step < world_size - 1:
                receive_chunk_rank = (rank - step - 1) % world_size
                receive_start, receive_end = spans[receive_chunk_rank]
                current_k, current_v = _rotate_kv_chunk(
                    current_k,
                    current_v,
                    rank=rank,
                    world_size=world_size,
                    group=group,
                    receive_len=receive_end - receive_start,
                )

        ctx.cp_metadata = metadata
        ctx.cp_rank = rank
        ctx.cp_world_size = world_size
        ctx.cp_group = group
        ctx.cp_scale = scale
        ctx.cp_reduce_kv_gradients = bool(reduce_kv_gradients)
        output = accumulator / running_sum
        ctx.save_for_backward(
            q,
            k,
            v,
            running_max.detach(),
            running_sum.detach(),
            output.detach(),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, running_max, running_sum, output = ctx.saved_tensors
        metadata = ctx.cp_metadata
        rank = ctx.cp_rank
        world_size = ctx.cp_world_size
        group = ctx.cp_group
        scale = ctx.cp_scale
        spans = split_sequence_indices(metadata.seq_len, world_size)
        local_start, local_end = spans[rank]

        grad_output = grad_output.contiguous()
        grad_q = torch.zeros_like(q)
        grad_chunk_k = [
            torch.zeros(
                (end - start,) + tuple(q.shape[1:]),
                dtype=q.dtype,
                device=q.device,
            )
            for start, end in spans
        ]
        grad_chunk_v = [tensor.clone() for tensor in grad_chunk_k]

        current_k = k
        current_v = v
        for step in range(world_size):
            chunk_rank = (rank - step) % world_size
            chunk_start, chunk_end = spans[chunk_rank]
            if chunk_end > chunk_start:
                scores = (
                    torch.einsum("qhd,khd->qhk", q, current_k) * scale
                )
                mask = _same_segment_mask(
                    metadata,
                    local_start,
                    local_end,
                    chunk_start,
                    chunk_end,
                    q.device,
                )
                scores = scores.masked_fill(~mask, float("-inf"))
                probs = torch.exp(scores - running_max) / running_sum
                d_probs = torch.einsum("qhd,khd->qhk", grad_output, current_v)
                global_mean = torch.einsum(
                    "qhd,qhd->qh", grad_output, output
                ).unsqueeze(-1)
                d_scores = probs * (
                    d_probs - global_mean
                )
                grad_q = grad_q + torch.einsum(
                    "qhk,khd->qhd", d_scores, current_k
                ) * scale
                grad_chunk_k[chunk_rank] = torch.einsum(
                    "qhk,qhd->khd", d_scores, q
                ) * scale
                grad_chunk_v[chunk_rank] = torch.einsum(
                    "qhk,qhd->khd", probs, grad_output
                )

            if step < world_size - 1:
                receive_chunk_rank = (rank - step - 1) % world_size
                receive_start, receive_end = spans[receive_chunk_rank]
                current_k, current_v = _rotate_kv_chunk(
                    current_k,
                    current_v,
                    rank=rank,
                    world_size=world_size,
                    group=group,
                    receive_len=receive_end - receive_start,
                )

        if ctx.cp_reduce_kv_gradients:
            grad_k_local, grad_v_local = _reduce_scatter_kv_grad_chunks(
                grad_chunk_k, grad_chunk_v, rank, world_size, group
            )

        return (
            grad_q,
            grad_k_local,
            grad_v_local,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def distributed_ring_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: PackedAttentionMetadata,
    rank: int,
    world_size: int,
    group,
    scale: float | None = None,
    reduce_kv_gradients: bool = True,
) -> torch.Tensor:
    """Run Ring CP with real collectives and return the local query shard.

    ``reduce_kv_gradients=True`` all-reduces key/value gradients inside the
    backward pass before returning each rank's local K/V shard. This is
    required to carry gradients from remote query shards into the rank that
    owns each K/V shard; optimizer/parameter-gradient reduction afterward then
    sums the local query shard and all K/V shard gradients once.
    """
    if world_size is None or world_size <= 1 or group is None:
        raise RuntimeError(
            "distributed_ring_varlen_attention requires world_size > 1 and a CP group"
        )
    kernel = (
        _DistributedFlashRingVarlenAttention
        if _can_use_flash(q)
        else _DistributedRingVarlenAttention
    )
    return kernel.apply(
        q,
        k,
        v,
        metadata,
        int(rank),
        int(world_size),
        group,
        scale,
        bool(reduce_kv_gradients),
    )


class _DistributedFlashRingVarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        metadata,
        rank,
        world_size,
        group,
        scale,
        reduce_kv_gradients,
    ):
        if group is None:
            raise RuntimeError("distributed_ring_varlen_attention requires a CP group")
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError("q/k/v must be [S, H, D]")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q/k/v must have identical shapes")

        rank = int(rank)
        world_size = int(world_size)
        spans = split_sequence_indices(metadata.seq_len, world_size)
        local_start, local_end = spans[rank]
        if q.shape[0] != local_end - local_start:
            raise ValueError(
                f"local q length {q.shape[0]} != local span length "
                f"{local_end - local_start}"
            )
        if scale is None:
            scale = q.shape[-1] ** -0.5

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        running_out = torch.zeros(
            q.shape, dtype=torch.float32, device=q.device
        )
        running_lse = torch.full(
            (q.shape[0], q.shape[1], 1),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        current_k = k
        current_v = v
        for step in range(world_size):
            chunk_rank = (rank - step) % world_size
            chunk_start, chunk_end = spans[chunk_rank]
            if chunk_end > chunk_start:
                q_segments = _span_segments(metadata, local_start, local_end)
                k_segments = _span_segments(metadata, chunk_start, chunk_end)
                for q_seg in q_segments:
                    q_segment_id = q_seg[0]
                    for k_seg in k_segments:
                        if k_seg[0] != q_segment_id:
                            continue
                        q_local_start, q_local_end = q_seg[1], q_seg[2]
                        k_local_start, k_local_end = k_seg[1], k_seg[2]
                        block_out, block_lse = _flash_varlen_segment_forward(
                            q[q_local_start:q_local_end],
                            current_k[k_local_start:k_local_end],
                            current_v[k_local_start:k_local_end],
                            scale,
                        )
                        _merge_flash_block(
                            running_out[q_local_start:q_local_end],
                            running_lse[q_local_start:q_local_end],
                            block_out,
                            block_lse,
                        )

            if step < world_size - 1:
                receive_chunk_rank = (rank - step - 1) % world_size
                receive_start, receive_end = spans[receive_chunk_rank]
                current_k, current_v = _rotate_kv_chunk(
                    current_k,
                    current_v,
                    rank=rank,
                    world_size=world_size,
                    group=group,
                    receive_len=receive_end - receive_start,
                )

        if not torch.isfinite(running_lse).all():
            raise RuntimeError(
                "flash Ring attention left query positions with no matching segment"
            )
        output = running_out.to(q.dtype)
        ctx.cp_metadata = metadata
        ctx.cp_rank = rank
        ctx.cp_world_size = world_size
        ctx.cp_group = group
        ctx.cp_scale = scale
        ctx.cp_reduce_kv_gradients = bool(reduce_kv_gradients)
        ctx.save_for_backward(
            q,
            k,
            v,
            running_lse.detach(),
            output.detach(),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, running_lse, output = ctx.saved_tensors
        metadata = ctx.cp_metadata
        rank = ctx.cp_rank
        world_size = ctx.cp_world_size
        group = ctx.cp_group
        scale = ctx.cp_scale
        spans = split_sequence_indices(metadata.seq_len, world_size)
        local_start, local_end = spans[rank]

        grad_output = grad_output.contiguous()
        grad_q = torch.zeros_like(q)
        grad_chunk_k = [
            torch.zeros(
                (end - start,) + tuple(q.shape[1:]),
                dtype=q.dtype,
                device=q.device,
            )
            for start, end in spans
        ]
        grad_chunk_v = [tensor.clone() for tensor in grad_chunk_k]

        current_k = k
        current_v = v
        for step in range(world_size):
            chunk_rank = (rank - step) % world_size
            chunk_start, chunk_end = spans[chunk_rank]
            if chunk_end > chunk_start:
                q_segments = _span_segments(metadata, local_start, local_end)
                k_segments = _span_segments(metadata, chunk_start, chunk_end)
                for q_seg in q_segments:
                    q_segment_id = q_seg[0]
                    for k_seg in k_segments:
                        if k_seg[0] != q_segment_id:
                            continue
                        q_local_start, q_local_end = q_seg[1], q_seg[2]
                        k_local_start, k_local_end = k_seg[1], k_seg[2]
                        q_seg_tensor = q[q_local_start:q_local_end]
                        k_seg_tensor = current_k[k_local_start:k_local_end]
                        v_seg_tensor = current_v[k_local_start:k_local_end]
                        dout_seg = grad_output[q_local_start:q_local_end]
                        out_seg = output[q_local_start:q_local_end]
                        lse_seg = (
                            running_lse[q_local_start:q_local_end]
                            .squeeze(-1)
                            .transpose(0, 1)
                            .contiguous()
                        )
                        dq, dk, dv = _flash_varlen_segment_backward(
                            dout_seg.contiguous(),
                            q_seg_tensor.contiguous(),
                            k_seg_tensor.contiguous(),
                            v_seg_tensor.contiguous(),
                            out_seg.contiguous(),
                            lse_seg,
                            scale,
                        )
                        grad_q[q_local_start:q_local_end] += dq
                        grad_chunk_k[chunk_rank][
                            k_local_start:k_local_end
                        ] += dk
                        grad_chunk_v[chunk_rank][
                            k_local_start:k_local_end
                        ] += dv

            if step < world_size - 1:
                receive_chunk_rank = (rank - step - 1) % world_size
                receive_start, receive_end = spans[receive_chunk_rank]
                current_k, current_v = _rotate_kv_chunk(
                    current_k,
                    current_v,
                    rank=rank,
                    world_size=world_size,
                    group=group,
                    receive_len=receive_end - receive_start,
                )

        if ctx.cp_reduce_kv_gradients:
            grad_k_local, grad_v_local = _reduce_scatter_kv_grad_chunks(
                grad_chunk_k, grad_chunk_v, rank, world_size, group
            )

        return (
            grad_q,
            grad_k_local,
            grad_v_local,
            None,
            None,
            None,
            None,
            None,
            None,
        )
