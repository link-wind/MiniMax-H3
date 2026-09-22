import torch
import torch.distributed as dist
import torch.nn.functional as F

from diffsynth.core.context_parallel import (
    PackedAttentionMetadata,
    PackedShardMetadata,
    exact_varlen_attention,
    logical_ring_varlen_attention,
    logical_ring_varlen_attention_all,
    split_sequence_indices,
)
from diffsynth.core.context_parallel.ring_attention import _rotate_kv_chunk


def _assert_close(actual, expected, atol=1e-6, rtol=1e-6):
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )


def test_split_sequence_indices_are_contiguous_and_balanced():
    spans = split_sequence_indices(10, 3)
    assert spans == [(0, 3), (3, 6), (6, 10)]
    assert [start for start, _ in spans] == [0, 3, 6]
    assert spans[-1][1] == 10


def test_packed_metadata_segment_mapping():
    metadata = PackedAttentionMetadata.from_cu_seqlens([0, 5, 10])
    assert metadata.segment_id_at(0) == 0
    assert metadata.segment_id_at(4) == 0
    assert metadata.segment_id_at(5) == 1
    assert metadata.segment_id_at(9) == 1
    assert metadata.segment_ids_for_span(0, 10) == (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)


def test_local_shard_metadata_maps_global_positions():
    metadata = PackedAttentionMetadata.from_cu_seqlens([0, 5, 10])
    shard = PackedShardMetadata(metadata, rank=1, world_size=3)
    assert shard.local_span == (3, 6)
    assert shard.local_segment_ids == (0, 0, 1)
    assert shard.chunk_span(2) == (6, 10)


def test_forward_matches_exact_reference_for_two_segments_and_pad():
    torch.manual_seed(0)
    metadata = PackedAttentionMetadata.from_cu_seqlens([0, 5, 13])
    q = torch.randn(13, 4, 8, dtype=torch.float64)
    k = torch.randn(13, 4, 8, dtype=torch.float64)
    v = torch.randn(13, 4, 8, dtype=torch.float64)

    reference = exact_varlen_attention(q, k, v, metadata.cu_seqlens)
    cp_output = logical_ring_varlen_attention_all(q, k, v, metadata, world_size=2)
    _assert_close(cp_output, reference)


def test_pad_region_is_kept_as_an_independent_attention_segment():
    torch.manual_seed(12)
    metadata = PackedAttentionMetadata.from_cu_seqlens([0, 5, 13])
    q = torch.randn(13, 2, 4, dtype=torch.float64)
    k = torch.randn(13, 2, 4, dtype=torch.float64)
    v = torch.randn(13, 2, 4, dtype=torch.float64)

    reference = exact_varlen_attention(q, k, v, metadata.cu_seqlens)
    cp_output = logical_ring_varlen_attention_all(q, k, v, metadata, world_size=2)
    _assert_close(cp_output, reference)
    reference_pad = exact_varlen_attention(
        q[5:], k[5:], v[5:], [0, 8]
    )
    _assert_close(reference[5:], reference_pad)


def test_forward_matches_with_uneven_ring_shards():
    torch.manual_seed(1)
    metadata = PackedAttentionMetadata.from_cu_seqlens([0, 6, 13])
    q = torch.randn(13, 2, 5, dtype=torch.float64)
    k = torch.randn(13, 2, 5, dtype=torch.float64)
    v = torch.randn(13, 2, 5, dtype=torch.float64)

    reference = exact_varlen_attention(q, k, v, metadata.cu_seqlens)
    cp_output = logical_ring_varlen_attention_all(q, k, v, metadata, world_size=3)
    _assert_close(cp_output, reference)


def test_gradient_parity_between_exact_and_logical_cp():
    torch.manual_seed(2)
    metadata = PackedAttentionMetadata.from_cu_seqlens([0, 7, 15])
    q = torch.randn(15, 3, 6, dtype=torch.float64, requires_grad=True)
    k = torch.randn(15, 3, 6, dtype=torch.float64, requires_grad=True)
    v = torch.randn(15, 3, 6, dtype=torch.float64, requires_grad=True)

    reference = exact_varlen_attention(q, k, v, metadata.cu_seqlens)
    ref_grads = torch.autograd.grad(
        reference.sum(), (q, k, v), retain_graph=True
    )

    cp_output = logical_ring_varlen_attention_all(q, k, v, metadata, world_size=2)
    cp_grads = torch.autograd.grad(cp_output.sum(), (q, k, v))
    for actual, expected in zip(cp_grads, ref_grads):
        _assert_close(actual, expected)


def test_logical_ring_preserves_segment_isolation():
    torch.manual_seed(3)
    metadata = PackedAttentionMetadata.from_cu_seqlens([0, 3, 6])
    q = torch.randn(6, 2, 4, dtype=torch.float64)
    k = torch.randn(6, 2, 4, dtype=torch.float64)
    v = torch.randn(6, 2, 4, dtype=torch.float64)

    reference = exact_varlen_attention(q, k, v, metadata.cu_seqlens)
    cp_output = logical_ring_varlen_attention_all(q, k, v, metadata, world_size=2)
    _assert_close(cp_output, reference)


def test_rotate_kv_chunk_uses_group_peer(monkeypatch):
    calls = []

    class FakeRequest:
        def wait(self):
            pass

    def fake_p2p(op, tensor, group=None, tag=0, peer=None, group_peer=None):
        calls.append((op, tuple(tensor.shape), group, group_peer))
        return object()

    def fake_batch(ops):
        return [FakeRequest() for _ in ops]

    monkeypatch.setattr(dist, "P2POp", fake_p2p)
    monkeypatch.setattr(dist, "batch_isend_irecv", fake_batch)
    k = torch.randn(3, 2, 4)
    v = torch.randn_like(k)
    group = object()

    _rotate_kv_chunk(
        k,
        v,
        rank=1,
        world_size=3,
        group=group,
        receive_len=3,
    )

    assert [call[3] for call in calls] == [2, 2, 0, 0]
    assert all(call[2] is group for call in calls)


def test_checkpointed_logical_cp_matches_uncheckpointed():
    torch.manual_seed(4)
    metadata = PackedAttentionMetadata.from_cu_seqlens([0, 8, 16])
    q = torch.randn(16, 3, 6, dtype=torch.float64, requires_grad=True)
    k = torch.randn(16, 3, 6, dtype=torch.float64, requires_grad=True)
    v = torch.randn(16, 3, 6, dtype=torch.float64, requires_grad=True)

    def run(q, k, v):
        return logical_ring_varlen_attention_all(q, k, v, metadata, world_size=2)

    uncheckpointed = run(q, k, v)
    checkpointed = torch.utils.checkpoint.checkpoint(
        run, q, k, v, use_reentrant=False
    )
    _assert_close(checkpointed, uncheckpointed)

    uncheckpointed.sum().backward(retain_graph=True)
    cp_grads = [q.grad.clone(), k.grad.clone(), v.grad.clone()]
    q.grad = None
    k.grad = None
    v.grad = None
    checkpointed.sum().backward()
    for actual, expected in zip(
        [q.grad, k.grad, v.grad], cp_grads
    ):
        _assert_close(actual, expected)
