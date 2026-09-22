import pytest
import torch

from diffsynth.core.context_parallel import (
    PackedAttentionMetadata,
    exact_varlen_attention,
)
from diffsynth.core.context_parallel.ring_attention import (
    _DistributedFlashRingVarlenAttention,
    _can_use_flash,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)
def test_flash_blockwise_ring_forward_and_gradients():
    probe = torch.empty(
        1,
        1,
        8,
        device="cuda",
        dtype=torch.bfloat16,
    )
    if not _can_use_flash(probe):
        pytest.skip("flash_attn is not available in this environment")

    torch.manual_seed(7)
    metadata = PackedAttentionMetadata.from_cu_seqlens([0, 8, 12])
    q = torch.randn(
        12,
        4,
        16,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        12,
        4,
        16,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn(
        12,
        4,
        16,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    reference = exact_varlen_attention(q, k, v, metadata.cu_seqlens)
    ref_grads = torch.autograd.grad(
        reference.float().sum(),
        (q, k, v),
        retain_graph=True,
    )

    flash_out = _DistributedFlashRingVarlenAttention.apply(
        q,
        k,
        v,
        metadata,
        0,
        1,
        object(),
        q.shape[-1] ** -0.5,
        False,
    )
    torch.testing.assert_close(
        flash_out,
        reference,
        atol=2e-2,
        rtol=2e-2,
    )

    flash_grads = torch.autograd.grad(
        flash_out.float().sum(),
        (q, k, v),
    )
    for actual, expected in zip(flash_grads, ref_grads):
        torch.testing.assert_close(
            actual,
            expected,
            atol=2e-2,
            rtol=2e-2,
        )
