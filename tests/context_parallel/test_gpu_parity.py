import os

os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"

import pytest
import torch

from diffsynth.core import attention as core_attention

core_attention.ATTENTION_IMPLEMENTATION = "torch"

from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT


def _make_model():
    return MiniMaxH3DiT(
        num_layers=1,
        token_refiner_num_layers=1,
        hidden_size=24,
        num_attention_heads=2,
        attention_head_dim=12,
        ffn_hidden_size=24,
        latents_dim=2,
        audio_latents_dim=4,
        patch_size=(1, 2, 2),
        text_dim=6,
        timestep_input_dim=4,
        time_embed_hidden_size=12,
        time_embed_dim=12,
        adaln_out_features=432,
        final_adaln_out_features=48,
        rope_inv_freq_len=2,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        final_norm_eps=1e-5,
    ).to(torch.bfloat16)


def _make_inputs(device):
    seq_len = 12
    text_len = 3
    img_pos = torch.tensor([3, 4, 5], dtype=torch.long, device=device)
    audio_pos = torch.tensor([6, 7], dtype=torch.long, device=device)
    text_pos = torch.arange(text_len, dtype=torch.long, device=device)
    cu = torch.tensor([0, 8, 12], dtype=torch.int32, device=device)
    token_tags = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    token_tags[text_pos] = 1
    token_tags[audio_pos] = 2
    token_tags[img_pos] = 0
    return {
        "x": torch.randn(1, seq_len, 8, device=device, dtype=torch.bfloat16),
        "audio_x": torch.randn(
            1, seq_len, 4, device=device, dtype=torch.bfloat16
        ),
        "img_position_ids": torch.randn(
            1, seq_len, 3, device=device, dtype=torch.bfloat16
        ),
        "unique_timesteps": torch.tensor(
            [0.2, 0.6], device=device, dtype=torch.bfloat16
        ),
        "inverse_indices": torch.tensor(
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], device=device
        ),
        "update_mask": None,
        "token_tags": token_tags,
        "prompt_embeds": torch.randn(
            text_len, 6, device=device, dtype=torch.bfloat16
        ),
        "img_pos_info": {"position_ids": img_pos},
        "audio_pos_info": {"position_ids": audio_pos},
        "text_pos_info": {"position_ids": text_pos},
        "img_pos_for_infer_output_info": {"position_ids": img_pos},
        "packed_seq_params": {
            "cu_seqlens_q": cu,
            "max_seqlen_q": 8,
        },
        "refiner_packed_seq_params": {
            "cu_seqlens_q": torch.tensor(
                [0, text_len, text_len], dtype=torch.int32, device=device
            ),
            "max_seqlen_q": text_len,
        },
        "skip_mask_out_condition": True,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_tiny_h3_cp1_cp2_forward_grad_and_loss():
    torch.manual_seed(5)
    model = _make_model().cuda()
    device = next(model.parameters()).device
    inputs = _make_inputs(device)
    video_ref, audio_ref = model(**inputs, cp_rank=0, cp_world_size=1)
    video_cp, audio_cp = model(**inputs, cp_rank=0, cp_world_size=2)
    torch.testing.assert_close(
        video_cp, video_ref, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        audio_cp, audio_ref, atol=2e-2, rtol=2e-2
    )

    target_video = torch.randn_like(video_ref)
    target_audio = torch.randn_like(audio_ref)
    loss_ref = torch.nn.functional.mse_loss(
        video_ref, target_video
    ) + torch.nn.functional.mse_loss(audio_ref, target_audio)
    loss_cp = torch.nn.functional.mse_loss(
        video_cp, target_video
    ) + torch.nn.functional.mse_loss(audio_cp, target_audio)
    torch.testing.assert_close(
        loss_cp, loss_ref, atol=2e-2, rtol=2e-2
    )

    grads_ref = torch.autograd.grad(
        loss_ref,
        [p for p in model.parameters() if p.requires_grad],
        retain_graph=True,
        allow_unused=True,
    )
    grads_cp = torch.autograd.grad(
        loss_cp,
        [p for p in model.parameters() if p.requires_grad],
        allow_unused=True,
    )
    for actual, expected in zip(grads_cp, grads_ref):
        if actual is not None or expected is not None:
            torch.testing.assert_close(
                actual, expected, atol=2e-2, rtol=2e-2
            )
