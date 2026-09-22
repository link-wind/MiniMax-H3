import os

os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"

import torch

from diffsynth.core import attention as core_attention

core_attention.ATTENTION_IMPLEMENTATION = "torch"

from diffsynth.models.minimax_h3_dit import (
    MiniMaxH3DiT,
    h3_shard_packed_fields,
)


def _make_tiny_h3():
    model = MiniMaxH3DiT(
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
    )
    return model.double()


def _make_tiny_h3_inputs():
    seq_len = 12
    text_len = 3
    img_pos = torch.tensor([3, 4, 5], dtype=torch.long)
    audio_pos = torch.tensor([6, 7], dtype=torch.long)
    text_pos = torch.arange(text_len, dtype=torch.long)
    cu = torch.tensor([0, 8, 12], dtype=torch.int32)
    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_pos] = 1
    token_tags[audio_pos] = 2
    token_tags[img_pos] = 0
    return {
        "x": torch.randn(1, seq_len, 8, dtype=torch.float64),
        "audio_x": torch.randn(1, seq_len, 4, dtype=torch.float64),
        "img_position_ids": torch.randn(1, seq_len, 3, dtype=torch.float64),
        "unique_timesteps": torch.tensor([0.2, 0.6], dtype=torch.float64),
        "inverse_indices": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        "update_mask": None,
        "token_tags": token_tags,
        "prompt_embeds": torch.randn(text_len, 6, dtype=torch.float64),
        "img_pos_info": {"position_ids": img_pos},
        "audio_pos_info": {"position_ids": audio_pos},
        "text_pos_info": {"position_ids": text_pos},
        "img_pos_for_infer_output_info": {"position_ids": img_pos},
        "packed_seq_params": {"cu_seqlens_q": cu, "max_seqlen_q": 8},
        "refiner_packed_seq_params": {
            "cu_seqlens_q": torch.tensor([0, text_len, text_len], dtype=torch.int32),
            "max_seqlen_q": text_len,
        },
        "skip_mask_out_condition": True,
    }


def test_tiny_h3_cp_forward_and_gradients_match():
    torch.manual_seed(5)
    model = _make_tiny_h3()
    inputs = _make_tiny_h3_inputs()
    target_video = torch.randn_like(model(**inputs, cp_rank=0, cp_world_size=1)[0])
    target_audio = torch.randn_like(model(**inputs, cp_rank=0, cp_world_size=1)[1])

    video_ref, audio_ref = model(**inputs, cp_rank=0, cp_world_size=1)
    video_cp, audio_cp = model(**inputs, cp_rank=0, cp_world_size=2)
    torch.testing.assert_close(video_cp, video_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(audio_cp, audio_ref, atol=1e-6, rtol=1e-6)

    loss_ref = torch.nn.functional.mse_loss(video_ref, target_video) + torch.nn.functional.mse_loss(
        audio_ref, target_audio
    )
    loss_cp = torch.nn.functional.mse_loss(video_cp, target_video) + torch.nn.functional.mse_loss(
        audio_cp, target_audio
    )
    torch.testing.assert_close(loss_cp, loss_ref, atol=1e-6, rtol=1e-6)

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
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_h3_packed_fields_map_to_local_shard():
    inputs = _make_tiny_h3_inputs()
    shard = h3_shard_packed_fields(
        local_start=4,
        local_end=10,
        inverse_indices=inputs["inverse_indices"],
        token_tags=inputs["token_tags"],
        img_position_ids=inputs["img_position_ids"],
        img_pos=inputs["img_pos_info"]["position_ids"],
        audio_pos=inputs["audio_pos_info"]["position_ids"],
        text_pos=inputs["text_pos_info"]["position_ids"],
        infer_out_pos=inputs["img_pos_for_infer_output_info"]["position_ids"],
    )
    assert shard["img_pos"].tolist() == [0, 1]
    assert shard["audio_pos"].tolist() == [2, 3]
    assert shard["text_pos"].tolist() == []
    assert shard["token_tags"].tolist() == [0, 0, 2, 2, -1, -1]
    assert shard["img_position_ids"].shape == (1, 6, 3)
