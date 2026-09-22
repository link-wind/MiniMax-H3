import os
import socket

os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffsynth.core import attention as core_attention

core_attention.ATTENTION_IMPLEMENTATION = "torch"

from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
from diffsynth.core.context_parallel import split_sequence_indices


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_model():
    torch.manual_seed(11)
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
    ).double()


def _make_global_inputs():
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


def _local_inputs(global_inputs, rank, world_size):
    inputs = dict(global_inputs)
    seq_len = int(global_inputs["x"].shape[1])
    local_start, local_end = split_sequence_indices(seq_len, world_size)[rank]
    inputs["packed_seq_params"] = dict(inputs["packed_seq_params"])
    inputs["packed_seq_params"]["seq_len"] = seq_len
    inputs["x"] = inputs["x"][:, local_start:local_end]
    inputs["audio_x"] = inputs["audio_x"][:, local_start:local_end]
    inputs["img_position_ids"] = inputs["img_position_ids"][
        :, local_start:local_end
    ]
    inputs["inverse_indices"] = inputs["inverse_indices"][local_start:local_end]
    inputs["token_tags"] = inputs["token_tags"][local_start:local_end]

    def local_positions(positions):
        positions = positions.view(-1).to(torch.long)
        mask = (positions >= local_start) & (positions < local_end)
        indices = torch.nonzero(mask).squeeze(1)
        return positions[mask] - local_start, indices

    local_img, img_indices = local_positions(
        inputs["img_pos_info"]["position_ids"]
    )
    local_audio, audio_indices = local_positions(
        inputs["audio_pos_info"]["position_ids"]
    )
    text_positions = inputs["text_pos_info"]["position_ids"].view(-1).to(torch.long)
    text_mask = (text_positions >= local_start) & (
        text_positions < local_end
    )
    local_text = text_positions[text_mask] - local_start
    global_text = text_positions[text_mask]
    inputs["img_pos_info"] = {"position_ids": local_img}
    inputs["audio_pos_info"] = {"position_ids": local_audio}
    inputs["text_pos_info"] = {"position_ids": local_text}
    inputs["img_pos_for_infer_output_info"] = {"position_ids": local_img}
    inputs["text_embed_select"] = global_text
    inputs["cp_rank"] = rank
    inputs["cp_world_size"] = world_size
    return inputs, img_indices, audio_indices


def _worker(rank, world_size, init_method):
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.new_group(list(range(world_size)))
    try:
        torch.manual_seed(10)
        global_inputs = _make_global_inputs()
        ref_model = _make_model()
        local_model = _make_model()

        ref_video, ref_audio = ref_model(
            **global_inputs,
            cp_rank=0,
            cp_world_size=1,
        )
        ref_loss = ref_video.sum() + ref_audio.sum()
        ref_grads = torch.autograd.grad(
            ref_loss,
            [param for param in ref_model.parameters() if param.requires_grad],
            retain_graph=True,
            allow_unused=True,
        )

        local_inputs, img_indices, audio_indices = _local_inputs(
            global_inputs, rank, world_size
        )
        torch.testing.assert_close(
            local_inputs["prompt_embeds"],
            global_inputs["prompt_embeds"],
        )
        torch.testing.assert_close(
            local_inputs["refiner_packed_seq_params"]["cu_seqlens_q"],
            global_inputs["refiner_packed_seq_params"]["cu_seqlens_q"],
        )
        local_video, local_audio_out = local_model(
            **local_inputs,
            cp_group=group,
        )
        torch.testing.assert_close(
            local_video,
            ref_video[img_indices],
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            local_audio_out,
            ref_audio[audio_indices],
            atol=1e-6,
            rtol=1e-6,
        )

        local_loss = local_video.sum() + local_audio_out.sum()
        local_grads = torch.autograd.grad(
            local_loss,
            [
                param
                for param in local_model.parameters()
                if param.requires_grad
            ],
            allow_unused=True,
        )
        for grad in local_grads:
            if grad is not None:
                dist.all_reduce(grad, group=group)

        for name, actual, expected in zip(
            [
                param_name
                for param_name, param in local_model.named_parameters()
                if param.requires_grad
            ],
            local_grads,
            ref_grads,
        ):
            if actual is None or expected is None:
                assert actual is expected
            else:
                torch.testing.assert_close(
                    actual, expected, atol=1e-6, rtol=1e-6
                )
    finally:
        dist.destroy_process_group()


def test_gloo_tiny_h3_local_shard_matches_global_reference():
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
