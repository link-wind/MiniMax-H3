#!/usr/bin/env python3
"""CPU-only checks for the long-horizon memory block (M1-fast, 方案 A).

Runs in seconds, loads **no weights and no media**, and verifies the three
layers that the training path touches:

1. ``MiniMaxH3Unit_PackedSequenceBuilder._build_packed_fl2va``: the memory block
   is a real slice of the packed sequence, it shifts every pre-existing row by
   exactly its own row count, it never enters ``img_pos``, and
   ``memory_latent_t=0`` reproduces the previous layout bit for bit.
2. ``model_fn_minimax_h3``: the memory rows land in ``x`` at ``mem_pos`` and are
   fed clean (``t = 1.0``).
3. ``MiniMaxH3DiT._embed``: memory rows go through ``video_patch_proj``, and the
   window's own rows keep identical embeddings.

Usage:
    PYTHONPATH="$PWD" python examples/minimax_h3/model_training/verify_memory_block.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT, patchify_video  # noqa: E402
from diffsynth.pipelines.minimax_h3_audio_video import (  # noqa: E402
    MiniMaxH3Unit_PackedSequenceBuilder,
    model_fn_minimax_h3,
)

# A 345-frame window at 480x832, matching the continuation pilot geometry.
WINDOW = dict(text_len=128, latent_t=102, latent_h=30, latent_w=52, audio_t=271, keyframe_indices=[])
MEMORY_STEPS = 12  # 39 frames = the M1-fast memory unit


def _builder() -> MiniMaxH3Unit_PackedSequenceBuilder:
    return MiniMaxH3Unit_PackedSequenceBuilder()


def check_packed_layout() -> None:
    builder = _builder()
    base = builder._build_packed_fl2va(**WINDOW)
    frame_rows = (WINDOW["latent_h"] // 2) * (WINDOW["latent_w"] // 2)

    for memory_steps in (MEMORY_STEPS, MEMORY_STEPS * 3):
        packed = builder._build_packed_fl2va(**WINDOW, memory_latent_t=memory_steps)
        rows = memory_steps * frame_rows
        used_delta = int(packed["cu_seqlens"][1]) - int(base["cu_seqlens"][1])
        assert used_delta == rows, (used_delta, rows)
        assert int(packed["seq_len"]) % 64 == 0
        assert torch.equal(packed["img_pos"] - rows, base["img_pos"])
        assert torch.equal(packed["audio_pos"] - rows, base["audio_pos"])
        assert packed["mem_pos"].numel() == rows
        assert set(packed["mem_pos"].tolist()) & set(packed["img_pos"].tolist()) == set()
        assert (packed["token_tags"][packed["mem_pos"]] == 0).all()
        grid = packed["img_position_ids"][0]
        gap = float(grid[packed["img_pos"][0], 0] - grid[packed["mem_pos"][-1], 0])
        assert 0 < gap < 7.0, gap  # one token span: the block sits flush
        print(f"  memory_steps={memory_steps:2d} rows={rows:5d} "
              f"used {int(base['cu_seqlens'][1])} -> {int(packed['cu_seqlens'][1])} "
              f"seq_len {int(base['seq_len'])} -> {int(packed['seq_len'])} gap={gap:.2f}")

    off = builder._build_packed_fl2va(**WINDOW, memory_latent_t=0)
    for key, value in base.items():
        other = off[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other), key
        else:
            assert value == other, key
    assert off["mem_pos"].numel() == 0

    window_span = builder._video_t_span(WINDOW["latent_t"])
    memory_span = builder._video_t_span(MEMORY_STEPS)
    print(f"  window span={window_span:.1f} memory span={memory_span:.1f} "
          f"relative distance={(window_span + memory_span) / window_span:.2f}x")
    print("  [1/3] packed layout + memory-off bit-identity OK")


class _DiTStub:
    """Records the keyword arguments ``model_fn_minimax_h3`` hands to the DiT."""

    def __init__(self) -> None:
        self.seen: dict = {}

    def __call__(self, **kwargs):
        self.seen = kwargs
        return (torch.zeros(kwargs["img_pos_info"]["position_ids"].numel(), 96),
                torch.zeros(kwargs["audio_pos_info"]["position_ids"].numel(), 32))


def check_model_fn() -> None:
    builder = _builder()
    frame_rows = (WINDOW["latent_h"] // 2) * (WINDOW["latent_w"] // 2)
    video = torch.randn(1, 24, WINDOW["latent_t"], WINDOW["latent_h"], WINDOW["latent_w"])
    audio = torch.randn(2, 32, WINDOW["audio_t"])

    seen = {}
    for name, memory_latents in (("off", None), ("on", torch.randn(1, 24, MEMORY_STEPS, 30, 52))):
        packed = builder._build_packed_fl2va(**WINDOW, memory_latent_t=0 if memory_latents is None else MEMORY_STEPS)
        dit = _DiTStub()
        model_fn_minimax_h3(
            dit, video_latents=video, audio_latents=audio, packed=packed, prompt_embeds=torch.randn(128, 64),
            timestep_video=torch.tensor(250.0), timestep_audio=torch.tensor(250.0),
            memory_latents=memory_latents,
        )
        seen[name] = dit.seen

    assert 1.0 not in seen["off"]["unique_timesteps"].tolist()
    info = seen["off"].get("mem_pos_info")
    assert info is None or info["position_ids"].numel() == 0

    kwargs = seen["on"]
    position_ids = kwargs["mem_pos_info"]["position_ids"]
    assert position_ids.numel() == MEMORY_STEPS * frame_rows
    written = kwargs["x"][0].view(-1, 96).index_select(0, position_ids)
    assert written.abs().max() > 0, "memory rows were not written into the packed input"
    assert kwargs["unique_timesteps"][-1].item() == 1.0, "memory rows are not fed clean"
    print(f"  memory rows={position_ids.numel()} written at mem_pos, timesteps={kwargs['unique_timesteps'].tolist()}")
    print("  [2/3] model_fn memory injection OK")


class _RefinerStub(torch.nn.Module):
    """The token refiner needs flash attention (CUDA-only); memory never uses it."""

    def forward(self, x, **kwargs):
        return x


def check_dit_embed() -> None:
    geometry = dict(WINDOW, text_len=8, latent_t=7, latent_h=4, latent_w=4, audio_t=6)
    frame_rows = (geometry["latent_h"] // 2) * (geometry["latent_w"] // 2)
    dit = MiniMaxH3DiT(
        num_layers=1, token_refiner_num_layers=1, hidden_size=64, num_attention_heads=2,
        attention_head_dim=16, ffn_hidden_size=128, latents_dim=24, audio_latents_dim=32,
        patch_size=(1, 2, 2), text_dim=48, timestep_input_dim=256, time_embed_hidden_size=64,
        time_embed_dim=32, adaln_out_features=6 * 64 * 3, final_adaln_out_features=2 * 64,
    ).eval()
    dit.token_refiner = _RefinerStub()
    builder = _builder()

    def embed(memory_steps):
        packed = builder._build_packed_fl2va(**geometry, memory_latent_t=memory_steps)
        seq_len = int(packed["seq_len"])
        x = torch.full((1, seq_len, 96), 0.25)
        memory = None
        if memory_steps:
            memory = torch.randn(1, 24, memory_steps, 4, 4)
            x[0].index_copy_(0, packed["mem_pos"], patchify_video(memory))
        timesteps = torch.full((seq_len,), 0.5)
        if memory_steps:
            timesteps[packed["mem_pos"]] = 1.0
        unique, inverse = torch.unique(timesteps, sorted=True, return_inverse=True)
        with torch.no_grad():
            embeddings, _ = dit._embed(
                x=x, audio_x=torch.full((1, seq_len, 32), 0.25),
                text_embeddings_selected=torch.randn(geometry["text_len"], 48),
                unique_timesteps=unique, img_pos=packed["img_pos"], audio_pos=packed["audio_pos"],
                text_pos=packed["text_pos"], text_embed_select=None,
                refiner_cu_seqlens=torch.tensor([0, geometry["text_len"], geometry["text_len"]]),
                refiner_max_seqlen=geometry["text_len"], seq_len=seq_len, device="cpu",
                mem_pos=None if not memory_steps else packed["mem_pos"],
            )
        return packed, embeddings, memory

    packed_off, embed_off, _ = embed(0)
    packed_on, embed_on, memory = embed(4)
    used_off = int(packed_off["cu_seqlens"][1])
    covered_off = (set(packed_off["text_pos"].tolist()) | set(packed_off["img_pos"].tolist())
                   | set(packed_off["audio_pos"].tolist()))
    assert covered_off == set(range(used_off)), "unexpected gap in the no-memory layout"
    used_on = int(packed_on["cu_seqlens"][1])
    covered_on = (set(packed_on["text_pos"].tolist()) | set(packed_on["img_pos"].tolist())
                  | set(packed_on["audio_pos"].tolist()) | set(packed_on["mem_pos"].tolist()))
    assert covered_on == set(range(used_on)), "gap in the memory layout"

    expected = dit.video_patch_proj(patchify_video(memory).to(torch.float32))
    assert torch.allclose(embed_on[packed_on["mem_pos"]], expected, atol=1e-5)
    assert (embed_on[packed_on["img_pos"]] - embed_off[packed_off["img_pos"]]).abs().max() < 1e-6
    assert (embed_on[packed_on["audio_pos"]] - embed_off[packed_off["audio_pos"]]).abs().max() < 1e-6
    print("  memory embeddings == video_patch_proj(patchified memory); window rows unchanged")
    print("  [3/3] DiT _embed memory branch OK")


LTM_LEAD_STEPS = 36  # constant anchor: the LTM slot never drifts with shot count


def check_memory_slots() -> None:
    """Dual-slot (STM + anchored LTM) and compressed-slot layouts."""
    builder = _builder()
    frame_rows = (WINDOW["latent_h"] // 2) * (WINDOW["latent_w"] // 2)
    stm = {"name": "stm", "tensor": torch.randn(1, 24, MEMORY_STEPS, 30, 52)}
    ltm = {"name": "ltm", "tensor": torch.randn(1, 24, MEMORY_STEPS, 30, 52), "lead_steps": LTM_LEAD_STEPS}

    packed = builder._build_packed_fl2va(**WINDOW, memory_slots=[stm, ltm])
    assert packed["mem_slot_names"] == ["stm", "ltm"]
    assert packed["mem_slot_slices"] == [
        (WINDOW["text_len"], WINDOW["text_len"] + MEMORY_STEPS * frame_rows),
        (WINDOW["text_len"] + MEMORY_STEPS * frame_rows, WINDOW["text_len"] + 2 * MEMORY_STEPS * frame_rows),
    ]
    grid = packed["img_position_ids"][0]
    window_t = float(grid[packed["img_pos"][0], 0])
    stm_start, stm_stop = packed["mem_slot_slices"][0]
    ltm_start, ltm_stop = packed["mem_slot_slices"][1]
    stm_t = float(grid[stm_start, 0])
    ltm_t = float(grid[ltm_start, 0])
    stm_end_t = float(grid[stm_stop - 1, 0])
    ltm_end_t = float(grid[ltm_stop - 1, 0])
    assert abs(stm_t - (window_t - builder._video_t_span(MEMORY_STEPS))) < 1e-9
    assert abs(ltm_t - (window_t - builder._video_t_span(LTM_LEAD_STEPS))) < 1e-9
    assert stm_end_t < window_t and ltm_end_t < stm_t, "slots are not disjoint from the window"
    assert int(packed["cu_seqlens"][1]) - int(builder._build_packed_fl2va(**WINDOW)["cu_seqlens"][1]) == 2 * MEMORY_STEPS * frame_rows
    print(f"  stm lead={window_t - stm_t:7.2f} ltm lead={window_t - ltm_t:7.2f} "
          f"slot gap={stm_t - ltm_end_t:5.2f} (constant, independent of shot count)")

    # A compressed slot carries m tokens (a compressor's output) at its own
    # anchor; its rows are read exactly like latent patch rows.
    compressed = {"name": "ltm-c", "kind": "rows", "tensor": torch.randn(8, 96), "lead_steps": LTM_LEAD_STEPS}
    packed_c = builder._build_packed_fl2va(**WINDOW, memory_slots=[stm, compressed])
    assert packed_c["mem_slot_slices"][1][1] - packed_c["mem_slot_slices"][1][0] == 8
    grid_c = packed_c["img_position_ids"][0]
    assert abs(float(grid_c[packed_c["mem_slot_slices"][1][0], 0]) - (float(grid_c[packed_c["img_pos"][0], 0]) - builder._video_t_span(LTM_LEAD_STEPS))) < 1e-9

    # Layout guards: a slot may not reach into the window and two slots may not
    # claim the same temporal position.
    for bad, message in (
        ([{"tensor": torch.randn(1, 24, MEMORY_STEPS, 30, 52), "lead_steps": MEMORY_STEPS - 1}], "into the window"),
        ([stm, {"tensor": torch.randn(1, 24, MEMORY_STEPS, 30, 52), "lead_steps": MEMORY_STEPS}], "overlap"),
    ):
        try:
            builder._build_packed_fl2va(**WINDOW, memory_slots=bad)
        except ValueError as error:
            assert message in str(error), error
        else:
            raise AssertionError(f"expected a ValueError mentioning {message!r}")
    print("  [4/4] dual-slot + compressed-slot + layout guards OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    torch.manual_seed(0)
    print("long-horizon memory block verification (CPU, no weights)")
    check_packed_layout()
    check_model_fn()
    check_dit_embed()
    check_memory_slots()
    print("all checks passed")


if __name__ == "__main__":
    main()
