import ast
from pathlib import Path

import pytest
import torch

from diffsynth.pipelines.minimax_h3_diffvf import (
    DiffVFConfig,
    DiffVFWindow,
    DiffVFTimeline,
    check_decode_budget,
    derive_rng_seed,
    estimate_decode_memory_gib,
    fuse_predictions,
    fusion_coefficient,
    hni_initialize,
    inverse_scatter,
    sparse_windows,
    temporal_permutation,
    build_wws_weight_map,
    build_paper_tes_windows,
    build_tes_windows,
    fuse_states,
    merge_window_states,
    paper_fusion_coefficient,
    paper_hni_initialize,
    paper_hni_permutation,
    tes_timestep_active,
    validate_audio_tes_alignment,
    merge_window_predictions,
)
from diffsynth.pipelines.minimax_h3_continuation import merge_joint_multidiffusion_predictions


def _timeline():
    return DiffVFTimeline(
        10,
        (
            DiffVFWindow(0, 0, 6, 0, 2),
            DiffVFWindow(1, 4, 10, 4, 6),
        ),
    )


def test_diffvf_config_validates_and_serializes():
    config = DiffVFConfig(tes_enabled=True, tes_window_steps=4, tes_stride=2).validate()
    assert config.to_dict()["tes_stride"] == 2
    with pytest.raises(ValueError, match="hni_weight"):
        DiffVFConfig(hni_weight=1.1).validate()
    with pytest.raises(ValueError, match="requires tes_enabled"):
        DiffVFConfig(audio_tes="experimental").validate()


def test_hni_is_deterministic_and_overlap_uses_one_global_value():
    timeline = _timeline()
    first, streams_a = hni_initialize((2, 10), timeline.windows, hni_weight=0.5, seed=7, time_dim=1)
    second, streams_b = hni_initialize((2, 10), timeline.windows, hni_weight=0.5, seed=7, time_dim=1)
    assert torch.equal(first, second)
    assert streams_a == streams_b
    assert torch.equal(first[:, 4:6], first[:, 4:6])
    with pytest.raises(ValueError):
        hni_initialize((2, 10), timeline.windows, hni_weight=-0.1, seed=7, time_dim=1)


def test_paper_hni_uses_first_clip_noise_and_cyclic_groups():
    timeline = _timeline()
    output, streams = paper_hni_initialize(
        (1, 6), timeline.windows, innovation_weight=0.0, seed=7, time_dim=1,
    )
    base = torch.randn((1, 6), generator=torch.Generator().manual_seed(streams["base"]))
    torch.testing.assert_close(output[:, :6], base)
    torch.testing.assert_close(output[:, 6:], base[:, [3, 2, 5, 4]])
    assert paper_hni_permutation(6, 2, 1) == (1, 0, 3, 2, 5, 4)
    assert paper_hni_permutation(5, 3, 2) == (2, 0, 1, 3, 4)


def test_paper_hni_is_deterministic_and_validates_strict_controls():
    first, streams_a = paper_hni_initialize(
        (2, 6), _timeline().windows, innovation_weight=1.0, seed=7, time_dim=1,
    )
    second, streams_b = paper_hni_initialize(
        (2, 6), _timeline().windows, innovation_weight=1.0, seed=7, time_dim=1,
    )
    assert torch.equal(first, second)
    assert streams_a == streams_b
    base = torch.randn((2, 6), generator=torch.Generator().manual_seed(streams_a["base"]))
    torch.testing.assert_close(first[:, :6], base)
    assert "innovation_0" not in streams_a
    assert "innovation_1" in streams_a
    with pytest.raises(ValueError, match="paper-strict sampling requires"):
        DiffVFConfig(sampling_semantics="paper-strict").validate()
    DiffVFConfig(
        sampling_semantics="paper-strict", wws_weighting="center-distance", tes_enabled=True,
    ).validate()


def test_wws_weights_cover_and_normalize_overlap():
    weights, mass = build_wws_weight_map(_timeline(), strategy="cosine-squared")
    assert torch.all(mass > 0)
    assert torch.allclose(weights.sum(dim=0), torch.ones(10))
    assert torch.all(weights[:, :4].sum(dim=0) == 1)
    with pytest.raises(ValueError):
        build_wws_weight_map(_timeline(), strategy="bad")


def test_paper_center_distance_weights_and_state_fusion():
    weights, mass = build_wws_weight_map(_timeline(), strategy="center-distance")
    torch.testing.assert_close(weights.sum(dim=0), torch.ones(10))
    torch.testing.assert_close(weights[:, 4], torch.tensor([2 / 3, 1 / 3]))
    torch.testing.assert_close(weights[:, 5], torch.tensor([1 / 3, 2 / 3]))
    states = [torch.full((1, 1, 6), 10.0), torch.full((1, 1, 6), 20.0)]
    merged = merge_window_states(states, _timeline().windows, timeline_length=10, time_dim=2)
    torch.testing.assert_close(merged[0, 0, 4:6], torch.tensor([40 / 3, 50 / 3]))
    assert torch.all(mass > 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fixed-seed GPU equivalence smoke requires CUDA")
def test_wws_kernel_matches_joint_multidiffusion_on_cuda():
    """The cosine WWS kernel must preserve the established joint merge numerics."""
    generator = torch.Generator(device="cuda").manual_seed(1234)
    predictions = [
        torch.randn((1, 2, 6, 1, 1), generator=generator, device="cuda", dtype=torch.float32),
        torch.randn((1, 2, 6, 1, 1), generator=generator, device="cuda", dtype=torch.float32),
    ]
    timeline = DiffVFTimeline(
        10,
        (
            DiffVFWindow(0, 0, 6, 0, 0),
            DiffVFWindow(1, 4, 10, 4, 6),
        ),
    )
    merged_wws, _ = merge_window_predictions(
        predictions, timeline.windows, timeline_length=timeline.length,
        time_dim=2, strategy="cosine-squared",
    )
    merged_reference = merge_joint_multidiffusion_predictions(
        predictions, overlap_steps=2, time_dim=2,
    )
    torch.testing.assert_close(merged_wws, merged_reference, rtol=1e-5, atol=1e-6)


def test_tes_permutation_sparse_windows_and_inverse_scatter():
    permutation = temporal_permutation(10, 3)
    assert sorted(permutation) == list(range(10))
    chunks = sparse_windows(permutation, 4)
    assert sum(map(len, chunks)) == 10
    values = torch.tensor([[float(index) for index in permutation]])
    restored = inverse_scatter(values, permutation, time_dim=1)
    assert torch.equal(restored, torch.arange(10, dtype=torch.float32).reshape(1, -1))


def test_tes_windows_have_explicit_masks_and_cover_without_padding():
    windows = build_tes_windows(10, window_steps=4, stride=3)
    assert [len(window.source_indices) for window in windows] == [4, 4, 4]
    assert [window.valid_length for window in windows] == [4, 4, 2]
    assert windows[-1].mask == (True, True, False, False)
    assert sorted(index for window in windows for index in window.source_indices[:window.valid_length]) == list(range(10))
    assert tes_timestep_active(0, 9)
    assert not tes_timestep_active(8, 9)
    assert tes_timestep_active(8, 9, configured_timesteps=(8,))


def test_paper_tes_interleave_is_a_bijection_with_tail_padding():
    windows = build_paper_tes_windows(10, interleave_count=3, window_steps=4)
    assert windows[0].source_indices == (0, 3, 6, 9)
    assert windows[1].source_indices == (1, 4, 7, -1)
    assert windows[2].source_indices == (2, 5, 8, -1)
    assert [window.valid_length for window in windows] == [4, 3, 3]
    assert sorted(index for window in windows for index in window.source_indices[:window.valid_length]) == list(range(10))


def test_center_distance_weights_keep_the_bfloat16_tail_covered():
    windows = (
        DiffVFWindow(index=0, start=0, end=603, overlap_start=0, overlap_end=0),
        DiffVFWindow(index=1, start=546, end=1149, overlap_start=546, overlap_end=603),
    )
    weights, mass = build_wws_weight_map(
        DiffVFTimeline(length=1149, windows=windows),
        strategy="center-distance",
        dtype=torch.bfloat16,
    )
    assert mass[-1].item() > 0
    assert weights[:, -1].sum().item() == pytest.approx(1.0)


def test_audio_tes_requires_physical_time_alignment():
    assert validate_audio_tes_alignment(34, round(34 / 24 * 40)) == round(34 / 24 * 40)
    with pytest.raises(ValueError, match="not aligned"):
        validate_audio_tes_alignment(34, 1)


def test_fusion_missing_tes_and_schedule():
    wws = torch.ones(2, 3)
    assert torch.equal(fuse_predictions(wws, None, coefficient=0.2), wws)
    tes = torch.zeros(2, 3)
    assert torch.equal(fuse_predictions(wws, tes, coefficient=0.25), torch.full((2, 3), 0.25))
    assert fusion_coefficient(0, 5, local_start=0.2, local_end=0.8) == pytest.approx(0.2)
    assert fusion_coefficient(4, 5, local_start=0.2, local_end=0.8) == pytest.approx(0.8)


def test_paper_state_fusion_schedule_and_values():
    assert paper_fusion_coefficient(0, 5, alpha=0.5, power=6) == pytest.approx(0.5)
    assert paper_fusion_coefficient(4, 5, alpha=0.5, power=6) == pytest.approx(0.0)
    local = torch.ones(2, 3)
    global_state = torch.zeros(2, 3)
    assert torch.equal(fuse_states(local, None, global_coefficient=0.0), local)
    assert torch.equal(fuse_states(local, global_state, global_coefficient=0.25), torch.full((2, 3), 0.75))


def test_decode_budget_guard():
    estimate = estimate_decode_memory_gib((1, 4, 8, 8), bytes_per_element=2)
    assert check_decode_budget(estimate, DiffVFConfig(max_decode_memory_gib=1)) == "unified"
    assert check_decode_budget(10, DiffVFConfig(max_decode_memory_gib=1, decode_strategy="temporal-chunk")) == "temporal-chunk"
    with pytest.raises(MemoryError, match="exceeds budget"):
        check_decode_budget(10, DiffVFConfig(max_decode_memory_gib=1))


def test_rng_stream_does_not_use_python_hash_randomization():
    assert derive_rng_seed(1, "base") == derive_rng_seed(1, "base")
    assert derive_rng_seed(1, "base") != derive_rng_seed(1, "other")


def test_diffvf_evaluation_matrix_is_fixed_and_conservative_for_audio():
    source_path = Path(__file__).parents[1] / "examples/minimax_h3/model_inference/MiniMax-H3-DiffVF-Eval.py"
    module = ast.parse(source_path.read_text())
    variants = next(
        node for node in module.body
        if (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "VARIANTS" for target in node.targets
        )) or (
            isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "VARIANTS"
        )
    )
    source = ast.unparse(variants)
    assert "video-tes" in source
    assert "--audio-tes" not in source


def test_continuation_cli_keeps_parsed_tes_schedule_on_args():
    source_path = Path(__file__).parents[1] / "examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py"
    source = source_path.read_text()
    assert "args.tes_timesteps = tuple" in source
    assert "tes_timesteps=args.tes_timesteps" in source
    assert "tes_timesteps values must be smaller than --num-inference-steps" in source


def test_joint_pipeline_rejects_tes_timestep_outside_schedule():
    source_path = Path(__file__).parents[1] / "diffsynth/pipelines/minimax_h3_audio_video.py"
    source = source_path.read_text()
    assert "tes_timesteps values must be smaller than num_inference_steps" in source


def test_video_tes_reference_check_accepts_reference_free_fl2va():
    source_path = Path(__file__).parents[1] / "diffsynth/pipelines/minimax_h3_audio_video.py"
    source = source_path.read_text()
    assert 'inputs_shared.get("ref_blocks")' in source


def _rebuild_dense_wws_packed_for_test(video_start, audio_start, *, text_lengths=(3, 5), ref_blocks=None):
    try:
        from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline
    except ImportError as error:
        pytest.skip(f"MiniMax-H3 packing dependencies are unavailable: {error}")

    pipe = MiniMaxH3Pipeline(device="cpu", torch_dtype=torch.float32)
    shared = {
        "video_latents": torch.zeros(1, 24, 6, 4, 4),
        "audio_latents": torch.zeros(1, 2, 10),
        "keyframe_cond_anchor": None,
        "keyframe_indices": None,
        "ref_blocks": ref_blocks,
    }
    conditions = []
    for text_length in text_lengths:
        conditions.append({
            "prompt_embeds": torch.zeros(text_length, 8),
            "text_token_tags": torch.ones(text_length, dtype=torch.long),
        })
    pipe._rebuild_dense_wws_packed(
        shared, conditions[0], conditions[1],
        video_start=video_start, audio_start=audio_start,
        total_video_steps=10, total_audio_steps=18,
    )
    return conditions


def test_dense_wws_packed_uses_global_offsets_and_agrees_on_overlap():
    first_posi, first_nega = _rebuild_dense_wws_packed_for_test(0, 0)
    second_posi, second_nega = _rebuild_dense_wws_packed_for_test(4, 8)

    first = first_posi["packed"]
    second = second_posi["packed"]
    frame_rows = first["img_pos"].numel() // 6
    first_overlap = first["img_position_ids"][0, first["img_pos"][4 * frame_rows:5 * frame_rows], 0]
    second_overlap = second["img_position_ids"][0, second["img_pos"][:frame_rows], 0]
    torch.testing.assert_close(first_overlap, second_overlap)

    first_audio = first["img_position_ids"][0, first["audio_pos"][:2], 0]
    second_audio = second["img_position_ids"][0, second["audio_pos"][:2], 0]
    torch.testing.assert_close(
        second_audio - second["text_pos"].numel(),
        torch.tensor([8.0, 9.0], dtype=second_audio.dtype),
    )
    assert torch.all(second_audio > first_audio)

    assert first_posi["packed"] is not first_nega["packed"]
    assert second_posi["packed"] is not second_nega["packed"]
    assert second_posi["packed"]["text_pos"].numel() == 3
    assert second_nega["packed"]["text_pos"].numel() == 5


def test_dense_wws_reference_blocks_reject_nonzero_global_offset():
    with pytest.raises(ValueError, match="absolute WWS positions.*reference blocks"):
        _rebuild_dense_wws_packed_for_test(4, 8, ref_blocks=[{"kind": "image"}])
