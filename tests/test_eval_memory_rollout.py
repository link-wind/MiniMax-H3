"""CPU-checkable pieces of the rollout memory ablation.

The ablation itself needs CUDA and a 14B checkpoint, but its decision logic --
which prompt text becomes the identity-free condition, how a donor slot is
matched by geometry, and how cross-shot drift is measured -- is exactly the part
that silently produces a wrong conclusion if it is off by one window.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from diffsynth.pipelines.minimax_h3_continuation import (
    ContinuationState,
    H3ContinuationConfig,
    H3Segment,
    H3SegmentPlan,
    H3WindowStateRecord,
    ResolvedContinuationWindow,
)
from diffsynth.utils.continuation_lora import CACHE_SCHEMA_VERSION


def _load_module():
    path = Path(__file__).resolve().parents[1] / "examples/minimax_h3/model_training/eval_memory_rollout.py"
    spec = importlib.util.spec_from_file_location("h3_eval_memory_rollout_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rollout = _load_module()


FULL_PROMPT = """<SUBJECT>: [subject1] (Real Person) is a man in a black suit.
[subject2] is a young man with a white bandage.
<Scene>: An indoor office with dark green curtains.
<Event>: [subject1] looks intently at [subject2]."""


def test_no_subject_mode_keeps_scene_and_event_only():
    stripped = rollout._strip_subject(FULL_PROMPT)
    assert "black suit" not in stripped
    assert "white bandage" not in stripped
    assert stripped.startswith("<Scene>:")
    assert "<Event>:" in stripped


def test_prompted_plan_strips_both_global_and_segment_prompts():
    plan = H3SegmentPlan(
        global_prompt=FULL_PROMPT,
        segments=(H3Segment(prompt="<SUBJECT>: a third person."), H3Segment()),
    )
    prompted = rollout._prompted_plan(plan, "no-subject")
    assert "black suit" not in prompted.global_prompt
    assert prompted.segments[0].prompt == ""
    # The full mode has to be a no-op, otherwise the two conditions differ by more
    # than the thing under test.
    assert rollout._prompted_plan(plan, "full") is plan


def test_prompted_plan_refuses_to_erase_the_whole_condition():
    with pytest.raises(ValueError, match="entire global prompt"):
        rollout._prompted_plan(
            H3SegmentPlan(global_prompt="<SUBJECT>: only a person.", segments=(H3Segment(),)),
            "no-subject",
        )


def _window(index, start, end):
    overlap = 0 if index == 0 else 39
    return ResolvedContinuationWindow(
        segment_index=index, segment_id=f"s{index}", requested_video_frames=end - start,
        resolved_video_frames=end - start, overlap_video_frames=overlap,
        timeline_start_video_frame=start, timeline_end_video_frame=end,
        new_video_frames=end - start - overlap, video_fps=24, audio_sample_rate=32_000,
        audio_latent_rate=40, resolved_audio_samples=0, overlap_audio_samples=0,
        resolved_audio_latents=0, overlap_audio_latents=0, video_latent_steps=0,
    )


class _Result:
    def __init__(self, video, records):
        self.video = video
        self.audio = None
        self.state = ContinuationState(
            plan_id="p", global_prompt="g",
            config=H3ContinuationConfig(requested_window_frames=345, overlap_frames=39),
            base_seed=0, records=records,
        )


def _flat_frame(value):
    return Image.new("RGB", (2, 2), (value, value, value))


def test_head_drift_is_measured_on_generated_frames_only():
    """Drift must come from frames a window produced, not from inherited overlap.

    Window 1 spans [61, 200) but inherits [61, 100) from window 0, so its
    generated range starts at 100.  Measuring the inherited frames would compare
    the head against window 0 twice and report a drift of zero for every model.
    """
    video = [_flat_frame(0)] * 100 + [_flat_frame(60)] * 100
    records = [
        H3WindowStateRecord(window=_window(0, 0, 100), seed=0, prompt="p"),
        H3WindowStateRecord(window=_window(1, 61, 200), seed=1, prompt="p"),
    ]
    report = rollout._memory_consistency_report(_Result(video, records))
    assert report["available"]
    assert report["windows"][0]["head_drift"] is None
    assert report["windows"][1]["generated_index"] == 100
    assert report["windows"][1]["generated_frames"] == 100
    assert report["windows"][1]["head_drift"] == pytest.approx(60.0)
    assert report["head_drift_mean"] == pytest.approx(60.0)
    assert report["head_drift_last"] == pytest.approx(60.0)
    # A static shot moves by zero, which is the scale the drift must be read against.
    assert report["motion_scale_mean"] == pytest.approx(0.0)


def test_memory_report_degrades_gracefully_without_decoded_video():
    report = rollout._memory_consistency_report(_Result(None, []))
    assert report == {"available": False, "reason": "no decoded video"}


def test_divergence_is_zero_for_identical_runs_and_positive_for_shifted_ones():
    frames = [_flat_frame(index % 60) for index in range(50)]
    other = [_flat_frame(index % 60) for index in range(50)]
    shifted = [_flat_frame((index + 10) % 60) for index in range(50)]
    assert rollout._divergence(_Result(frames, []), _Result(other, [])) == pytest.approx(0.0)
    assert rollout._divergence(_Result(frames, []), _Result(shifted, [])) > 0
    assert rollout._divergence(_Result(frames, []), _Result(None, [])) is None


def _donor_cache(root, *, shape, count, schema=CACHE_SCHEMA_VERSION):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        payload = {
            "video_latents": torch.zeros(1, 24, 12, shape[3], shape[4]),
            "audio_latents": None,
            "memory_latents": None,
            "memory_slots": [torch.full(shape, float(index))],
            "metadata": {
                "schema_version": schema, "sample_id": f"donor{index}",
                "video_shape": [1, 24, 12, shape[3], shape[4]], "video_dtype": str(torch.float32),
                "video_fps": 24, "audio_sample_rate": 32_000, "audio_latent_rate": 40,
                "window_frames": 345, "overlap_frames": 39,
            },
        }
        torch.save(payload, root / f"donor{index}.pt")
    return root


def test_donor_slots_are_selected_by_latent_geometry(tmp_path):
    cache = _donor_cache(tmp_path, shape=(1, 24, 12, 2, 2), count=3)
    slots = rollout._load_donor_slots(cache, window_count=2, reference_shape=(1, 24, 12, 2, 2))
    assert len(slots) == 2
    assert {float(slot.mean()) for slot in slots} == {0.0, 1.0}


def test_donor_cache_with_the_wrong_geometry_is_rejected_loudly(tmp_path):
    """A slot is a raw latent block; transplanting it across resolutions is a bug."""
    cache = _donor_cache(tmp_path, shape=(1, 24, 12, 2, 2), count=2)
    with pytest.raises(ValueError, match="no usable slot matching"):
        rollout._load_donor_slots(cache, window_count=2, reference_shape=(1, 24, 12, 3, 3))


def test_empty_donor_cache_is_rejected(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="contains no"):
        rollout._load_donor_slots(tmp_path / "empty", window_count=1, reference_shape=(1, 24, 12, 2, 2))


def test_sharded_donor_layout_is_read_from_the_requested_split(tmp_path):
    """The cache on disk is ``shard_N/<split>/*.pt``; the train split must not leak in."""
    _donor_cache(tmp_path / "shard_0" / "validation", shape=(1, 24, 12, 2, 2), count=2)
    _donor_cache(tmp_path / "shard_1" / "train", shape=(1, 24, 12, 2, 2), count=2)
    entries = rollout._donor_entries(tmp_path, "validation")
    assert [entry.parent.name for entry in entries] == ["validation", "validation"]
    assert all(entry.parent.parent.name == "shard_0" for entry in entries)
    slots = rollout._load_donor_slots(tmp_path, window_count=2)
    assert len(slots) == 2


def test_flat_donor_directory_still_works(tmp_path):
    _donor_cache(tmp_path, shape=(1, 24, 12, 2, 2), count=2)
    assert len(rollout._load_donor_slots(tmp_path, window_count=2)) == 2


def test_donor_shape_can_be_inferred_when_no_reference_arm_ran(tmp_path):
    """The swap arm has to be launchable on its own GPU, without a 'both' arm."""
    cache = _donor_cache(tmp_path, shape=(1, 24, 12, 2, 2), count=3)
    slots = rollout._load_donor_slots(cache, window_count=3)
    assert len(slots) == 3
    assert {tuple(slot.shape) for slot in slots} == {(1, 24, 12, 2, 2)}


def test_donor_cache_mixing_latent_shapes_is_rejected(tmp_path):
    """A mix would silently condition different windows on different geometries."""
    _donor_cache(tmp_path / "a", shape=(1, 24, 12, 2, 2), count=1)
    _donor_cache(tmp_path / "b", shape=(1, 24, 12, 3, 3), count=1)
    with pytest.raises(ValueError, match="mixes latent shapes"):
        rollout._load_donor_slots(tmp_path, window_count=2)


def test_replay_encoder_cycles_donors_and_clones_them():
    donors = [torch.ones(1, 24, 12, 2, 2), torch.full((1, 24, 12, 2, 2), 2.0)]
    encoder = rollout._ReplaySlotEncoder(donors)
    first = encoder([], height=480, width=832)
    second = encoder([], height=480, width=832)
    third = encoder([], height=480, width=832)
    assert float(first.mean()) == 1.0 and float(second.mean()) == 2.0
    assert float(third.mean()) == 1.0
    # A donor that leaks the caller's mutation back into the pool would make the
    # swap arm depend on how many times a slot happens to be requested.
    first.add_(100)
    assert float(donors[0].mean()) == 1.0


def test_replay_encoder_rejects_an_empty_donor_pool():
    with pytest.raises(ValueError, match="at least one donor"):
        rollout._ReplaySlotEncoder([])


def test_slot_observer_records_names_and_copies():
    observer = rollout._SlotObserver()
    observer("stm", torch.zeros(1, 24, 12, 2, 2))
    observer("ltm", torch.ones(1, 24, 12, 2, 2))
    assert [name for name, _tensor in observer.calls] == ["stm", "ltm"]
    assert observer.shapes() == [(1, 24, 12, 2, 2), (1, 24, 12, 2, 2)]
    # The recording has to be a copy, not a view: the donor pool is reused for
    # every later window, so a downstream mutation must not reach it.
    observer.calls[0][1].add_(50)
    assert float(observer.calls[0][1].mean()) == 50.0


def test_arm_table_matches_the_teacher_forced_experiment():
    """The two ablations must vary the same axes, or their results cannot be read together."""
    assert set(rollout.ARMS) == {"both", "stm", "ltm", "none", "swap", "base"}
    assert rollout.ARMS["none"] == {"stm": False, "ltm": False, "swap": False, "lora_scale": 1.0}
    assert rollout.ARMS["base"]["lora_scale"] == 0.0
    assert rollout.ARMS["swap"]["stm"] and rollout.ARMS["swap"]["ltm"] and rollout.ARMS["swap"]["swap"]
    # Every LoRA arm must keep the scale at 1.0 so 'base' is the only arm that
    # changes the weights.
    for name, arm in rollout.ARMS.items():
        if name != "base":
            assert arm["lora_scale"] == 1.0


def test_summary_reports_divergence_for_every_arm_pair():
    payload = {
        "both": {"result": _Result([_flat_frame(0)] * 20, []), "joins": {"joins": []}, "memory": {"available": True}},
        "none": {"result": _Result([_flat_frame(0)] * 20, []), "joins": {"joins": []}, "memory": {"available": True}},
    }
    summary = rollout._summarize(payload)
    assert summary["divergence"] == {"both_vs_none": pytest.approx(0.0)}
    assert set(summary["arms"]) == {"both", "none"}


def test_join_mean_ignores_unavailable_measurements():
    joins = {"joins": [
        {"video_mean_absolute_difference": 0.2},
        {"video_mean_absolute_difference": "unavailable"},
        {"video_mean_absolute_difference": 0.4},
    ]}
    assert rollout._join_mean(joins, "video_mean_absolute_difference") == pytest.approx(0.3)
    assert rollout._join_mean({"joins": []}, "video_mean_absolute_difference") is None


def test_plan_can_override_the_cli_overlap():
    plan = H3SegmentPlan(
        global_prompt="g", segments=(H3Segment(),), global_controls={"overlap_frames": 90},
    )
    assert rollout.manifest_overlap(plan) == 90
    assert rollout.manifest_overlap(H3SegmentPlan(global_prompt="g", segments=(H3Segment(),))) is None


def test_dry_run_writes_nothing(tmp_path, capsys, monkeypatch):
    argv = [
        "--dry-run", "--segment-plan", str(tmp_path / "p.json"), "--checkpoint", "ckpt",
        "--h3-base", str(tmp_path), "--lora", str(tmp_path), "--arms", "both,none",
        "--output-dir", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", ["eval_memory_rollout.py", *argv])
    rollout.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["arms"] == ["both", "none"]
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- #
# Launcher contract
#
# The launcher's job beyond "call python" is to hand the loader a shard *set*.
# A sharded transformer is one model, and the pattern must reach python already
# expanded: passing the literal ``model*.safetensors`` makes the loader try to
# open that string as a path, which fails only after the text encoder has
# finished loading.
# --------------------------------------------------------------------------- #

LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "examples/minimax_h3/model_training/run_memory_rollout_eval_16gpu.sh"
)


def _fake_venv(tmp_path):
    """A venv whose ``python`` records argv instead of loading a 14B model."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    # argv goes to a shared file (the single-arm test reads it); the GPU goes to
    # stdout, which the launcher redirects per arm, so concurrent arms cannot
    # overwrite each other's record.
    python.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$@" > "${FAKE_PYTHON_ARGS_OUT}"\n'
        'echo "gpu=${CUDA_VISIBLE_DEVICES:-unset}"\n'
    )
    python.chmod(0o755)
    return venv


def _launcher_fixture(tmp_path):
    ckpt_dir = tmp_path / "transformer"
    ckpt_dir.mkdir()
    for index in (1, 2):
        (ckpt_dir / f"model-0000{index}-of-00002.safetensors").write_bytes(b"")
    h3_base = tmp_path / "h3base"
    h3_base.mkdir()
    lora = tmp_path / "step-375.safetensors"
    lora.write_bytes(b"")
    plan = tmp_path / "plan.json"
    plan.write_text('{"global_prompt": "<SUBJECT>: a man.", "segments": [{"requested_frames": 345}]}')
    donor = tmp_path / "donors"
    (donor / "shard_0" / "validation").mkdir(parents=True)
    return {
        "H3_VENV": str(_fake_venv(tmp_path)),
        "CHECKPOINT": str(ckpt_dir / "model*.safetensors"),
        "H3_BASE": str(h3_base),
        "LORA_CHECKPOINT": str(lora),
        "SEGMENT_PLAN": str(plan),
        "DONOR_CACHE": str(donor),
        "OUTPUT_DIR": str(tmp_path / "out"),
        "FAKE_PYTHON_ARGS_OUT": str(tmp_path / "argv.txt"),
    }


def _run_launcher(tmp_path, env):
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("GEMINI_")}
    base_env.update(env)
    base_env["ROLLOUT_ARMS"] = env.get("ROLLOUT_ARMS", "both")
    completed = subprocess.run(
        ["bash", str(LAUNCHER)], env=base_env, capture_output=True, text=True
    )
    argv_path = Path(env["FAKE_PYTHON_ARGS_OUT"])
    argv = argv_path.read_text().splitlines() if argv_path.exists() else []
    return completed, argv


def test_launcher_expands_the_checkpoint_pattern_before_calling_python(tmp_path):
    """The loader must receive files, never the glob string it cannot open."""
    env = _launcher_fixture(tmp_path)
    completed, argv = _run_launcher(tmp_path, env)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--checkpoint" in argv
    checkpoint = argv[argv.index("--checkpoint") + 1:]
    checkpoint = checkpoint[: next((i for i, item in enumerate(checkpoint) if item.startswith("--")), len(checkpoint))]
    assert checkpoint == [
        str(Path(env["CHECKPOINT"]).parent / "model-00001-of-00002.safetensors"),
        str(Path(env["CHECKPOINT"]).parent / "model-00002-of-00002.safetensors"),
    ]
    assert not any("*" in item for item in argv)


def test_launcher_fails_before_loading_when_no_shard_matches(tmp_path):
    env = _launcher_fixture(tmp_path)
    env["CHECKPOINT"] = str(tmp_path / "missing" / "model*.safetensors")
    completed, argv = _run_launcher(tmp_path, env)
    assert completed.returncode != 0
    assert "no H3 transformer checkpoint matched" in completed.stderr
    assert argv == []


def test_launcher_puts_each_arm_on_its_own_gpu(tmp_path):
    env = _launcher_fixture(tmp_path)
    env["ROLLOUT_ARMS"] = "both,stm,ltm"
    completed, _argv = _run_launcher(tmp_path, env)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    gpus = [
        (Path(env["OUTPUT_DIR"]) / arm / "run.log").read_text().strip()
        for arm in ("both", "stm", "ltm")
    ]
    assert gpus == ["gpu=0", "gpu=1", "gpu=2"]
    # Each arm writes into its own directory, so a partial failure stays readable.
    for arm in ("both", "stm", "ltm"):
        assert (Path(env["OUTPUT_DIR"]) / arm).is_dir()


def test_launcher_rejects_a_missing_donor_cache_for_the_swap_arm(tmp_path):
    env = _launcher_fixture(tmp_path)
    env["ROLLOUT_ARMS"] = "swap"
    env["DONOR_CACHE"] = str(tmp_path / "no-such-donor-cache")
    completed, argv = _run_launcher(tmp_path, env)
    assert completed.returncode != 0
    assert "donor cache not found" in completed.stderr
    assert argv == []


def _stub_cuda(monkeypatch, *, allocated_gib=41.5, reserved_gib=47.0, free_gib=30.0):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: allocated_gib * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: reserved_gib * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (free_gib * 1024 ** 3, 80 * 1024 ** 3))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)


def test_window_progress_records_shape_and_previous_peak(tmp_path, monkeypatch):
    """A rollout that dies in the DiT must still leave its window geometry behind."""
    _stub_cuda(monkeypatch)
    progress = rollout._WindowProgress(tmp_path / "progress_both.jsonl")
    progress(window=_window(0, 0, 345), seed=0, controls={})
    record = json.loads((tmp_path / "progress_both.jsonl").read_text().splitlines()[0])
    assert record["event"] == "window_start"
    assert record["resolved_video_frames"] == 345
    assert record["overlap_video_frames"] == 0
    assert record["previous_peak_allocated_gib"] == 41.5
    assert record["free_gib_at_start"] == 30.0


def test_window_progress_keeps_the_high_water_mark_across_windows(tmp_path, monkeypatch):
    """Per-window peaks are reset, but the reported maximum must not shrink."""
    peaks = iter([20.0, 55.0])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: next(peaks) * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0.0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1.0, 80 * 1024 ** 3))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    progress = rollout._WindowProgress(tmp_path / "progress.jsonl")
    progress(window=_window(0, 0, 345), seed=0, controls={})
    progress(window=_window(1, 306, 651), seed=1, controls={})
    # finish() consumes the last peak, which is the smaller of the two.
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 5.0 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0.0)
    done = progress.finish(video_frames=651)
    assert done["peak_allocated_gib"] == 55.0
    assert done["decoded_video_frames"] == 651
    assert len((tmp_path / "progress.jsonl").read_text().splitlines()) == 3


def test_summary_surfaces_peak_memory_when_available():
    payload = {
        "both": {
            "result": _Result([_flat_frame(0)] * 20, []),
            "joins": {"joins": []},
            "memory": {"available": True},
            "progress": {"peak_allocated_gib": 61.5, "peak_reserved_gib": 70.0},
        },
    }
    summary = rollout._summarize(payload)
    assert summary["arms"]["both"]["peak_allocated_gib"] == 61.5
    assert summary["arms"]["both"]["peak_reserved_gib"] == 70.0


# --------------------------------------------------------------------------- #
# Resolution / step forwarding
#
# The pipeline carries its own default resolution and step count, and neither is
# the training-time one.  A rollout that forgets to forward them allocates a much
# larger sequence *and* breaks the memory-slot spatial check, because a slot is a
# raw latent block that cannot cross grids.  Both regressions look like unrelated
# failures (OOM, shape mismatch), so the forwarding itself is pinned here.
# --------------------------------------------------------------------------- #


def test_dry_run_reports_the_pipeline_kwargs_it_will_forward(tmp_path, capsys, monkeypatch):
    argv = [
        "--dry-run", "--segment-plan", str(tmp_path / "p.json"), "--checkpoint", "ckpt",
        "--h3-base", str(tmp_path), "--lora", str(tmp_path), "--arms", "both",
        "--height", "480", "--width", "832", "--num-inference-steps", "8",
        "--seed", "7", "--output-dir", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", ["eval_memory_rollout.py", *argv])
    rollout.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["pipeline_kwargs"] == {
        "height": 480, "width": 832, "num_inference_steps": 8, "cfg_scale": 1.0,
    }
    assert payload["seed"] == 7


def test_run_arm_forwards_resolution_steps_and_seed(monkeypatch):
    seen = {}

    class _Runner:
        def run(self, plan, *, base_seed=42, pipeline_kwargs=None, state=None):
            seen["base_seed"] = base_seed
            seen["pipeline_kwargs"] = dict(pipeline_kwargs or {})
            return _Result([], [])

    monkeypatch.setattr(rollout, "evaluate_continuation_joins", lambda result: {})
    monkeypatch.setattr(rollout, "_memory_consistency_report", lambda result: {})

    kwargs = {"height": 480, "width": 832, "num_inference_steps": 8, "cfg_scale": 1.0}
    entry = rollout._run_arm(
        _Runner(), object(), pipeline_kwargs=kwargs, base_seed=7,
    )
    assert seen["pipeline_kwargs"] == kwargs
    assert seen["base_seed"] == 7
    assert "result" in entry


def test_summary_path_is_unique_per_arm_subset(tmp_path):
    """Two nodes share one output directory; one shared name loses an arm set."""
    first = rollout._summary_path(tmp_path, ["both", "stm", "ltm"])
    second = rollout._summary_path(tmp_path, ["none", "swap", "base"])
    assert first != second
    assert first.name == "rollout_ablation_both-stm-ltm.json"
    assert second.name == "rollout_ablation_none-swap-base.json"


def test_save_arm_video_writes_a_named_file(tmp_path, monkeypatch):
    import diffsynth.utils.data as data_utils

    written = {}
    monkeypatch.setattr(
        data_utils, "save_video",
        lambda frames, path, fps, **kwargs: written.update(
            frames=list(frames), path=path, fps=fps,
        ),
    )

    class _Result:
        video = ["a", "b", "c"]

    saved = rollout._save_arm_video(_Result(), tmp_path, "both", fps=24)
    assert saved == str(tmp_path / "videos" / "both.mp4")
    assert written["fps"] == 24
    assert written["frames"] == ["a", "b", "c"]


def test_save_arm_video_is_a_noop_without_frames(tmp_path):
    class _Result:
        video = None

    assert rollout._save_arm_video(_Result(), tmp_path, "both", fps=24) is None
    assert not (tmp_path / "videos").exists()


def test_summary_surfaces_the_saved_video_path():
    payload = {
        "both": {
            "result": _Result([], []),
            "joins": {"joins": []},
            "memory": {"available": True},
            "video_path": "/tmp/videos/both.mp4",
        },
    }
    summary = rollout._summarize(payload)
    assert summary["arms"]["both"]["video_path"] == "/tmp/videos/both.mp4"


# --------------------------------------------------------------------------- #
# LoRA comparison arms (--lora-arms)
#
# Comparing two training runs is only meaningful if the inference-time
# conditioning is identical across arms and the only thing that varies is the
# checkpoint.  These tests pin that: memory is forced off for every arm, each arm
# carries its own checkpoint, and the pipeline cache is keyed by the checkpoint
# rather than by the scale (several arms commonly share scale 1.0).
# --------------------------------------------------------------------------- #


def _write_lora(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"stub")
    return path


def test_parse_lora_arms_reads_name_path_and_scale(tmp_path):
    v5 = _write_lora(tmp_path, "v5.safetensors")
    table = rollout._parse_lora_arms(f"base=,v5={v5},v9={v5}@0.75")
    assert list(table) == ["base", "v5", "v9"]
    assert table["base"]["lora_path"] is None
    assert table["v5"]["lora_path"] == str(v5.resolve())
    assert table["v5"]["lora_scale"] == 1.0
    assert table["v9"]["lora_scale"] == 0.75


def test_parse_lora_arms_turns_memory_off_for_every_arm(tmp_path):
    """The comparison isolates the checkpoint, so nothing else may be conditioned."""
    v5 = _write_lora(tmp_path, "v5.safetensors")
    table = rollout._parse_lora_arms(f"base=,v5={v5}")
    for arm in table.values():
        assert arm["stm"] is False
        assert arm["ltm"] is False
        assert arm["swap"] is False


def test_parse_lora_arms_rejects_a_missing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="no such LoRA file"):
        rollout._parse_lora_arms(f"v5={tmp_path / 'absent.safetensors'}")


def test_parse_lora_arms_rejects_duplicates_and_bad_syntax(tmp_path):
    v5 = _write_lora(tmp_path, "v5.safetensors")
    with pytest.raises(ValueError, match="repeats arm"):
        rollout._parse_lora_arms(f"v5={v5},v5={v5}")
    with pytest.raises(ValueError, match="NAME=PATH"):
        rollout._parse_lora_arms("v5")
    with pytest.raises(ValueError, match="names no arms"):
        rollout._parse_lora_arms(" , ")


def _dry_run_argv(tmp_path, extra):
    return [
        "--dry-run", "--segment-plan", str(tmp_path / "p.json"), "--checkpoint", "ckpt",
        "--h3-base", str(tmp_path), *extra, "--output-dir", str(tmp_path / "out"),
    ]


def test_dry_run_reports_the_lora_arm_table(tmp_path, capsys, monkeypatch):
    v5 = _write_lora(tmp_path, "v5.safetensors")
    monkeypatch.setattr(sys, "argv", [
        "eval_memory_rollout.py",
        *_dry_run_argv(tmp_path, ["--lora-arms", f"base=,v5={v5}"]),
    ])
    rollout.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["arms"] == ["base", "v5"]
    assert payload["lora_arms"]["base"]["lora_path"] is None
    assert payload["lora_arms"]["v5"]["lora_scale"] == 1.0


def test_lora_arms_makes_the_lora_flag_unnecessary(tmp_path, capsys, monkeypatch):
    """--lora and --lora-arms are alternatives; neither one alone is an error."""
    v5 = _write_lora(tmp_path, "v5.safetensors")
    monkeypatch.setattr(sys, "argv", [
        "eval_memory_rollout.py",
        *_dry_run_argv(tmp_path, ["--lora-arms", f"v5={v5}"]),
    ])
    rollout.main()
    assert json.loads(capsys.readouterr().out)["arms"] == ["v5"]


def test_neither_lora_flag_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "eval_memory_rollout.py", *_dry_run_argv(tmp_path, []),
    ])
    with pytest.raises(ValueError, match="either --lora or --lora-arms"):
        rollout.main()
