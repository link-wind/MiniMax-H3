#!/usr/bin/env python3
"""Teacher-forced memory-slot ablation for the H3 continuation LoRA.

Answers "did the model actually learn to use the memory slots?" without any
autoregressive rollout.  The validation cache already carries real slots and a
real target, so the only variable is *which* slots the model may see, and the
prediction error on the target is a clean causal readout.

Arms
----
both   stm + ltm                 the trained configuration
stm    stm only                  does the near-history slot carry weight?
ltm    ltm only                  does the sequence-head slot carry weight?
none   no memory at all          pure text-to-shot baseline
swap   another sequence's slots  the causal test: error must follow the slot

Why the seeds are pinned
------------------------
The loss draws the diffusion timestep with ``torch.randint`` and the noise with
``torch.randn_like`` from the global RNG.  Left alone, the arm-to-arm gap is
dominated by t/noise variance rather than by the slots.  Every (sample, repeat)
therefore gets a deterministic seed, identical across arms and across ranks, so
the five arms differ in exactly one thing.

Sharding
--------
Samples are sharded by ``dp_rank`` so a CP group always processes one sample
together, matching the training launch shape.  The swap partner is chosen among
slots with the same latent geometry, because a slot is a raw latent block and
cannot be transplanted across resolutions.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import accelerate  # noqa: E402

from diffsynth.core.context_parallel import (  # noqa: E402
    cp_process_layout,
    create_cp_process_group,
    validate_cp_topology,
)
from diffsynth.utils.continuation_lora import ContinuationLatentDataset  # noqa: E402

# arm name -> (continuation_memory_mode, expected slot names)
ARMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "both": ("memory-only", ("stm", "ltm")),
    "stm": ("memory-only", ("stm",)),
    "ltm": ("memory-only", ("ltm",)),
    "none": ("off", ()),
    "swap": ("memory-only", ("stm", "ltm")),
}


def _load_training_module():
    """Import train.py without running its __main__ block."""
    spec = importlib.util.spec_from_file_location("h3_train_under_eval", _HERE / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_for(index: int, repeat: int) -> int:
    """Deterministic, arm-independent seed for one (sample, repeat) pair."""
    return 20260921 + int(index) * 1009 + int(repeat) * 17


def _pin_seed(index: int, repeat: int) -> None:
    seed = _seed_for(index, repeat)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _swap_partners(dataset: ContinuationLatentDataset, indices: list[int]) -> dict[int, int]:
    """Pair each slotted sample with another slotted sample of the same geometry."""
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in indices:
        record = dataset.records[index]
        metadata = record.get("metadata") or {}
        if not (metadata.get("memory_slots") or []):
            continue
        shape = tuple(metadata.get("video_shape") or ())
        if len(shape) < 2:
            continue
        groups[(int(shape[-2]), int(shape[-1]))].append(index)
    partners: dict[int, int] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        for position, index in enumerate(group):
            partners[index] = group[(position + 1) % len(group)]
    return partners


def _optimizer_params(model) -> list[torch.nn.Parameter]:
    """Trainable parameters for the placeholder optimizer (LoRA adapters only)."""
    trainable_modules = getattr(model, "trainable_modules", None)
    if callable(trainable_modules):
        params = [p for p in trainable_modules() if p.requires_grad]
        if params:
            return params
    return [p for p in model.parameters() if p.requires_grad]


def _loss_components(loss: torch.Tensor) -> dict[str, float]:
    components = getattr(loss, "continuation_components", None)
    if not components:
        try:
            import diffsynth.diffusion.loss as _loss_module

            components = _loss_module._LAST_CONTINUATION_COMPONENTS
        except Exception:
            components = None
    return dict(components or {})


def main() -> None:
    train_module = _load_training_module()

    parser = train_module.minimax_h3_parser()
    parser.description = "Teacher-forced memory-slot ablation for the H3 continuation LoRA."
    parser.add_argument("--eval-arms", type=str, default="both,stm,ltm,none,swap",
                        help="Comma-separated subset of: both,stm,ltm,none,swap")
    parser.add_argument("--eval-repeat", type=int, default=3,
                        help="Fixed-seed repeats per sample; the loss variance across t/noise is large.")
    parser.add_argument("--eval-max-samples", type=int, default=None,
                        help="Score at most this many validation samples (per DP group is derived from it).")
    parser.add_argument("--eval-skip-unslotted", action="store_true",
                        help="Only score samples that actually carry slots.")
    parser.add_argument("--eval-json", type=str, default=None,
                        help="Report path; defaults to <output_path>/memory_ablation.json")
    args = parser.parse_args()

    arms = [name.strip() for name in args.eval_arms.split(",") if name.strip()]
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        raise SystemExit(f"unknown eval arms: {unknown}; known: {sorted(ARMS)}")

    if args.task != "continuation_sft":
        args.task = "continuation_sft"
    if args.lora_base_model is None:
        args.lora_base_model = "dit"
    if args.continuation_manifest is None:
        raise SystemExit("--continuation_manifest is required (the dual-slot cache root)")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16" if args.bf16 else None,
    )
    topology = validate_cp_topology(accelerator.num_processes, args.cp_world_size)
    args.cp_world_size = topology["cp_world_size"]
    args.dp_world_size = topology["dp_world_size"]
    args.cp_rank = accelerator.process_index % args.cp_world_size
    args.dp_rank = accelerator.process_index // args.cp_world_size
    args.cp_group = create_cp_process_group(accelerator.process_index, args.cp_world_size)
    if accelerator.is_main_process:
        print(
            f"[eval] CP topology: processes={accelerator.num_processes} "
            f"cp={args.cp_world_size} dp={args.dp_world_size}",
            flush=True,
        )

    dataset = ContinuationLatentDataset(
        args.continuation_manifest, split=args.continuation_split, max_items=None,
    )
    all_indices = list(range(len(dataset)))
    if args.eval_skip_unslotted:
        all_indices = [
            index for index in all_indices
            if ((dataset.records[index].get("metadata") or {}).get("memory_slots") or [])
        ]
    if args.eval_max_samples is not None:
        all_indices = all_indices[: max(1, int(args.eval_max_samples))]

    # Shard by dp_rank: every rank inside a CP group sees the same sample.
    my_indices = all_indices[args.dp_rank :: args.dp_world_size]
    partners = _swap_partners(dataset, all_indices)
    if accelerator.is_main_process:
        slotted = sum(
            1 for index in all_indices
            if ((dataset.records[index].get("metadata") or {}).get("memory_slots") or [])
        )
        print(
            f"[eval] {len(all_indices)} samples selected ({slotted} slotted), "
            f"{len(my_indices)} per DP group, arms={arms}, repeats={args.eval_repeat}, "
            f"swap partners available for {len(partners)} samples",
            flush=True,
        )

    model = train_module.MiniMaxH3TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        template_model_id_or_path=args.template_model_id_or_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        silent_on_missing_audio=args.silent_on_missing_audio,
        training_cfg_scale=args.training_cfg_scale,
        continuation_memory_mode="memory-only",
        continuation_memory_expect_slots=("stm", "ltm"),
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        cp_rank=args.cp_rank,
        cp_world_size=args.cp_world_size,
        cp_group=args.cp_group,
    )
    model.continuation_overlap_steps = args.continuation_overlap_steps
    model.continuation_hard_core_steps = args.continuation_hard_core_steps
    model.continuation_transition_steps = args.continuation_transition_steps
    model.continuation_first_suffix_steps = args.continuation_first_suffix_steps
    model.continuation_transition_weight = args.continuation_transition_weight
    model.continuation_first_suffix_weight = args.continuation_first_suffix_weight
    model.continuation_suffix_weight = args.continuation_suffix_weight
    model.continuation_lambda_audio = args.continuation_lambda_audio
    model.continuation_conditioning = args.continuation_conditioning
    model.continuation_prefix_present_mode = args.continuation_prefix_present_mode
    model.use_gradient_checkpointing = False

    from diffsynth.diffusion.runner import (
        _deepspeed_zero_stage,
        _ensure_zero3_cpu_partition,
        _z3_ensure_stub_device,
        _z3_text_encoder_selftest,
        ensure_deepspeed_micro_batch_size,
    )

    ensure_deepspeed_micro_batch_size(accelerator)

    trainable = _optimizer_params(model)
    if not trainable:
        raise SystemExit(
            "no trainable parameters found -- the LoRA was not injected, so there is "
            "nothing to evaluate. Check --lora_checkpoint and --lora_target_modules."
        )
    # The DeepSpeed config carries no optimizer section, so prepare() needs one.
    # It is never stepped: this run only does forward passes.
    placeholder_optimizer = torch.optim.SGD(trainable, lr=0.0)
    if args.initialize_model_on_cpu:
        # Keep the full model on CPU while ZeRO-3 partitions it; moving the
        # whole 57GB transformer to one GPU first would not fit.
        _ensure_zero3_cpu_partition(accelerator, model, cp_world_size=args.cp_world_size)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
    model, _optimizer = accelerator.prepare(model, placeholder_optimizer)

    # DeepSpeed leaves every partitioned parameter with an empty CPU stub.
    # Modules holding a single direct parameter (Embedding, RMSNorm, bias-free
    # Linear) then have their all-gathered tensor copied with ``.to(param.device)``
    # and materialize on CPU, so the first CUDA op fails with a device mismatch
    # even though the parameter reports AVAILABLE.  Training has always repointed
    # those stubs; the eval path needs the same step or it dies inside the text
    # encoder's embed_tokens on the very first forward.
    engine_module = getattr(model, "module", None)
    if engine_module is not None and _deepspeed_zero_stage(accelerator) == 3:
        _z3_ensure_stub_device(engine_module, accelerator)
    _z3_text_encoder_selftest(accelerator, model, args)

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.use_gradient_checkpointing = False
    accelerator.wait_for_everyone()

    details: list[dict[str, object]] = []
    cache: dict[int, dict] = {}

    for position, index in enumerate(my_indices, start=1):
        if index not in cache:
            cache[index] = dataset[index]
        primary = cache[index]
        has_slots = bool((dataset.records[index].get("metadata") or {}).get("memory_slots") or [])

        swapped = None
        if "swap" in arms:
            partner = partners.get(index)
            if partner is not None:
                if partner not in cache:
                    cache[partner] = dataset[partner]
                swapped = cache[partner].get("continuation_memory_video_latents")

        for arm in arms:
            mode, slots = ARMS[arm]
            unwrapped.continuation_memory_mode = mode
            unwrapped.continuation_memory_expect_slots = slots
            inputs = primary
            if arm == "swap" and swapped is not None:
                inputs = dict(primary)
                inputs["continuation_memory_video_latents"] = swapped
            for repeat in range(int(args.eval_repeat)):
                _pin_seed(index, repeat)
                with torch.no_grad():
                    loss = model({}, inputs=inputs)
                components = _loss_components(loss)
                details.append({
                    "arm": arm,
                    "index": int(index),
                    "sample_id": dataset.records[index].get("sample_id"),
                    "window_frames": int(dataset.records[index].get("window_frames") or 0),
                    "has_slots": has_slots,
                    # A slot-free sample has nothing to transplant, so the swap
                    # arm falls back to its own (absent) slot there.  Flagging it
                    # keeps those samples out of the swap contrast instead of
                    # letting them dilute it toward zero.
                    "swap_applied": bool(arm == "swap" and swapped is not None),
                    "repeat": repeat,
                    "loss": float(loss.detach().float().item()),
                    "video_loss": components.get("video_loss"),
                    "audio_loss": components.get("audio_loss"),
                })
        if position % 25 == 0 or position == len(my_indices):
            print(f"[eval] rank {accelerator.process_index}: {position}/{len(my_indices)} samples", flush=True)

    gathered: list[list[dict[str, object]]] | None = [None] * accelerator.num_processes
    if accelerator.num_processes > 1:
        torch.distributed.gather_object(details, gathered if accelerator.is_main_process else None, dst=0)
    else:
        gathered = [details]

    if accelerator.is_main_process:
        merged = [item for chunk in (gathered or []) if chunk for item in chunk]
        report = _summarize(merged, arms, int(args.eval_repeat))
        output_path = Path(args.eval_json) if args.eval_json else Path(args.output_path) / "memory_ablation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _print_report(report)
        print(f"[eval] report written to {output_path}", flush=True)
        with open(output_path.with_suffix(".details.jsonl"), "w", encoding="utf-8") as handle:
            for item in merged:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    accelerator.end_training()


def _summarize(details: list[dict[str, object]], arms: list[str], repeats: int) -> dict[str, object]:
    """Per-arm means plus the paired contrasts that decide the question."""
    per_arm: dict[str, list[float]] = defaultdict(list)
    per_arm_video: dict[str, list[float]] = defaultdict(list)
    per_arm_audio: dict[str, list[float]] = defaultdict(list)
    by_sample: dict[tuple[str, int], list[float]] = defaultdict(list)
    for item in details:
        arm = str(item["arm"])
        if arm == "swap" and not item.get("swap_applied", False):
            # No partner existed for this sample; it is not part of the swap test.
            continue
        per_arm[arm].append(float(item["loss"]))
        if item.get("video_loss") is not None:
            per_arm_video[arm].append(float(item["video_loss"]))
        if item.get("audio_loss") is not None:
            per_arm_audio[arm].append(float(item["audio_loss"]))
        by_sample[(arm, int(item["index"]))].append(float(item["loss"]))

    def mean(values):
        return sum(values) / len(values) if values else None

    sample_mean = {key: mean(values) for key, values in by_sample.items()}
    indices = sorted({index for _, index in by_sample})

    def paired(arm_a: str, arm_b: str) -> dict[str, object]:
        """Mean of per-sample (a - b); positive means a is worse."""
        diffs = [
            sample_mean[(arm_a, index)] - sample_mean[(arm_b, index)]
            for index in indices
            if (arm_a, index) in sample_mean and (arm_b, index) in sample_mean
        ]
        if not diffs:
            return {"n": 0, "mean": None, "stderr": None, "wins": None}
        avg = mean(diffs)
        if len(diffs) > 1:
            var = sum((value - avg) ** 2 for value in diffs) / (len(diffs) - 1)
            stderr = (var / len(diffs)) ** 0.5
        else:
            stderr = 0.0
        return {
            "n": len(diffs),
            "mean": avg,
            "stderr": stderr,
            "wins": sum(1 for value in diffs if value > 0),
        }

    return {
        "repeats_per_sample": repeats,
        "scored_samples": len(indices),
        "arms": {
            arm: {
                "n": len(per_arm[arm]),
                "loss_mean": mean(per_arm[arm]),
                "video_loss_mean": mean(per_arm_video[arm]),
                "audio_loss_mean": mean(per_arm_audio[arm]),
            }
            for arm in arms
        },
        "paired_differences": {
            # Restricted to samples whose slots were actually transplanted.
            "swap_minus_both": paired("swap", "both"),
            "none_minus_both": paired("none", "both"),
            "stm_minus_both": paired("stm", "both"),
            "ltm_minus_both": paired("ltm", "both"),
        },
    }


def _print_report(report: dict[str, object]) -> None:
    print("\n=== memory-slot ablation (teacher-forced) ===", flush=True)
    print(f"scored samples: {report['scored_samples']}  repeats/sample: {report['repeats_per_sample']}")
    print(f"{'arm':<6} {'n':>6} {'loss':>9} {'video':>9} {'audio':>9}")
    for arm, stats in report["arms"].items():
        def fmt(value):
            return "   n/a  " if value is None else f"{value:9.5f}"
        print(
            f"{arm:<6} {stats['n']:>6} {fmt(stats['loss_mean'])} "
            f"{fmt(stats['video_loss_mean'])} {fmt(stats['audio_loss_mean'])}"
        )
    print("\npaired per-sample contrasts (positive = first arm worse):")
    for name, stats in report["paired_differences"].items():
        if stats["mean"] is None:
            print(f"  {name:<18} n=0 (no overlapping samples)")
            continue
        print(
            f"  {name:<18} n={stats['n']:<4} mean={stats['mean']:+.5f} "
            f"stderr={stats['stderr']:.5f} wins={stats['wins']}/{stats['n']}"
        )


if __name__ == "__main__":
    main()
