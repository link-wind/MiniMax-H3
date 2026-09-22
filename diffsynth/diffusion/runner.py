import os, json, random, torch, importlib, time
import contextlib
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager
from diffsynth.core.context_parallel import (
    build_cp_training_dataloader,
    reduce_cp_parameter_gradients,
    should_reduce_cp_parameter_gradients,
    should_prepare_dataloader,
)
from diffsynth.utils.continuation_lora import (
    ContinuationRegionConfig,
    load_continuation_training_checkpoint,
    manifest_sha256,
    save_continuation_training_checkpoint,
)


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def save_training_args(args):
    output_path = getattr(args, "output_path", None) if args is not None else None
    if output_path is None:
        return
    try:
        os.makedirs(args.output_path, exist_ok=True)
        save_path = os.path.join(args.output_path, "training_args.json")
        args_to_save = vars(args).copy() if hasattr(args, "__dict__") else vars(args)
        args_to_save.pop("cp_group", None)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(args_to_save, f, indent=4, ensure_ascii=False, default=str)
        print(f"Training arguments saved to `{save_path}`.")
    except Exception as e:
        print(f"Warning: failed to save training arguments: {e}")


def ensure_deepspeed_micro_batch_size(accelerator: Accelerator):
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        return
    config = plugin.deepspeed_config
    if config.get("train_micro_batch_size_per_gpu") in (None, "auto"):
        config["train_micro_batch_size_per_gpu"] = 1


def _cp_trace_enabled() -> bool:
    return os.environ.get("H3_CP_TRACE") == "1"


def _deepspeed_zero_stage(accelerator: Accelerator) -> int:
    """Return the DeepSpeed ZeRO stage from the accelerate plugin config.

    accelerate keeps the raw ds_config (nested ``zero_optimization.stage``) and
    never adds a top-level ``zero_stage`` key, so lookups must descend into
    ``zero_optimization``.
    """
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        return 0
    config = plugin.deepspeed_config
    zero_optimization = config.get("zero_optimization", {}) or {}
    try:
        return int(zero_optimization.get("stage", 0))
    except (TypeError, ValueError):
        return 0


def _flush_cuda_cache_before_backward(accelerator: Accelerator) -> bool:
    """Flush the PyTorch caching allocator before backward on ZeRO-3 runs.

    DeepSpeed ZeRO-3 + activation checkpointing leave many small cached blocks
    behind after the forward pass. Backward then needs a handful of large
    contiguous allocations and can OOM even though the cached bytes would be
    enough. DeepSpeed itself recommends an explicit empty_cache() before the
    backward step; this keeps that flush to ZeRO-3 runs only.
    """
    if not torch.cuda.is_available():
        return False
    return _deepspeed_zero_stage(accelerator) == 3


def _param_device_census(model, accelerator: Accelerator, tag: str) -> None:
    """Bucket parameter bytes by device and requires_grad, printed once per rank."""
    # After ``accelerator.prepare`` with DeepSpeed, the object returned is the
    # DeepSpeedEngine; its actual module tree lives under ``engine.module``.
    if type(model).__name__ == "DeepSpeedEngine" and getattr(model, "module", None) is not None:
        model = model.module
    buckets: dict[tuple[str, bool], int] = {}
    for name, param in model.named_parameters():
        # ZeRO-3 partitioned parameters keep ``param.data`` empty; report the
        # partition slice instead of the logical full parameter.
        numel = getattr(param, "ds_numel", None) or param.numel()
        ds_tensor = getattr(param, "ds_tensor", None)
        device = ds_tensor.device if (ds_tensor is not None and ds_tensor.numel() > 0) else param.device
        key = (str(device), bool(param.requires_grad))
        buckets[key] = buckets.get(key, 0) + numel * param.element_size()
    parts = ", ".join(
        f"{dev}{'/grad' if rg else '/frozen'}={nbytes / 2**30:.2f}GiB"
        for (dev, rg), nbytes in sorted(buckets.items())
    )
    print(
        f"[param][rank{accelerator.process_index}] {tag}: {parts}",
        flush=True,
    )


def _zero3_partition_stats(model, accelerator: Accelerator, tag: str) -> dict:
    """Report how many model parameters are ZeRO-3 partitioned (``ds_id``)."""
    partitioned = 0
    partitioned_numel = 0
    plain = 0
    plain_numel = 0
    plain_samples: list[str] = []
    for name, param in model.named_parameters():
        numel = getattr(param, "ds_numel", None) or param.numel()
        if getattr(param, "ds_id", None) is not None:
            partitioned += 1
            partitioned_numel += numel
        else:
            plain += 1
            plain_numel += numel
            if len(plain_samples) < 6:
                plain_samples.append(name)
    stats = {
        "tag": tag,
        "partitioned": partitioned,
        "partitioned_numel": partitioned_numel,
        "plain": plain,
        "plain_numel": plain_numel,
        "plain_samples": plain_samples,
    }
    if accelerator.is_main_process:
        print(
            f"[z3][rank{accelerator.process_index}] {tag}: "
            f"partitioned={partitioned} ({partitioned_numel / 1e9:.2f}B elems), "
            f"plain={plain} ({plain_numel / 1e9:.2f}B elems)"
            + (f"; sample plain params: {plain_samples}" if plain_samples else ""),
            flush=True,
        )
    return stats


def _print_key_param_state(model, accelerator: Accelerator, tag: str) -> None:
    """Dump device/dtype/partition state of representative params."""
    if not accelerator.is_main_process:
        return
    state_dict = dict(model.named_parameters())
    printed = 0
    for name, param in state_dict.items():
        if not (name.startswith("pipe.text_encoder.") or name.startswith("pipe.dit.")):
            continue
        if printed >= 8:
            break
        ds_tensor = getattr(param, "ds_tensor", None)
        pg_world = None
        try:
            import torch.distributed as dist
            pg = getattr(param, "ds_process_group", None)
            if pg is not None:
                pg_world = dist.get_world_size(pg)
        except Exception:
            pass
        print(
            f"[z3param][rank{accelerator.process_index}] {tag} {name}: "
            f"device={param.device} dtype={param.dtype} grad={param.requires_grad} "
            f"shape={tuple(param.shape)} ds_id={getattr(param, 'ds_id', None)} "
            f"ds_status={getattr(param, 'ds_status', None)} "
            f"ds_numel={getattr(param, 'ds_numel', None)} "
            f"ds_tensor={None if ds_tensor is None else tuple(ds_tensor.shape)} "
            f"partition_world_at_convert={getattr(param, 'ds_zero_partition_world_size', None)} "
            f"pg_world_now={pg_world}",
            flush=True,
        )
        printed += 1


def _report_z3_partition_footprint(engine_module, accelerator: Accelerator) -> None:
    """Sum actual per-rank ZeRO-3 storage (ds_tensor bytes) by device."""
    if not accelerator.is_main_process:
        return
    buckets: dict[str, int] = {}
    for _name, param in engine_module.named_parameters():
        ds_tensor = getattr(param, "ds_tensor", None)
        if ds_tensor is None or ds_tensor.numel() == 0:
            continue
        key = str(ds_tensor.device)
        buckets[key] = buckets.get(key, 0) + ds_tensor.numel() * ds_tensor.element_size()
    parts = ", ".join(f"{dev}={nbytes / 2**30:.2f}GiB" for dev, nbytes in sorted(buckets.items()))
    print(
        f"[z3][rank{accelerator.process_index}] ds_tensor per-rank footprint: {parts}",
        flush=True,
    )


def _z3_ensure_stub_device(engine_module, accelerator: Accelerator) -> None:
    """Repoint empty ZeRO-3 partition stubs at the accelerator device.

    Models constructed under ``deepspeed.zero.Init(remote_device='cpu')`` leave
    each partitioned parameter with an empty ``param.data`` stub on CPU. For
    modules with a single direct parameter (Embedding, RMSNorm, bias-free
    Linear, ...) DeepSpeed's sequential gather copies the all-gathered tensor
    with ``.to(param.device)``, so the full tensor is materialized on CPU and
    the first CUDA op crashes with a CPU/CUDA device mismatch even though the
    parameter reports ``AVAILABLE``. Only the meaningless empty stub is moved;
    the real partition storage (``param.ds_tensor``) stays on CPU.
    """
    if not torch.cuda.is_available():
        return
    moved = 0
    device = accelerator.device
    for name, p in engine_module.named_parameters():
        if not hasattr(p, "ds_id") or getattr(p, "ds_tensor", None) is None:
            continue
        status = getattr(p, "ds_status", None)
        if status is None or getattr(status, "name", str(status)) != "NOT_AVAILABLE":
            continue
        if p.data.numel() != 0 or p.device.type == "cuda":
            continue
        p.data = torch.empty(0, dtype=p.dtype, device=device)
        moved += 1
    if accelerator.is_main_process:
        print(
            f"[z3] repointed {moved} empty partition stubs to {device} "
            f"(ds_tensor storage stays on CPU)",
            flush=True,
        )


def _report_z3_hook_coverage(engine_module, accelerator: Accelerator) -> None:
    """Report ZeRO-3 forward-hook registration along the text encoder path."""
    if not accelerator.is_main_process:
        return
    wanted = {
        "pipe.text_encoder",
        "pipe.text_encoder.model",
        "pipe.text_encoder.model.visual",
        "pipe.text_encoder.model.language_model",
    }
    embed_printed = False
    for path, mod in engine_module.named_modules():
        if path in wanted or path.endswith("embed_tokens"):
            direct = len(list(mod.parameters(recurse=False)))
            print(
                f"[z3hookcov][rank{accelerator.process_index}] {path} class={type(mod).__name__} "
                f"fwd_pre={len(getattr(mod, '_forward_pre_hooks', {}))} "
                f"fwd_post={len(getattr(mod, '_forward_hooks', {}))} "
                f"direct_params={direct} ds_id={getattr(mod, 'ds_id', None)}",
                flush=True,
            )
            if path.endswith("embed_tokens"):
                embed_printed = True
    if not embed_printed:
        for path, mod in engine_module.named_modules():
            if type(mod).__name__ in ("Embedding", "Qwen3VLEmbedding") and "text_encoder" in path:
                print(
                    f"[z3hookcov][rank{accelerator.process_index}] {path} class={type(mod).__name__} "
                    f"fwd_pre={len(getattr(mod, '_forward_pre_hooks', {}))} "
                    f"fwd_post={len(getattr(mod, '_forward_hooks', {}))}",
                    flush=True,
                )


def _install_z3_forward_tracer(accelerator: Accelerator) -> None:
    """Trace ZeRO-3 pre/post module hooks for the text encoder path.

    Enabled by default for continuation_sft runs so a failing first step prints
    the smoking gun: whether DeepSpeed's per-module fetch hooks actually fire
    and whether parameters are gathered before use. Disable with
    H3_DS_Z3_TRACE=0.
    """
    env = os.environ.get("H3_DS_Z3_TRACE")
    if env is not None and env not in ("1", "true", "True"):
        return
    if getattr(accelerator.state, "_h3_z3_tracer_installed", False):
        return
    try:
        from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
    except Exception:
        return
    key_terms = ("TextEncoder", "Qwen3VL", "Embedding")
    # Cap trace volume: only the first forward(s) matter for this diagnosis.
    budget = [240]

    def _emit(line: str) -> None:
        if budget[0] <= 0:
            return
        budget[0] -= 1
        print(line, flush=True)

    def _fmt(p) -> str:
        ds_t = getattr(p, "ds_tensor", None)
        return (
            f"dev={p.device} shape={tuple(p.shape)} "
            f"st={getattr(p, 'ds_status', None)} "
            f"ds_id={getattr(p, 'ds_id', None)} "
            f"ds_t={None if ds_t is None else tuple(ds_t.shape)}"
        )

    orig_pre = DeepSpeedZeRoOffload.pre_sub_module_forward_function

    def traced_pre(self, sub_module):
        cls = type(sub_module).__name__
        trace = any(k in cls for k in key_terms)
        if not trace:
            return orig_pre(self, sub_module)
        params = [p for p in sub_module.parameters(recurse=False)]
        if accelerator.is_main_process:
            first = _fmt(params[0]) if params else "no-direct-params"
            _emit(
                f"[z3trace][rank{accelerator.process_index}][pre] {cls} "
                f"ds_id={getattr(sub_module, 'ds_id', None)} params={len(params)} first={first}"
            )
        out = orig_pre(self, sub_module)
        if accelerator.is_main_process and params:
            _emit(
                f"[z3trace][rank{accelerator.process_index}][post] {cls} "
                f"ds_id={getattr(sub_module, 'ds_id', None)} params={len(params)} first={_fmt(params[0])}"
            )
        return out

    DeepSpeedZeRoOffload.pre_sub_module_forward_function = traced_pre
    accelerator.state._h3_z3_tracer_installed = True

    # If an Embedding module is invoked while its weight is still off-GPU, dump
    # the exact hook state at the moment of use. This distinguishes "DeepSpeed
    # hooks are gone" from "hooks are registered but did not gather".
    try:
        import torch.nn as nn

        orig_embed_fwd = nn.Embedding.forward

        def traced_embed_fwd(self, input):
            if (
                budget[0] > 0
                and isinstance(input, torch.Tensor)
                and input.is_cuda
                and self.weight.device.type != "cuda"
                and accelerator.is_main_process
            ):
                hooks = [
                    getattr(fn, "__qualname__", repr(fn))
                    for fn in self._forward_pre_hooks.values()
                ]
                _emit(
                    "[z3embed][rank0] Embedding.forward with weight off-GPU: "
                    f"weight_dev={self.weight.device} shape={tuple(self.weight.shape)} "
                    f"ds_status={getattr(self.weight, 'ds_status', None)} "
                    f"ds_id={getattr(self.weight, 'ds_id', None)} "
                    f"pre_hooks={hooks}"
                )
            return orig_embed_fwd(self, input)

        nn.Embedding.forward = traced_embed_fwd
    except Exception:
        pass

    if accelerator.is_main_process:
        print(
            "[z3trace][rank0] DeepSpeedZeRoOffload.pre_sub_module_forward_function tracer installed.",
            flush=True,
        )


def _z3_text_encoder_selftest(accelerator: Accelerator, model, args) -> None:
    """Run a tiny text-encoder forward to surface ZeRO-3 gather failures early.

    Enabled by default for continuation_sft runs; disable with
    H3_DS_Z3_SELFTEST=0.
    """
    env = os.environ.get("H3_DS_Z3_SELFTEST")
    if env is not None and env not in ("1", "true", "True"):
        return
    if getattr(args, "task", None) != "continuation_sft":
        return
    engine_module = getattr(model, "module", None) or model
    text_encoder = engine_module.pipe.text_encoder
    language_model = getattr(getattr(text_encoder, "model", None), "language_model", None)
    embed = getattr(language_model, "embed_tokens", None) if language_model is not None else None
    if accelerator.is_main_process:
        if embed is not None:
            w = embed.weight
            hooks = [
                getattr(fn, "__qualname__", repr(fn))
                for fn in embed._forward_pre_hooks.values()
            ]
            print(
                "[z3selftest][pre] embed_tokens.weight: "
                f"dev={w.device} shape={tuple(w.shape)} dtype={w.dtype} "
                f"ds_id={getattr(w, 'ds_id', None)} ds_status={getattr(w, 'ds_status', None)} "
                f"pre_hooks={hooks}",
                flush=True,
            )
        print("[z3selftest] running text encoder forward with tiny inputs", flush=True)
    ids = torch.randint(0, 1000, (1, 8), device=accelerator.device)
    attention_mask = torch.ones_like(ids)
    try:
        with torch.no_grad():
            text_encoder(input_ids=ids, attention_mask=attention_mask)
        if accelerator.is_main_process:
            print("[z3selftest] text encoder forward OK", flush=True)
    except Exception as exc:
        if accelerator.is_main_process:
            print(f"[z3selftest] text encoder forward FAILED: {exc!r}", flush=True)
            if embed is not None:
                w = embed.weight
                ds_t = getattr(w, "ds_tensor", None)
                hooks = [
                    getattr(fn, "__qualname__", repr(fn))
                    for fn in embed._forward_pre_hooks.values()
                ]
                print(
                    "[z3selftest] embed_tokens.weight: "
                    f"dev={w.device} shape={tuple(w.shape)} dtype={w.dtype} "
                    f"ds_id={getattr(w, 'ds_id', None)} ds_status={getattr(w, 'ds_status', None)} "
                    f"ds_t={None if ds_t is None else tuple(ds_t.shape)} "
                    f"ds_t_dev={None if ds_t is None else ds_t.device}",
                    flush=True,
                )
                print(
                    f"[z3selftest] embed_tokens hooks: fwd_pre={len(getattr(embed, '_forward_pre_hooks', {}))} "
                    f"fwd_post={len(getattr(embed, '_forward_hooks', {}))} "
                    f"pre_hooks={hooks}",
                    flush=True,
                )
            # Dump a compact census of text-encoder params that are not on GPU.
            shown = 0
            for name, p in text_encoder.named_parameters():
                if p.device.type != "cuda":
                    print(
                        f"[z3selftest] non-cuda param {name}: dev={p.device} "
                        f"shape={tuple(p.shape)} status={getattr(p, 'ds_status', None)}",
                        flush=True,
                    )
                    shown += 1
                    if shown >= 30:
                        break
        raise


def _ensure_zero3_cpu_partition(accelerator: Accelerator, model, cp_world_size: int = 1) -> None:
    """Partition CPU-initialized parameters for DeepSpeed ZeRO-3.

    DiffSynth normally constructs every sub-model inside ``deepspeed.zero.Init``
    (see ``diffsynth.core.loader.model.get_init_context``), which leaves each
    parameter with a ``ds_id``/``ds_tensor`` partition slice on the rank's local
    device. If that load-time context was not active (for example the HF
    DeepSpeed config weakref was not registered yet), all parameters stay as
    full plain CPU tensors. A real optimizer then only makes the *trainable*
    parameters ZeRO-3 partitioned (via DeepSpeed stage-3), while frozen towers
    such as the text encoder remain plain CPU tensors - and the first forward
    crashes with a CPU/CUDA device mismatch.

    This helper partitions the whole model with ``deepspeed.zero.Init`` when no
    parameter carries a ``ds_id`` yet, making the frozen towers ZeRO-3 managed
    too. It must run before the optimizer is created so the optimizer keeps
    references to the (mutated) partitioned parameters, mirroring the standard
    transformers + stage-3 flow.
    """
    import deepspeed

    stats = _zero3_partition_stats(model, accelerator, "before zero3 partition")
    _print_key_param_state(model, accelerator, "before zero3 partition")
    total = stats["partitioned"] + stats["plain"]
    if total == 0:
        return
    if cp_world_size > 1:
        if accelerator.is_main_process:
            print(
                "[z3] WARNING: CP_WORLD_SIZE>1 detected; skipping in-runner ZeRO-3 "
                "partition fallback (partition group semantics are CP-specific). "
                f"partitioned={stats['partitioned']} plain={stats['plain']}.",
                flush=True,
            )
        return
    if stats["partitioned"] == total:
        if accelerator.is_main_process:
            print("[z3] All parameters are already ZeRO-3 partitioned; nothing to do.", flush=True)
        return
    if stats["partitioned"] > 0:
        if accelerator.is_main_process:
            print(
                "[z3] WARNING: mixed partition state detected "
                f"({stats['partitioned']} partitioned / {stats['plain']} plain). "
                "ZeRO-3 can only partition an all-plain model here; plain frozen "
                "parameters may still fail in forward. Proceeding without repartition.",
                flush=True,
            )
        return
    if accelerator.is_main_process:
        print("[z3] No ZeRO-3 partitioned parameters found; partitioning full model on CPU.", flush=True)
    # Partition all parameters (frozen + trainable) across the DP group. The
    # plain parameters currently live on CPU and ``remote_device="cpu"`` keeps
    # the local slices in host memory until the engine is initialized.
    with torch.no_grad():
        deepspeed.zero.Init(
            module=model,
            remote_device="cpu",
            dtype=torch.bfloat16,
            mem_efficient_linear=False,
        )
    _zero3_partition_stats(model, accelerator, "after zero3 partition")
    _print_key_param_state(model, accelerator, "after zero3 partition")


def _report_oom_context(accelerator: Accelerator, model, where: str) -> None:
    """Best-effort memory summary printed right before the OOM propagates."""
    try:
        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        print(
            f"[oom][rank{accelerator.process_index}] {where}: "
            f"allocated={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
            f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB "
            f"free={free / 2**30:.2f}GiB "
            f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )
        _param_device_census(model, accelerator, f"census at {where}")
    except Exception as exc:  # never mask the original OOM
        print(
            f"[oom][rank{accelerator.process_index}] failed to report context: {exc}",
            flush=True,
        )


def _log_gpu_memory(accelerator: Accelerator, tag: str) -> None:
    if os.environ.get("DIFFSYNTH_MEMORY_LOG") != "1":
        return
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    print(
        f"[mem][rank{accelerator.process_index}] {tag}: "
        f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB "
        f"free={free / 2**30:.2f}GiB",
        flush=True,
    )


def _cp_trace(accelerator: Accelerator, message: str) -> None:
    if _cp_trace_enabled():
        print(
            f"[H3CP][rank {accelerator.process_index}] {message}",
            flush=True,
        )


def _continuation_region_config(args) -> ContinuationRegionConfig | None:
    if getattr(args, "task", None) != "continuation_sft":
        return None
    return ContinuationRegionConfig(
        overlap_video_steps=getattr(args, "continuation_overlap_steps", 12),
        hard_core_video_steps=getattr(args, "continuation_hard_core_steps", 12),
        transition_video_steps=getattr(args, "continuation_transition_steps", 0),
        first_suffix_clip_steps=getattr(args, "continuation_first_suffix_steps", 5),
        transition_weight=getattr(args, "continuation_transition_weight", 0.5),
        first_suffix_weight=getattr(args, "continuation_first_suffix_weight", 3.0),
        suffix_weight=getattr(args, "continuation_suffix_weight", 1.0),
        lambda_audio=getattr(args, "continuation_lambda_audio", 0.5),
        conditioning_mode=getattr(args, "continuation_conditioning", "masked-av-v14"),
    )


def _continuation_manifest_hash(args) -> str | None:
    manifest = getattr(args, "continuation_manifest", None)
    if not manifest:
        return None
    # ContinuationLatentDataset accepts either a split manifest or the cache
    # root. Resolve the latter to the selected split before hashing so resume
    # metadata identifies the exact records used by this run.
    manifest_path = Path(manifest)
    if manifest_path.is_dir():
        split = getattr(args, "continuation_split", "train")
        manifest_path = manifest_path / split / "manifest.jsonl"
    return manifest_sha256(manifest_path)


def _continuation_ckpt_rank_path(path, process_index, num_processes):
    """ZeRO-3 state dicts are per-rank partitions, so every rank must write and
    read its own checkpoint file. Single-process runs keep the plain path."""
    path = str(path)
    if num_processes <= 1:
        return path
    return f"{path}.rank{process_index}"


def _save_continuation_checkpoint(
    accelerator: Accelerator,
    model: DiffusionTrainingModule,
    args,
    optimizer,
    scheduler,
    *,
    step_count: int,
    manifest_hash: str | None,
    region_config: ContinuationRegionConfig | None,
) -> None:
    save_path = getattr(args, "continuation_checkpoint_save_path", None)
    if not save_path:
        return
    # DeepSpeed ZeRO-3 partitions model and optimizer state across ranks, so a
    # single shared file is both a write race (rank1-7 FileNotFoundError above)
    # and semantically wrong (one rank's partition overwriting the others).
    # Write one file per rank and resume from the same per-rank files.
    save_path = _continuation_ckpt_rank_path(
        save_path, accelerator.process_index, accelerator.num_processes
    )
    unwrapped = accelerator.unwrap_model(model)
    trainable_state = unwrapped.export_trainable_state_dict(
        unwrapped.state_dict(),
        remove_prefix=getattr(args, "remove_prefix_in_ckpt", None),
    )
    save_continuation_training_checkpoint(
        save_path,
        model_state_dict=trainable_state,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        global_step=step_count,
        seed=getattr(args, "seed", 42),
        manifest_hash=manifest_hash,
        region_config=region_config,
        rng_state=torch.get_rng_state(),
        random_state=random.getstate(),
        extra_metadata={
            "cp_world_size": getattr(args, "cp_world_size", 1),
            "training_cfg_scale": getattr(args, "training_cfg_scale", 1.0),
            "bf16": bool(getattr(args, "bf16", False)),
            "gradient_checkpointing": bool(getattr(args, "use_gradient_checkpointing", False)),
        },
    )
    if accelerator.is_main_process:
        logical = getattr(args, "continuation_checkpoint_save_path", None)
        suffix = "" if accelerator.num_processes <= 1 else ".rank<proc>"
        print(
            f"Full continuation checkpoint saved to {logical}{suffix} "
            f"at step {step_count} (one file per rank).",
            flush=True,
        )


def _should_save_continuation_checkpoint(args, step_count: int, save_steps: int | None) -> bool:
    if not getattr(args, "continuation_checkpoint_save_path", None):
        return False
    interval = getattr(args, "continuation_checkpoint_interval", None) or save_steps
    return interval is not None and step_count > 0 and step_count % interval == 0


def _average_log_loss(accelerator, loss):
    """Average the scalar loss across every process for stable training logs.

    Log-only aid; gradients are untouched. Each rank of a CP group already
    holds the same loss for its own sample, so the world average is the mean
    over that step's DP samples instead of one noisy per-rank sample.
    """
    value = loss.detach().float().reshape(-1).mean()
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and accelerator.num_processes > 1
    ):
        value = value.to(device=accelerator.device)
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.AVG)
    return value


_BACKWARD_PROFILED = [False]


def _run_backward_step(accelerator, loss, output_dir=None):
    """Run ``accelerator.backward`` once with optional one-shot memory profiling.

    With ``DIFFSYNTH_MEM_PROFILE=1`` the first backward of the run is wrapped
    in a CUDA memory profiler and an op-level table is written on the main
    process (``<output_path>/mem_profile_step0.txt``). With
    ``DIFFSYNTH_MEMORY_LOG=1`` every rank reports the allocated peak reached
    during the backward (the peak counter is reset right before it), which is
    the number that decides whether the run OOMs.
    """
    mem_log = os.environ.get("DIFFSYNTH_MEMORY_LOG") == "1"
    if mem_log and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    profiling = (
        os.environ.get("DIFFSYNTH_MEM_PROFILE") == "1"
        and not _BACKWARD_PROFILED[0]
    )
    if profiling:
        _BACKWARD_PROFILED[0] = True
        from torch.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
        ) as prof:
            accelerator.backward(loss)
        torch.cuda.synchronize()
        if accelerator.is_main_process and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "mem_profile_step0.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("===== by self cuda time =====\n")
                handle.write(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=120
                    )
                )
                handle.write("\n===== by self cuda memory =====\n")
                handle.write(
                    prof.key_averages().table(
                        sort_by="self_cuda_memory_usage", row_limit=120
                    )
                )
    else:
        accelerator.backward(loss)
    if mem_log and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(
            f"[mem][rank{accelerator.process_index}] backward peak "
            f"allocated={peak:.2f}GiB",
            flush=True,
        )


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    max_steps: int = None,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    args = None,
    cp_world_size: int = 1,
    cp_group=None,
    cp_gradient_reduction: str = "global-group",
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        max_steps = getattr(args, "max_steps", None)
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer
        cp_world_size = getattr(args, "cp_world_size", 1)
        cp_group = getattr(args, "cp_group", None)
        cp_gradient_reduction = getattr(
            args, "cp_gradient_reduction", "global-group"
        )

    if accelerator.is_main_process:
        save_training_args(args)
        if _deepspeed_zero_stage(accelerator) == 3:
            try:
                import accelerate
                import deepspeed
                import transformers

                print(
                    f"[z3] deepspeed={deepspeed.__version__} accelerate={accelerate.__version__} "
                    f"transformers={transformers.__version__} cp_world_size={cp_world_size}",
                    flush=True,
                )
            except Exception:
                pass

    # ZeRO-3 + CPU-initialized models: make sure the frozen towers (text
    # encoder, VAEs, ...) are actually ZeRO-3 partitioned before the optimizer
    # is created. Without this, DeepSpeed stage-3 with a real optimizer only
    # partitions the trainable parameters and the frozen ones stay as full CPU
    # tensors, so the first forward dies with a CPU/CUDA device mismatch.
    if (
        not enable_model_cpu_offload
        and getattr(args, "initialize_model_on_cpu", False)
        and _deepspeed_zero_stage(accelerator) == 3
    ):
        _ensure_zero3_cpu_partition(accelerator, model, cp_world_size=cp_world_size)

    optimizer_class = get_optimizer_class(customized_optimizer)
    optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = build_cp_training_dataloader(
        dataset,
        process_index=accelerator.process_index,
        process_count=accelerator.num_processes,
        cp_world_size=cp_world_size,
        num_workers=num_workers,
        cp_seed=getattr(args, "cp_seed", 42),
    )
    manifest_hash = _continuation_manifest_hash(args)
    region_config = _continuation_region_config(args)
    prepare_dataloader = should_prepare_dataloader(cp_world_size)
    ensure_deepspeed_micro_batch_size(accelerator)

    if enable_model_cpu_offload:
        if not prepare_dataloader:
            optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
        else:
            optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        deepspeed_zero3_cpu_init = (
            _deepspeed_zero_stage(accelerator) == 3
            and getattr(args, "initialize_model_on_cpu", False)
        )
        if deepspeed_zero3_cpu_init:
            # Keep the full model on CPU while DeepSpeed ZeRO-3 partitions it.
            # Moving it to GPU before prepare temporarily needs ~full 62GB per
            # rank and leaves almost no room for activations/backward recompute.
            if not prepare_dataloader:
                model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
            else:
                model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
        else:
            model.to(device=accelerator.device)
            if not prepare_dataloader:
                model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
            else:
                model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    _log_gpu_memory(accelerator, "after prepare")
    if accelerator.is_main_process:
        _param_device_census(model, accelerator, "after prepare")
    # After ``accelerator.prepare`` the object is the DeepSpeedEngine; report
    # the partition state of the wrapped module tree for comparison with the
    # pre-prepare diagnostics above.
    engine_module = getattr(model, "module", None)
    if engine_module is not None and _deepspeed_zero_stage(accelerator) == 3:
        _z3_ensure_stub_device(engine_module, accelerator)
        _zero3_partition_stats(engine_module, accelerator, "after prepare (engine.module)")
        _print_key_param_state(engine_module, accelerator, "after prepare (engine.module)")
        _report_z3_partition_footprint(engine_module, accelerator)
        _report_z3_hook_coverage(engine_module, accelerator)
    _install_z3_forward_tracer(accelerator)
    _z3_text_encoder_selftest(accelerator, model, args)
    step_count = 0
    resume_path = getattr(args, "continuation_resume_path", None)
    if resume_path:
        resume_rank_path = _continuation_ckpt_rank_path(
            resume_path, accelerator.process_index, accelerator.num_processes
        )
        checkpoint = load_continuation_training_checkpoint(
            resume_rank_path,
            expected_manifest_hash=manifest_hash,
            expected_region_config=region_config,
        )
        unwrapped = accelerator.unwrap_model(model)
        load_result = unwrapped.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if load_result.unexpected_keys:
            raise ValueError(
                f"Cannot resume continuation checkpoint {resume_path}: "
                f"{len(load_result.unexpected_keys)} unexpected model keys."
            )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step_count = int(checkpoint.get("global_step", 0))
        if checkpoint.get("rng_state") is not None:
            torch.set_rng_state(checkpoint["rng_state"])
        if checkpoint.get("random_state") is not None:
            random.setstate(checkpoint["random_state"])
        model_logger.num_steps = step_count
        if accelerator.is_main_process:
            print(
                f"Resumed continuation training from {resume_path} at global step {step_count}.",
                flush=True,
            )
    # DeepSpeed owns gradient accumulation internally (managed gradient
    # accumulation): ``accelerator.backward`` already runs the engine step and
    # ``Accelerator.accumulate`` would wrap the model in ``no_sync``, which is
    # illegal under ZeRO-3. Only the non-DeepSpeed paths use ``accumulate``.
    deepspeed_managed_gas = _deepspeed_zero_stage(accelerator) > 0
    gradient_accumulation_steps = max(
        1, int(getattr(args, "gradient_accumulation_steps", 1))
    )
    micro_steps_since_opt = 0
    for epoch_id in range(num_epochs):
        progress = tqdm(dataloader, disable=not accelerator.is_main_process,
                        desc=f"epoch {epoch_id}", dynamic_ncols=True)
        # Accumulate per-micro-step loss components across the gradient-
        # accumulation window, so the logged video/audio losses reflect the
        # whole optimizer step instead of only the final micro-step's
        # instantaneous values.
        component_accum: dict[str, list[float]] = {}
        # Optional rank0-only diagnostic for "why audio_loss never logs".
        # Enabled only when DIFFSYNTH_AUDIO_DEBUG=1; off by default so it
        # never perturbs normal runs.
        import os as _os
        _audio_debug = _os.environ.get("DIFFSYNTH_AUDIO_DEBUG", "0") == "1"
        _mic_none = _mic_comp = _mic_total = 0
        for data in progress:
            step_cm = (
                contextlib.nullcontext()
                if deepspeed_managed_gas
                else accelerator.accumulate(model)
            )
            with step_cm:
                step_start_time = time.monotonic()
                _cp_trace(
                    accelerator,
                    f"step {step_count} forward start",
                )
                try:
                    if dataset.load_from_cache:
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                except torch.cuda.OutOfMemoryError:
                    _report_oom_context(accelerator, model, "after forward")
                    raise
                _cp_trace(
                    accelerator,
                    f"step {step_count} forward done "
                    f"({time.monotonic() - step_start_time:.2f}s)",
                )
                _log_gpu_memory(accelerator, f"step {step_count} after forward")
                if _flush_cuda_cache_before_backward(accelerator):
                    # ZeRO-3 + activation checkpointing leave fragmented cached
                    # blocks behind; flush before backward needs large chunks.
                    torch.cuda.empty_cache()
                _log_gpu_memory(accelerator, f"step {step_count} before backward")
                try:
                    _run_backward_step(
                        accelerator, loss,
                        output_dir=getattr(args, "output_path", None),
                    )
                except torch.cuda.OutOfMemoryError:
                    _report_oom_context(accelerator, model, "during backward")
                    raise
                _cp_trace(
                    accelerator,
                    f"step {step_count} backward done "
                    f"({time.monotonic() - step_start_time:.2f}s)",
                )
                _log_gpu_memory(accelerator, f"step {step_count} after backward")
                if should_reduce_cp_parameter_gradients(
                    cp_world_size, cp_gradient_reduction
                ):
                    reduce_cp_parameter_gradients(model, cp_group)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                # Under accelerate+DeepSpeed the optimizer/scheduler/zero_grad
                # wrappers are no-ops: DeepSpeed's engine.step already ran inside
                # accelerator.backward and applies the weight update only on its
                # gradient-accumulation boundary. Track optimizer steps here so
                # logging/checkpoint/resume stay on optimizer-step semantics.
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # Fold this micro-step's loss components into the window
                # accumulator. The attribute set on the loss tensor can be
                # dropped when the wrapper/Distributed graph returns the loss,
                # so fall back to the module-level side channel when missing.
                step_components = getattr(loss, "continuation_components", None)
                if step_components is None:
                    try:
                        import diffsynth.diffusion.loss as _loss_mod
                        step_components = _loss_mod._LAST_CONTINUATION_COMPONENTS
                    except Exception:
                        step_components = None
                if step_components:
                    for key, val in step_components.items():
                        try:
                            fval = float(val)
                        except (TypeError, ValueError):
                            continue
                        buf = component_accum.setdefault(key, [0.0, 0])
                        buf[0] += fval
                        buf[1] += 1
                if _audio_debug and accelerator.is_main_process:
                    _mic_total += 1
                    if data.get("audio_input_latents") is None:
                        _mic_none += 1
                    if step_components is not None:
                        _mic_comp += 1
                if deepspeed_managed_gas:
                    micro_steps_since_opt += 1
                    is_optimizer_step = (
                        micro_steps_since_opt % gradient_accumulation_steps == 0
                    )
                else:
                    is_optimizer_step = accelerator.sync_gradients
                if is_optimizer_step:
                    log_loss = _average_log_loss(accelerator, loss)
                    components = {
                        key: (acc[0] / acc[1])
                        for key, acc in component_accum.items() if acc[1] > 0
                    } or None
                    component_accum.clear()
                    if _audio_debug and accelerator.is_main_process and _mic_total:
                        print(
                            f"[audiodbg] opt_step={step_count} audio=None "
                            f"micro={_mic_none}/{_mic_total} "
                            f"comp={_mic_comp}/{_mic_total}",
                            flush=True,
                        )
                        _mic_none = _mic_comp = _mic_total = 0
                    model_logger.on_step_end(
                        accelerator, model, save_steps, loss=log_loss,
                        continuation_components=components,
                    )
                    if accelerator.is_main_process:
                        progress.set_postfix(loss=f"{log_loss.item():.5f}")
                    step_count += 1
                    _cp_trace(
                        accelerator,
                        f"optimizer step done "
                        f"({time.monotonic() - step_start_time:.2f}s)",
                    )
                    if _should_save_continuation_checkpoint(args, step_count, save_steps):
                        _save_continuation_checkpoint(
                            accelerator, model, args, optimizer, scheduler,
                            step_count=step_count,
                            manifest_hash=manifest_hash,
                            region_config=region_config,
                        )
            if max_steps is not None and step_count >= max_steps:
                break
        progress.close()
        if max_steps is not None and step_count >= max_steps:
            break
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    if getattr(args, "continuation_checkpoint_save_path", None) and (
        save_steps is None
        or not _should_save_continuation_checkpoint(args, step_count, save_steps)
    ):
        _save_continuation_checkpoint(
            accelerator, model, args, optimizer, scheduler,
            step_count=step_count,
            manifest_hash=manifest_hash,
            region_config=region_config,
        )
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        model, dataloader = accelerator.prepare(model, dataloader)
    
    deepspeed_managed_gas = _deepspeed_zero_stage(accelerator) > 0
    for data_id, data in enumerate(tqdm(dataloader)):
        step_cm = (
            contextlib.nullcontext()
            if deepspeed_managed_gas
            else accelerator.accumulate(model)
        )
        with step_cm:
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
