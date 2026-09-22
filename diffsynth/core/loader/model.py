from ..vram.initialization import skip_model_initialization
from ..vram.disk_map import DiskMap
from ..vram.layers import enable_vram_management
from .file import load_state_dict, load_metadata_from_safetensors
import json
import os
import torch
from contextlib import contextmanager
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import ContextManagers


def _is_main_rank() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return os.environ.get("LOCAL_RANK", "0") == "0"


# ``is_deepspeed_zero3_enabled()`` relies on a *weak* reference to the active
# transformers HfDeepSpeedConfig. Keep our re-registered copy alive for the
# whole process so parameters created later still partition under ZeRO-3.
_HF_DS_CONFIG_KEEPALIVE = None


def _re_register_zero3_config_if_needed() -> bool:
    """Return True when ZeRO-3 construction is (or becomes) active.

    ``load_model`` constructs every sub-model under
    ``deepspeed.zero.Init(remote_device=...)`` only when
    ``transformers.integrations.is_deepspeed_zero3_enabled()`` reports stage 3.
    That flag is backed by a weak reference that accelerate registers when its
    DeepSpeed plugin is selected; if model loading happens before that weak
    reference is (re)established, every parameter silently becomes a full plain
    CPU tensor and ZeRO-3 training later crashes on the first forward with a
    CPU/CUDA device mismatch. As a safety net, re-register the HfDeepSpeedConfig
    from the ``ACCELERATE_DEEPSPEED_CONFIG_FILE`` environment variable here.
    """
    global _HF_DS_CONFIG_KEEPALIVE
    if is_deepspeed_zero3_enabled():
        return True
    config_file = os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE", "none")
    if config_file in (None, "", "none"):
        return False
    # Only self-heal when this process is actually running an accelerate +
    # DeepSpeed session with ZeRO-3 model construction enabled. Other flows
    # (e.g. DiffSynth's OffloadTrainingManager) load plain CPU models while the
    # environment may still carry stale DeepSpeed variables.
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") != "true":
        return False
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        try:
            import yaml

            with open(config_file, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            config = raw.get("deepspeed_config", raw)
        except Exception:
            return False
    try:
        stage = int((config.get("zero_optimization") or {}).get("stage", 0))
    except (TypeError, ValueError):
        return False
    if stage != 3:
        return False
    zero3_init_env = os.environ.get("ACCELERATE_DEEPSPEED_ZERO3_INIT")
    if zero3_init_env is not None and zero3_init_env.lower() not in ("true", "1"):
        return False
    try:
        from transformers.integrations import HfDeepSpeedConfig
    except ImportError:
        try:
            from transformers.integrations.deepspeed import HfDeepSpeedConfig
        except ImportError:
            return False
    # Mirror accelerate's normalization in ``DeepSpeedPlugin.set_deepspeed_weakref``:
    # ``auto`` batch sizes are not valid inside HfDeepSpeedConfig.
    config = dict(config)
    if config.get("train_batch_size") == "auto":
        config.pop("train_batch_size", None)
    if config.get("gradient_accumulation_steps") in (None, "auto"):
        config["gradient_accumulation_steps"] = 1
    if config.get("train_micro_batch_size_per_gpu") in (None, "auto"):
        config["train_micro_batch_size_per_gpu"] = 1
    try:
        _HF_DS_CONFIG_KEEPALIVE = HfDeepSpeedConfig(config)
    except Exception:
        return False
    return is_deepspeed_zero3_enabled()


def _print_partition_probe(model, tag: str) -> None:
    """Print ZeRO-3 partition state of the first few params (main rank only)."""
    if not _is_main_rank():
        return
    try:
        import torch.distributed as dist

        world = dist.get_world_size() if dist.is_initialized() else None
    except Exception:
        world = None
    printed = 0
    for name, param in model.named_parameters():
        if printed >= 4:
            break
        ds_tensor = getattr(param, "ds_tensor", None)
        print(
            f"[z3load][probe-{tag}] {name}: dev={param.device} shape={tuple(param.shape)} "
            f"ds_id={getattr(param, 'ds_id', None)} st={getattr(param, 'ds_status', None)} "
            f"ds_numel={getattr(param, 'ds_numel', None)} "
            f"ds_t={None if ds_tensor is None else tuple(ds_tensor.shape)} "
            f"pw={getattr(param, 'ds_zero_partition_world_size', None)} world={world}",
            flush=True,
        )
        printed += 1


def load_model(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, use_disk_map=False, module_map=None, vram_config=None, vram_limit=None, state_dict=None, quantize=None):
    config = {} if config is None else config
    if _is_main_rank():
        print(
            f"[z3load] load_model {model_class} zero3_enabled={is_deepspeed_zero3_enabled()} "
            f"device={device} torch_dtype={torch_dtype}",
            flush=True,
        )
    with ContextManagers(get_init_context(torch_dtype=torch_dtype, device=device)):
        model = model_class(**config)
    if is_deepspeed_zero3_enabled():
        _print_partition_probe(model, "after-construct")
    # What is `module_map`?
    # This is a module mapping table for VRAM management.
    if module_map is not None and quantize is None:
        devices = [vram_config["offload_device"], vram_config["onload_device"], vram_config["preparing_device"], vram_config["computation_device"]]
        device = [d for d in devices if d != "disk"][0]
        dtypes = [vram_config["offload_dtype"], vram_config["onload_dtype"], vram_config["preparing_dtype"], vram_config["computation_dtype"]]
        dtype = [d for d in dtypes if d != "disk"][0]
        if vram_config["offload_device"] != "disk":
            if state_dict is None: state_dict = DiskMap(path, device, torch_dtype=dtype)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            else:
                state_dict = {i: state_dict[i] for i in state_dict}
            model.load_state_dict(state_dict, assign=True)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit)
        else:
            disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit)
    elif quantize is not None and module_map is not None:
        if "disk" in vram_config.values():
            if not quantize.load_prequantized:
                raise ValueError("Disk offload with quantization is only supported for pre-quantized checkpoints (load_prequantized=True).")
            devices = [vram_config[k] for k in ("offload_device", "onload_device", "preparing_device", "computation_device")]
            load_device = [d for d in devices if d != "disk"][0]
            disk_map = DiskMap(path, load_device, torch_dtype=None, state_dict_converter=state_dict_converter)
            metadata = load_metadata_from_safetensors(path)
            model = quantize.prepare_for_prequantized_load(model, compute_dtype=vram_config["computation_dtype"])
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit, quantize=quantize, metadata=metadata)
        else:
            offload_device = vram_config["offload_device"]
            computation_device = vram_config["computation_device"]
            computation_dtype = vram_config["computation_dtype"]
            load_dtype = None if quantize.load_prequantized else computation_dtype
            if state_dict is None: state_dict = DiskMap(path, offload_device, torch_dtype=load_dtype)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            else:
                state_dict = {i: state_dict[i] for i in state_dict}

            if quantize.load_prequantized:
                model = quantize.prepare_for_prequantized_load(model, compute_dtype=computation_dtype)
                state_dict = quantize.unflatten_state_dict(state_dict, load_metadata_from_safetensors(path))

            model.load_state_dict(state_dict, assign=True)
            state_dict = None

            model = quantize.quantize_model(model, compute_device=computation_device, model_device=offload_device)
            model = quantize.dequantize_model(model, compute_dtype=computation_dtype, compute_device=computation_device, model_device=offload_device)
            model = model.to(dtype=computation_dtype, device=offload_device)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit, quantize=quantize)
    elif quantize is not None:
        # Weight-only quantization (see `diffsynth.core.quant`), isolated from the normal path below.
        if quantize.load_prequantized:
            load_device, load_dtype = device, None
        else:
            load_device, load_dtype = "cpu", torch_dtype

        if state_dict is not None:
            pass
        elif use_disk_map:
            state_dict = DiskMap(path, load_device, torch_dtype=load_dtype)
        else:
            state_dict = load_state_dict(path, load_dtype, load_device)

        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        else:
            state_dict = {i: state_dict[i] for i in state_dict}

        if quantize.load_prequantized:
            model = quantize.prepare_for_prequantized_load(model, compute_dtype=torch_dtype or torch.bfloat16)
            state_dict = quantize.unflatten_state_dict(state_dict, load_metadata_from_safetensors(path))

        model.load_state_dict(state_dict, assign=True)
        model = quantize.quantize_model(model, compute_device=device, model_device=device)
        model = quantize.dequantize_model(model, compute_dtype=torch_dtype or torch.bfloat16)
        model = model.to(dtype=torch_dtype, device=device)
    else:
        # Why do we use `DiskMap`?
        # Sometimes a model file contains multiple models,
        # and DiskMap can load only the parameters of a single model,
        # avoiding the need to load all parameters in the file.
        if state_dict is not None:
            pass
        elif use_disk_map:
            state_dict = DiskMap(path, device, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict(path, torch_dtype, device)
        # Why do we use `state_dict_converter`?
        # Some models are saved in complex formats,
        # and we need to convert the state dict into the appropriate format.
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        else:
            state_dict = {i: state_dict[i] for i in state_dict}
        # Why does DeepSpeed ZeRO Stage 3 need to be handled separately?
        # Because at this stage, model parameters are partitioned across multiple GPUs.
        # Loading them directly could lead to excessive GPU memory consumption.
        if is_deepspeed_zero3_enabled():
            if _is_main_rank():
                print(f"[z3load] loading {model_class} through zero3 state dict loader", flush=True)
            from transformers.integrations.deepspeed import _load_state_dict_into_zero3_model
            _load_state_dict_into_zero3_model(model, state_dict)
            _print_partition_probe(model, "after-zero3-load")
        else:
            if _is_main_rank():
                print(f"[z3load] loading {model_class} through plain load_state_dict(assign=True)", flush=True)
            model.load_state_dict(state_dict, assign=True)
        # Why do we call `to()`?
        # Because some models override the behavior of `to()`,
        # especially those from libraries like Transformers.
        model = model.to(dtype=torch_dtype, device=device)
    if quantize is not None:
        # Downstream steps (e.g. LoRA hot-loading) need the config to handle the quantized layers.
        model.quantize_config = quantize
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def load_model_with_disk_offload(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, module_map=None):
    if isinstance(path, str):
        path = [path]
    config = {} if config is None else config
    with skip_model_initialization():
        model = model_class(**config)
    if hasattr(model, "eval"):
        model = model.eval()
    disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": device,
        "computation_dtype": torch_dtype,
        "computation_device": device,
    }
    enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=80)
    return model


def get_init_context(torch_dtype, device):
    zero3_active = _re_register_zero3_config_if_needed()
    if zero3_active:
        if _is_main_rank():
            print(
                f"[z3load] constructing under deepspeed.zero.Init(remote_device={device}, dtype={torch_dtype})",
                flush=True,
            )
        from transformers.modeling_utils import set_zero3_state
        import deepspeed
        # Why do we use "deepspeed.zero.Init"?
        # Weight segmentation of the model can be performed on the CPU side
        # and loading the segmented weights onto the computing card
        init_kwargs = {"remote_device": device, "dtype": torch_dtype}
        try:
            import torch.distributed as dist

            # Never rely on DeepSpeed's lazily-initialized world group for the
            # load-time partition: if it is not ready yet DeepSpeed can end up
            # partitioning each parameter "across" a single rank, leaving full
            # per-rank copies on CPU. Pin the explicit torch WORLD group when
            # distributed training is already active.
            if dist.is_initialized() and dist.get_world_size() > 1:
                init_kwargs["data_parallel_group"] = dist.group.WORLD
        except Exception:
            pass
        init_contexts = [deepspeed.zero.Init(**init_kwargs), set_zero3_state()]
    else:
        if _is_main_rank():
            print(
                f"[z3load] constructing {torch_dtype} with skip_model_initialization (zero3 disabled)",
                flush=True,
            )
        # Why do we use `skip_model_initialization`?
        # It skips the random initialization of model parameters,
        # thereby speeding up model loading and avoiding excessive memory usage.
        init_contexts = [skip_model_initialization()]

    return init_contexts
