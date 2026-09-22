#!/usr/bin/env python3
"""Build MiniMax-H3 continuation latent caches from a JSONL index.

The command is intentionally a separate stage from training.  It streams the
index, encodes one window at a time, and writes ``<split>/<short>.pt`` plus
``manifest.jsonl``.  Use ``--fake`` for a codec/model-free smoke run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.utils.continuation_lora import (
    DEFAULT_LTM_LEAD_STEPS,
    ContinuationPair,
    ContinuationSample,
    build_latent_cache,
    build_latent_pair_cache,
    dual_memory_slot_specs,
    iter_continuation_samples,
)


class _FakeVideoEncoder:
    def __init__(self, channels: int = 24, height: int = 32, width: int = 32):
        self.channels, self.height, self.width = channels, height, width

    def __call__(self, frames):
        # The fake shape follows H3's temporal contract; values are irrelevant.
        steps = ((len(frames) - 5) // 17) * 5 + 2
        return torch.zeros(1, self.channels, steps, self.height, self.width)


class _FakeAudioEncoder:
    def __call__(self, waveform, sample_rate=None):
        return torch.zeros(2, 32, round(waveform.shape[-1] / 800))


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no H3 VAE checkpoint under {root}: {pattern}")
    return matches[0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fake", action="store_true", help="Use CPU fake encoders; no H3 checkpoint is loaded.")
    parser.add_argument("--h3-base", type=Path, default=None, help="Local FL2VA root containing video_vae/ and audio_vae/.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0, help="Stable cache shard index for multi-GPU runs.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of stable cache shards.")
    parser.add_argument("--height", type=int, default=480, help="Encoded video height (H3 pilot default: 480).")
    parser.add_argument("--width", type=int, default=832, help="Encoded video width (H3 pilot default: 832).")
    parser.add_argument(
        "--disable-video-tiling", action="store_true",
        help="Disable H3 Video VAE spatial tiling. Use only when the full 17-frame clip fits GPU memory (tested for 832x480 on H100 80GB).",
    )
    parser.add_argument(
        "--video-tile-height", type=int, default=None,
        help="Override H3 Video VAE tile height (recommended 480 for H100 80GB at 832x480).",
    )
    parser.add_argument(
        "--video-tile-width", type=int, default=None,
        help="Override H3 Video VAE tile width (recommended 832 for H100 80GB at 832x480; use 480 if VRAM headroom is tight).",
    )
    parser.add_argument("--progress", action="store_true", help="Show a tqdm progress bar while encoding.")
    parser.add_argument(
        "--memory-frames", type=int, default=0,
        help="Long-horizon memory block (M1-fast, K=1): encode the N frames immediately "
             "before each window as a separate clean latent block. N must satisfy 17n+5; "
             "39 (12 latent steps) is the M1-fast unit. 0 disables the block.",
    )
    parser.add_argument(
        "--ltm-frames", type=int, default=0,
        help="Second memory slot (LTM): encode the opening N frames of the source clip as its own "
             "clean block, anchored at --ltm-lead-steps instead of at the true (unbounded) distance. "
             "0 disables it. Use together with --memory-frames to build the dual-slot STM+LTM layout "
             "that memory-only training reads.",
    )
    parser.add_argument(
        "--ltm-lead-steps", type=int, default=None,
        help="Constant anchor of the LTM slot, in video latent steps before the window "
             "(default: diffsynth.utils.continuation_lora.DEFAULT_LTM_LEAD_STEPS).",
    )
    parser.add_argument(
        "--memory-anchor", choices=("window-start", "sequence-head"), default="window-start",
        help="Which frames the memory block is cut from. 'window-start' (STM-style) takes the "
             "N frames immediately before the window; 'sequence-head' (LTM / origin anchor) takes "
             "the opening N frames of the source clip, which is the memory-only objective's "
             "per-sequence anchor.",
    )
    parser.add_argument(
        "--cpu-prefetch", type=int, default=0,
        help="Bounded CPU media prefetch depth. 1 overlaps decode/resize/resample of the next sample with VAE encoding.",
    )
    parser.add_argument(
        "--cpu-prefetch-workers", type=int, default=1,
        help="CPU decode/resize workers per GPU process; use 2-4 only when host RAM and storage bandwidth permit.",
    )
    parser.add_argument(
        "--cpu-audio-vae", action="store_true",
        help="Keep Audio VAE on CPU and move it per sample; required with --disable-video-tiling on 80GB GPUs.",
    )
    parser.add_argument(
        "--expanded-index",
        action="store_true",
        help="Read an already expanded continuation index (video_path/start_sec fields).",
    )
    parser.add_argument(
        "--directional-pairs",
        action="store_true",
        help="Read an A/B pair index emitted with build_continuation_dataset.py --directional-pairs.",
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=None,
        help="Directory of per-fragment wavs (``<clip_stem>_origin.wav``). Defaults to "
             "``<clip_dir>/../audio/raw`` for records with ``audio_from_video``.",
    )
    parser.add_argument(
        "--audio-suffix", default="_origin.wav",
        help="Suffix appended to the clip stem when resolving fragment audio.",
    )
    parser.add_argument(
        "--reuse-root", type=Path, default=None,
        help="Root used to resolve relative ``cache_path`` values on reused records.",
    )
    parser.add_argument(
        "--no-reuse", action="store_true",
        help="Re-encode even records that point at an existing cache entry.",
    )
    return parser.parse_args()


def resize_h3_frame(frame, *, height: int, width: int):
    """Center-crop and resize one RGB frame to the H3 VAE geometry."""
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError(f"height/width must be positive multiples of 16, got {height}x{width}")
    from PIL import Image, ImageOps

    return ImageOps.fit(
        frame.convert("RGB"), (width, height), method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )


def iter_expanded_index(path: str | Path, reuse_root: Path | None = None):
    """Read records emitted by build_continuation_dataset.py.

    The expanded index already contains integer-window metadata and must not be
    passed through the raw annotation parser, which expects ``file_path`` and
    shot headers in ``prompt``.
    """
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = ("sample_id", "sequence_id", "video_path")
            missing = [key for key in required if key not in record]
            if missing:
                raise ValueError(f"expanded index line {line_number} missing: {', '.join(missing)}")
            # Reused records were already encoded by an earlier cache job and only
            # carry a pointer to that ``.pt``; everything else is re-encoded.
            reused_path = None
            if record.get("reused") and record.get("cache_path"):
                reused_path = Path(str(record["cache_path"]))
                if not reused_path.is_absolute() and reuse_root is not None:
                    reused_path = reuse_root / reused_path
            window_frames = int(record.get("window_frames", 345))
            start_sec = float(record.get("start_sec", 0.0))
            end_sec = float(record.get("end_sec", start_sec + window_frames / 24.0))
            yield ContinuationSample(
                sample_id=str(record["sample_id"]),
                sequence_id=str(record["sequence_id"]),
                video_path=str(record["video_path"]),
                audio_path=record.get("audio_path") or None,
                shot_index=int(record.get("shot_index", 0)),
                shot_start_sec=float(record.get("shot_start_sec", start_sec)),
                shot_end_sec=float(record.get("shot_end_sec", end_sec)),
                start_sec=start_sec,
                end_sec=end_sec,
                window_frames=window_frames,
                overlap_frames=int(record.get("overlap_frames", 34)),
                hard_core_frames=int(record.get("hard_core_frames", 17)),
                transition_frames=int(record.get("transition_frames", 17)),
                prompt=str(record.get("prompt", "")),
                prompt_source=str(record.get("prompt_source", "index")),
                prompt_cleaning_version=str(record.get("prompt_cleaning_version", "v1")),
                audio_missing=bool(record.get("audio_missing", not record.get("audio_path"))),
                split=str(record.get("split", "train")),
                status=str(record.get("status", "accepted")),
                conditioning_mode=str(record.get("conditioning_mode", "legacy")),
                segments=tuple(record.get("segments") or ()),
                audio_from_video=bool(record.get("audio_from_video", False)),
                prefix_mode=str(record.get("prefix_mode", "masked")),
                reused_cache_path=str(reused_path) if reused_path else None,
                cache_metadata=record.get("cache_metadata"),
            )


def iter_expanded_pairs(path: str | Path):
    """Read directional A/B records while retaining their individual cache ids."""
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("history"), dict) or not isinstance(record.get("target"), dict):
                raise ValueError(f"pair index line {line_number} is missing history or target")
            try:
                history = ContinuationSample(**record["history"])
                target = ContinuationSample(**record["target"])
                yield ContinuationPair(
                    pair_id=str(record["pair_id"]), sequence_id=str(record["sequence_id"]),
                    split=str(record["split"]), history=history, target=target,
                    schema_version=str(record.get("schema_version", "h3-continuation-pair-v1")),
                )
            except (KeyError, TypeError) as exc:
                raise ValueError(f"invalid pair index line {line_number}: {exc}") from exc


def main():
    args = parse_args()
    if args.num_shards <= 0 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise SystemExit("--height and --width must be positive multiples of 16")
    if args.cpu_prefetch < 0:
        raise SystemExit("--cpu-prefetch must be non-negative")
    if args.cpu_prefetch_workers <= 0:
        raise SystemExit("--cpu-prefetch-workers must be positive")
    if args.directional_pairs and args.cpu_prefetch:
        raise SystemExit("--cpu-prefetch is currently supported for single-window cache indexes only")
    frame_processor = partial(resize_h3_frame, height=args.height, width=args.width)
    if not args.fake and args.h3_base is None:
        raise SystemExit("provide --h3-base for real VAE encoding or use --fake")
    if args.fake:
        video_encoder, audio_encoder, video_preprocessor = _FakeVideoEncoder(), _FakeAudioEncoder(), None
    else:
        # Load only the two codecs.  Importing MiniMaxH3Pipeline also imports
        # the Qwen3-VL text encoder, which is unnecessary for cache building
        # and is not available in all training environments.
        if (args.h3_base / "vae").is_dir() and (args.h3_base / "audio_vae").is_dir():
            # The current lsj/MiniMax-H3 environment stores the released
            # Diffusers checkpoint.  Its VAE classes are independent of the
            # text encoder and support sharded safetensors directly.
            from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

            video_vae = AutoencoderKLMiniMaxH3.from_pretrained(
                str(args.h3_base / "vae"), torch_dtype=torch.bfloat16
            ).to(args.device).eval()
            if args.disable_video_tiling:
                video_vae.disable_tiling()
            elif args.video_tile_height is not None or args.video_tile_width is not None:
                tile_height = args.video_tile_height or video_vae.tile_sample_min_height
                tile_width = args.video_tile_width or video_vae.tile_sample_min_width
                video_vae.enable_tiling(
                    tile_sample_min_height=tile_height,
                    tile_sample_min_width=tile_width,
                )
            audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
                str(args.h3_base / "audio_vae"), torch_dtype=torch.bfloat16
            ).to("cpu" if args.cpu_audio_vae else args.device).eval()

            def video_preprocessor(frames):
                # Diffusers H3 VAE expects ImageNet-normalized RGB pixels.
                import numpy as np
                pixels = torch.stack(
                    [torch.from_numpy(np.asarray(frame, dtype=np.float32) / 255.0).permute(2, 0, 1) for frame in frames],
                    dim=1,
                ).unsqueeze(0).to(device=args.device, dtype=torch.float32)
                mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
                std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
                return ((pixels - mean) / std).to(dtype=torch.bfloat16)

            def video_wrapper(video_input):
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    posterior = video_vae.encode(video_input, return_dict=False)[0]
                latents = posterior.mode()
                mean = latents.new_tensor(video_vae.config.latents_mean).view(1, -1, 1, 1, 1)
                std = latents.new_tensor(video_vae.config.latents_std).view(1, -1, 1, 1, 1)
                return ((latents - mean) / std).to(torch.bfloat16)

            def audio_wrapper(waveform, _sample_rate=None):
                sample = waveform.to(device=args.device, dtype=torch.float32)[:, None]
                sample = sample.to(dtype=torch.bfloat16)
                if args.cpu_audio_vae:
                    audio_vae.to(args.device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    posterior = audio_vae.encode(sample, return_dict=False)[0]
                if args.cpu_audio_vae:
                    audio_vae.to("cpu")
                latents = posterior.mode()
                mean = latents.new_tensor(audio_vae.config.latents_mean).view(1, -1, 1)
                std = latents.new_tensor(audio_vae.config.latents_std).view(1, -1, 1)
                return ((latents - mean) / std).to(torch.bfloat16)

            video_encoder, audio_encoder = video_wrapper, audio_wrapper
        else:
            from diffsynth.core.loader.model import load_model
            from diffsynth.models.minimax_h3_video_vae import MiniMaxH3VideoVAE
            from diffsynth.models.minimax_h3_audio_vae import MiniMaxH3AudioVAE
            from diffsynth.utils.state_dict_converters.minimax_h3_video_vae import MiniMaxH3VideoVAEStateDictConverter
            from diffsynth.utils.state_dict_converters.minimax_h3_audio_vae import MiniMaxH3AudioVAEStateDictConverter

            video_path = _find_one(args.h3_base, "video_vae/source/model*.safetensors")
            audio_path = _find_one(args.h3_base, "audio_vae/model*.safetensors")
            video_encoder = load_model(
                MiniMaxH3VideoVAE, str(video_path), torch_dtype=torch.bfloat16,
                device=args.device, state_dict_converter=MiniMaxH3VideoVAEStateDictConverter,
                use_disk_map=True,
            )
            audio_vae = load_model(
                MiniMaxH3AudioVAE, str(audio_path), torch_dtype=torch.bfloat16,
                device=args.device, state_dict_converter=MiniMaxH3AudioVAEStateDictConverter,
                use_disk_map=True,
            )

            def video_preprocessor(frames):
                import numpy as np
                return torch.stack(
                    [torch.from_numpy(np.asarray(frame, dtype=np.float32) / 255.0).permute(2, 0, 1) for frame in frames],
                    dim=1,
                ).unsqueeze(0).to(device=args.device, dtype=torch.float32)

            def audio_wrapper(waveform, _sample_rate=None):
                return audio_vae.encode_audio(waveform.to(device=args.device, dtype=torch.float32), dtype=torch.bfloat16)

            audio_encoder = audio_wrapper

    if args.directional_pairs and not args.expanded_index:
        raise SystemExit("--directional-pairs requires --expanded-index")
    source_samples = (
        iter_expanded_pairs(args.index_jsonl)
        if args.directional_pairs else (
            iter_expanded_index(args.index_jsonl, reuse_root=args.reuse_root)
            if args.expanded_index else iter_continuation_samples(args.index_jsonl, require_paths=True)
        )
    )
    if args.num_shards > 1:
        def shard_samples():
            for sample in source_samples:
                identifier = sample.pair_id if args.directional_pairs else sample.sample_id
                digest = hashlib.sha256(identifier.encode("utf-8")).digest()
                if int.from_bytes(digest[:8], "big") % args.num_shards == args.shard_index:
                    yield sample
        samples = shard_samples()
    else:
        samples = source_samples
    if args.progress:
        try:
            from tqdm import tqdm
            samples = tqdm(samples, desc=f"shard {args.shard_index}/{args.num_shards}", unit="sample")
        except ImportError:
            pass
    # The VAE remains single-sample to bound GPU memory.  Optional CPU prefetch
    # overlaps only media decode/resize/resample with the prior VAE encode.
    build = build_latent_pair_cache if args.directional_pairs else build_latent_cache
    limit_name = "max_pairs" if args.directional_pairs else "max_samples"
    build_kwargs = {
        "video_encoder": video_encoder,
        "audio_encoder": audio_encoder,
        "frame_processor": frame_processor,
        "video_preprocessor": video_preprocessor,
        "audio_dir": args.audio_dir,
        "audio_suffix": args.audio_suffix,
        "split": args.split,
        limit_name: args.max_samples,
        "overwrite": args.overwrite,
        "video_height": args.height,
        "video_width": args.width,
        "video_resize_mode": "center_crop_bicubic",
    }
    if args.memory_frames or args.ltm_frames:
        # Memory slots are extra, much smaller VAE clips; they are encoded inside
        # the same sample loop so a cache entry is always self-contained.
        ltm_lead = DEFAULT_LTM_LEAD_STEPS if args.ltm_lead_steps is None else args.ltm_lead_steps
        if args.ltm_frames:
            build_kwargs["memory_specs"] = dual_memory_slot_specs(
                args.memory_frames, args.ltm_frames, ltm_lead_steps=ltm_lead,
            )
        else:
            build_kwargs["memory_frames"] = args.memory_frames
            build_kwargs["memory_anchor"] = args.memory_anchor
    if not args.directional_pairs:
        build_kwargs["cpu_prefetch"] = args.cpu_prefetch
        build_kwargs["cpu_prefetch_workers"] = args.cpu_prefetch_workers
        build_kwargs["reuse_cache"] = not args.no_reuse
    stats = build(samples, args.output_dir, **build_kwargs)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
