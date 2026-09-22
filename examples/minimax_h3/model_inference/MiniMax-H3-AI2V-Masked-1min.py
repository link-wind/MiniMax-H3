"""One-minute AI2V generation with native masked latent continuation.

The first window uses an image keyframe and driving audio. Later windows use
the previous clean video latent tail as a hard 39-frame prefix and receive the
next audio segment as a fixed AI2V condition. All windows are assembled in
latent space and decoded once at the end.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import soundfile as sf
from PIL import Image

from diffsynth.core import ModelConfig
from diffsynth.pipelines.h3_appearance_memory import (
    H3AppearanceMemoryBank,
    H3AppearanceMemoryConfig,
)
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline
from diffsynth.pipelines.minimax_h3_continuation import (
    H3VideoLatentContinuation,
    h3_continuation_video_latent_steps,
    is_masked_av_context_frames,
)
from diffsynth.utils.data.audio_video import write_video_audio
from diffsynth.utils.continuation_lora import apply_continuation_lora


H3_ROOT = Path("/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/models/MiniMax/MiniMax-H3/FL2VA")
IMAGE_PATH = Path("/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/data/diffsynth_example_dataset/ai2v/dianxin.png")
AUDIO_PATH = Path("/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/data/diffsynth_example_dataset/ai2v/视频3.wav")
CHECKPOINT_PATH = H3_ROOT / "transformer_lora600_v4"

PROMPT_WITH_IMAGE = (
    "For the target video, at 0.00 seconds into the target video, <Picture 1> "
    "(from [Shot 1]) is fully referenced. integrated_multimodal_description: "
    "[Shot 1] photorealistic live-action video of the same real adult Chinese man "
    "shown in <Picture 1>, preserving his exact face identity, short black hair, "
    "skin texture, facial proportions, cream jacket, light shirt, olive trousers, "
    "and original mountain-and-vehicle background. Use a natural medium portrait "
    "framing with realistic sunset lighting and a documentary music-video look. "
    "He faces the camera and sings the provided song with passionate, energetic "
    "emotional expression and lively, natural body movement throughout. The mouth "
    "is the key focus: keep the lips clearly visible and make the mouth open, close, "
    "shape, and articulate with a wide, precise range that mirrors every syllable and "
    "vowel of the sung audio, locking each lip and jaw movement tightly to the vocal "
    "timing, accents, and melodic contour of the song in real time. Mouth motion is "
    "choreographed to the audio so lips and voice stay perfectly in sync. Add strong "
    "performance energy: expressive eyes, strong but natural facial emotion, visible "
    "breaths, expressive eyebrows, rhythmic head bobs and shoulder swaying that follow "
    "the musical beat, and confident, animated hand gestures that expand and contract "
    "with the melody. Keep the whole body subtly grooving to the rhythm so the shot "
    "feels alive and dynamic, not stiff, while staying photorealistic and natural. "
    "Preserve a real human appearance and realistic skin, hair, fabric, and lighting "
    "throughout. Do not stylize, cartoonize, animate, or replace the person. Use a "
    "completely fixed, locked-off camera: the camera position, angle, zoom, and framing "
    "stay perfectly still for the entire shot, identical to the composition in <Picture 1> "
    "across every frame. No camera movement of any kind: no pan, tilt, dolly, track, "
    "crane, handheld wobble, push-in, pull-out, zoom, or rotation. The camera stays "
    "locked while the singer freely sings, gestures, and moves with lively energy inside "
    "the frame. No other characters, subtitles, logos, watermarks, or written text appear."
)
PROMPT_CONTINUATION = PROMPT_WITH_IMAGE.replace(
    "at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced. ",
    ""
).replace("established by <Picture 1>", "established at the beginning of the shot")


# Contrast prompt for normal (non-locked-off) camera motion: a natural, musical
# music-video camera language (subtle push-in / pull-out, gentle tracking and
# framing shifts that follow the artist and the song's rhythm), instead of the
# fully fixed, locked-off camera in PROMPT_WITH_IMAGE.
NORMAL_CAMERA_PROMPT_WITH_IMAGE = (
    "For the target video, at 0.00 seconds into the target video, <Picture 1> "
    "(from [Shot 1]) is fully referenced. integrated_multimodal_description: "
    "[Shot 1] photorealistic live-action music-video of the same real adult Chinese man "
    "shown in <Picture 1>, preserving his exact face identity, short black hair, "
    "skin texture, facial proportions, cream jacket, light shirt, olive trousers, "
    "and original mountain-and-vehicle background. Use a natural medium portrait "
    "framing with realistic sunset lighting and a documentary music-video look. "
    "He faces the camera and sings the provided song with passionate, energetic "
    "emotional expression and lively, natural body movement throughout. The mouth "
    "is the key focus: keep the lips clearly visible and make the mouth open, close, "
    "shape, and articulate with a wide, precise range that mirrors every syllable and "
    "vowel of the sung audio, locking each lip and jaw movement tightly to the vocal "
    "timing, accents, and melodic contour of the song in real time. Mouth motion is "
    "choreographed to the audio so lips and voice stay perfectly in sync. Add strong "
    "performance energy: expressive eyes, strong but natural facial emotion, visible "
    "breaths, expressive eyebrows, rhythmic head bobs and shoulder swaying that follow "
    "the musical beat, and confident, animated hand gestures that expand and contract "
    "with the melody. Keep the whole body subtly grooving to the rhythm so the shot "
    "feels alive and dynamic, not stiff, while staying photorealistic and natural. "
    "Use a natural, musical camera language appropriate for a music video: allow "
    "a slow, subtle push-in or pull-out, gentle tracking that follows the singer, "
    "and smooth, motivated framing shifts that accent the song's build-ups and "
    "choruses. Camera moves are smooth and steady, matching the calm-but-alive "
    "mood; no jarring, chaotic, or handheld shake and no hard cuts between different "
    "locations or shot sizes. Preserve a real human appearance and realistic skin, "
    "hair, fabric, and lighting throughout. Do not stylize, cartoonize, animate, "
    "or replace the person. No other characters, subtitles, logos, watermarks, or "
    "written text appear."
)
NORMAL_CAMERA_PROMPT_CONTINUATION = NORMAL_CAMERA_PROMPT_WITH_IMAGE.replace(
    "at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced. ",
    ""
).replace("established by <Picture 1>", "established at the beginning of the shot")
MULTI_SHOT_PERSON_ANCHOR = (
    "the same real adult Chinese man shown in <Picture 1>, preserving his exact "
    "face identity, short black hair, skin texture, facial proportions, cream "
    "jacket, light shirt, olive trousers, and the same vintage off-road vehicle "
    "from the reference image, using a matching cabin layout whenever he is inside "
    "it. The mountain world established by the reference image remains the setting. "
    "He sings the provided song with passionate, energetic emotional expression "
    "while performing the action required by the current shot. He may walk, dance, drive the vehicle, "
    "or perform beside it, but his identity, hairstyle, and outfit remain exact. "
    "The mouth is the key focus: keep the lips clearly visible and make the mouth "
    "open, close, shape, and articulate with a wide, precise range that mirrors "
    "every syllable and vowel of the sung audio, locking each lip and jaw movement "
    "tightly to the vocal timing, accents, and melodic contour of the song in real "
    "time; mouth motion is choreographed to the audio so lips and voice stay "
    "perfectly in sync. Preserve a real human appearance and realistic skin, hair, "
    "fabric, and lighting throughout; do not stylize, cartoonize, animate, or "
    "replace the person."
)

MULTI_SHOT_PERSON_CONTINUATION = (
    "the same real adult Chinese man from earlier in the video, with the exact "
    "same face identity, short black hair, skin texture, facial proportions, cream "
    "jacket, light shirt, olive trousers, and the same vintage off-road vehicle, "
    "with a matching cabin layout whenever he is inside it. The mountain world "
    "established earlier in the video remains the setting. He continues singing the "
    "provided song with passionate, energetic emotional expression while carrying "
    "out the action required by the current shot. He may walk, dance, drive, or "
    "perform beside the vehicle without changing his identity, hairstyle, or "
    "outfit. The mouth is the key focus: keep the lips clearly visible and make "
    "the mouth open, close, shape, and articulate with a wide, precise range that "
    "mirrors every syllable and vowel of the sung audio, locking each lip and jaw "
    "movement tightly to the vocal timing, accents, and melodic contour of the "
    "song in real time; mouth motion is choreographed to the audio so lips and "
    "voice stay perfectly in sync. Preserve a real human appearance and realistic "
    "skin, hair, fabric, and lighting throughout; do not stylize, cartoonize, "
    "animate, or replace the person."
)

# One camera/framing description per shot. Shot N runs from
# (cut + 12.75*(N-1)) seconds to (cut + 12.75*N) seconds, where
# cut = window_seconds / 2 (e.g. 7.1875s for a 345-frame window), except Shot 1
# which starts at 0s. Cuts therefore fall at every window center and never at a
# window boundary, so the latent continuation always stays inside one shot.
MULTI_SHOT_SHOT_SPECS = {
    1: "an opening medium portrait from the chest up with realistic sunset "
       "lighting and a documentary music-video look, the vintage vehicle and "
       "mountain landscape held behind him as he begins the song with a direct, "
       "confident performance",
    2: "a waist-up medium tracking shot as he walks a few steps toward the "
       "camera while singing, shoulders, hands, and head moving naturally with "
       "the beat, the vehicle and mountain landscape remaining visible behind him",
    3: "a knees-up medium-wide dance shot beside the vehicle, the camera "
       "tracking and circling with him so his face stays clearly readable while "
       "his arms, shoulders, torso, and legs move strongly with the rhythm",
    4: "a driver's-seat medium close-up inside the same vintage vehicle, viewed "
       "from the passenger side, his hands on the steering wheel as he sings, "
       "head and shoulders grooving subtly to the beat while sunlit mountain "
       "roads move beyond the windshield and side windows",
    5: "a low-angle knees-up dance shot beside the moving vehicle, a smooth "
       "push-in accenting a chorus as he sings and dances without pulling the "
       "camera away, the mountain sky and vehicle framing him from behind",
    6: "a medium side-tracking shot moving parallel with him as he dances or "
       "walks beside the slowly moving vehicle, his face and mouth staying "
       "clearly visible while the vehicle and mountain landscape travel through "
       "the background",
    7: "a compact medium close-up through the open passenger-side window while "
       "he drives and sings, his face and mouth unobstructed, both hands on the "
       "wheel, the cabin and moving road visible around him, no other occupant",
    8: "a medium shot as he steps out of the driver's door and continues singing, "
       "the camera arcs with him while keeping him close and centered, the same "
       "vehicle, clothing, and mountain landscape remaining consistent",
    9: "a closing knees-up medium-wide performance shot as he stands beside the "
       "front fender with one hand resting on the vehicle, the grand mountain "
       "landscape filling the background as he finishes the song, with no pull-out "
       "and his face remaining clearly readable",
}

MULTI_SHOT_FACE_SCALE_CONSTRAINT = (
    " Throughout every shot, keep the singer in the foreground at a close, "
    "medium-close, or medium shot; a knees-up medium-wide shot is the widest "
    "allowed framing. His face must remain clearly readable and must never become "
    "a small distant figure. Keep the camera close to him during dancing and "
    "driving, and treat the mountain landscape, road, and vehicle as background "
    "or nearby action rather than pulling far away."
)


def _fmt_ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def _build_multi_shot_window_prompt(
    shot_in: int,
    shot_out: int,
    start_seconds: float,
    window_seconds: float,
    *,
    use_image: bool,
) -> str:
    """Build the per-window multi-shot prompt.

    A window continues the incoming shot from its start, hard-cuts to the next
    shot at the window center (offset = window_seconds / 2), then holds the
    outgoing shot through the window end so the following window can continue
    it. The cut is thus placed strictly at a window center and never at a
    window boundary.
    """
    cut_seconds = start_seconds + window_seconds / 2.0
    if use_image:
        first_bits = (
            "For the target video, at 0.00 seconds into the target video, <Picture 1> "
            f"(from [Shot {shot_in}]) is fully referenced. integrated_multimodal_description: "
            f"[Shot {shot_in}] photorealistic live-action music-video of "
            f"{MULTI_SHOT_PERSON_ANCHOR} "
        )
    else:
        first_bits = (
            f"integrated_multimodal_description: [Shot {shot_in}] "
            "photorealistic live-action music-video of "
            f"{MULTI_SHOT_PERSON_CONTINUATION} "
        )
    return (
        f"{first_bits}"
        f"[Shot {shot_in}]: {MULTI_SHOT_SHOT_SPECS[shot_in]}. Keep the same man, "
        "the identical face, jacket, and synchronized mouth throughout. "
        f"At {_fmt_ts(cut_seconds)}, the camera cuts to a clean hard cut to a "
        f"different camera angle and framing for [Shot {shot_out}]: "
        f"{MULTI_SHOT_SHOT_SPECS[shot_out]}. Keep his identity, face, jacket, "
        "and lip-sync exactly as before even if the camera angle, framing, "
        "vehicle position, or performance action changes. The new shot may move "
        "him between dancing, the driver's seat, and the area around the vehicle, "
        "but his face, clothing, and voice remain consistent."
        f"{MULTI_SHOT_FACE_SCALE_CONSTRAINT} No other characters, subtitles, "
        "logos, watermarks, or written text appear."
    )


def _parse_shot_window_starts(text: str, windows: int) -> list[int]:
    """Parse 0-based window indices where new shots (hard cuts) begin.

    Window 0 is always a shot start.  Example: "0,3,6" with 8 windows means
    three shots covering windows 0-2, 3-5 and 6-7.
    """
    starts = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if 0 not in starts:
        starts = [0] + starts
    if any(start < 0 or start >= windows for start in starts):
        raise ValueError(f"shot-window-starts out of range for {windows} windows: {starts}")
    return starts


def _build_shot_cut_window_prompt(
    shot_no: int,
    *,
    is_cut_window: bool,
    start_seconds: float,
    use_image: bool,
) -> str:
    """Build the per-window prompt for window-boundary shot cuts.

    Shots span whole windows.  The first window of a shot is generated without
    the previous window's latent tail (hard cut at the window boundary); later
    windows of the same shot keep the identical camera framing and continue via
    the masked latent prefix.  The identity of the man is anchored either by the
    keyframe image (window 0) or by the Ref2VA reference image on cut windows.
    """
    spec = MULTI_SHOT_SHOT_SPECS[shot_no]
    if use_image:
        head = (
            "For the target video, at 0.00 seconds into the target video, <Picture 1> "
            f"(from [Shot {shot_no}]) is fully referenced. integrated_multimodal_description: "
            f"[Shot {shot_no}] photorealistic live-action music-video of "
            f"{MULTI_SHOT_PERSON_ANCHOR} "
        )
    elif is_cut_window:
        head = (
            f"At {_fmt_ts(start_seconds)}, <Picture 1> is fully referenced as the identity "
            "anchor of the new shot. The camera cuts with a clean hard cut to a new shot of "
            "the same man. "
            f"integrated_multimodal_description: [Shot {shot_no}] photorealistic live-action "
            f"music-video of {MULTI_SHOT_PERSON_ANCHOR} "
        )
    else:
        head = (
            "This window directly continues the previous window without any cut, keeping the "
            "exact same camera angle, framing, and shot size. "
            f"integrated_multimodal_description: [Shot {shot_no}] photorealistic live-action "
            f"music-video of {MULTI_SHOT_PERSON_CONTINUATION} "
        )
    return (
        f"{head}"
        f"[Shot {shot_no}]: {spec}. Keep the same man, the identical face, jacket, and "
        "synchronized mouth throughout while performing the action described for this "
        "shot. The action and vehicle position may differ from the reference image, but "
        "his identity, hairstyle, clothing, and voice stay consistent."
        f"{MULTI_SHOT_FACE_SCALE_CONSTRAINT} No other characters, subtitles, logos, "
        "watermarks, or written text appear."
    )


# H3 otherwise tends to interpret a white cutout-style keyframe as an
# underspecified subject image and completes it with a plausible gray studio.
WHITE_BACKGROUND_CONSTRAINT = (
    " Scene constraint: the entire area outside the boy is a perfectly uniform, "
    "featureless pure white (#FFFFFF) seamless background for every frame. The "
    "ground is also pure white and visually merges with the background. There is "
    "no gray studio, no colored backdrop, no floor plane, no horizon, no wall, no "
    "texture, no vignette, no gradient, no cast shadow, and no reflected shadow. "
    "Keep this pure-white isolation exactly unchanged for the entire shot."
)

GRAY_BACKGROUND_CONSTRAINT = (
    " Scene constraint: the entire area outside the boy is a perfectly uniform, "
    "featureless neutral light-gray (#D9D9D9) seamless background for every frame. "
    "The ground is also the same #D9D9D9 gray and visually merges with the background. "
    "There is no white backdrop, no gray studio gradient, no colored backdrop, no floor "
    "plane, no horizon, no wall, no texture, no vignette, no cast shadow, and no reflected "
    "shadow. Keep this uniform gray isolation exactly unchanged for the entire shot."
)

DARK_GRAY_BACKGROUND_CONSTRAINT = (
    " Scene constraint: the entire area outside the boy is a perfectly uniform, "
    "featureless neutral dark-gray (#666666) seamless background for every frame. "
    "The ground is also the same #666666 gray and visually merges with the background. "
    "There is no white backdrop, no light-gray backdrop, no gray studio gradient, no "
    "colored backdrop, no floor plane, no horizon, no wall, no texture, no vignette, "
    "no cast shadow, and no reflected shadow. Keep this uniform dark-gray isolation "
    "exactly unchanged for the entire shot."
)


def _vram_config() -> dict:
    return {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }


def _load_pipeline(h3_root: Path, checkpoint_path: Path) -> MiniMaxH3Pipeline:
    vram = _vram_config()
    return MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=[str(p) for p in sorted((h3_root / "text_encoder").glob("model*.safetensors"))], **vram),
            ModelConfig(path=[str(p) for p in sorted(checkpoint_path.glob("model-*.safetensors"))], **vram),
            ModelConfig(path=str(h3_root / "video_vae" / "source" / "model.safetensors"), **vram),
            ModelConfig(path=str(h3_root / "audio_vae" / "model.safetensors"), **vram),
        ],
        processor_config=ModelConfig(path=str(h3_root / "processor")),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 4,
    )


def _read_audio_segment(audio_path: Path, start_seconds: float, duration: float):
    # soundfile avoids the torchcodec binary dependency in the H3 environment.
    waveform, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
    start = max(0, int(round(start_seconds * sample_rate)))
    length = max(1, int(round(duration * sample_rate)))
    segment = torch.from_numpy(waveform[start:start + length].T.copy())
    if segment.shape[-1] < length:
        segment = torch.nn.functional.pad(segment, (0, length - segment.shape[-1]))
    return segment, int(sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/h3_ai2v_masked_1min_lora600_v4.mp4"))
    parser.add_argument("--h3-root", type=Path, default=H3_ROOT, help="FL2VA model root")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Transformer shard directory")
    parser.add_argument("--image", type=Path, default=IMAGE_PATH, help="First-frame image")
    parser.add_argument("--audio", type=Path, default=AUDIO_PATH, help="Driving audio")
    parser.add_argument("--lora", type=Path, default=None, help="Optional continuation LoRA safetensors")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="Continuation LoRA scale")
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--window-frames", type=int, default=345)
    parser.add_argument("--overlap-frames", type=int, default=39)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--global-scene-reference", action="store_true",
        help="Pass the original image as a Ref2VA appearance and background anchor to later windows.",
    )
    parser.add_argument(
        "--inject-image-every-window", action="store_true",
        help="Pass the original first-frame image as a Ref2VA reference to EVERY window, not just the first.",
    )
    parser.add_argument(
        "--reference-short-edge", type=int, default=512,
        help="Short edge used to encode the global scene reference.",
    )
    parser.add_argument(
        "--appearance-memory-bank", action="store_true",
        help="Use dynamic decoded-frame references from completed windows for later windows.",
    )
    parser.add_argument("--appearance-trusted-anchor-frames", type=int, default=2)
    parser.add_argument("--appearance-memory-frames", type=int, default=1)
    parser.add_argument("--appearance-boundary-reference", type=int, choices=(0, 1), default=1)
    parser.add_argument("--appearance-max-visual-references", type=int, default=4)
    parser.add_argument(
        "--white-background-prompt", action="store_true",
        help="Add an explicit pure-white, no-floor/no-shadow scene constraint to every window prompt.",
    )
    parser.add_argument(
        "--gray-background-prompt", action="store_true",
        help="Add an explicit neutral light-gray (#D9D9D9) no-floor/no-shadow constraint to every window prompt.",
    )
    parser.add_argument(
        "--dark-gray-background-prompt", action="store_true",
        help="Add an explicit neutral dark-gray (#666666) no-floor/no-shadow constraint to every window prompt.",
    )
    parser.add_argument(
        "--keep-full-timeline", action="store_true",
        help="Keep the complete assembled timeline instead of cropping to 60 seconds.",
    )
    parser.add_argument(
        "--normal-camera-motion", action="store_true",
        help="Use the normal (non-locked-off) music-video camera prompt instead of the fixed locked-off camera prompt.",
    )
    parser.add_argument(
        "--multi-shot", action="store_true",
        help="Use the multi-shot (Shot 1..N hard cuts) prompt instead of the fixed single locked-off camera prompt.",
    )
    parser.add_argument(
        "--shot-window-starts", type=str, default="",
        help="Comma-separated 0-based window indices where new shots begin (hard cuts at "
             "window boundaries); requires --multi-shot. Example: 0,3,6. Empty keeps the "
             "legacy center-of-window cut behaviour.",
    )
    parser.add_argument(
        "--match-audio-duration", action="store_true",
        help="Crop the assembled video/audio to the exact duration of the input audio.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MiniMax-H3 AI2V inference")
    if not is_masked_av_context_frames(args.overlap_frames):
        raise ValueError("overlap-frames must be a native Masked AV context such as 39")
    if args.windows < 2:
        raise ValueError("at least two windows are required for continuation")
    if args.appearance_memory_bank and args.global_scene_reference:
        raise ValueError(
            "appearance-memory-bank and global-scene-reference are separate comparisons; "
            "run only one reference policy at a time"
        )
    if sum((args.white_background_prompt, args.gray_background_prompt, args.dark_gray_background_prompt)) > 1:
        raise ValueError("background prompt constraints are mutually exclusive")
    if args.lora_scale < 0:
        raise ValueError("lora-scale must be non-negative")
    if not args.h3_root.is_dir() or not args.checkpoint.is_dir():
        raise FileNotFoundError(f"missing H3 root or transformer checkpoint: {args.h3_root}, {args.checkpoint}")
    if not args.image.exists() or not args.audio.exists():
        raise FileNotFoundError(f"missing AI2V input: {args.image} or {args.audio}")
    if args.lora is not None and not args.lora.is_file():
        raise FileNotFoundError(f"LoRA file not found: {args.lora}")

    pipe = _load_pipeline(args.h3_root, args.checkpoint)
    if args.lora is not None:
        print(json.dumps(apply_continuation_lora(pipe, args.lora, args.lora_scale), ensure_ascii=False))
    if args.multi_shot:
        prompt_with_image = None
        prompt_continuation = None
    elif args.normal_camera_motion:
        prompt_with_image = NORMAL_CAMERA_PROMPT_WITH_IMAGE
        prompt_continuation = NORMAL_CAMERA_PROMPT_CONTINUATION
    else:
        prompt_with_image = PROMPT_WITH_IMAGE
        prompt_continuation = PROMPT_CONTINUATION
    if args.white_background_prompt:
        if prompt_continuation is not None:
            prompt_continuation += WHITE_BACKGROUND_CONSTRAINT
        if prompt_with_image is not None:
            prompt_with_image += WHITE_BACKGROUND_CONSTRAINT
    if args.gray_background_prompt:
        if prompt_continuation is not None:
            prompt_continuation += GRAY_BACKGROUND_CONSTRAINT
        if prompt_with_image is not None:
            prompt_with_image += GRAY_BACKGROUND_CONSTRAINT
    if args.dark_gray_background_prompt:
        if prompt_continuation is not None:
            prompt_continuation += DARK_GRAY_BACKGROUND_CONSTRAINT
        if prompt_with_image is not None:
            prompt_with_image += DARK_GRAY_BACKGROUND_CONSTRAINT
    first_frame = Image.open(args.image).convert("RGB")
    overlap_video_steps = h3_continuation_video_latent_steps(args.overlap_frames)
    stride_frames = args.window_frames - args.overlap_frames
    window_seconds = args.window_frames / 24.0
    stride_seconds = stride_frames / 24.0

    # Optional window-boundary shot plan: each listed window index starts a new
    # shot (hard cut at the window boundary, no latent tail).  Inside a shot the
    # windows keep the masked latent continuation.
    shot_starts: list[int] | None = None
    if args.multi_shot and args.shot_window_starts:
        shot_starts = _parse_shot_window_starts(args.shot_window_starts, args.windows)
        if len(shot_starts) > len(MULTI_SHOT_SHOT_SPECS):
            raise ValueError(
                f"too many shots ({len(shot_starts)}) for available shot specs "
                f"({len(MULTI_SHOT_SHOT_SPECS)})"
            )
    shot_starts_set = set(shot_starts) if shot_starts is not None else None

    # Per-window prompts for the multi-shot (hard-cut) mode.  With a shot plan
    # the cut sits at a window boundary and each shot spans whole windows; with
    # the legacy plan every window continues its incoming shot, hard-cuts at its
    # own center, then holds the outgoing shot so the latent continuation never
    # crosses a cut.
    multi_shot_prompts = None
    if args.multi_shot:
        if shot_starts is not None:
            multi_shot_prompts = {
                idx: _build_shot_cut_window_prompt(
                    shot_no=1 + sum(1 for start in shot_starts if start <= idx),
                    is_cut_window=(idx in shot_starts_set) and idx > 0,
                    start_seconds=idx * stride_seconds,
                    use_image=(idx == 0),
                )
                for idx in range(args.windows)
            }
        else:
            multi_shot_prompts = {
                idx: _build_multi_shot_window_prompt(
                    shot_in=idx + 1,
                    shot_out=idx + 2,
                    start_seconds=idx * stride_seconds,
                    window_seconds=window_seconds,
                    use_image=(idx == 0),
                )
                for idx in range(args.windows)
            }

    # Decode one continuation suffix at a time. Keeping the whole 140-second
    # latent timeline until the end causes a large VAE peak at this resolution.
    assembled_video = []
    assembled_audio = None
    previous_video_tail = None
    previous_audio_tail = None
    overlap_audio_steps = round(args.overlap_frames / 24.0 * 40.0)
    overlap_audio_samples = round(args.overlap_frames / 24.0 * pipe.audio_vae.sample_rate)
    appearance_bank = None
    if args.appearance_memory_bank:
        appearance_bank = H3AppearanceMemoryBank(
            H3AppearanceMemoryConfig(
                mode="dynamic",
                trusted_anchor_frames=args.appearance_trusted_anchor_frames,
                memory_frame_budget=args.appearance_memory_frames,
                boundary_reference_frames=bool(args.appearance_boundary_reference),
                max_visual_references=args.appearance_max_visual_references,
            )
        )

    for index in range(args.windows):
        start_seconds = index * stride_seconds
        audio = _read_audio_segment(args.audio, start_seconds, window_seconds)
        is_cut_window = bool(shot_starts_set is not None and index in shot_starts_set and index > 0)
        if index == 0 or is_cut_window:
            # First window of a shot: no latent tail.  The very first window uses
            # the keyframe image; later hard-cut windows reuse the first-frame
            # image as a Ref2VA identity/reference anchor so the same person
            # carries across the cut.
            continuation = None
            if multi_shot_prompts is not None:
                prompt = multi_shot_prompts[index]
            else:
                prompt = prompt_with_image if index == 0 else prompt_continuation
            if index == 0:
                keyframes = [first_frame]
                keyframe_indices = [0]
            else:
                keyframes = None
                keyframe_indices = None
            references = (
                [{"type": "image", "image": first_frame}]
                if (args.inject_image_every_window or is_cut_window) else None
            )
        else:
            continuation = H3VideoLatentContinuation(
                latents=previous_video_tail,
                overlap_frames=args.overlap_frames,
                overlap_mode="hard",
            )
            if multi_shot_prompts is not None:
                prompt = multi_shot_prompts[index]
            else:
                prompt = prompt_continuation
            keyframes = None
            keyframe_indices = None
            if args.inject_image_every_window:
                references = [{"type": "image", "image": first_frame}]
            elif appearance_bank is not None:
                references = appearance_bank.build_references()
            else:
                references = (
                    [{"type": "image", "image": first_frame}]
                    if args.global_scene_reference else None
                )

        print(f"[AI2V mask] window={index + 1}/{args.windows} start={start_seconds:.3f}s")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        result = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.window_frames,
            num_inference_steps=args.steps,
            seed=args.seed,
            keyframes=keyframes,
            keyframe_indices=keyframe_indices,
            references=references,
            ref_image_short_edge=args.reference_short_edge,
            ai2v_audio=audio,
            continuation_video_latents=continuation,
            return_latents=True,
            decode_output=False,
        )
        if appearance_bank is not None:
            # These decoded frames are reference-only. They are kept per-window
            # so the appearance bank never requires a full-timeline decode.
            decoded_window, _ = pipe.decode_latent_timeline(
                video_latents=result.video_latents,
                audio_latents=None,
                tiled=True,
                tile_size=256,
                tile_overlap=64,
            )
            if index == 0:
                appearance_bank.initialize_anchors(decoded_window, segment_index=index)
            else:
                appearance_bank.update_from_window(
                    decoded_window,
                    segment_index=index,
                    overlap_frames=args.overlap_frames,
                )
            if args.appearance_boundary_reference:
                appearance_bank.set_boundary(
                    decoded_window[-1],
                    segment_index=index,
                    frame_index=len(decoded_window) - 1,
                )
            print(
                "[AI2V appearance bank] "
                + json.dumps(appearance_bank.provenance_metadata(), ensure_ascii=False)
            )
            del decoded_window
        current_video_latents = result.video_latents.detach().clone()
        current_audio_latents = result.audio_latents.detach().clone()
        if index == 0:
            decoded_video, decoded_audio = pipe.decode_latent_timeline(
                video_latents=current_video_latents,
                audio_latents=current_audio_latents,
                tiled=True,
                tile_size=256,
                tile_overlap=64,
            )
        elif is_cut_window:
            # Hard-cut window: decode the whole new window standalone and drop
            # its head (which overlaps the previous shot in wall-clock time), so
            # the assembled timeline stays on the same per-window stride grid.
            decoded_video, decoded_audio = pipe.decode_latent_timeline(
                video_latents=current_video_latents,
                audio_latents=current_audio_latents,
                tiled=True,
                tile_size=256,
                tile_overlap=64,
            )
            decoded_video = list(decoded_video)[args.overlap_frames:]
            if decoded_audio is not None:
                decoded_audio = decoded_audio[..., overlap_audio_samples:]
        else:
            decoded_video, decoded_audio = pipe.decode_continuation_suffix(
                previous_video_latents=previous_video_tail,
                current_video_latents=current_video_latents,
                previous_audio_latents=previous_audio_tail,
                current_audio_latents=current_audio_latents,
                overlap_video_steps=overlap_video_steps,
                overlap_audio_steps=overlap_audio_steps,
                overlap_video_frames=args.overlap_frames,
                overlap_audio_samples=overlap_audio_samples,
                tiled=True,
                tile_size=256,
                tile_overlap=64,
            )
        assembled_video.extend(decoded_video)
        if assembled_audio is None:
            assembled_audio = decoded_audio
        else:
            assembled_audio = torch.cat([assembled_audio, decoded_audio], dim=-1)
        # Update continuation context only after decoding this window, so the
        # next window sees this window's clean latent tail.
        previous_video_tail = current_video_latents[:, :, -overlap_video_steps:].detach().clone()
        previous_audio_tail = current_audio_latents[..., -overlap_audio_steps:].detach().clone()
        del decoded_video, decoded_audio, current_video_latents, current_audio_latents
        del result
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    video, audio = assembled_video, assembled_audio
    if args.match_audio_duration:
        # Use the source file's exact sample count as the duration authority.
        source_info = sf.info(str(args.audio))
        source_duration = float(source_info.frames) / float(source_info.samplerate)
        # The decoded H3 waveform is at audio_vae.sample_rate (32 kHz),
        # whereas the input WAV may use another rate (this one is 16 kHz).
        target_samples = int(round(source_duration * pipe.audio_vae.sample_rate))
        target_frames = min(len(video), int(round(source_duration * 24)))
    else:
        target_frames = len(video) if args.keep_full_timeline else min(len(video), 1440)
        target_samples = audio.shape[-1] if args.keep_full_timeline else 60 * 32000
    video = list(video[:target_frames])
    audio = audio[..., :target_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_video_audio(video=video, audio=audio, output_path=str(args.output), fps=24, audio_sample_rate=32000)
    if appearance_bank is not None:
        bank_report = args.output.with_suffix(".appearance_memory.json")
        bank_report.write_text(
            json.dumps(
                {
                    "overlap_frames": args.overlap_frames,
                    "window_frames": args.window_frames,
                    "windows": args.windows,
                    "bank": appearance_bank.provenance_metadata(),
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"saved {bank_report}")
    print(f"saved {args.output} frames={len(video)} audio_samples={audio.shape[-1]}")


if __name__ == "__main__":
    main()
