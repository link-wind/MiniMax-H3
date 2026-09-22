# MiniMax-H3 continuation LoRA training commands

These commands use the `masked-av-v14` teacher-forced continuation task. The
default LoRA targets for `continuation_sft` are:

```text
attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2
```

The default region configuration is:

```json
{
  "overlap_video_steps": 12,
  "hard_core_video_steps": 12,
  "transition_video_steps": 0,
  "first_suffix_clip_steps": 5,
  "transition_weight": 0.5,
  "first_suffix_weight": 3.0,
  "suffix_weight": 1.0,
  "lambda_audio": 0.5,
  "conditioning_mode": "masked-av-v14"
}
```

## 124 frame CPU config check

This command validates window arithmetic, region masks, LoRA targets, loss
configuration, and CP topology without loading MiniMax-H3 weights.

```bash
PYTHONPATH="$PWD" python examples/minimax_h3/model_training/train.py \
  --dataset_base_path . \
  --num_frames 124 \
  --task continuation_sft \
  --validate_continuation_config \
  --continuation_overlap_steps 12 \
  --continuation_hard_core_steps 12 \
  --continuation_transition_steps 0 \
  --continuation_first_suffix_steps 5 \
  --continuation_transition_weight 0.5 \
  --continuation_first_suffix_weight 3.0 \
  --continuation_suffix_weight 1.0 \
  --continuation_lambda_audio 0.5 \
  --continuation_conditioning masked-av-v14
```

## 345 frame single GPU smoke

Run one optimizer step against a small latent-cache manifest and write a full
resumable checkpoint.

```bash
PYTHONPATH="$PWD" accelerate launch --num_processes 1 \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path . \
  --dataset_metadata_path /unused \
  --num_frames 345 \
  --task continuation_sft \
  --continuation_manifest outputs/h3_continuation_cache/train \
  --continuation_split train \
  --continuation_max_items 4 \
  --max_steps 1 \
  --save_steps 1 \
  --bf16 \
  --use_gradient_checkpointing \
  --training_cfg_scale 1.0 \
  --continuation_conditioning masked-av-v14 \
  --continuation_overlap_steps 12 \
  --continuation_hard_core_steps 12 \
  --continuation_transition_steps 0 \
  --lora_base_model dit \
  --lora_target_modules attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2 \
  --lora_rank 32 \
  --output_path outputs/h3_continuation_lora_smoke \
  --enable_csv_log \
  --enable_tensorboard_log \
  --continuation_checkpoint_save_path outputs/h3_continuation_lora_smoke/continuation.pt \
  --continuation_checkpoint_interval 1
```

## 345 frame formal training

The formal run can use the same cache manifest and checkpointing flags.  Adjust
`--num_processes`, `--cp_world_size`, `--learning_rate`, and `--max_steps` for
the target hardware.

```bash
PYTHONPATH="$PWD" torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path . \
  --dataset_metadata_path /unused \
  --num_frames 345 \
  --task continuation_sft \
  --continuation_manifest outputs/h3_continuation_cache/train \
  --continuation_split train \
  --bf16 \
  --use_gradient_checkpointing \
  --training_cfg_scale 1.0 \
  --continuation_conditioning masked-av-v14 \
  --continuation_overlap_steps 12 \
  --continuation_hard_core_steps 12 \
  --continuation_transition_steps 0 \
  --lora_base_model dit \
  --lora_target_modules attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2 \
  --lora_rank 32 \
  --learning_rate 1e-5 \
  --cp_world_size 2 \
  --save_steps 100 \
  --output_path outputs/h3_continuation_lora \
  --enable_csv_log \
  --enable_tensorboard_log \
  --continuation_checkpoint_save_path outputs/h3_continuation_lora/continuation.pt \
  --continuation_checkpoint_interval 100
```

## Resume from a full checkpoint

Resume must keep the same manifest, seed, region weights, LoRA target/rank, and
training scale.  The runner restores optimizer/scheduler state, global step,
RNG state, and verifies the manifest hash and region config before continuing.

```bash
PYTHONPATH="$PWD" torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path . \
  --dataset_metadata_path /unused \
  --num_frames 345 \
  --task continuation_sft \
  --continuation_manifest outputs/h3_continuation_cache/train \
  --continuation_split train \
  --bf16 \
  --use_gradient_checkpointing \
  --training_cfg_scale 1.0 \
  --continuation_conditioning masked-av-v14 \
  --continuation_overlap_steps 12 \
  --continuation_hard_core_steps 12 \
  --continuation_transition_steps 0 \
  --lora_base_model dit \
  --lora_target_modules attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2 \
  --lora_rank 32 \
  --learning_rate 1e-5 \
  --cp_world_size 2 \
  --seed 42 \
  --output_path outputs/h3_continuation_lora \
  --continuation_checkpoint_save_path outputs/h3_continuation_lora/continuation.pt \
  --continuation_checkpoint_interval 100 \
  --continuation_resume_path outputs/h3_continuation_lora/continuation.pt
```

These examples do not launch GPU training automatically from unit tests.  The
124-frame CPU config check and the new continuation checkpoint/LoRA tests are
the CI-safe subset.


## Shot-level (stage-1) variable-context build

The stage-1 task is `p(target shot | previous shot, text)` with a **per-sample**
context: `min(previous shot, 141 frames)` quantised down onto the `17n+5`
overlap grid, and a target that is the whole shot quantised down to `17k`
frames. Because every sample carries its own overlap, the run-level
`--continuation_overlap_steps` becomes a fallback only.

```bash
# 1) index: one row per adjacent shot pair, plus one prefix-free row per record
PYTHONPATH="$PWD" python examples/minimax_h3/model_training/build_continuation_dataset.py \
  --source-jsonl /path/to/data_with_face_and_speech_and_caption.jsonl \
  --output-jsonl outputs/shot_continuation_index/train.jsonl \
  --mode masked-av-v14 \
  --shot-level \
  --include-plain \
  --max-context-frames 141 \
  --require-paths

# 2) supply statistics (replays the training sampler; no media decode)
PYTHONPATH="$PWD" python examples/minimax_h3/model_training/plan_shot_continuation.py \
  --source-jsonl /path/to/data_with_face_and_speech_and_caption.jsonl \
  --output-json outputs/shot_window_plans/shot_continuation_all.json

# 3) latent cache (per-sample overlap is written into the manifest metadata)
PYTHONPATH="$PWD" python examples/minimax_h3/model_training/build_continuation_cache.py \
  --index-jsonl outputs/shot_continuation_index/train.jsonl \
  --output-dir outputs/shot_continuation_cache \
  --expanded-index \
  --h3-base /path/to/FL2VA \
  --shard-index 0 --num-shards 8

# 4) train: the loss resolves each sample's overlap from its cache metadata,
#    so no per-bucket config is needed
```

Acceptance checks before training:

- every emitted row passes `validate_continuation_layout`;
- `window_frames <= 345`, `target % 17 == 0`, `context` on `{22, 39, ..., 141}`;
- resume across a different `--continuation_overlap_steps` works (only the
  conditioning contract is fingerprinted);
- audio overlap is frame-anchored (`39 frames -> 65 steps`), never
  `steps * 65 / 12`.
