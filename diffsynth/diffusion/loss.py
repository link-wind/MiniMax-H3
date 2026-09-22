from .base_pipeline import BasePipeline
from ..models.minimax_h3_dit import h3_local_target_rows
from ..core.context_parallel import (
    broadcast_cp_tensor,
    h3_cp_loss,
    mse_local_sum_count,
    reduce_cp_weighted_mean,
    split_sequence_indices,
)
from dataclasses import replace

import torch

from ..utils.continuation_lora import (
    ContinuationRegionConfig,
    continuation_flow_loss,
    overlap_frames_to_audio_steps,
    resolve_overlap_video_steps,
    v3_continuation_inputs,
    video_steps_to_frames,
)


# Explicit side channel for per-modality loss components. The custom attribute
# set on the returned loss tensor is dropped when the tensor is wrapped by the
# accelerator/DeepSpeed graph, so logging reads this module-level variable
# instead when the attribute is missing.
_LAST_CONTINUATION_COMPONENTS = None

def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    if "lora" in inputs:
        # Image-to-LoRA models need to load lora here.
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def FlowMatchSFTMiniMaxH3AudioVideoLoss(
    pipe: BasePipeline,
    training_cfg_scale: float = 1.0,
    inputs_nega: dict | None = None,
    cp_rank=0,
    cp_world_size=1,
    cp_group=None,
    **inputs,
):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    use_cp_broadcast = cp_world_size is not None and cp_world_size > 1 and cp_group is not None
    if cp_rank == 0 or not use_cp_broadcast:
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep_video = pipe.scheduler.timesteps[timestep_id].to(dtype=torch.float32, device=pipe.device)
        timestep_audio = pipe.scheduler_audio.timesteps[timestep_id].to(dtype=torch.float32, device=pipe.device)
        noise = torch.randn_like(inputs["input_latents"])
        audio_noise = (
            torch.randn_like(inputs["audio_input_latents"])
            if "audio_input_latents" in inputs
            else None
        )
    else:
        timestep_video = torch.empty((1,), dtype=torch.float32, device=pipe.device)
        timestep_audio = torch.empty((1,), dtype=torch.float32, device=pipe.device)
        noise = torch.empty_like(inputs["input_latents"])
        audio_noise = (
            torch.empty_like(inputs["audio_input_latents"])
            if "audio_input_latents" in inputs
            else None
        )
    timestep_video = broadcast_cp_tensor(timestep_video, cp_rank, cp_world_size, cp_group)
    timestep_audio = broadcast_cp_tensor(timestep_audio, cp_rank, cp_world_size, cp_group)
    noise = broadcast_cp_tensor(noise, cp_rank, cp_world_size, cp_group)
    if audio_noise is not None:
        audio_noise = broadcast_cp_tensor(audio_noise, cp_rank, cp_world_size, cp_group)

    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep_video)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep_video)

    if "audio_input_latents" in inputs:
        inputs["audio_latents"] = pipe.scheduler_audio.add_noise(inputs["audio_input_latents"], audio_noise, timestep_audio)
        training_target_audio = pipe.scheduler_audio.training_target(inputs["audio_input_latents"], audio_noise, timestep_audio)

    inputs["cp_rank"] = cp_rank
    inputs["cp_world_size"] = cp_world_size
    inputs["cp_group"] = cp_group

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    if training_cfg_scale > 1.0:
        if not inputs_nega:
            raise ValueError(
                "MiniMax-H3 CFG-aware training requires unconditional inputs. "
                "When using split training, rebuild the data cache with the same "
                "--training_cfg_scale value."
            )
        inputs_uncond = {**inputs, **inputs_nega}
        inputs_uncond["use_gradient_checkpointing"] = False
        inputs_uncond["use_gradient_checkpointing_offload"] = False
        with torch.no_grad():
            noise_pred_uncond, noise_pred_audio_uncond = pipe.model_fn(
                **models, **inputs_uncond,
                timestep_video=timestep_video, timestep_audio=timestep_audio,
            )

    noise_pred, noise_pred_audio = pipe.model_fn(
        **models, **inputs,
        timestep_video=timestep_video, timestep_audio=timestep_audio,
    )

    if training_cfg_scale > 1.0:
        # The checkpoint's conditional prediction has CFG distilled into it.
        # Rearrange the CFG equation to recover the raw velocity fitted to the
        # standard flow-matching target, using the current model as the teacher.
        noise_pred = (noise_pred + (training_cfg_scale - 1.0) * noise_pred_uncond) / training_cfg_scale
        noise_pred_audio = (noise_pred_audio + (training_cfg_scale - 1.0) * noise_pred_audio_uncond) / training_cfg_scale

    use_local_cp = cp_world_size is not None and cp_world_size > 1 and cp_group is not None
    if "audio_input_latents" in inputs:
        if use_local_cp:
            cond_anchor = inputs.get("ref_visual_anchor")
            if cond_anchor is None:
                cond_anchor = inputs.get("keyframe_cond_anchor")
            cond_rows_count = (
                0 if cond_anchor is None else cond_anchor.shape[0]
            )
            ref_audio_anchor = inputs.get("ref_audio_anchor")
            ref_audio_rows_count = (
                0
                if ref_audio_anchor is None
                else ref_audio_anchor.shape[0]
            )
            target_video_local, target_audio_local = h3_local_target_rows(
                training_target,
                training_target_audio,
                inputs["packed"],
                cp_rank=cp_rank,
                cp_world_size=cp_world_size,
                cond_rows_count=cond_rows_count,
                ref_audio_rows_count=ref_audio_rows_count,
            )
            video_sum, video_count = mse_local_sum_count(
                noise_pred,
                target_video_local,
                weight=pipe.scheduler.training_weight(timestep_video),
            )
            audio_sum, audio_count = mse_local_sum_count(
                noise_pred_audio,
                target_audio_local,
                weight=pipe.scheduler_audio.training_weight(timestep_audio),
            )
            return h3_cp_loss(
                video_sum,
                video_count,
                audio_sum,
                audio_count,
                group=cp_group,
            )

        video_sum, video_count = mse_local_sum_count(
            noise_pred,
            training_target,
            weight=pipe.scheduler.training_weight(timestep_video),
        )
        audio_sum, audio_count = mse_local_sum_count(
            noise_pred_audio,
            training_target_audio,
            weight=pipe.scheduler_audio.training_weight(timestep_audio),
        )
        return h3_cp_loss(
            video_sum,
            video_count,
            audio_sum,
            audio_count,
            group=cp_group,
        )
    video_sum, video_count = mse_local_sum_count(
        noise_pred,
        training_target,
        weight=pipe.scheduler.training_weight(timestep_video),
    )
    return reduce_cp_weighted_mean(video_sum, video_count, group=cp_group)


def _draw_clean_prefix(config, cp_rank, cp_world_size, cp_group, use_cp, device):
    """Draw whether this micro-batch presents the overlap prefix as clean
    (denoise_mask=0 / timestep=1.0, as masked-av-v14 inference does) instead of
    noised-at-t.  ``mixed`` flips a per-micro-batch coin that stays identical
    inside each CP group, mirroring the timestep draw."""
    mode = config.prefix_present_mode
    if mode == "noised":
        return False
    if mode == "clean":
        return True
    # mixed: ~50/50, synchronized across the CP group like the timestep draw.
    if cp_rank == 0 or not use_cp:
        draw = (torch.rand(1, device=device) < 0.5).to(torch.long)
    else:
        draw = torch.empty((1,), dtype=torch.long, device=device)
    if use_cp:
        draw = broadcast_cp_tensor(draw, cp_rank, cp_world_size, cp_group)
    return bool(draw.item())


def _clean_prefix_injection(history_clean, full_latents, time_dim):
    """Return (input_latents, denoise_mask) reproducing the masked-av-v14
    inference embedder: prefix rows are injected as clean latents with a
    0-valued denoise mask (model_fn then marks their timestep as 1.0), while
    the remaining rows keep the noised latents (denoise mask 1)."""
    tdim = time_dim % full_latents.ndim
    prefix_steps = history_clean.shape[tdim]
    input_latents = torch.zeros_like(full_latents)
    input_latents.narrow(tdim, 0, prefix_steps).copy_(history_clean)
    mask = torch.ones(full_latents.shape[tdim], dtype=full_latents.dtype, device=full_latents.device)
    mask[:prefix_steps] = 0.0
    if full_latents.ndim >= 5:
        denoise_mask = mask.view(1, 1, -1, 1, 1).expand(-1, -1, -1, *full_latents.shape[-2:])
    else:
        denoise_mask = mask.view(1, 1, -1).expand(full_latents.shape[0], -1, -1)
    return input_latents, denoise_mask


def ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
    pipe: BasePipeline,
    training_cfg_scale: float = 1.0,
    inputs_nega: dict | None = None,
    region_config: ContinuationRegionConfig | None = None,
    cp_rank=0,
    cp_world_size=1,
    cp_group=None,
    **inputs,
):
    """Teacher-forced continuation loss for the v2/v3 LoRA objectives.

    This is opt-in and deliberately separate from the existing H3 SFT loss. The
    caller supplies clean target latents plus optional clean history latents;
    when history is absent, the target overlap is used as a deterministic teacher.
    """
    global _LAST_CONTINUATION_COMPONENTS
    _LAST_CONTINUATION_COMPONENTS = None
    if training_cfg_scale < 1.0:
        raise ValueError("training_cfg_scale must be at least 1.0")
    config = region_config or ContinuationRegionConfig()
    # Shot-level training gives every sample the overlap its own context needs,
    # so the run-level ``overlap_video_steps`` is only the fallback for caches
    # that do not record one.  masked-av-v14 keeps hard_core == overlap and
    # transition == 0, so the per-sample value replaces both.
    sample_overlap = resolve_overlap_video_steps(
        inputs.get("continuation_cache_metadata"), config.overlap_video_steps,
    )
    if sample_overlap != config.overlap_video_steps:
        config = replace(
            config,
            overlap_video_steps=sample_overlap,
            hard_core_video_steps=sample_overlap,
            transition_video_steps=0,
        )
    target_video = inputs["input_latents"]
    history_video = inputs.get("continuation_history_video_latents", target_video)
    # "masked" is the v3/v5 teacher-forced continuation window (39-frame prefix
    # present).  "none" trains a bare flow-matching window with no prefix at all,
    # i.e. the hard-cut / new-shot case the inference script can also request.
    prefix_mode = str(inputs.get("continuation_prefix_mode", "masked"))
    if prefix_mode not in ("masked", "none"):
        raise ValueError(f"unsupported continuation prefix_mode={prefix_mode!r}")
    plain_window = prefix_mode == "none"
    config.validate(target_video.shape[-3])
    use_cp = cp_world_size is not None and cp_world_size > 1 and cp_group is not None
    if cp_rank == 0 or not use_cp:
        timestep_id = torch.randint(0, len(pipe.scheduler.timesteps), (1,), device=target_video.device)
    else:
        timestep_id = torch.empty((1,), dtype=torch.long, device=target_video.device)
    if use_cp:
        timestep_id = broadcast_cp_tensor(timestep_id, cp_rank, cp_world_size, cp_group)
    # masked-av-v14 inference feeds the overlap prefix to the DiT as clean
    # latents (input_latents_* / denoise_mask_* with timestep=1.0), while the
    # v3 objective only presents a noised-at-t prefix.  Sampling the clean form
    # (prefix_present_mode=clean/mixed) closes that train/inference gap.
    use_clean_prefix = False if plain_window else _draw_clean_prefix(
        config, cp_rank, cp_world_size, cp_group, use_cp, target_video.device,
    )
    # Schedulers keep their lookup tables on CPU, while cached latents are
    # transferred to the training device.  Move the small timestep tables (or
    # the index) before indexing to avoid a cross-device advanced-index error.
    scheduler_timesteps = pipe.scheduler.timesteps.to(device=target_video.device)
    scheduler_audio_timesteps = pipe.scheduler_audio.timesteps.to(device=target_video.device)
    timestep_id = timestep_id.to(device=target_video.device)
    timestep_video = scheduler_timesteps[timestep_id].to(dtype=torch.float32)
    timestep_audio = scheduler_audio_timesteps[timestep_id].to(dtype=torch.float32)
    video_noise = torch.randn_like(target_video)
    if config.conditioning_mode == "masked-av-v14":
        if plain_window:
            # No history rows: every video latent is a noised prediction target.
            video_latents = pipe.scheduler.add_noise(target_video, video_noise, timestep_video)
            video_target = video_noise - target_video
        else:
            # v14: hard masked AV prefix.  History is only the overlap prefix and
            # shares the exact noise used by the target prefix, matching inference.
            overlap = config.overlap_video_steps
            if history_video is None:
                raise ValueError("masked-av-v14 requires continuation_history_video_latents")
            if history_video.shape == target_video.shape:
                history_video = history_video.narrow(-3, 0, overlap)
            if history_video.shape[-3] != overlap:
                raise ValueError("masked-av-v14 history must contain exactly the overlap prefix")
            video_latents, video_target, _ = v3_continuation_inputs(
                target_video, history_video, video_noise, timestep_video, pipe.scheduler.add_noise,
                hard_core=overlap, transition=0, transition_start_weight=1.0,
                transition_end_weight=1.0, time_dim=-3,
            )
            if use_clean_prefix:
                inputs["input_latents_video"], inputs["denoise_mask_video"] = _clean_prefix_injection(
                    history_video, video_latents, time_dim=-3,
                )
    inputs["video_latents"] = video_latents
    audio_target_clean = inputs.get("audio_input_latents")
    import os as _os_dbg
    if (_os_dbg.environ.get("DIFFSYNTH_AUDIO_DEBUG") == "1"
            and not getattr(ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss, "_adbg0", False)):
        ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss._adbg0 = True
        _ac = audio_target_clean
        print(f"[adbg] audio_target_clean=None/"
              f"{None if _ac is None else tuple(_ac.shape)}", flush=True)
    audio_target = None
    audio_weights = None
    audio_noise = None
    if audio_target_clean is not None:
        audio_history = inputs.get("continuation_history_audio_latents", audio_target_clean)
        audio_noise = torch.randn_like(audio_target_clean)
        if config.conditioning_mode == "masked-av-v14":
            # The video->audio overlap is anchored on frames, not on video
            # latent tokens: 39 output frames = 65 audio steps exactly, while
            # 12 tokens already over-cover 39 frames (12 * 17/5 = 40.8), so a
            # token-linear rule drifts by 7 audio steps (0.175 s) at a 141-frame
            # context.  This is the same rule the masked-av-v14 inference
            # pipeline uses, which is what keeps lip-sync aligned.
            audio_hard = 0 if plain_window else overlap_frames_to_audio_steps(
                video_steps_to_frames(config.overlap_video_steps)
            )
            audio_transition = 0
        if config.conditioning_mode == "masked-av-v14":
            if plain_window:
                audio_latents = pipe.scheduler_audio.add_noise(
                    audio_target_clean, audio_noise, timestep_audio,
                )
                audio_target = audio_noise - audio_target_clean
            else:
                if audio_history is None:
                    raise ValueError("masked-av-v14 requires continuation_history_audio_latents")
                if audio_history.shape == audio_target_clean.shape:
                    audio_history = audio_history.narrow(-1, 0, audio_hard)
                if audio_history.shape[-1] != audio_hard:
                    raise ValueError("masked-av-v14 audio history must contain exactly the overlap prefix")
                audio_latents, audio_target, _ = v3_continuation_inputs(
                    audio_target_clean, audio_history, audio_noise, timestep_audio,
                    pipe.scheduler_audio.add_noise, hard_core=audio_hard, transition=0,
                    transition_start_weight=1.0, transition_end_weight=1.0, time_dim=-1,
                )
                if use_clean_prefix:
                    inputs["input_latents_audio"], inputs["denoise_mask_audio"] = _clean_prefix_injection(
                        audio_history, audio_latents, time_dim=-1,
                    )
        inputs["audio_latents"] = audio_latents
        if plain_window:
            # Every audio step is a target; there is no seam to protect.
            audio_weights = torch.full(
                (audio_target.shape[-1],), config.suffix_weight,
                device=audio_target.device, dtype=audio_target.dtype,
            )
        else:
            audio_weights = torch.ones(audio_target.shape[-1], device=audio_target.device, dtype=audio_target.dtype)
            audio_weights[:audio_hard] = 0
            # The transition ramp is shared with video in physical time.
            if audio_transition:
                audio_weights[audio_hard:audio_hard + audio_transition] = torch.linspace(
                    config.transition_weight, 0.0, audio_transition,
                    device=audio_weights.device, dtype=audio_weights.dtype,
                )
            audio_suffix_start = audio_hard + audio_transition
            audio_first_end = min(audio_weights.numel(), audio_suffix_start + round(config.first_suffix_clip_steps / 5 * 17 / 24 * 40))
            audio_weights[audio_suffix_start:audio_first_end] = config.first_suffix_weight
            audio_weights[audio_first_end:] = config.suffix_weight
    # Forward the CP layout into the model call, mirroring the SFT loss.
    # Without these keys model_fn_minimax_h3 runs the FULL window on every
    # rank (memory identical for CP=1/2/4); with them the DiT sequence is
    # sharded across the CP group and per-rank memory shrinks with CP.
    inputs["cp_rank"] = cp_rank
    inputs["cp_world_size"] = cp_world_size
    inputs["cp_group"] = cp_group
    if cp_world_size > 1:
        if not getattr(ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss, "_cp_diag", False):
            state = "sharded" if cp_group is not None else "NOT SHARDED (cp_group is None)"
            print(f"[cont-cp] cp_world_size={cp_world_size} -> model_fn will run {state}", flush=True)
            ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss._cp_diag = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    if training_cfg_scale > 1.0:
        if not inputs_nega:
            raise ValueError(
                "MiniMax-H3 CFG-aware continuation training requires unconditional inputs."
            )
        inputs_uncond = {**inputs, **inputs_nega}
        inputs_uncond["use_gradient_checkpointing"] = False
        inputs_uncond["use_gradient_checkpointing_offload"] = False
        with torch.no_grad():
            noise_pred_video_uncond, noise_pred_audio_uncond = pipe.model_fn(
                **models, **inputs_uncond,
                timestep_video=timestep_video, timestep_audio=timestep_audio,
            )
        noise_pred_video, noise_pred_audio = pipe.model_fn(
            **models, **inputs,
            timestep_video=timestep_video, timestep_audio=timestep_audio,
        )
        # H3 checkpoints are CFG-distilled. Recover the raw conditional velocity
        # before applying the teacher-forced continuation objective.
        noise_pred_video = (
            noise_pred_video + (training_cfg_scale - 1.0) * noise_pred_video_uncond
        ) / training_cfg_scale
        noise_pred_audio = (
            noise_pred_audio + (training_cfg_scale - 1.0) * noise_pred_audio_uncond
        ) / training_cfg_scale
    else:
        noise_pred_video, noise_pred_audio = pipe.model_fn(
            **models, **inputs, timestep_video=timestep_video, timestep_audio=timestep_audio,
        )
    video_weights = torch.zeros(video_target.shape[-3], device=video_target.device, dtype=video_target.dtype)
    if plain_window:
        # No prefix rows and no seam: a uniform flow-matching window.
        video_weights[:] = config.suffix_weight
    else:
        if config.transition_video_steps:
            video_weights[config.hard_core_video_steps:config.hard_core_video_steps + config.transition_video_steps] = torch.linspace(config.transition_weight, 0.0, config.transition_video_steps, device=video_target.device, dtype=video_target.dtype)
        suffix_start = config.overlap_video_steps
        first_end = min(video_weights.numel(), suffix_start + config.first_suffix_clip_steps)
        video_weights[suffix_start:first_end] = config.first_suffix_weight
        video_weights[first_end:] = config.suffix_weight
    if use_cp:
        # H3's CP model returns patch rows local to each sequence shard. Map the
        # temporal region weights to those rows before global sum/count reduce.
        spans = split_sequence_indices(int(inputs["packed"]["seq_len"]), cp_world_size)
        local_start, local_end = spans[cp_rank]
        img_pos = inputs["packed"]["img_pos"].view(-1).to(torch.long)
        audio_pos = inputs["packed"]["audio_pos"].view(-1).to(torch.long)
        local_img = (img_pos >= local_start) & (img_pos < local_end)
        local_audio = (audio_pos >= local_start) & (audio_pos < local_end)
        # No reference rows are used by continuation, so all packed image/audio
        # positions correspond to prediction rows. Video patch rows are frame
        # major (t * spatial block) and audio rows are channel major
        # (ch * steps), so the per-frame weights repeat over the spatial block
        # size and over the audio channels respectively.
        vh, vw = target_video.shape[-2:]
        video_row_weights = video_weights.repeat_interleave((vh // 2) * (vw // 2))
        audio_channels = int(audio_target.shape[0]) if audio_target is not None else 2
        audio_row_weights = audio_weights.repeat(audio_channels)
        local_video_weights = video_row_weights[torch.arange(video_row_weights.numel(), device=video_weights.device)[local_img]]
        local_audio_weights = audio_row_weights[torch.arange(audio_row_weights.numel(), device=audio_weights.device)[local_audio]]
        # The model may return local rows rather than dense latents under CP.
        target_video_rows = target_audio_rows = None
        if noise_pred_video.ndim == 2 or (audio_target is not None and noise_pred_audio.ndim == 2):
            target_video_rows, target_audio_rows = h3_local_target_rows(
                video_target, audio_target, inputs["packed"], cp_rank, cp_world_size,
                cond_rows_count=0, ref_audio_rows_count=0,
            )
        if not getattr(ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss, "_cp_audio_diag", False):
            print(
                f"[cont-cp] pred_video={tuple(noise_pred_video.shape)} "
                f"pred_audio={tuple(noise_pred_audio.shape)} "
                f"local img/audio rows={int(local_img.sum())}/{int(local_audio.sum())} "
                f"target rows={None if target_video_rows is None else tuple(target_video_rows.shape)}/"
                f"{None if target_audio_rows is None else tuple(target_audio_rows.shape)} "
                f"weights img/audio={int(local_video_weights.numel())}/{int(local_audio_weights.numel())}",
                flush=True,
            )
            ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss._cp_audio_diag = True
        if noise_pred_video.ndim == 2:
            diff = (noise_pred_video.float() - target_video_rows.float()).square()
            row_weights = local_video_weights.to(diff.device, diff.dtype).view(-1, 1)
            video_sum, video_count = (diff * row_weights).sum(), row_weights.sum() * diff.shape[-1]
        else:
            video_sum = ((noise_pred_video.float() - video_target.float()).square() * video_weights.view(1, 1, -1, 1, 1)).sum()
            video_count = video_weights.sum()
        if (_os_dbg.environ.get("DIFFSYNTH_AUDIO_DEBUG") == "1"
                and not getattr(ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss, "_adbg1", False)):
            ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss._adbg1 = True
            _at = audio_target
            print(f"[adbg] audio_target=None/"
                  f"{None if _at is None else tuple(_at.shape)} "
                  f"hmm_entered_audio_gate={audio_target is not None}", flush=True)
        if audio_target is not None:
            if noise_pred_audio.ndim == 2:
                diff = (noise_pred_audio.float() - target_audio_rows.float()).square()
                row_weights = local_audio_weights.to(diff.device, diff.dtype).view(-1, 1)
                audio_sum, audio_count = (diff * row_weights).sum(), row_weights.sum() * diff.shape[-1]
            else:
                audio_sum = ((noise_pred_audio.float() - audio_target.float()).square() * audio_weights.view(1, -1)).sum()
                audio_count = audio_weights.sum()
            video_mean = reduce_cp_weighted_mean(video_sum, video_count, group=cp_group)
            audio_mean = reduce_cp_weighted_mean(audio_sum, audio_count, group=cp_group)
            total_loss = video_mean + config.lambda_audio * audio_mean
            # Expose per-modality components for logging without perturbing
            # the returned loss or gradient path.
            _components = {
                "video_loss": float(video_mean.detach().item()),
                "audio_loss": float(audio_mean.detach().item()),
                "lambda_audio": float(config.lambda_audio),
            }
            try:
                total_loss.continuation_components = _components
            except Exception as _exc:
                if _os_dbg.environ.get("DIFFSYNTH_AUDIO_DEBUG") == "1":
                    import traceback as _tb
                    print(f"[adbg] COMPONENTS-EXC {type(_exc).__name__}: {_exc}", flush=True)
                    print(_tb.format_exc()[-2000:], flush=True)
            _LAST_CONTINUATION_COMPONENTS = _components
            return total_loss
        return reduce_cp_weighted_mean(video_sum, video_count, group=cp_group)
    return continuation_flow_loss(
        noise_pred_video, video_target, video_weights,
        audio_prediction=noise_pred_audio if audio_target is not None else None,
        audio_target=audio_target, audio_weights=audio_weights,
        lambda_audio=config.lambda_audio,
    )[0]


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
