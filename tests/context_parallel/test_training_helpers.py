from pathlib import Path

import yaml
import torch
import torch.nn as nn

from diffsynth.core.context_parallel import (
    CPAwareSampler,
    broadcast_cp_tensor,
    cp_process_layout,
    create_cp_process_group,
    h3_cp_loss,
    mse_local_sum_count,
    reduce_cp_parameter_gradients,
    reduce_cp_weighted_mean,
    should_reduce_cp_parameter_gradients,
    validate_cp_setup,
    validate_cp_topology,
)


def test_cp_process_layout_is_cp_contiguous():
    assert cp_process_layout(0, 2) == (0, 0)
    assert cp_process_layout(1, 2) == (0, 1)
    assert cp_process_layout(2, 2) == (1, 0)


def test_cp_aware_sampler_replicates_within_cp_group():
    data = list(range(10))
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    sampler_a = CPAwareSampler(
        data, process_index=0, cp_world_size=2, process_count=4, generator=generator_a
    )
    sampler_b = CPAwareSampler(
        data, process_index=1, cp_world_size=2, process_count=4, generator=generator_b
    )
    sampler_c = CPAwareSampler(
        data, process_index=2, cp_world_size=2, process_count=4
    )
    assert list(sampler_a) == list(sampler_b)
    assert list(sampler_c) != list(sampler_a)


def test_cp_aware_sampler_equalizes_dp_epoch_length():
    data = list(range(10))
    samplers = [
        CPAwareSampler(
            data,
            process_index=rank,
            cp_world_size=1,
            process_count=3,
            shuffle=False,
        )
        for rank in range(3)
    ]
    assert [len(sampler) for sampler in samplers] == [4, 4, 4]
    for sampler in samplers:
        assert len(list(sampler)) == 4


def test_broadcast_cp_tensor_is_noop_without_real_group():
    tensor = torch.tensor([1.0, 2.0])
    assert broadcast_cp_tensor(tensor, cp_rank=1, cp_world_size=2) is tensor


def test_broadcast_cp_tensor_uses_group_local_leader(monkeypatch):
    import diffsynth.core.context_parallel.training as cp_training

    calls = []

    def fake_broadcast(tensor, group_src, group):
        calls.append((tensor.clone(), group_src, group))

    monkeypatch.setattr(cp_training.dist, "broadcast", fake_broadcast)
    tensor = torch.tensor([1.0, 2.0])
    group = object()
    result = broadcast_cp_tensor(
        tensor, cp_rank=2, cp_world_size=3, group=group
    )
    assert result is tensor
    assert len(calls) == 1
    assert calls[0][1] == 0
    assert calls[0][2] is group


def test_create_cp_process_group_returns_none_for_cpu_single_process():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return
    assert create_cp_process_group(1, 2) is None
    assert create_cp_process_group(1, 1) is None


def test_reduce_cp_parameter_gradients_all_reduces_trainable_grads(monkeypatch):
    import diffsynth.core.context_parallel.training as cp_training

    calls = []

    def fake_all_reduce(tensor, group):
        calls.append((tensor, group))

    monkeypatch.setattr(cp_training.dist, "all_reduce", fake_all_reduce)
    model = nn.Linear(2, 2)
    grad = torch.ones_like(model.weight)
    model.weight.grad = grad
    group = object()
    reduce_cp_parameter_gradients(model, group)
    assert len(calls) == 1
    assert calls[0][0] is grad
    assert calls[0][1] is group


def test_h3_cp_loss_preserves_global_video_plus_audio_mean():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    video_global = pred[:1]
    video_target_global = target[:1]
    audio_global = pred[1:]
    audio_target_global = target[1:]

    v_local_sum, v_local_count = mse_local_sum_count(
        video_global, video_target_global
    )
    a_local_sum, a_local_count = mse_local_sum_count(
        audio_global, audio_target_global
    )
    cp_loss = h3_cp_loss(v_local_sum, v_local_count, a_local_sum, a_local_count)

    expected = (
        torch.nn.functional.mse_loss(video_global, video_target_global)
        + torch.nn.functional.mse_loss(audio_global, audio_target_global)
    ).double()
    torch.testing.assert_close(cp_loss, expected, atol=1e-6, rtol=1e-6)


def test_reduce_weighted_mean_handles_uneven_local_counts():
    local_sums = torch.tensor([3.0, 7.0])
    local_counts = torch.tensor([2.0, 8.0])
    result = reduce_cp_weighted_mean(
        local_sums.sum(), local_counts.sum(), group=None
    )
    assert result.item() == 1.0


def test_validate_cp_topology_accepts_24_processes_as_cp8_dp3():
    topology = validate_cp_topology(24, 8, dp_world_size=3)
    assert topology == {
        "process_count": 24,
        "cp_world_size": 8,
        "dp_world_size": 3,
    }


def test_validate_cp_topology_rejects_mismatch():
    try:
        validate_cp_topology(24, 8, dp_world_size=4)
    except ValueError:
        return
    raise AssertionError("expected mismatch to be rejected")


def test_cp8_dp3_accelerate_config_matches_documented_launch_shape():
    config_path = (
        Path(__file__).parents[2]
        / "examples/minimax_h3/model_training/full"
        / "accelerate_config_zero3_cp8_dp3.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["num_processes"] == 24
    assert config["distributed_type"] == "DEEPSPEED"
    topology = validate_cp_topology(config["num_processes"], 8)
    assert topology == {
        "process_count": 24,
        "cp_world_size": 8,
        "dp_world_size": 3,
    }


def test_cp_deepspeed_configs_enable_cpu_activation_checkpointing():
    for config_name in (
        "accelerate_config_zero3_cp8_dp1.yaml",
        "accelerate_config_zero3_cp8_dp3.yaml",
    ):
        config_path = (
            Path(__file__).parents[2]
            / "examples/minimax_h3/model_training/full"
            / config_name
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        activation = config["deepspeed_config"]["activation_checkpointing"]
        assert activation["cpu_checkpointing"] is True
        assert activation["contiguous_memory_optimization"] is False


def test_validate_cp_setup_uses_zero3_global_group():
    report = validate_cp_setup(
        process_count=24,
        cp_world_size=8,
        deepspeed_config={"zero_stage": 3},
    )
    assert report["dp_world_size"] == 3
    assert report["prepare_dataloader"] is False
    assert report["loss_reduction"] == "cp-weighted-video-audio"
    assert report["parameter_gradient_reduction"] == (
        "deepspeed-zero3-global-group"
    )


def test_validate_cp_setup_marks_dp_only_fallback_for_lower_zero_stages():
    report = validate_cp_setup(
        process_count=24,
        cp_world_size=8,
        deepspeed_config={"zero_stage": 1},
    )
    assert report["parameter_gradient_reduction"] == (
        "deepspeed-zero1-with-cp-reduction"
    )


def test_should_reduce_cp_parameter_gradients_is_explicit_fallback_only():
    assert should_reduce_cp_parameter_gradients(8, "global-group") is False
    assert should_reduce_cp_parameter_gradients(8, "dp-only") is True
    assert should_reduce_cp_parameter_gradients(1, "dp-only") is False
