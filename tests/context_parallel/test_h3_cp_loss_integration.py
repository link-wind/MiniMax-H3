import torch

from diffsynth.diffusion.loss import FlowMatchSFTMiniMaxH3AudioVideoLoss


class _FakeScheduler:
    def __init__(self, num_timesteps=1000):
        self.timesteps = torch.arange(num_timesteps, dtype=torch.float32)
        self.linear_timesteps_weights = torch.ones(
            num_timesteps, dtype=torch.float32
        )

    def training_weight(self, timestep):
        return torch.ones_like(timestep)

    def add_noise(self, original_samples, noise, timestep):
        return original_samples + noise

    def training_target(self, sample, noise, timestep):
        return noise - sample


class _FakePipe:
    device = torch.device("cpu")
    torch_dtype = torch.float32
    scheduler = _FakeScheduler()
    scheduler_audio = _FakeScheduler()
    in_iteration_models = ("dit",)
    dit = None

    def model_fn(self, **kwargs):
        self.called_with = dict(kwargs)
        return kwargs["video_latents"], kwargs["audio_latents"]


def test_h3_loss_passes_cp_rank_to_model_and_returns_finite_loss():
    pipe = _FakePipe()
    loss = FlowMatchSFTMiniMaxH3AudioVideoLoss(
        pipe,
        input_latents=torch.zeros(2, 3),
        audio_input_latents=torch.zeros(2, 4),
        cp_rank=1,
        cp_world_size=2,
        cp_group=None,
    )
    assert pipe.called_with["cp_rank"] == 1
    assert pipe.called_with["cp_world_size"] == 2
    assert torch.isfinite(loss)
