"""Tests for the portable PyAV video writer used by VBench-Long.

``vbench2_beta_long`` writes every intermediate clip with
``torchvision.io.write_video``, which on this image only works because of a
container-local torchvision patch (PyAV 14 wants a ``PictureType`` on
``frame.pict_type``, torchvision 0.20 hands over the string ``"NONE"``).  That
patch sits on the node-local overlay, so the clip bytes used to depend on which
node happened to write them -- and the VBench stage is sharded across two
machines.

``vbench-portable/site-packages/vbench2_beta_long/_vbench_portable_video.py``
replaces that call with plain PyAV.  These tests pin its contract: the frames it
writes must be readable back, the layouts the call sites actually use must be
accepted, and it must not reach for torchvision at all.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PORTABLE_SITE_PACKAGES = REPO_ROOT.parent / "vbench-portable" / "site-packages"
MODULE_PATH = PORTABLE_SITE_PACKAGES / "vbench2_beta_long" / "_vbench_portable_video.py"

pytestmark = pytest.mark.skipif(
    not MODULE_PATH.is_file(),
    reason=f"portable vbench environment not present at {MODULE_PATH}",
)


@pytest.fixture(scope="module")
def writer():
    if str(PORTABLE_SITE_PACKAGES) not in sys.path:
        sys.path.insert(0, str(PORTABLE_SITE_PACKAGES))
    spec = importlib.util.spec_from_file_location("_vbench_portable_video_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decode(path):
    av = pytest.importorskip("av")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        rate = float(stream.average_rate) if stream.average_rate else None
    return np.stack(frames), rate


def _gradient(frames=6, height=64, width=96):
    rows = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    cols = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    base = ((rows + cols) / 2.0).astype(np.uint8)
    return np.stack([np.stack([base, base, base], axis=-1)] * frames)


def test_writes_a_decodable_video_with_the_right_length(writer, tmp_path):
    frames = _gradient(frames=6)
    path = tmp_path / "out.mp4"

    writer.write_video(path, frames, fps=24)

    decoded, rate = _decode(path)
    assert len(decoded) == 6
    assert rate == pytest.approx(24.0)


def test_round_trip_is_close_to_the_source(writer, tmp_path):
    frames = _gradient(frames=6)
    path = tmp_path / "rt.mp4"

    writer.write_video(path, frames, fps=24)

    decoded, _ = _decode(path)
    assert decoded.shape == frames.shape
    # Lossy, but a smooth gradient should survive h264 easily.
    assert np.abs(decoded.astype(np.int16) - frames.astype(np.int16)).mean() < 3.0


def test_accepts_torch_tensors_in_both_layouts(writer, tmp_path):
    torch = pytest.importorskip("torch")
    frames = _gradient(frames=4)

    tchw = tmp_path / "tchw.mp4"
    writer.write_video(tchw, torch.from_numpy(frames.transpose(0, 3, 1, 2)), fps=8)
    assert len(_decode(tchw)[0]) == 4

    thwc = tmp_path / "thwc.mp4"
    writer.write_video(thwc, torch.from_numpy(frames), fps=8)
    assert len(_decode(thwc)[0]) == 4


def test_accepts_float_frames_scaled_0_255(writer, tmp_path):
    # vbench.utils.load_video does torch.Tensor(uint8_array), which yields float32
    # in [0, 255] -- the call sites feed that straight in.
    frames = _gradient(frames=4).astype(np.float32)

    path = tmp_path / "float.mp4"
    writer.write_video(path, frames, fps=8)

    decoded, _ = _decode(path)
    assert len(decoded) == 4
    assert np.abs(decoded.astype(np.int16) - frames.astype(np.int16)).mean() < 3.0


def test_accepts_grayscale_video(writer, tmp_path):
    gray = _gradient(frames=4)[..., :1]

    path = tmp_path / "gray.mp4"
    writer.write_video(path, gray, fps=8)

    decoded, _ = _decode(path)
    assert decoded.shape[1:] == gray.shape[1:3] + (3,)
    assert np.array_equal(decoded[..., 0], decoded[..., 2])


def test_odd_extents_are_padded_to_even(writer, tmp_path):
    # yuv420p cannot represent odd extents; upstream padded too.
    frames = _gradient(frames=4, height=65, width=97)

    path = tmp_path / "odd.mp4"
    writer.write_video(path, frames, fps=8)

    decoded, _ = _decode(path)
    assert decoded.shape == (4, 66, 98, 3)


def test_rejects_an_empty_clip(writer, tmp_path):
    with pytest.raises(ValueError):
        writer.write_video(tmp_path / "empty.mp4", np.zeros((0, 8, 8, 3), dtype=np.uint8))


def test_never_touches_the_torchvision_writer(writer, tmp_path, monkeypatch):
    """The whole point: the image's torchvision patch must be irrelevant."""
    torchvision_io = pytest.importorskip("torchvision.io")

    def explode(*args, **kwargs):
        raise TypeError("an integer is required")

    monkeypatch.setattr(torchvision_io, "write_video", explode)

    path = tmp_path / "no_torchvision.mp4"
    writer.write_video(path, _gradient(frames=4), fps=8)
    assert len(_decode(path)[0]) == 4


def test_uses_exactly_the_official_encoder_settings(writer, tmp_path, monkeypatch):
    """The deterministic half of "same output as the official path".

    Every VBench-Long score is read off the decoded clips, so the writer has to
    encode the way the official pipeline does: libx264 with *no* options of its
    own (torchvision sets ``stream.options = options or {}``), yuv420p, and the
    requested rate.  Adding e.g. a crf would move the scores, not just the size.
    """
    av = pytest.importorskip("av")

    captured = {}
    real_open = av.open

    class _StreamSpy:
        """Records the settings the writer puts on the PyAV stream."""

        def __init__(self, stream, sink):
            object.__setattr__(self, "_stream", stream)
            object.__setattr__(self, "_sink", sink)

        def __setattr__(self, name, value):
            self._sink[name] = value
            setattr(self._stream, name, value)

        def __getattr__(self, name):
            return getattr(self._stream, name)

    class _ContainerSpy:
        def __init__(self, container, sink):
            object.__setattr__(self, "_container", container)
            object.__setattr__(self, "_sink", sink)

        def add_stream(self, codec, **kwargs):
            self._sink["codec"] = codec
            self._sink["rate"] = kwargs.get("rate")
            return _StreamSpy(self._container.add_stream(codec, **kwargs), self._sink)

        def __getattr__(self, name):
            return getattr(self._container, name)

    monkeypatch.setattr(av, "open", lambda *a, **k: _ContainerSpy(real_open(*a, **k), captured))
    writer.write_video(tmp_path / "spy.mp4", _gradient(frames=4), fps=24)

    assert captured["codec"] == "libx264"
    assert captured["pix_fmt"] == "yuv420p"
    assert float(captured["rate"]) == 24.0
    assert captured["options"] == {}, "the writer must not impose its own x264 options"


def test_matches_the_official_torchvision_output(writer, tmp_path):
    """The output half: same bytes, or if x264 wiggles, at least the same pixels.

    libx264 has been observed to emit a bitstream 1-2 bytes off for identical
    input, so byte equality is not something to hard-assert; what has to hold is
    that the decoded clip is the same picture, which is what gets scored.
    """
    torchvision_io = pytest.importorskip("torchvision.io")
    torch = pytest.importorskip("torch")

    frames = _gradient(frames=8)

    reference = tmp_path / "reference.mp4"
    try:
        torchvision_io.write_video(reference, torch.from_numpy(frames), fps=24)
    except TypeError as exc:  # the node-local torchvision is unpatched here
        pytest.skip(f"torchvision writer unusable on this image: {exc}")

    ours = tmp_path / "ours.mp4"
    writer.write_video(ours, frames, fps=24)

    if ours.read_bytes() == reference.read_bytes():
        return  # byte-identical: nothing more to check

    size_delta = abs(ours.stat().st_size - reference.stat().st_size)
    assert size_delta <= max(16, reference.stat().st_size // 1000), (
        f"file sizes should agree to within a rounding error, got {size_delta} B "
        f"({ours.stat().st_size} vs {reference.stat().st_size})"
    )

    a, rate_a = _decode(ours)
    b, rate_b = _decode(reference)
    assert rate_a == pytest.approx(rate_b), f"fps {rate_a} vs {rate_b}"
    assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    assert diff.mean() < 0.1 and diff.max() <= 2, (
        f"decoded frames must match: mean={diff.mean():.5f} max={diff.max()} "
        f"size_delta={size_delta} B"
    )


def test_fps_defaults_and_rejects_garbage(writer, tmp_path):
    path = tmp_path / "defaultfps.mp4"
    writer.write_video(path, _gradient(frames=4))
    assert _decode(path)[1] == pytest.approx(24.0)

    bad = tmp_path / "badfps.mp4"
    writer.write_video(bad, _gradient(frames=4), fps="not-a-number")
    assert _decode(bad)[1] == pytest.approx(24.0)
