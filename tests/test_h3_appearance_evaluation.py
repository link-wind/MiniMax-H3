import json

import pytest
from PIL import Image

from diffsynth.pipelines.h3_appearance_evaluation import (
    format_appearance_metric,
    histogram_appearance_score,
    write_appearance_drift_report,
)


def test_histogram_metric_measures_similar_frames():
    frame = Image.new("L", (8, 8), 128)
    score = histogram_appearance_score([frame], [frame])
    assert score == pytest.approx(1.0)

    other = Image.new("L", (8, 8), 0)
    low_score = histogram_appearance_score([frame], [other])
    assert low_score is not None
    assert 0.0 <= low_score < 1.0


def test_appearance_metric_reports_unavailable_without_image_representation():
    metric = format_appearance_metric(["raw-string"], ["raw-string"], metric="histogram")
    assert metric["status"] == "unavailable"
    with pytest.raises(ValueError, match="unsupported appearance metric"):
        format_appearance_metric([], [], metric="unknown")


def test_appearance_drift_report_is_json_safe(tmp_path):
    rows = [
        {
            "mode": "no-bank",
            "seed": 0,
            "references_per_window": [0],
            "appearance_metric": {
                "metric": "histogram",
                "status": "unavailable",
                "reason": "no frames supplied",
            },
        }
    ]
    path = write_appearance_drift_report(
        tmp_path / "appearance_drift_report.json",
        rows,
        metadata={"checkpoint": "fake-checkpoint"},
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["metadata"]["checkpoint"] == "fake-checkpoint"
    assert data["rows"][0]["mode"] == "no-bank"
