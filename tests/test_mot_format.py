"""Unit tests for MOT format file input/output and coordinate transformations."""

import os
import tempfile
import numpy as np
import pytest

from src.utils.mot_format import xyxy_to_xywh, xywh_to_xyxy, save_mot_results, load_mot_results


def test_xyxy_to_xywh_and_back():
    original = np.array([
        [10.0, 20.0, 50.0, 100.0],
        [100.0, 150.0, 220.0, 310.0],
    ])

    xywh = xyxy_to_xywh(original)
    expected_xywh = np.array([
        [10.0, 20.0, 40.0, 80.0],
        [100.0, 150.0, 120.0, 160.0],
    ])
    np.testing.assert_allclose(xywh, expected_xywh)

    recovered = xywh_to_xyxy(xywh)
    np.testing.assert_allclose(recovered, original)


def test_mot_save_and_load_roundtrip():
    records = [
        (1, 1, 10.0, 20.0, 50.0, 80.0, 0.95),
        (1, 2, 100.0, 120.0, 40.0, 90.0, 0.88),
        (2, 1, 12.0, 20.0, 50.0, 80.0, 0.96),
        (2, 2, 105.0, 120.0, 40.0, 90.0, 0.89),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = os.path.join(tmpdir, "test_mot.txt")
        save_mot_results(tmp_file, records)

        assert os.path.exists(tmp_file)

        # Inspect raw lines
        with open(tmp_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 4
        assert lines[0] == "1,1,10.00,20.00,50.00,80.00,0.9500,-1,-1,-1"

        # Load with load_mot_results
        df = load_mot_results(tmp_file)
        assert len(df) == 4
        assert list(df["frame"]) == [1, 1, 2, 2]
        assert list(df["id"]) == [1, 2, 1, 2]
        assert pytest.approx(df["x"].iloc[0], rel=1e-4) == 10.0
        assert pytest.approx(df["conf"].iloc[0], rel=1e-4) == 0.95
