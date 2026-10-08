"""Unit tests for MOT evaluation and failure analysis on synthetic ground truth."""

import os
import tempfile
import numpy as np
import pytest

from src.evaluation.mot_evaluator import MOTEvaluator
from src.evaluation.failure_analysis import FailureAnalyzer
from src.utils.mot_format import save_mot_results


def test_mot_evaluation_synthetic_perfect_tracking():
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_file = os.path.join(tmpdir, "gt.txt")
        pred_file = os.path.join(tmpdir, "pred.txt")

        # Create 10 frames of 2 objects moving
        gt_data = []
        pred_data = []
        for f in range(1, 11):
            # Target 1
            x1 = 50.0 + f * 5.0
            gt_data.append((f, 1, x1, 50.0, 30.0, 60.0, 1.0))
            pred_data.append((f, 1, x1, 50.0, 30.0, 60.0, 0.95))

            # Target 2
            x2 = 200.0 - f * 4.0
            gt_data.append((f, 2, x2, 100.0, 30.0, 60.0, 1.0))
            pred_data.append((f, 2, x2, 100.0, 30.0, 60.0, 0.92))

        save_mot_results(gt_file, gt_data)
        save_mot_results(pred_file, pred_data)

        evaluator = MOTEvaluator(iou_threshold=0.50)
        summary, metrics = evaluator.evaluate_sequence(gt_file, pred_file, "synth_perfect")

        # Perfect tracking should have MOTA = 1.0, IDF1 = 1.0, 0 switches, 0 FP, 0 FN
        assert pytest.approx(metrics["mota"], rel=1e-4) == 1.0
        assert pytest.approx(metrics["idf1"], rel=1e-4) == 1.0
        assert metrics["id_switches"] == 0
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0


def test_failure_analysis_detects_switch():
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_file = os.path.join(tmpdir, "gt.txt")
        pred_file = os.path.join(tmpdir, "pred.txt")

        # Target 1 has an ID switch at frame 5 (predicted ID changes from 10 to 11)
        gt_data = []
        pred_data = []
        for f in range(1, 10):
            x = 50.0 + f * 2.0
            gt_data.append((f, 1, x, 50.0, 30.0, 60.0, 1.0))
            tracker_id = 10 if f < 5 else 11  # Switched ID
            pred_data.append((f, tracker_id, x, 50.0, 30.0, 60.0, 0.9))

        save_mot_results(gt_file, gt_data)
        save_mot_results(pred_file, pred_data)

        analyzer = FailureAnalyzer(match_iou_thresh=0.50)
        switches = analyzer.analyze_switches(gt_file, pred_file)

        assert len(switches) == 1
        assert switches[0]["frame"] == 5
        assert switches[0]["gt_id"] == 1
        assert switches[0]["prev_tracker_id"] == 10
        assert switches[0]["new_tracker_id"] == 11
