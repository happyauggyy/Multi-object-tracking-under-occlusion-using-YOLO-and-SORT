"""End-to-end verification of tracking, evaluation, visualization, and failure analysis pipelines."""

import os
import tempfile
import cv2
import numpy as np
import pytest

from scripts.run_tracking import run_tracking
from scripts.evaluate import run_evaluation
from scripts.visualize_trajectories import render_video_trajectories
from scripts.analyze_failures import run_failure_analysis
from src.utils.mot_format import save_mot_results


def test_full_pipeline_synthetic_sequence():
    """
    Creates a temporary synthetic image sequence, runs the tracker,
    evaluates against synthetic ground truth, produces trajectory visualization,
    and executes failure analysis.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "seq_images")
        os.makedirs(img_dir, exist_ok=True)

        # Generate 15 frames of synthetic frames (solid background with moving rectangles)
        gt_annotations = []
        for f in range(1, 16):
            frame = np.full((360, 640, 3), 40, dtype=np.uint8)

            # Person 1: moving left-to-right
            x1 = 100 + (f - 1) * 15
            y1 = 120
            w1 = 40
            h1 = 90
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (200, 200, 200), -1)
            gt_annotations.append((f, 1, float(x1), float(y1), float(w1), float(h1), 1.0))

            # Person 2: moving right-to-left
            x2 = 450 - (f - 1) * 15
            y2 = 120
            w2 = 40
            h2 = 90
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (180, 180, 180), -1)
            gt_annotations.append((f, 2, float(x2), float(y2), float(w2), float(h2), 1.0))

            img_path = os.path.join(img_dir, f"{f:06d}.jpg")
            cv2.imwrite(img_path, frame)

        gt_file = os.path.join(tmpdir, "gt.txt")
        save_mot_results(gt_file, gt_annotations)

        # 1. Run Tracking pipeline
        track_out_txt = os.path.join(tmpdir, "tracking_out.txt")
        track_out_video = os.path.join(tmpdir, "tracking_out.mp4")

        # Note: In synthetic drawn rectangles without real COCO person textures,
        # YOLO might or might not detect synthetic monochrome boxes, so we test
        # run_tracking with the detector on the sequence
        run_tracking(
            input_path=img_dir,
            output_txt=track_out_txt,
            output_video=track_out_video,
            conf_thresh=0.20,
        )
        assert os.path.exists(track_out_txt)
        assert os.path.exists(track_out_video)

        # 2. Evaluate tracking on synthetic GT
        # Create a matching prediction file to test evaluate pipeline
        pred_file = os.path.join(tmpdir, "pred.txt")
        save_mot_results(pred_file, gt_annotations)

        json_out = os.path.join(tmpdir, "metrics.json")
        csv_out = os.path.join(tmpdir, "metrics.csv")
        metrics = run_evaluation(
            gt_path=gt_file,
            pred_path=pred_file,
            output_json=json_out,
            output_csv=csv_out,
        )
        assert os.path.exists(json_out)
        assert os.path.exists(csv_out)
        assert metrics["mota"] == 1.0

        # 3. Trajectory visualization
        vis_video = os.path.join(tmpdir, "vis_demo.mp4")
        vis_plot = os.path.join(tmpdir, "vis_plot.png")
        render_video_trajectories(
            input_media=img_dir,
            tracking_file=pred_file,
            output_video=vis_video,
            output_plot=vis_plot,
        )
        assert os.path.exists(vis_video)
        assert os.path.exists(vis_plot)

        # 4. Failure Analysis
        fail_csv = os.path.join(tmpdir, "failure_report.csv")
        fail_plot = os.path.join(tmpdir, "failure_plot.png")
        df_fail = run_failure_analysis(
            gt_path=gt_file,
            pred_path=pred_file,
            output_csv=fail_csv,
            output_plot=fail_plot,
        )
        assert os.path.exists(fail_csv)
