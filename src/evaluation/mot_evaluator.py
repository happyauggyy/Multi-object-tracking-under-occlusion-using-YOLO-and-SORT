"""MOT (Multiple Object Tracking) standard metrics evaluator."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# NumPy 2.x compatibility patch for motmetrics which calls np.asfarray
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm
from src.tracking.iou import compute_iou
from src.utils.mot_format import load_mot_results, xywh_to_xyxy


class MOTEvaluator:
    """Evaluates tracking predictions against ground-truth MOT annotations."""

    def __init__(self, iou_threshold: float = 0.50):
        """
        Initialize the MOT evaluator.

        Args:
            iou_threshold: Minimum IoU overlap required between detection and GT for a valid match (default: 0.50).
        """
        self.iou_threshold = float(iou_threshold)
        # In motmetrics, distance is (1 - iou), so max acceptable distance is (1.0 - iou_threshold)
        self.max_iou_distance = 1.0 - self.iou_threshold

    def evaluate_sequence(
        self,
        gt_path: str,
        pred_path: str,
        seq_name: str = "sequence",
        filter_gt_pedestrians: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evaluate tracker predictions against ground truth for a sequence.

        Args:
            gt_path: Path to ground truth file (e.g. gt/gt.txt).
            pred_path: Path to tracking results file.
            seq_name: Sequence identifier name.
            filter_gt_pedestrians: If True and GT has class column, filter for pedestrian (class == 1).

        Returns:
            Tuple of (summary_df, metrics_dict).
        """
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")

        gt_df = load_mot_results(gt_path)
        pred_df = load_mot_results(pred_path)

        # In MOT17/MOT20, col 7 is class (1: pedestrian, 2: person on vehicle, etc.) and col 6 is mark (1: consider, 0: ignore)
        if filter_gt_pedestrians and "x3" in gt_df.columns and "y3" in gt_df.columns:
            # col 6 is conf/mark, col 7 is class
            # Only keep class == 1 (pedestrian) and mark != 0
            is_ped = gt_df["x3"] == 1
            is_valid = gt_df["conf"] != 0
            if (is_ped & is_valid).sum() > 0:
                gt_df = gt_df[is_ped & is_valid]

        acc = mm.MOTAccumulator(auto_id=True)

        all_frames = sorted(list(set(gt_df["frame"].unique()) | set(pred_df["frame"].unique())))

        for f in all_frames:
            gt_frame = gt_df[gt_df["frame"] == f]
            pred_frame = pred_df[pred_df["frame"] == f]

            gt_ids = gt_frame["id"].tolist()
            pred_ids = pred_frame["id"].tolist()

            # Coordinates in [x, y, w, h] format
            gt_boxes = gt_frame[["x", "y", "w", "h"]].to_numpy(dtype=float)
            pred_boxes = pred_frame[["x", "y", "w", "h"]].to_numpy(dtype=float)

            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                distances = np.empty((0, 0), dtype=float)
            elif len(gt_boxes) == 0:
                distances = np.empty((0, len(pred_boxes)), dtype=float)
            elif len(pred_boxes) == 0:
                distances = np.empty((len(gt_boxes), 0), dtype=float)
            else:
                gt_xyxy = xywh_to_xyxy(gt_boxes)
                pred_xyxy = xywh_to_xyxy(pred_boxes)
                iou_mat = compute_iou(gt_xyxy, pred_xyxy)
                distances = 1.0 - iou_mat
                distances[iou_mat < self.iou_threshold] = np.nan

            acc.update(gt_ids, pred_ids, distances)

        mh = mm.metrics.create()
        summary = mh.compute(
            acc,
            metrics=[
                "mota",
                "idf1",
                "num_switches",
                "num_false_positives",
                "num_misses",
                "num_objects",
                "mostly_tracked",
                "partially_tracked",
                "mostly_lost",
            ],
            name=seq_name,
        )

        metrics_dict = {
            "sequence": seq_name,
            "iou_threshold": self.iou_threshold,
            "mota": float(summary["mota"].iloc[0]),
            "idf1": float(summary["idf1"].iloc[0]),
            "id_switches": int(summary["num_switches"].iloc[0]),
            "false_positives": int(summary["num_false_positives"].iloc[0]),
            "false_negatives": int(summary["num_misses"].iloc[0]),
            "ground_truth_objects": int(summary["num_objects"].iloc[0]),
            "mostly_tracked": int(summary["mostly_tracked"].iloc[0]),
            "partially_tracked": int(summary["partially_tracked"].iloc[0]),
            "mostly_lost": int(summary["mostly_lost"].iloc[0]),
        }

        return summary, metrics_dict

    def save_metrics(
        self,
        metrics_dict: Dict[str, Any],
        output_json_path: str,
        output_csv_path: Optional[str] = None,
    ) -> None:
        """Save computed evaluation metrics to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4)

        if output_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            df = pd.DataFrame([metrics_dict])
            df.to_csv(output_csv_path, index=False)
