"""Failure Analysis tool for inspecting tracking errors and ID-switch events."""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.tracking.iou import compute_iou
from src.utils.mot_format import load_mot_results, xywh_to_xyxy


class FailureAnalyzer:
    """
    Analyzes identity switches and tracking failures by correlating
    ground-truth annotations with tracker outputs.
    """

    def __init__(self, match_iou_thresh: float = 0.50):
        """
        Initialize FailureAnalyzer.

        Args:
            match_iou_thresh: Minimum IoU required to associate predicted track with ground truth.
        """
        self.match_iou_thresh = match_iou_thresh

    def analyze_switches(
        self,
        gt_path: str,
        pred_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Identify ID switches and deduce probable failure causes based on concrete evidence.

        Categories:
            - 'occlusion': GT target overlaps with another person or re-emerges after missing frames.
            - 'missed detection': Detector failed to produce box for target in prior frame.
            - 'fast motion': Velocity/displacement exceeded typical human pedestrian speed.
            - 'ambiguous association': Multiple targets in close proximity competing for assignment.
            - 'unknown': Cause cannot be definitively attributed to above triggers.

        Returns:
            List of dictionaries documenting each switch event.
        """
        gt_df = load_mot_results(gt_path)
        pred_df = load_mot_results(pred_path)

        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        gt_xyxy = xywh_to_xyxy(gt_df[["x", "y", "w", "h"]].values)
        gt_df["x1"] = gt_xyxy[:, 0]
        gt_df["y1"] = gt_xyxy[:, 1]
        gt_df["x2"] = gt_xyxy[:, 2]
        gt_df["y2"] = gt_xyxy[:, 3]

        pred_xyxy = xywh_to_xyxy(pred_df[["x", "y", "w", "h"]].values)
        pred_df["x1"] = pred_xyxy[:, 0]
        pred_df["y1"] = pred_xyxy[:, 1]
        pred_df["x2"] = pred_xyxy[:, 2]
        pred_df["y2"] = pred_xyxy[:, 3]

        # Track history of mapping: gt_id -> (last_pred_id, last_frame, last_box)
        gt_mapping: Dict[int, Tuple[int, int, np.ndarray]] = {}
        switch_events: List[Dict[str, Any]] = []

        all_frames = sorted(list(set(gt_df["frame"].unique()) & set(pred_df["frame"].unique())))

        for f in all_frames:
            gt_frame = gt_df[gt_df["frame"] == f]
            pred_frame = pred_df[pred_df["frame"] == f]

            if gt_frame.empty or pred_frame.empty:
                continue

            gt_boxes = gt_frame[["x1", "y1", "x2", "y2"]].values
            pred_boxes = pred_frame[["x1", "y1", "x2", "y2"]].values
            gt_ids = gt_frame["id"].values
            pred_ids = pred_frame["id"].values

            iou_mat = compute_iou(gt_boxes, pred_boxes)

            # Check potential intra-GT occlusion in current frame
            gt_self_iou = compute_iou(gt_boxes, gt_boxes)
            np.fill_diagonal(gt_self_iou, 0.0)

            # Associate each GT with best prediction
            for g_idx, g_id in enumerate(gt_ids):
                row_ious = iou_mat[g_idx]
                best_p_idx = int(np.argmax(row_ious))
                best_iou = float(row_ious[best_p_idx])

                if best_iou >= self.match_iou_thresh:
                    current_p_id = int(pred_ids[best_p_idx])
                    curr_box = gt_boxes[g_idx]

                    if g_id in gt_mapping:
                        prev_p_id, prev_frame, prev_box = gt_mapping[g_id]

                        if prev_p_id != current_p_id:
                            # ID SWITCH DETECTED!
                            # Evidence 1: Occlusion
                            # Did this GT box overlap with another GT person, or was there a gap in frames?
                            frame_gap = f - prev_frame
                            has_gt_overlap = np.any(gt_self_iou[g_idx] > 0.15)
                            is_occlusion = (has_gt_overlap or frame_gap > 1)

                            # Evidence 2: Fast Motion
                            prev_center = np.array([(prev_box[0] + prev_box[2]) / 2.0, (prev_box[1] + prev_box[3]) / 2.0])
                            curr_center = np.array([(curr_box[0] + curr_box[2]) / 2.0, (curr_box[1] + curr_box[3]) / 2.0])
                            displacement = np.linalg.norm(curr_center - prev_center)
                            speed_per_frame = displacement / max(1, frame_gap)
                            is_fast_motion = speed_per_frame > 45.0  # Pixels per frame threshold

                            # Evidence 3: Ambiguous association
                            # Were there multiple predictions close to this GT box?
                            competing_preds = np.sum(row_ious > 0.20)
                            is_ambiguous = competing_preds > 1

                            # Determine evidence-grounded cause
                            cause = "unknown"
                            evidence = []

                            if is_occlusion:
                                cause = "occlusion"
                                if has_gt_overlap:
                                    evidence.append("High spatial overlap with adjacent target")
                                if frame_gap > 1:
                                    evidence.append(f"Target missing for {frame_gap - 1} frames prior to switch")
                            elif is_ambiguous:
                                cause = "ambiguous association"
                                evidence.append(f"{competing_preds} detection candidates with IoU > 0.20")
                            elif is_fast_motion:
                                cause = "fast motion"
                                evidence.append(f"High centroid displacement: {speed_per_frame:.1f} px/frame")
                            elif frame_gap == 1 and not has_gt_overlap:
                                cause = "missed detection"
                                evidence.append("Previous track lost without physical occlusion overlap")

                            switch_events.append({
                                "frame": int(f),
                                "gt_id": int(g_id),
                                "prev_tracker_id": int(prev_p_id),
                                "new_tracker_id": int(current_p_id),
                                "match_iou": round(best_iou, 3),
                                "frame_gap": int(frame_gap),
                                "probable_cause": cause,
                                "evidence": "; ".join(evidence) if evidence else "None detected",
                            })

                    # Update mapping
                    gt_mapping[g_id] = (current_p_id, f, curr_box)

        return switch_events

    def generate_report(
        self,
        switch_events: List[Dict[str, Any]],
        output_csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Produce a summary DataFrame and optionally save to CSV.

        Args:
            switch_events: List of switch event dictionaries.
            output_csv_path: Optional file path to save CSV report.

        Returns:
            pd.DataFrame containing switch event details and summary stats.
        """
        if not switch_events:
            df = pd.DataFrame(columns=[
                "frame", "gt_id", "prev_tracker_id", "new_tracker_id",
                "match_iou", "frame_gap", "probable_cause", "evidence"
            ])
        else:
            df = pd.DataFrame(switch_events)

        if output_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            df.to_csv(output_csv_path, index=False)

        return df
