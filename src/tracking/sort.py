"""SORT: Simple Online and Realtime Tracking under Occlusion."""

from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.iou import compute_iou
from src.tracking.track import Track, TrackState


class SortTracker:
    """
    SORT tracker implementation with occlusion persistence and Hungarian matching.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.30,
        two_stage: bool = False,
        high_conf_threshold: float = 0.35,
        low_conf_threshold: float = 0.10,
        iou_threshold_second: Optional[float] = None,
    ):
        """
        Initialize the SORT tracker.

        Args:
            max_age: Maximum consecutive missing frames to tolerate before track deletion.
                     Preserves ID during brief occlusions.
            min_hits: Minimum detection hits before track is confirmed and output.
            iou_threshold: Minimum IoU overlap required for detection-to-track association.
            two_stage: If True, enable ByteTrack-style two-stage association.
            high_conf_threshold: High confidence threshold for stage 1 association.
            low_conf_threshold: Low confidence threshold for stage 2 track recovery.
            iou_threshold_second: Optional IoU threshold for stage 2 (defaults to iou_threshold).
        """
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.two_stage = bool(two_stage)
        self.high_conf_threshold = float(high_conf_threshold)
        self.low_conf_threshold = float(low_conf_threshold)
        self.iou_threshold_second = (
            float(iou_threshold_second) if iou_threshold_second is not None else float(iou_threshold)
        )

        self.tracks: List[Track] = []
        self.frame_count: int = 0
        self.diagnostics = {
            "total_frames": 0,
            "high_dets": 0,
            "low_dets": 0,
            "stage1_matches": 0,
            "stage2_matches": 0,
            "unmatched_high_dets": 0,
            "unmatched_low_dets": 0,
        }

    def reset(self) -> None:
        """Reset tracker state and track IDs."""
        self.tracks.clear()
        self.frame_count = 0
        self.diagnostics = {
            "total_frames": 0,
            "high_dets": 0,
            "low_dets": 0,
            "stage1_matches": 0,
            "stage2_matches": 0,
            "unmatched_high_dets": 0,
            "unmatched_low_dets": 0,
        }
        Track.reset_counter()

    def get_diagnostics(self) -> dict:
        """Return cumulative association diagnostics."""
        return dict(self.diagnostics)

    def _associate_detections_to_tracks(
        self,
        detections: np.ndarray,
        predicted_boxes: np.ndarray,
        iou_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Associate detections to existing tracks using Hungarian algorithm on IoU matrix.

        Args:
            detections: Array of shape (M, 4+) [x1, y1, x2, y2, ...]
            predicted_boxes: Array of shape (N, 4) [x1, y1, x2, y2]
            iou_threshold: Optional override for matching IoU threshold.

        Returns:
            Tuple of:
                matches: (K, 2) array of [track_idx, detection_idx]
                unmatched_tracks: (U_t,) 1D array of track indices
                unmatched_detections: (U_d,) 1D array of detection indices
        """
        thresh = float(iou_threshold) if iou_threshold is not None else self.iou_threshold
        num_tracks = len(predicted_boxes)
        num_dets = len(detections)

        if num_tracks == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.empty(0, dtype=int),
                np.arange(num_dets, dtype=int),
            )

        if num_dets == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(num_tracks, dtype=int),
                np.empty(0, dtype=int),
            )

        # Compute IoU matrix (shape: num_tracks x num_dets)
        iou_matrix = compute_iou(predicted_boxes, detections[:, :4])

        # Fast path: if no pairs meet the threshold, all tracks and detections are unmatched
        if not np.any(iou_matrix >= thresh):
            return (
                np.empty((0, 2), dtype=int),
                np.arange(num_tracks, dtype=int),
                np.arange(num_dets, dtype=int),
            )

        # Cost matrix: minimize (1.0 - IoU) for valid associations.
        # Assign high cost to invalid pairs (IoU < threshold) so Hungarian matching
        # strictly maximizes valid associations without allowing rejected pairs to bias matches.
        cost_matrix = 1.0 - iou_matrix
        invalid_cost = 1e5
        cost_matrix[iou_matrix < thresh] = invalid_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = set(range(num_tracks))
        unmatched_dets = set(range(num_dets))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= thresh:
                matches.append([r, c])
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        matches_arr = np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
        unmatched_tracks_arr = np.array(sorted(unmatched_tracks), dtype=int)
        unmatched_dets_arr = np.array(sorted(unmatched_dets), dtype=int)

        return matches_arr, unmatched_tracks_arr, unmatched_dets_arr

    def update(
        self,
        detections: Optional[np.ndarray] = None,
        return_occluded: bool = False,
    ) -> np.ndarray:
        """
        Process a new frame with detections and update all active tracks.

        Args:
            detections: Array of shape (M, 5) where each row is [x1, y1, x2, y2, score].
                        If None or empty, tracks will still predict forward.
            return_occluded: If True, returns tracks even if currently occluded
                             (as long as time_since_update <= max_age). Default is False.

        Returns:
            np.ndarray of shape (K, 6): [[x1, y1, x2, y2, track_id, conf], ...]
            If no tracks qualify, returns empty array of shape (0, 6).
        """
        self.frame_count += 1

        if detections is None or len(detections) == 0:
            dets = np.empty((0, 5), dtype=np.float64)
        else:
            dets = np.asarray(detections, dtype=np.float64)
            if dets.ndim == 1:
                dets = dets.reshape(1, -1)
            # Ensure at least 5 columns [x1, y1, x2, y2, conf]
            if dets.shape[1] < 5:
                ones = np.ones((len(dets), 1), dtype=np.float64)
                dets = np.hstack([dets[:, :4], ones])

        # 1. Kalman prediction for all existing tracks
        predicted_boxes = []
        valid_track_indices = []

        for i, trk in enumerate(self.tracks):
            pred_box = trk.predict()
            if not np.any(np.isnan(pred_box)) and not np.any(np.isinf(pred_box)):
                predicted_boxes.append(pred_box)
                valid_track_indices.append(i)

        # Retain only tracks with valid predictions
        self.tracks = [self.tracks[i] for i in valid_track_indices]
        pred_boxes_arr = np.array(predicted_boxes, dtype=np.float64) if predicted_boxes else np.empty((0, 4), dtype=np.float64)

        # 2. Association (Single-stage or ByteTrack Two-stage)
        self.diagnostics["total_frames"] += 1

        if self.two_stage and dets.shape[0] > 0 and dets.shape[1] >= 5:
            high_mask = dets[:, 4] >= self.high_conf_threshold
            low_mask = (dets[:, 4] >= self.low_conf_threshold) & (~high_mask)

            dets_high = dets[high_mask]
            dets_low = dets[low_mask]

            self.diagnostics["high_dets"] += int(len(dets_high))
            self.diagnostics["low_dets"] += int(len(dets_low))

            # Stage 1: Associate existing tracks with high-confidence detections
            matches_high, unmatched_trks_1, unmatched_dets_high = self._associate_detections_to_tracks(
                dets_high, pred_boxes_arr, iou_threshold=self.iou_threshold
            )
            self.diagnostics["stage1_matches"] += int(len(matches_high))
            self.diagnostics["unmatched_high_dets"] += int(len(unmatched_dets_high))

            # Stage 2: Associate remaining tracks with low-confidence detections
            matches_low = []
            unmatched_trks = list(unmatched_trks_1)

            if len(unmatched_trks_1) > 0 and len(dets_low) > 0:
                unmatched_pred_boxes = pred_boxes_arr[unmatched_trks_1]
                matches_low_sub, unmatched_sub_indices, _ = self._associate_detections_to_tracks(
                    dets_low, unmatched_pred_boxes, iou_threshold=self.iou_threshold_second
                )

                for sub_trk_idx, low_det_idx in matches_low_sub:
                    orig_trk_idx = unmatched_trks_1[sub_trk_idx]
                    matches_low.append((orig_trk_idx, low_det_idx))

                unmatched_trks = [unmatched_trks_1[i] for i in unmatched_sub_indices]

            self.diagnostics["stage2_matches"] += int(len(matches_low))
            self.diagnostics["unmatched_low_dets"] += int(len(dets_low) - len(matches_low))

            # 3. Update matched tracks from both stages
            for trk_idx, det_idx in matches_high:
                det_box = dets_high[det_idx, :4]
                det_conf = float(dets_high[det_idx, 4])
                self.tracks[trk_idx].update(det_box, confidence=det_conf)

            for trk_idx, det_idx in matches_low:
                det_box = dets_low[det_idx, :4]
                det_conf = float(dets_low[det_idx, 4])
                self.tracks[trk_idx].update(det_box, confidence=det_conf)

            # 4. Initialize new tracks ONLY from unmatched high-confidence detections
            # (Unmatched low-confidence detections are explicitly discarded to prevent false tracks)
            for det_idx in unmatched_dets_high:
                det_box = dets_high[det_idx, :4]
                det_conf = float(dets_high[det_idx, 4])
                new_trk = Track(det_box, confidence=det_conf)
                self.tracks.append(new_trk)

        else:
            # Single-stage standard SORT association
            matches, unmatched_trks, unmatched_dets = self._associate_detections_to_tracks(
                dets, pred_boxes_arr, iou_threshold=self.iou_threshold
            )

            # 3. Update matched tracks
            for trk_idx, det_idx in matches:
                det_box = dets[det_idx, :4]
                det_conf = float(dets[det_idx, 4])
                self.tracks[trk_idx].update(det_box, confidence=det_conf)

            # 4. Initialize new tracks for unmatched detections
            for det_idx in unmatched_dets:
                det_box = dets[det_idx, :4]
                det_conf = float(dets[det_idx, 4])
                new_trk = Track(det_box, confidence=det_conf)
                self.tracks.append(new_trk)

        # 5. Filter out dead tracks (exceeded max_age)
        self.tracks = [trk for trk in self.tracks if not trk.is_dead(self.max_age)]

        # 6. Gather tracks for output
        output_records = []
        for trk in self.tracks:
            # Condition to report track:
            # - Must be confirmed (hits >= min_hits) or early frames in video
            # - If not return_occluded, must have been updated in this frame
            is_confirmed = (trk.hits >= self.min_hits) or (self.frame_count <= self.min_hits)
            is_visible = (trk.time_since_update == 0)

            should_report = is_confirmed and (is_visible or return_occluded)

            if should_report:
                box = trk.get_state()
                output_records.append([
                    box[0], box[1], box[2], box[3],
                    float(trk.track_id),
                    trk.confidence,
                ])

        if not output_records:
            return np.empty((0, 6), dtype=np.float64)

        return np.array(output_records, dtype=np.float64)

    def get_tracks(self) -> List[Track]:
        """Return the current list of all active track objects."""
        return self.tracks
