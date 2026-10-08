"""Track representation for Multi-Object Tracking."""

from typing import List, Optional, Tuple, Union
import numpy as np

from src.tracking.kalman_filter import KalmanBoxTracker


class TrackState:
    """Enumeration of track states."""
    TENTATIVE = "Tentative"
    CONFIRMED = "Confirmed"
    DELETED = "Deleted"


class Track:
    """
    Represents an individual tracked target throughout video frames.
    Maintains Kalman filter state, trajectory history, and occlusion status.
    """

    _count = 0  # Global class counter for assigning sequential unique track IDs

    def __init__(self, bbox: Union[np.ndarray, list], confidence: float = 1.0):
        """
        Initialize a new Track.

        Args:
            bbox: Initial detection bounding box [x1, y1, x2, y2].
            confidence: Detection confidence score.
        """
        Track._count += 1
        self.track_id: int = Track._count
        self.kf: KalmanBoxTracker = KalmanBoxTracker(bbox)

        self.hits: int = 1  # Total detection matches
        self.hit_streak: int = 1  # Consecutive detection matches
        self.age: int = 0  # Total lifespan in frames
        self.time_since_update: int = 0  # Frames elapsed since last detection update

        self.state: str = TrackState.TENTATIVE
        self.confidence: float = float(confidence)

        # Store trajectory history as list of (centroid_x, centroid_y)
        initial_box = np.asarray(bbox, dtype=np.float64).flatten()[:4]
        cx = (initial_box[0] + initial_box[2]) / 2.0
        cy = (initial_box[1] + initial_box[3]) / 2.0
        self.history: List[Tuple[float, float]] = [(cx, cy)]
        self.last_bbox: np.ndarray = initial_box.copy()

    @classmethod
    def reset_counter(cls, start_id: int = 0) -> None:
        """Reset the global track ID counter (useful for reproducible tests/runs)."""
        cls._count = start_id

    def predict(self) -> np.ndarray:
        """
        Advance state estimation using Kalman filter prediction.

        Returns:
            Predicted bounding box [x1, y1, x2, y2].
        """
        predicted_bbox = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.last_bbox = predicted_bbox.copy()
        cx = (predicted_bbox[0] + predicted_bbox[2]) / 2.0
        cy = (predicted_bbox[1] + predicted_bbox[3]) / 2.0
        self.history.append((cx, cy))

        return predicted_bbox

    def update(self, bbox: Union[np.ndarray, list], confidence: float = 1.0) -> np.ndarray:
        """
        Update track state with a matched detection.

        Args:
            bbox: Detected bounding box [x1, y1, x2, y2].
            confidence: Detection confidence score.

        Returns:
            Updated bounding box [x1, y1, x2, y2].
        """
        updated_bbox = self.kf.update(bbox)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = float(confidence)

        self.last_bbox = updated_bbox.copy()
        # Update last historical centroid with refined estimate
        cx = (updated_bbox[0] + updated_bbox[2]) / 2.0
        cy = (updated_bbox[1] + updated_bbox[3]) / 2.0
        if self.history:
            self.history[-1] = (cx, cy)
        else:
            self.history.append((cx, cy))

        return updated_bbox

    def get_state(self) -> np.ndarray:
        """
        Return the current bounding box estimate.

        Returns:
            np.ndarray of shape (4,): [x1, y1, x2, y2].
        """
        return self.kf.get_state()

    def is_confirmed(self, min_hits: int = 3) -> bool:
        """
        Check if track is confirmed.

        Args:
            min_hits: Minimum hits required to confirm track.
        """
        return self.hits >= min_hits

    def is_dead(self, max_age: int = 30) -> bool:
        """
        Check if track has exceeded maximum allowable missing frames.

        Args:
            max_age: Maximum frames to tolerate occlusion / missing detection.
        """
        return self.time_since_update > max_age

    def is_occluded(self) -> bool:
        """Return True if the track is currently coasting without detection."""
        return self.time_since_update > 0
