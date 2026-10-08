"""Unit tests for SORT Tracker including track confirmation, ID persistence, and occlusion recovery."""

import numpy as np
import pytest

from src.tracking.sort import SortTracker
from src.tracking.track import Track


@pytest.fixture(autouse=True)
def reset_track_counter():
    """Reset Track ID counter before each test."""
    Track.reset_counter(0)


def test_track_creation_and_confirmation():
    tracker = SortTracker(max_age=10, min_hits=3, iou_threshold=0.3)
    det1 = np.array([[50.0, 50.0, 100.0, 100.0, 0.9]])

    # Frame 1: Track is created, but not confirmed yet (min_hits=3)
    # Note: in early frames <= min_hits, SORT returns active tracks so they aren't lost at startup
    out1 = tracker.update(det1)
    assert len(out1) == 1
    assert out1[0, 4] == 1  # ID = 1

    # Frame 2
    out2 = tracker.update(det1)
    assert len(out2) == 1
    assert out2[0, 4] == 1

    # Frame 3: Hits reaches 3 -> Confirmed
    out3 = tracker.update(det1)
    assert len(out3) == 1
    assert out3[0, 4] == 1
    assert tracker.tracks[0].is_confirmed(min_hits=3)


def test_id_persistence_with_moving_target():
    tracker = SortTracker(max_age=10, min_hits=2, iou_threshold=0.3)

    # Simulate object moving right: x shifts by +4 pixels each frame
    for f in range(10):
        x = 50.0 + f * 4.0
        det = np.array([[x, 50.0, x + 50.0, 100.0, 0.95]])
        out = tracker.update(det)
        assert len(out) == 1
        assert int(out[0, 4]) == 1  # Track ID must remain consistently 1


def test_occlusion_recovery_within_max_age():
    """
    Test that when an object disappears for 5 frames (occlusion),
    the Kalman filter continues predicting, and when re-detected,
    SORT re-associates the detection and preserves the original track ID.
    """
    tracker = SortTracker(max_age=10, min_hits=3, iou_threshold=0.2)

    # 1. Establish track for 5 frames
    for f in range(5):
        det = np.array([[100.0 + f * 2.0, 100.0, 150.0 + f * 2.0, 200.0, 0.9]])
        out = tracker.update(det)
        assert len(out) == 1
        assert int(out[0, 4]) == 1

    initial_id = int(out[0, 4])

    # 2. Occlusion: 4 frames with NO detections
    for f in range(4):
        empty_out = tracker.update(np.empty((0, 5)))
        assert len(empty_out) == 0  # Not returned when invisible
        assert len(tracker.tracks) == 1  # But still alive internally in memory!
        assert tracker.tracks[0].is_occluded()

    # 3. Target re-emerges at frame 10 near predicted position
    # Speed was ~2 px/frame. 4 missed frames => shift roughly +8 to +10 px
    re_emerged_det = np.array([[120.0, 100.0, 170.0, 200.0, 0.92]])
    recovery_out = tracker.update(re_emerged_det)

    assert len(recovery_out) == 1
    recovered_id = int(recovery_out[0, 4])
    # The crucial assertion: ID must NOT change across the occlusion!
    assert recovered_id == initial_id


def test_track_deletion_after_max_age():
    """
    Test that when an object disappears longer than max_age frames,
    the track is removed from memory.
    """
    tracker = SortTracker(max_age=5, min_hits=2, iou_threshold=0.3)

    # Establish track
    tracker.update(np.array([[50.0, 50.0, 100.0, 100.0, 0.9]]))
    tracker.update(np.array([[50.0, 50.0, 100.0, 100.0, 0.9]]))
    assert len(tracker.tracks) == 1

    # Disappear for max_age + 1 frames
    for _ in range(6):
        tracker.update(np.empty((0, 5)))

    # Track must now be pruned
    assert len(tracker.tracks) == 0


def test_multiple_distinct_tracks():
    tracker = SortTracker(max_age=10, min_hits=1, iou_threshold=0.3)

    dets_frame1 = np.array([
        [10.0, 10.0, 50.0, 50.0, 0.9],
        [200.0, 200.0, 250.0, 250.0, 0.85],
    ])
    out1 = tracker.update(dets_frame1)
    assert len(out1) == 2
    ids_frame1 = set(out1[:, 4].astype(int))
    assert len(ids_frame1) == 2

    dets_frame2 = np.array([
        [12.0, 10.0, 52.0, 50.0, 0.9],
        [202.0, 200.0, 252.0, 250.0, 0.85],
    ])
    out2 = tracker.update(dets_frame2)
    assert len(out2) == 2
    ids_frame2 = set(out2[:, 4].astype(int))
    assert ids_frame1 == ids_frame2
