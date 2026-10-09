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


def test_associate_straightforward_match():
    tracker = SortTracker(iou_threshold=0.3)
    predicted_boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
    detections = np.array([[12.0, 10.0, 52.0, 50.0, 0.9]])
    matches, unmatched_tracks, unmatched_dets = tracker._associate_detections_to_tracks(
        detections, predicted_boxes
    )
    assert matches.shape == (1, 2)
    assert np.array_equal(matches[0], [0, 0])
    assert len(unmatched_tracks) == 0
    assert len(unmatched_dets) == 0


def test_associate_below_threshold():
    tracker = SortTracker(iou_threshold=0.5)
    predicted_boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
    detections = np.array([[40.0, 40.0, 80.0, 80.0, 0.9]])
    matches, unmatched_tracks, unmatched_dets = tracker._associate_detections_to_tracks(
        detections, predicted_boxes
    )
    assert len(matches) == 0
    assert np.array_equal(unmatched_tracks, [0])
    assert np.array_equal(unmatched_dets, [0])


def test_associate_valid_match_not_lost_to_invalid_competing_pair():
    tracker = SortTracker(iou_threshold=0.30)
    t0 = [0.0, 0.0, 100.0, 100.0]
    t1 = [-50.0, 0.0, 50.0, 100.0]
    d0 = [-20.0, 0.0, 80.0, 100.0, 0.9]
    d1 = [70.0, 0.0, 170.0, 100.0, 0.9]

    predicted_boxes = np.array([t0, t1])
    detections = np.array([d0, d1])

    matches, unmatched_tracks, unmatched_dets = tracker._associate_detections_to_tracks(
        detections, predicted_boxes
    )
    assert len(matches) == 1
    assert matches[0, 0] == 0
    assert matches[0, 1] == 0
    assert np.array_equal(unmatched_tracks, [1])
    assert np.array_equal(unmatched_dets, [1])


def test_associate_more_detections_than_tracks():
    tracker = SortTracker(iou_threshold=0.3)
    predicted_boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
    detections = np.array([
        [10.0, 10.0, 50.0, 50.0, 0.9],
        [200.0, 200.0, 250.0, 250.0, 0.8],
        [300.0, 300.0, 350.0, 350.0, 0.7],
    ])
    matches, unmatched_tracks, unmatched_dets = tracker._associate_detections_to_tracks(
        detections, predicted_boxes
    )
    assert len(matches) == 1
    assert np.array_equal(matches[0], [0, 0])
    assert len(unmatched_tracks) == 0
    assert np.array_equal(unmatched_dets, [1, 2])


def test_associate_more_tracks_than_detections():
    tracker = SortTracker(iou_threshold=0.3)
    predicted_boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [200.0, 200.0, 250.0, 250.0],
        [300.0, 300.0, 350.0, 350.0],
    ])
    detections = np.array([[10.0, 10.0, 50.0, 50.0, 0.9]])
    matches, unmatched_tracks, unmatched_dets = tracker._associate_detections_to_tracks(
        detections, predicted_boxes
    )
    assert len(matches) == 1
    assert np.array_equal(matches[0], [0, 0])
    assert np.array_equal(unmatched_tracks, [1, 2])
    assert len(unmatched_dets) == 0


def test_associate_empty_detections():
    tracker = SortTracker(iou_threshold=0.3)
    predicted_boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
    detections = np.empty((0, 5))
    matches, unmatched_tracks, unmatched_dets = tracker._associate_detections_to_tracks(
        detections, predicted_boxes
    )
    assert len(matches) == 0
    assert np.array_equal(unmatched_tracks, [0])
    assert len(unmatched_dets) == 0


def test_associate_empty_tracks():
    tracker = SortTracker(iou_threshold=0.3)
    predicted_boxes = np.empty((0, 4))
    detections = np.array([[10.0, 10.0, 50.0, 50.0, 0.9]])
    matches, unmatched_tracks, unmatched_dets = tracker._associate_detections_to_tracks(
        detections, predicted_boxes
    )
    assert len(matches) == 0
    assert len(unmatched_tracks) == 0
    assert np.array_equal(unmatched_dets, [0])


def test_associate_no_valid_associations():
    tracker = SortTracker(iou_threshold=0.5)
    predicted_boxes = np.array([
        [0.0, 0.0, 50.0, 50.0],
        [100.0, 100.0, 150.0, 150.0],
    ])
    detections = np.array([
        [500.0, 500.0, 550.0, 550.0, 0.9],
        [700.0, 700.0, 750.0, 750.0, 0.8],
    ])
    matches, unmatched_tracks, unmatched_dets = tracker._associate_detections_to_tracks(
        detections, predicted_boxes
    )
    assert len(matches) == 0
    assert np.array_equal(unmatched_tracks, [0, 1])
    assert np.array_equal(unmatched_dets, [0, 1])


def test_associate_no_duplicate_matching():
    tracker = SortTracker(iou_threshold=0.2)
    predicted_boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [15.0, 15.0, 55.0, 55.0],
        [100.0, 100.0, 150.0, 150.0],
    ])
    detections = np.array([
        [12.0, 12.0, 52.0, 52.0, 0.9],
        [14.0, 14.0, 54.0, 54.0, 0.9],
        [102.0, 102.0, 152.0, 152.0, 0.9],
    ])
    matches, unmatched_tracks, unmatched_dets = tracker._associate_detections_to_tracks(
        detections, predicted_boxes
    )
    assert len(matches) == 3
    matched_tracks = matches[:, 0]
    matched_dets = matches[:, 1]
    assert len(matched_tracks) == len(set(matched_tracks))
    assert len(matched_dets) == len(set(matched_dets))
    assert len(unmatched_tracks) == 0
    assert len(unmatched_dets) == 0


def test_two_stage_high_confidence_matching():
    """Verify that high-confidence detections match existing tracks and spawn new tracks in stage 1."""
    tracker = SortTracker(
        max_age=10,
        min_hits=1,
        iou_threshold=0.30,
        two_stage=True,
        high_conf_threshold=0.35,
        low_conf_threshold=0.10,
    )
    # Frame 1: Create 2 tracks from high-confidence detections
    dets1 = np.array([
        [10.0, 10.0, 50.0, 50.0, 0.90],
        [200.0, 200.0, 250.0, 250.0, 0.85],
    ])
    out1 = tracker.update(dets1)
    assert len(out1) == 2
    ids1 = set(out1[:, 4].astype(int))

    # Frame 2: Both receive high-confidence updates
    dets2 = np.array([
        [12.0, 12.0, 52.0, 52.0, 0.88],
        [202.0, 202.0, 252.0, 252.0, 0.70],
    ])
    out2 = tracker.update(dets2)
    assert len(out2) == 2
    ids2 = set(out2[:, 4].astype(int))
    assert ids1 == ids2


def test_two_stage_low_confidence_recovery():
    """
    Verify ByteTrack stage 2: an existing track missed in high-conf stage
    is successfully recovered by a lower-confidence detection [0.10, 0.35).
    """
    tracker = SortTracker(
        max_age=10,
        min_hits=2,
        iou_threshold=0.30,
        two_stage=True,
        high_conf_threshold=0.35,
        low_conf_threshold=0.10,
    )
    # Frames 1 & 2: Establish confirmed track with high confidence
    box = [50.0, 50.0, 100.0, 100.0]
    tracker.update(np.array([[50.0, 50.0, 100.0, 100.0, 0.90]]))
    tracker.update(np.array([[52.0, 50.0, 102.0, 100.0, 0.85]]))
    assert len(tracker.tracks) == 1
    orig_id = tracker.tracks[0].track_id

    # Frame 3: Target suffers partial occlusion / blur, detection confidence drops to 0.20
    # High-conf pool is empty; low-conf pool has the detection
    det_low = np.array([[54.0, 50.0, 104.0, 100.0, 0.20]])
    out3 = tracker.update(det_low)

    assert len(out3) == 1
    assert int(out3[0, 4]) == orig_id
    # Track hits increased and time_since_update reset to 0
    assert tracker.tracks[0].time_since_update == 0
    assert tracker.tracks[0].hits == 3


def test_two_stage_rejected_low_confidence_detections():
    """
    Verify that unmatched low-confidence detections [0.10, 0.35) are explicitly discarded
    and NEVER spawn new tracks. Detections < low_conf_threshold are also ignored.
    """
    tracker = SortTracker(
        max_age=10,
        min_hits=1,
        iou_threshold=0.30,
        two_stage=True,
        high_conf_threshold=0.35,
        low_conf_threshold=0.10,
    )
    # Supply only low-confidence and sub-threshold detections, no existing tracks
    dets = np.array([
        [10.0, 10.0, 50.0, 50.0, 0.25],   # Low-confidence: should NOT spawn track
        [100.0, 100.0, 150.0, 150.0, 0.05], # Sub-threshold: ignored
    ])
    out = tracker.update(dets)

    assert len(out) == 0
    assert len(tracker.tracks) == 0

    # Supply 1 high-conf and 1 unmatched low-conf
    dets2 = np.array([
        [300.0, 300.0, 350.0, 350.0, 0.80], # High-conf: SHOULD spawn track
        [500.0, 500.0, 550.0, 550.0, 0.20], # Unmatched low-conf: should NOT spawn track
    ])
    out2 = tracker.update(dets2)
    assert len(out2) == 1
    assert len(tracker.tracks) == 1
    assert out2[0, 0] == 300.0


def test_two_stage_duplicate_assignment_prevention():
    """
    Verify that in two-stage mode, tracks matched in stage 1 are excluded from stage 2,
    and each detection/track pair is assigned at most once.
    """
    tracker = SortTracker(
        max_age=10,
        min_hits=1,
        iou_threshold=0.30,
        two_stage=True,
        high_conf_threshold=0.35,
        low_conf_threshold=0.10,
    )
    # Establish 2 tracks
    tracker.update(np.array([
        [10.0, 10.0, 50.0, 50.0, 0.90],
        [60.0, 10.0, 100.0, 50.0, 0.90],
    ]))
    assert len(tracker.tracks) == 2

    # Frame 2: Track 0 matches high-conf det; Track 1 matches low-conf det
    # Also supply an overlapping low-conf det near Track 0 to ensure Track 0 cannot be double-matched
    dets = np.array([
        [11.0, 11.0, 51.0, 51.0, 0.85],  # High conf -> matches Track 0
        [12.0, 12.0, 52.0, 52.0, 0.20],  # Low conf near Track 0 -> must NOT match Track 0 (already matched!)
        [61.0, 11.0, 101.0, 51.0, 0.25], # Low conf -> matches Track 1 in stage 2
    ])
    out = tracker.update(dets)
    assert len(out) == 2
    matched_ids = [int(r[4]) for r in out]
    assert len(matched_ids) == len(set(matched_ids)), "Duplicate track ID assignment detected!"
    # Total tracks in memory should still be 2 (the extra low-conf det was discarded)
    assert len(tracker.tracks) == 2


def test_two_stage_empty_inputs():
    """Verify that two-stage association handles empty detections, None, and empty tracks gracefully."""
    tracker = SortTracker(
        max_age=10,
        min_hits=2,
        iou_threshold=0.30,
        two_stage=True,
        high_conf_threshold=0.35,
        low_conf_threshold=0.10,
    )
    # Empty array
    out_empty = tracker.update(np.empty((0, 5)))
    assert len(out_empty) == 0

    # None
    out_none = tracker.update(None)
    assert len(out_none) == 0

    # Populate 1 track
    tracker.update(np.array([[20.0, 20.0, 60.0, 60.0, 0.95]]))
    # Update with empty
    out_empty2 = tracker.update(np.empty((0, 5)))
    assert len(out_empty2) == 0
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].time_since_update == 1


def test_two_stage_unmatched_tracks_and_detections():
    """
    Verify proper handling of unmatched tracks (age increments)
    and unmatched detections (high-conf initializes, low-conf discarded).
    """
    tracker = SortTracker(
        max_age=5,
        min_hits=1,
        iou_threshold=0.30,
        two_stage=True,
        high_conf_threshold=0.35,
        low_conf_threshold=0.10,
    )
    # Establish Track 1 at (0, 0, 50, 50)
    tracker.update(np.array([[0.0, 0.0, 50.0, 50.0, 0.90]]))
    assert len(tracker.tracks) == 1
    t1_id = tracker.tracks[0].track_id

    # Frame 2:
    # - Far high-conf det at (200, 200, 250, 250, 0.90) -> unmatched high det -> new track
    # - Far low-conf det at (400, 400, 450, 450, 0.25) -> unmatched low det -> discarded
    # - Original Track 1 at (0,0) receives NO matches in stage 1 or stage 2 -> unmatched track
    dets = np.array([
        [200.0, 200.0, 250.0, 250.0, 0.90],
        [400.0, 400.0, 450.0, 450.0, 0.25],
    ])
    out = tracker.update(dets)

    # Output only contains visible confirmed tracks:
    # New track at 200,200 is visible (time_since_update == 0)
    # Track 1 was unmatched, so time_since_update == 1 (not returned if not return_occluded)
    assert len(out) == 1
    new_id = int(out[0, 4])
    assert new_id != t1_id

    # Internally, tracker tracks list should contain Track 1 and New Track, but NOT a track for the 400,400 low-conf det
    assert len(tracker.tracks) == 2
    trk_ids = {trk.track_id for trk in tracker.tracks}
    assert t1_id in trk_ids
    assert new_id in trk_ids
