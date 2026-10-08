"""Unit tests for the SORT Kalman Filter."""

import numpy as np
import pytest

from src.tracking.kalman_filter import KalmanBoxTracker, bbox_to_z, x_to_bbox


def test_bbox_state_conversions():
    original_bbox = np.array([50.0, 100.0, 150.0, 300.0])  # w=100, h=200, cx=100, cy=200
    z = bbox_to_z(original_bbox)

    assert z.shape == (4, 1)
    assert pytest.approx(z[0, 0], rel=1e-4) == 100.0  # cx
    assert pytest.approx(z[1, 0], rel=1e-4) == 200.0  # cy
    assert pytest.approx(z[2, 0], rel=1e-4) == 20000.0  # area: 100 * 200
    assert pytest.approx(z[3, 0], rel=1e-4) == 0.5  # aspect ratio: 100 / 200

    recovered_bbox = x_to_bbox(z)
    np.testing.assert_allclose(recovered_bbox, original_bbox, rtol=1e-4, atol=1e-4)


def test_kalman_prediction():
    initial_bbox = [10.0, 20.0, 50.0, 100.0]
    kf = KalmanBoxTracker(initial_bbox)

    pred = kf.predict()
    assert pred.shape == (4,)
    # Initial velocity is zero, so prediction should be very close to initial box
    np.testing.assert_allclose(pred, initial_bbox, rtol=1e-3, atol=1e-3)


def test_kalman_update_velocity_learning():
    # Simulate a target moving rightwards (+10 pixels/frame)
    kf = KalmanBoxTracker([0.0, 0.0, 20.0, 40.0])

    for step in range(1, 6):
        kf.predict()
        x_offset = float(step * 10)
        measured_box = [x_offset, 0.0, x_offset + 20.0, 40.0]
        updated_box = kf.update(measured_box)
        # Position should follow the measurement
        assert updated_box[0] > 0.0

    # Next step: predict without measurement; it should extrapolate rightward velocity!
    pred_extrapolated = kf.predict()
    assert pred_extrapolated[0] > 50.0  # Should continue forward


def test_numerical_stability():
    kf = KalmanBoxTracker([100.0, 100.0, 200.0, 200.0])

    # Perform 100 predict/update steps with slight noise
    for i in range(100):
        pred = kf.predict()
        assert not np.any(np.isnan(pred))
        assert not np.any(np.isinf(pred))

        meas = [100.0 + i * 0.5, 100.0, 200.0 + i * 0.5, 200.0]
        updated = kf.update(meas)
        assert not np.any(np.isnan(updated))
        assert not np.any(np.isinf(updated))

    # Verify covariance matrix P remains positive semi-definite
    eigenvalues = np.linalg.eigvals(kf.P)
    assert np.all(eigenvalues.real >= -1e-6)
