"""Kalman Filter implementation for bounding box tracking in SORT."""

from typing import Optional, Tuple, Union
import numpy as np


def bbox_to_z(bbox: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert a bounding box [x1, y1, x2, y2] to measurement vector z:
    [center_x, center_y, scale (area), aspect_ratio]^T.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        np.ndarray of shape (4, 1)
    """
    bbox = np.asarray(bbox, dtype=np.float64).flatten()
    w = max(1e-4, float(bbox[2] - bbox[0]))
    h = max(1e-4, float(bbox[3] - bbox[1]))
    x = float(bbox[0]) + w / 2.0
    y = float(bbox[1]) + h / 2.0
    s = w * h
    r = w / h
    return np.array([[x], [y], [s], [r]], dtype=np.float64)


def x_to_bbox(x: np.ndarray) -> np.ndarray:
    """
    Convert state vector x or measurement vector z [u, v, s, r, ...]
    back to bounding box [x1, y1, x2, y2].

    Args:
        x: State vector of shape (7, 1) or (4, 1) or 1D array.

    Returns:
        np.ndarray of shape (4,) representing [x1, y1, x2, y2].
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    u, v, s, r = x[0], x[1], max(1e-4, x[2]), max(1e-4, x[3])

    # s = w * h, r = w / h => w^2 = s * r => w = sqrt(s * r), h = s / w
    w = np.sqrt(s * r)
    h = s / w if w > 1e-4 else 1e-4

    x1 = u - w / 2.0
    y1 = v - h / 2.0
    x2 = u + w / 2.0
    y2 = v + h / 2.0

    return np.array([x1, y1, x2, y2], dtype=np.float64)


class KalmanBoxTracker:
    """
    Standard SORT Kalman Filter for 2D bounding boxes.
    State: [u, v, s, r, u_dot, v_dot, s_dot]^T (dim: 7)
    Measurement: [u, v, s, r]^T (dim: 4)
    """

    def __init__(self, bbox: Union[np.ndarray, list]):
        """
        Initialize Kalman filter state from an initial bounding box.

        Args:
            bbox: [x1, y1, x2, y2]
        """
        # State transition matrix F (7x7) - constant velocity model
        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = 1.0  # u += u_dot
        self.F[1, 5] = 1.0  # v += v_dot
        self.F[2, 6] = 1.0  # s += s_dot

        # Measurement matrix H (4x7)
        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[0, 0] = 1.0  # measure u
        self.H[1, 1] = 1.0  # measure v
        self.H[2, 2] = 1.0  # measure s
        self.H[3, 3] = 1.0  # measure r

        # Measurement noise covariance R (4x4)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)

        # Process noise covariance Q (7x7)
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001]).astype(np.float64)

        # State estimation covariance P (7x7)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0]).astype(np.float64)

        # State vector x (7x1)
        z = bbox_to_z(bbox)
        self.x = np.zeros((7, 1), dtype=np.float64)
        self.x[0:4] = z

    def predict(self) -> np.ndarray:
        """
        Predict state forward by one time step.

        Returns:
            Predicted bounding box [x1, y1, x2, y2] as np.ndarray of shape (4,).
        """
        # Prevent negative scale collapse
        if (self.x[2, 0] + self.x[6, 0]) <= 0.0:
            self.x[6, 0] = 0.0

        # State prediction: x = F * x
        self.x = np.dot(self.F, self.x)

        # Covariance prediction: P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # Ensure scale remains non-negative
        self.x[2, 0] = max(1e-4, float(self.x[2, 0]))
        self.x[3, 0] = max(1e-4, float(self.x[3, 0]))

        return x_to_bbox(self.x)

    def update(self, bbox: Union[np.ndarray, list]) -> np.ndarray:
        """
        Update state estimate with observed bounding box measurement.

        Args:
            bbox: Measured bounding box [x1, y1, x2, y2].

        Returns:
            Updated bounding box [x1, y1, x2, y2] as np.ndarray of shape (4,).
        """
        z = bbox_to_z(bbox)

        # Innovation: y = z - H * x
        y = z - np.dot(self.H, self.x)

        # Innovation covariance: S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman gain: K = P * H^T * S^-1
        # Use np.linalg.solve for numerical stability over direct matrix inversion
        try:
            K = np.linalg.solve(S.T, np.dot(self.P, self.H.T).T).T
        except np.linalg.LinAlgError:
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.pinv(S))

        # State update: x = x + K * y
        self.x = self.x + np.dot(K, y)

        # Covariance update using numerically stable Joseph form:
        # P = (I - K * H) * P * (I - K * H)^T + K * R * K^T
        I = np.eye(7, dtype=np.float64)
        I_KH = I - np.dot(K, self.H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, self.R), K.T)

        # Ensure scale and aspect ratio remain valid
        self.x[2, 0] = max(1e-4, float(self.x[2, 0]))
        self.x[3, 0] = max(1e-4, float(self.x[3, 0]))

        return x_to_bbox(self.x)

    def get_state(self) -> np.ndarray:
        """
        Return the current bounding box estimate [x1, y1, x2, y2].

        Returns:
            Bounding box array of shape (4,).
        """
        return x_to_bbox(self.x)
