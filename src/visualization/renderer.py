"""Trajectory and bounding box renderer for Multi-Object Tracking visualization."""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.tracking.track import Track


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """
    Generate a deterministic, vibrant BGR color for a given track ID.

    Args:
        track_id: Unique integer identifier.

    Returns:
        (B, G, R) color tuple.
    """
    # Use golden ratio color generation in HSV space for distinct hues
    hue = int((track_id * 137.508) % 180)  # OpenCV hue range [0, 179]
    sat = 220
    val = 240
    hsv_pixel = np.uint8([[[hue, sat, val]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


class TrajectoryRenderer:
    """Renders bounding boxes, IDs, frame counters, and motion histories onto video frames."""

    def __init__(
        self,
        trajectory_length: int = 50,
        draw_boxes: bool = True,
        draw_labels: bool = True,
        draw_trajectories: bool = True,
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        """
        Initialize the renderer.

        Args:
            trajectory_length: Maximum number of historical trajectory points to keep and render.
            draw_boxes: Whether to draw bounding boxes.
            draw_labels: Whether to render track ID badges.
            draw_trajectories: Whether to draw past motion paths.
            line_thickness: Thickness of bounding box borders.
            font_scale: OpenCV font scale for text badges.
        """
        self.trajectory_length = trajectory_length
        self.draw_boxes = draw_boxes
        self.draw_labels = draw_labels
        self.draw_trajectories = draw_trajectories
        self.line_thickness = line_thickness
        self.font_scale = font_scale

        # Persistent history buffer: track_id -> List[(x, y)]
        self.track_histories: Dict[int, List[Tuple[int, int]]] = {}

    def reset(self) -> None:
        """Clear all stored trajectory histories."""
        self.track_histories.clear()

    def render_frame(
        self,
        frame: np.ndarray,
        tracks: Union[List[Track], np.ndarray],
        frame_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw annotations for the given tracks onto a copy of the input frame.

        Args:
            frame: Input BGR image (H, W, 3).
            tracks: Either a list of Track objects or an array of shape (N, 5+) [x1, y1, x2, y2, id, ...].
            frame_idx: Optional frame number to display.

        Returns:
            Annotated BGR frame copy.
        """
        canvas = frame.copy()
        active_ids = set()

        # Parse track data whether passed as Track objects or numeric arrays
        track_items = []
        if isinstance(tracks, list) and (len(tracks) == 0 or isinstance(tracks[0], Track)):
            for trk in tracks:
                # Include confirmed or visible tracks
                if trk.hits >= 2 or trk.time_since_update == 0:
                    box = trk.get_state()
                    track_items.append((int(trk.track_id), box, trk.is_occluded()))
        elif isinstance(tracks, np.ndarray) and len(tracks) > 0:
            for row in tracks:
                t_id = int(row[4])
                box = row[:4]
                track_items.append((t_id, box, False))

        # 1. Update and draw trajectories
        for t_id, box, is_occluded in track_items:
            active_ids.add(t_id)
            cx = int((box[0] + box[2]) / 2.0)
            cy = int((box[1] + box[3]) / 2.0)

            if t_id not in self.track_histories:
                self.track_histories[t_id] = []
            self.track_histories[t_id].append((cx, cy))

            if len(self.track_histories[t_id]) > self.trajectory_length:
                self.track_histories[t_id].pop(0)

            color = get_color_for_id(t_id)

            if self.draw_trajectories and len(self.track_histories[t_id]) > 1:
                pts = self.track_histories[t_id]
                for p_idx in range(1, len(pts)):
                    thickness = max(1, int(self.line_thickness * (p_idx / len(pts))))
                    cv2.line(canvas, pts[p_idx - 1], pts[p_idx], color, thickness)
                    # Draw subtle head circle at current centroid
                cv2.circle(canvas, (cx, cy), 3, color, -1)

            # 2. Draw bounding boxes
            if self.draw_boxes:
                x1, y1, x2, y2 = map(int, box)
                box_color = color if not is_occluded else (100, 100, 100)
                box_thickness = self.line_thickness if not is_occluded else 1
                cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, box_thickness)

                # 3. Draw ID badge
                if self.draw_labels:
                    label = f"ID: {t_id}" + (" (Occluded)" if is_occluded else "")
                    (tw, th), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                    )
                    top_left_y = max(0, y1 - th - baseline - 4)
                    # Badge background
                    cv2.rectangle(
                        canvas,
                        (x1, top_left_y),
                        (x1 + tw + 6, top_left_y + th + baseline + 4),
                        box_color,
                        -1,
                    )
                    # Badge text
                    text_color = (0, 0, 0) if sum(box_color) > 380 else (255, 255, 255)
                    cv2.putText(
                        canvas,
                        label,
                        (x1 + 3, top_left_y + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        text_color,
                        1,
                        cv2.LINE_AA,
                    )

        # Clean up stale trajectories for IDs that haven't appeared in a long while
        dead_ids = [tid for tid in self.track_histories if tid not in active_ids and len(self.track_histories[tid]) > 0]
        for tid in dead_ids:
            self.track_histories[tid].pop(0)

        # 4. Info banner in corner
        if frame_idx is not None:
            banner_text = f"Frame: {frame_idx:04d} | Active Tracks: {len(track_items)}"
            cv2.rectangle(canvas, (10, 10), (320, 42), (20, 20, 20), -1)
            cv2.putText(
                canvas,
                banner_text,
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )

        return canvas
