"""Generate annotated diagnostic sample frames from tracking results for visual inspection."""

import os
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.mot_format import load_mot_results, xywh_to_xyxy
from src.visualization.renderer import TrajectoryRenderer


def render_diagnostic_frames(
    img_dir: str,
    pred_path: str,
    output_dir: str = "outputs/plots",
    target_frames: list = None,
):
    """Render and save selected diagnostic frames showing bounding boxes, IDs, and trajectories."""
    if target_frames is None:
        target_frames = [50, 90, 321, 517]

    os.makedirs(output_dir, exist_ok=True)
    df = load_mot_results(pred_path)
    renderer = TrajectoryRenderer(
        trajectory_length=50,
        draw_boxes=True,
        draw_labels=True,
        draw_trajectories=True,
        line_thickness=2,
        font_scale=0.5,
    )

    # Process sequentially up to max target frame to maintain accurate trajectory history
    max_frame = max(target_frames)
    saved_paths = []

    print(f"Loading tracking results from: {pred_path}")
    print(f"Reading frames from: {img_dir}")
    print(f"Target diagnostic frames: {target_frames}")

    for frame_idx in range(1, max_frame + 1):
        frame_file = os.path.join(img_dir, f"{frame_idx:06d}.jpg")
        if not os.path.exists(frame_file):
            continue

        frame_data = df[df["frame"] == frame_idx]
        if len(frame_data) > 0:
            boxes = frame_data[["x", "y", "w", "h"]].to_numpy(dtype=float)
            xyxy = xywh_to_xyxy(boxes)
            ids = frame_data["id"].to_numpy(dtype=float).reshape(-1, 1)
            tracks_arr = np.hstack([xyxy, ids])
        else:
            tracks_arr = np.empty((0, 5), dtype=float)

        # Only load image from disk when needed or near target
        if frame_idx in target_frames:
            frame_bgr = cv2.imread(frame_file)
            rendered = renderer.render_frame(frame_bgr, tracks_arr, frame_idx=frame_idx)
            out_file = os.path.join(output_dir, f"diagnostic_frame_{frame_idx:04d}.jpg")
            cv2.imwrite(out_file, rendered)
            print(f"Saved annotated diagnostic frame: {out_file} (Active tracks: {len(tracks_arr)})")
            saved_paths.append(out_file)
        else:
            # Update trajectory history without rendering full canvas to save CPU time
            for row in tracks_arr:
                t_id = int(row[4])
                cx = int((row[0] + row[2]) / 2.0)
                cy = int((row[1] + row[3]) / 2.0)
                if t_id not in renderer.track_histories:
                    renderer.track_histories[t_id] = []
                renderer.track_histories[t_id].append((cx, cy))
                if len(renderer.track_histories[t_id]) > renderer.trajectory_length:
                    renderer.track_histories[t_id].pop(0)

    print(f"Successfully generated {len(saved_paths)} diagnostic frames in {output_dir}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render diagnostic frames.")
    parser.add_argument("--img-dir", default=r"D:\MOT17\MOT17\train\MOT17-02-FRCNN\img1", help="Image directory.")
    parser.add_argument("--pred", default=r"outputs\MOT17-02-FRCNN_conf035_maxage60_imgsz960.txt", help="Predictions file.")
    parser.add_argument("--output-dir", default="outputs/plots", help="Output directory.")
    parser.add_argument("--prefix", default="diagnostic_frame_imgsz960_", help="Output filename prefix.")
    args = parser.parse_args()

    # Customized render with prefix
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_mot_results(args.pred)
    renderer = TrajectoryRenderer(trajectory_length=50)
    target_frames = [50, 90, 321, 517]
    for frame_idx in range(1, max(target_frames) + 1):
        frame_file = os.path.join(args.img_dir, f"{frame_idx:06d}.jpg")
        if not os.path.exists(frame_file):
            continue
        frame_data = df[df["frame"] == frame_idx]
        if len(frame_data) > 0:
            boxes = frame_data[["x", "y", "w", "h"]].to_numpy(dtype=float)
            xyxy = xywh_to_xyxy(boxes)
            ids = frame_data["id"].to_numpy(dtype=float).reshape(-1, 1)
            tracks_arr = np.hstack([xyxy, ids])
        else:
            tracks_arr = np.empty((0, 5), dtype=float)

        if frame_idx in target_frames:
            frame_bgr = cv2.imread(frame_file)
            rendered = renderer.render_frame(frame_bgr, tracks_arr, frame_idx=frame_idx)
            out_file = os.path.join(args.output_dir, f"{args.prefix}{frame_idx:04d}.jpg")
            cv2.imwrite(out_file, rendered)
            print(f"Saved: {out_file} (Active: {len(tracks_arr)})")
        else:
            for row in tracks_arr:
                t_id = int(row[4])
                cx = int((row[0] + row[2]) / 2.0)
                cy = int((row[1] + row[3]) / 2.0)
                if t_id not in renderer.track_histories:
                    renderer.track_histories[t_id] = []
                renderer.track_histories[t_id].append((cx, cy))
                if len(renderer.track_histories[t_id]) > renderer.trajectory_length:
                    renderer.track_histories[t_id].pop(0)
