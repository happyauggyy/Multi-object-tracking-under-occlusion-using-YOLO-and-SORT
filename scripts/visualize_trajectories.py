"""Script to render bounding boxes, IDs, and trajectory paths from tracking results onto video."""

import argparse
import os
import sys
import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.video import VideoReader, VideoWriter
from src.utils.mot_format import load_mot_results, xywh_to_xyxy
from src.visualization.renderer import TrajectoryRenderer, get_color_for_id


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def render_video_trajectories(
    input_media: str,
    tracking_file: str,
    output_video: str = "outputs/videos/rendered_tracking.mp4",
    output_plot: str = "outputs/plots/trajectory_summary.png",
    config_path: str = "configs/config.yaml",
    display: bool = False,
) -> None:
    """Render tracking results onto video frames and generate a motion path plot."""
    if not os.path.exists(input_media):
        print(f"[ERROR] Input video/sequence not found: {input_media}")
        sys.exit(1)

    if not os.path.exists(tracking_file):
        print(f"[ERROR] Tracking file not found: {tracking_file}")
        sys.exit(1)

    cfg = load_config(config_path)
    vis_cfg = cfg.get("visualization", {})

    renderer = TrajectoryRenderer(
        trajectory_length=vis_cfg.get("trajectory_length", 50),
        draw_boxes=vis_cfg.get("draw_boxes", True),
        draw_labels=vis_cfg.get("draw_labels", True),
        draw_trajectories=vis_cfg.get("draw_trajectories", True),
        line_thickness=vis_cfg.get("line_thickness", 2),
        font_scale=vis_cfg.get("font_scale", 0.5),
    )

    track_df = load_mot_results(tracking_file)
    track_by_frame = {}
    for frame_id, group in track_df.groupby("frame"):
        xyxy = xywh_to_xyxy(group[["x", "y", "w", "h"]].values)
        ids = group["id"].values.reshape(-1, 1)
        confs = group["conf"].values.reshape(-1, 1) if "conf" in group.columns else np.ones((len(group), 1))
        # shape: (N, 6) [x1, y1, x2, y2, id, conf]
        track_by_frame[int(frame_id)] = np.hstack([xyxy, ids, confs])

    reader = VideoReader(input_media)
    writer = None
    if output_video:
        writer = VideoWriter(output_video, fps=reader.fps, width=reader.width, height=reader.height)

    print("==================================================")
    print("Rendering Trajectory Visualizations")
    print("==================================================")
    print(f"Input Media:    {input_media}")
    print(f"Tracking File:  {tracking_file}")
    print(f"Output Video:   {output_video if output_video else 'Disabled'}")
    print(f"Output Plot:    {output_plot if output_plot else 'Disabled'}")
    print("==================================================")

    first_frame = None
    total_frames = 0

    try:
        for f_idx, frame in reader:
            total_frames += 1
            if first_frame is None:
                first_frame = frame.copy()

            frame_tracks = track_by_frame.get(f_idx, np.empty((0, 6)))
            rendered = renderer.render_frame(frame, frame_tracks, frame_idx=f_idx)

            if writer is not None:
                writer.write(rendered)

            if display:
                cv2.imshow("Trajectory Playback", rendered)
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break
    finally:
        reader.close()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    if output_video:
        print(f"Rendered video saved to: {output_video}")

    # Generate 2D Trajectory Summary Plot
    if output_plot and first_frame is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)
        plt.figure(figsize=(10, 6), dpi=150)
        # Background: grayscale initial frame
        bg_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        plt.imshow(bg_rgb, alpha=0.6)

        # Plot full trajectory paths for all tracked IDs
        unique_ids = track_df["id"].unique()
        for u_id in unique_ids:
            u_data = track_df[track_df["id"] == u_id].sort_values("frame")
            c_x = u_data["x"] + u_data["w"] / 2.0
            c_y = u_data["y"] + u_data["h"] / 2.0
            bgr = get_color_for_id(int(u_id))
            rgb = (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)

            plt.plot(c_x, c_y, marker="o", markersize=3, label=f"ID {u_id}", color=rgb, linewidth=1.5)
            # Annotate start position
            plt.text(c_x.iloc[0], c_y.iloc[0], f" {u_id}", color=rgb, fontsize=8, weight="bold")

        plt.title(f"Full Trajectory Paths ({len(unique_ids)} targets over {total_frames} frames)")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()
        print(f"Trajectory plot saved to: {output_plot}")


def main():
    parser = argparse.ArgumentParser(description="Visualize tracking trajectories on video or plot.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video or sequence directory.")
    parser.add_argument("--tracking-file", "-t", type=str, required=True, help="Path to MOT tracking text file.")
    parser.add_argument("--output-video", "-o", type=str, default="outputs/videos/rendered_tracking.mp4", help="Path to output video file.")
    parser.add_argument("--output-plot", "-p", type=str, default="outputs/plots/trajectory_summary.png", help="Path to output trajectory 2D plot.")
    parser.add_argument("--config", "-c", type=str, default="configs/config.yaml", help="Path to configuration YAML.")
    parser.add_argument("--display", action="store_true", help="Display window during rendering.")

    args = parser.parse_args()
    render_video_trajectories(
        input_media=args.input,
        tracking_file=args.tracking_file,
        output_video=args.output_video,
        output_plot=args.output_plot,
        config_path=args.config,
        display=args.display,
    )


if __name__ == "__main__":
    main()
