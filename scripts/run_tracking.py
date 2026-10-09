"""Main execution script for running Multi-Object Tracking on a video or image sequence."""

import argparse
import os
import sys
import time
import yaml
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection.yolo_detector import YOLODetector
from src.tracking.sort import SortTracker
from src.utils.video import VideoReader, VideoWriter
from src.utils.mot_format import xyxy_to_xywh, save_mot_results
from src.visualization.renderer import TrajectoryRenderer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration YAML if present, else fallback to defaults."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_tracking(
    input_path: str,
    output_txt: str = "outputs/tracking_results/tracking_output.txt",
    output_video: str = "",
    config_path: str = "configs/config.yaml",
    model_name: str = "",
    conf_thresh: float = -1.0,
    max_age: int = -1,
    min_hits: int = -1,
    iou_thresh: float = -1.0,
    device: str = "",
    display: bool = False,
    imgsz: int = -1,
    two_stage: bool = False,
    low_conf_thresh: float = -1.0,
    iou_thresh_second: float = -1.0,
) -> str:
    """
    Execute full YOLO + SORT tracking pipeline.

    Args:
        input_path: Path to video file or MOT image folder.
        output_txt: Path to write MOT-format tracking results.
        output_video: Optional path to write rendered tracking video.
        config_path: Path to YAML config.
        model_name: Optional override for detector model.
        conf_thresh: Optional override for confidence threshold.
        max_age: Optional override for tracker max age.
        min_hits: Optional override for tracker min hits.
        iou_thresh: Optional override for tracker IoU threshold.
        device: Hardware device for YOLO.
        display: If True, show live OpenCV display window.
        imgsz: Optional override for detector inference image size (e.g. 640, 960, 1280).
        two_stage: If True, enable ByteTrack-style two-stage association.
        low_conf_thresh: Low confidence threshold for stage-2 recovery.
        iou_thresh_second: Optional override for stage-2 IoU threshold.

    Returns:
        Path to output tracking text file.
    """
    cfg = load_config(config_path)

    # Resolve settings (CLI overrides config file, which overrides defaults)
    det_cfg = cfg.get("detector", {})
    trk_cfg = cfg.get("tracker", {})
    vis_cfg = cfg.get("visualization", {})

    model = model_name if model_name else det_cfg.get("model", "yolo11n.pt")
    conf = conf_thresh if conf_thresh >= 0.0 else det_cfg.get("confidence_threshold", 0.40)
    dev = device if device else det_cfg.get("device", "")
    person_cls = det_cfg.get("person_class_id", 0)
    i_size = imgsz if imgsz > 0 else det_cfg.get("imgsz", None)

    m_age = max_age if max_age >= 0 else trk_cfg.get("max_age", 30)
    m_hits = min_hits if min_hits >= 0 else trk_cfg.get("min_hits", 3)
    i_thresh = iou_thresh if iou_thresh >= 0.0 else trk_cfg.get("iou_threshold", 0.30)

    # Two-stage parameters
    is_two_stage = bool(two_stage or trk_cfg.get("two_stage", False))
    low_conf = low_conf_thresh if low_conf_thresh >= 0.0 else trk_cfg.get("low_conf_threshold", 0.10)
    i_thresh_second = iou_thresh_second if iou_thresh_second >= 0.0 else trk_cfg.get("iou_threshold_second", None)

    # If two-stage is active, detector must produce candidates down to low_conf
    det_conf = min(conf, low_conf) if is_two_stage else conf

    print("==================================================")
    print("Multi-Object Tracking Under Occlusion")
    print("==================================================")
    print(f"Input:         {input_path}")
    print(f"Output TXT:    {output_txt}")
    print(f"Output Video:  {output_video if output_video else 'Disabled'}")
    print(f"Detector:      {model} (conf={conf}, class={person_cls}, imgsz={i_size if i_size else 'default'})")
    if is_two_stage:
        print(f"Association:   ByteTrack Two-Stage (high_conf={conf}, low_conf={low_conf})")
    else:
        print("Association:   Single-Stage SORT")
    print(f"SORT Tracker:  max_age={m_age}, min_hits={m_hits}, iou_thresh={i_thresh}")
    print("==================================================")

    # Initialize components
    detector = YOLODetector(
        model_name=model,
        conf_thresh=det_conf,
        person_class_id=person_cls,
        device=dev,
        imgsz=i_size,
    )
    tracker = SortTracker(
        max_age=m_age,
        min_hits=m_hits,
        iou_threshold=i_thresh,
        two_stage=is_two_stage,
        high_conf_threshold=conf,
        low_conf_threshold=low_conf,
        iou_threshold_second=i_thresh_second,
    )
    renderer = TrajectoryRenderer(
        trajectory_length=vis_cfg.get("trajectory_length", 50),
        draw_boxes=vis_cfg.get("draw_boxes", True),
        draw_labels=vis_cfg.get("draw_labels", True),
        draw_trajectories=vis_cfg.get("draw_trajectories", True),
        line_thickness=vis_cfg.get("line_thickness", 2),
        font_scale=vis_cfg.get("font_scale", 0.5),
    )

    reader = VideoReader(input_path)
    writer = None
    if output_video:
        writer = VideoWriter(output_video, fps=reader.fps, width=reader.width, height=reader.height)

    mot_results = []
    start_time = time.time()
    total_processed = 0

    try:
        for frame_idx, frame in reader:
            total_processed += 1

            # 1. Detect person bounding boxes
            detections = detector.detect(frame)  # Shape: (M, 5) [x1, y1, x2, y2, score]

            # 2. Update SORT Tracker
            active_tracks = tracker.update(detections, return_occluded=False)

            # 3. Record MOT-format detections: (frame, id, x, y, w, h, conf)
            for trk in active_tracks:
                x1, y1, x2, y2, track_id, score = trk[:6]
                w = x2 - x1
                h = y2 - y1
                mot_results.append((int(frame_idx), int(track_id), float(x1), float(y1), float(w), float(h), float(score)))

            # 4. Optional Rendering
            if writer is not None or display:
                rendered = renderer.render_frame(frame, tracker.get_tracks(), frame_idx=frame_idx)
                if writer is not None:
                    writer.write(rendered)
                if display:
                    cv2.imshow("Tracking", rendered)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\nInterrupted by user.")
                        break

            if total_processed % 25 == 0:
                elapsed = time.time() - start_time
                fps_calc = total_processed / max(1e-4, elapsed)
                print(f"Processed frame {frame_idx} | Active tracks: {len(active_tracks)} | Speed: {fps_calc:.1f} FPS")

    finally:
        reader.close()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    avg_fps = total_processed / max(1e-4, elapsed)
    print("--------------------------------------------------")
    print(f"Finished! Processed {total_processed} frames in {elapsed:.2f}s (Avg {avg_fps:.1f} FPS)")
    print(f"Total track annotations recorded: {len(mot_results)}")

    if is_two_stage:
        diag = tracker.get_diagnostics()
        print("--------------------------------------------------")
        print("Two-Stage Association Diagnostics:")
        print(f"  Total frames:                {diag.get('total_frames', 0)}")
        print(f"  High-confidence detections:  {diag.get('high_dets', 0)}")
        print(f"  Low-confidence detections:   {diag.get('low_dets', 0)}")
        print(f"  Stage-1 matches (High-conf): {diag.get('stage1_matches', 0)}")
        print(f"  Stage-2 matches (Low-conf):  {diag.get('stage2_matches', 0)}")
        print(f"  Unmatched high-conf dets:    {diag.get('unmatched_high_dets', 0)} (spawned new tracks)")
        print(f"  Unmatched low-conf dets:     {diag.get('unmatched_low_dets', 0)} (discarded)")
        print("--------------------------------------------------")

    save_mot_results(output_txt, mot_results)
    print(f"Tracking output saved to: {output_txt}")
    if output_video:
        print(f"Rendered video saved to: {output_video}")

    return output_txt


def main():
    parser = argparse.ArgumentParser(description="Run YOLO + SORT Multi-Object Tracking.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video file or image sequence directory.")
    parser.add_argument("--output-txt", "-o", type=str, default="outputs/tracking_results/tracking_output.txt", help="Path to save MOT tracking text file.")
    parser.add_argument("--output-video", "-v", type=str, default="", help="Optional path to save rendered tracking video.")
    parser.add_argument("--config", "-c", type=str, default="configs/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--model", type=str, default="", help="Ultralytics YOLO model name/path.")
    parser.add_argument("--conf-thresh", type=float, default=-1.0, help="Confidence threshold override.")
    parser.add_argument("--max-age", type=int, default=-1, help="Tracker max_age override.")
    parser.add_argument("--min-hits", type=int, default=-1, help="Tracker min_hits override.")
    parser.add_argument("--iou-thresh", type=float, default=-1.0, help="Tracker IoU threshold override.")
    parser.add_argument("--device", type=str, default="", help="Device: 'cpu', 'cuda:0', etc.")
    parser.add_argument("--display", action="store_true", help="Display tracking output window in real-time.")
    parser.add_argument("--imgsz", type=int, default=-1, help="Inference image size (e.g. 640, 960, 1280). Default: model default.")
    parser.add_argument("--two-stage", action="store_true", help="Enable ByteTrack-style two-stage association.")
    parser.add_argument("--low-conf-thresh", type=float, default=-1.0, help="Low confidence threshold for two-stage association.")
    parser.add_argument("--iou-thresh-second", type=float, default=-1.0, help="Stage 2 IoU threshold override.")

    args = parser.parse_args()
    run_tracking(
        input_path=args.input,
        output_txt=args.output_txt,
        output_video=args.output_video,
        config_path=args.config,
        model_name=args.model,
        conf_thresh=args.conf_thresh,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_thresh=args.iou_thresh,
        device=args.device,
        display=args.display,
        imgsz=args.imgsz,
        two_stage=args.two_stage,
        low_conf_thresh=args.low_conf_thresh,
        iou_thresh_second=args.iou_thresh_second,
    )


if __name__ == "__main__":
    main()
