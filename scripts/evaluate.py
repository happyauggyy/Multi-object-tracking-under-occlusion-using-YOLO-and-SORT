"""Evaluation script for calculating MOT benchmark metrics (MOTA, IDF1, ID switches, FP, FN)."""

import argparse
import json
import os
import sys
import yaml
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.mot_evaluator import MOTEvaluator


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_evaluation(
    gt_path: str,
    pred_path: str,
    output_json: str = "outputs/metrics/metrics.json",
    output_csv: str = "outputs/metrics/metrics.csv",
    iou_thresh: float = -1.0,
    config_path: str = "configs/config.yaml",
    seq_name: str = "evaluation_seq",
) -> dict:
    """Run MOT evaluation comparing ground truth and tracker predictions."""
    cfg = load_config(config_path)
    eval_cfg = cfg.get("evaluation", {})
    thresh = iou_thresh if iou_thresh >= 0.0 else eval_cfg.get("iou_threshold", 0.50)

    if not os.path.exists(gt_path):
        print(f"\n[ERROR] Ground truth file not found: {gt_path}")
        print("To evaluate on MOT17, ensure the dataset is placed locally (e.g. data/MOT17/train/<seq>/gt/gt.txt)")
        print("and specify the correct path using --gt <path_to_gt.txt>.\n")
        sys.exit(1)

    if not os.path.exists(pred_path):
        print(f"\n[ERROR] Predictions file not found: {pred_path}")
        print("Please run tracking first using scripts/run_tracking.py or specify a valid output file.\n")
        sys.exit(1)

    print("==================================================")
    print("Evaluating Tracking Performance")
    print("==================================================")
    print(f"Ground Truth:  {gt_path}")
    print(f"Predictions:   {pred_path}")
    print(f"IoU Threshold: {thresh}")
    print("==================================================")

    evaluator = MOTEvaluator(iou_threshold=thresh)
    summary_df, metrics = evaluator.evaluate_sequence(
        gt_path=gt_path,
        pred_path=pred_path,
        seq_name=seq_name,
    )

    evaluator.save_metrics(metrics, output_json, output_csv)

    print("\nBenchmark Evaluation Results:")
    print("-" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<24}: {v:.4f}")
        else:
            print(f"  {k:<24}: {v}")
    print("-" * 50)
    print(f"Metrics saved to: {output_json} and {output_csv}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOT tracking results against ground truth.")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth MOT file (e.g. gt/gt.txt).")
    parser.add_argument("--pred", type=str, required=True, help="Path to tracking predictions file.")
    parser.add_argument("--output-json", type=str, default="outputs/metrics/metrics.json", help="Path to output JSON metrics.")
    parser.add_argument("--output-csv", type=str, default="outputs/metrics/metrics.csv", help="Path to output CSV metrics.")
    parser.add_argument("--iou-thresh", type=float, default=-1.0, help="Matching IoU threshold (default: 0.50).")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration YAML.")
    parser.add_argument("--seq-name", type=str, default="sequence", help="Name of sequence being evaluated.")

    args = parser.parse_args()
    run_evaluation(
        gt_path=args.gt,
        pred_path=args.pred,
        output_json=args.output_json,
        output_csv=args.output_csv,
        iou_thresh=args.iou_thresh,
        config_path=args.config,
        seq_name=args.seq_name,
    )


if __name__ == "__main__":
    main()
