"""Script to identify, categorize, and report identity switches and tracking failures."""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.failure_analysis import FailureAnalyzer


def run_failure_analysis(
    gt_path: str,
    pred_path: str,
    output_csv: str = "outputs/failure_cases/id_switches.csv",
    output_plot: str = "outputs/plots/failure_distribution.png",
    match_iou_thresh: float = 0.50,
) -> pd.DataFrame:
    """Analyze tracking failure modes and export report."""
    if not os.path.exists(gt_path):
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        print("Please provide a valid MOT ground truth file using --gt <path_to_gt.txt>.")
        sys.exit(1)

    if not os.path.exists(pred_path):
        print(f"[ERROR] Predictions file not found: {pred_path}")
        print("Please provide a valid tracking predictions file using --pred <path_to_pred.txt>.")
        sys.exit(1)

    print("==================================================")
    print("Tracking Failure & ID-Switch Analysis")
    print("==================================================")
    print(f"Ground Truth:  {gt_path}")
    print(f"Predictions:   {pred_path}")
    print(f"IoU Threshold: {match_iou_thresh}")
    print("==================================================")

    analyzer = FailureAnalyzer(match_iou_thresh=match_iou_thresh)
    switches = analyzer.analyze_switches(gt_path, pred_path)
    df_report = analyzer.generate_report(switches, output_csv_path=output_csv)

    print(f"\nTotal ID Switches Detected: {len(switches)}")
    if not df_report.empty:
        cause_counts = df_report["probable_cause"].value_counts()
        print("\nBreakdown by Probable Cause:")
        print("-" * 40)
        for cause, count in cause_counts.items():
            pct = (count / len(df_report)) * 100.0
            print(f"  {cause:<24}: {count:>3} ({pct:.1f}%)")
        print("-" * 40)
        print(f"Full breakdown exported to: {output_csv}")

        # Generate bar plot of failure causes
        if output_plot:
            os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)
            plt.figure(figsize=(8, 4.5), dpi=150)
            cause_counts.plot(kind="bar", color="#3b82f6", edgecolor="#1e3a8a")
            plt.title("Distribution of ID-Switch Causes")
            plt.xlabel("Attributed Cause")
            plt.ylabel("Number of Occurrences")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(output_plot)
            plt.close()
            print(f"Failure distribution plot saved to: {output_plot}")
    else:
        print("No ID switches found! Perfect identity preservation on matched targets.")

    return df_report


def main():
    parser = argparse.ArgumentParser(description="Analyze tracking ID switches and failure causes.")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth MOT annotations file.")
    parser.add_argument("--pred", type=str, required=True, help="Path to tracker output file.")
    parser.add_argument("--output-csv", type=str, default="outputs/failure_cases/id_switches.csv", help="Path to save switch report CSV.")
    parser.add_argument("--output-plot", type=str, default="outputs/plots/failure_distribution.png", help="Path to save cause distribution plot.")
    parser.add_argument("--iou-thresh", type=float, default=0.50, help="Matching IoU threshold (default: 0.50).")

    args = parser.parse_args()
    run_failure_analysis(
        gt_path=args.gt,
        pred_path=args.pred,
        output_csv=args.output_csv,
        output_plot=args.output_plot,
        match_iou_thresh=args.iou_thresh,
    )


if __name__ == "__main__":
    main()
