"""
CMR-RIS Unified Execution & Training Pipeline CLI
CardioMetabolic-Renal Risk Intelligence System

Usage:
    python -m src.pipeline --mode all
    python -m src.pipeline --mode train
    python -m src.pipeline --mode calibrate --confidence 0.90
    python -m src.pipeline --mode fairness
    python -m src.pipeline --mode eval
"""

import argparse
import os
import sys
import pandas as pd

from src.modeling import run_full_modeling_suite
from src.uncertainty import calibrate_all_champions
from src.fairness import run_full_fairness_audit


def main():
    parser = argparse.ArgumentParser(
        description="CMR-RIS End-to-End Clinical ML Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "train", "calibrate", "fairness", "eval"],
        default="all",
        help="Execution mode for the pipeline"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save/load trained model serialized artifacts"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save benchmarks and evaluation reports"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.90,
        help="Conformal prediction confidence level (1 - alpha)"
    )

    args = parser.parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 70)
    print("CMR-RIS Clinical Decision-Support Pipeline")
    print(f"Mode: {args.mode.upper()} | Models Dir: {args.models_dir} | Results Dir: {args.results_dir}")
    print("=" * 70)

    if args.mode in ["all", "train"]:
        print("\n[1/3] Training Disease-Specific Models & Multi-Task Shared Representation...")
        df_comp, champions = run_full_modeling_suite(
            results_dir=args.results_dir,
            models_dir=args.models_dir
        )
        print("[OK] All models trained, comparison exported, champions selected.")

    if args.mode in ["all", "calibrate"]:
        print(f"\n[2/3] Performing Split Conformal Calibration (Confidence={args.confidence*100:.0f}%)...")
        calibrator = calibrate_all_champions(
            models_dir=args.models_dir,
            confidence_level=args.confidence
        )
        print("[OK] Conformal calibrator saved to models/conformal_calibrator.pkl")

    if args.mode in ["all", "fairness"]:
        print("\n[3/3] Running Demographic Subgroup Fairness & Disparity Audit...")
        df_fair = run_full_fairness_audit(
            models_dir=args.models_dir,
            results_dir=args.results_dir
        )
        print("[OK] Fairness audit complete and exported.")

    if args.mode in ["all", "eval"]:
        print("\n" + "=" * 70)
        print("PIPELINE BENCHMARK SUMMARY")
        print("=" * 70)
        comp_path = os.path.join(args.results_dir, "model_comparison.csv")
        if os.path.exists(comp_path):
            df_comp = pd.read_csv(comp_path)
            print("\nModel Comparison Table:")
            print(df_comp[["Disease", "Model_Type", "Algorithm", "Test_Recall", "Test_ROC_AUC", "Test_F1"]].to_string(index=False))

        fair_path = os.path.join(args.results_dir, "fairness_audit.csv")
        if os.path.exists(fair_path):
            df_fair = pd.read_csv(fair_path)
            print("\nDemographic Fairness Audit Preview:")
            print(df_fair.head(6).to_string(index=False))

    print("\n[SUCCESS] CMR-RIS Pipeline execution finished successfully.\n")


if __name__ == "__main__":
    main()
