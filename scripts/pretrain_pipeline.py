"""
Full pre-training pipeline for improved molecular property prediction.

This script runs the complete pre-training workflow:
1. Download pre-training data (ZINC 250K molecules)
2. Pre-train GNN encoder with self-supervised learning
3. Fine-tune on Tox21 with pre-trained weights
4. Compare results with baseline (no pre-training)

Expected improvement: +3-5% AUC-ROC

Usage:
    python scripts/pretrain_pipeline.py
    python scripts/pretrain_pipeline.py --skip_download  # If data already downloaded
    python scripts/pretrain_pipeline.py --quick  # Quick test with smaller data
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import argparse
import subprocess
import json
from datetime import datetime

from src.constants import MODELS_DIR, DATA_DIR


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))

    if result.returncode != 0:
        print(f"\nError: {description} failed with code {result.returncode}")
        return False

    print(f"\n{description} completed successfully!")
    return True


def main(args):
    start_time = datetime.now()
    print(f"\n{'#'*60}")
    print("# GNN Pre-training Pipeline")
    print(f"# Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    results = {}

    # Step 1: Download pre-training data
    if not args.skip_download:
        data_path = DATA_DIR / "zinc" / "zinc_pretrain.csv"
        if not data_path.exists():
            num_mols = 10000 if args.quick else 50000
            success = run_command(
                [sys.executable, "scripts/download_pretrain_data.py",
                 "--num_molecules", str(num_mols)],
                "Download pre-training data"
            )
            if not success:
                return
        else:
            print(f"\nPre-training data already exists at {data_path}")
    else:
        print("\nSkipping download (--skip_download flag)")

    # Step 2: Pre-train GNN encoder
    if not args.skip_pretrain:
        pretrain_epochs = 20 if args.quick else 100
        max_mols = 5000 if args.quick else 0  # 0 = all

        cmd = [
            sys.executable, "scripts/pretrain_gnn.py",
            "--epochs", str(pretrain_epochs),
            "--batch_size", "256" if not args.quick else "128",
        ]
        if max_mols > 0:
            cmd.extend(["--max_molecules", str(max_mols)])

        success = run_command(cmd, "Pre-train GNN encoder")
        if not success:
            print("Pre-training failed. Continuing with fine-tuning from scratch...")
    else:
        print("\nSkipping pre-training (--skip_pretrain flag)")

    # Step 3: Fine-tune on Tox21 WITH pre-trained weights
    finetune_epochs = 30 if args.quick else 50

    print("\n" + "="*60)
    print("Fine-tuning WITH pre-trained weights")
    print("="*60)

    pretrained_path = MODELS_DIR / "pretrained_gnn_encoder.pt"
    if pretrained_path.exists():
        success = run_command(
            [sys.executable, "scripts/train_gnn.py",
             "--epochs", str(finetune_epochs),
             "--pretrained"],
            "Fine-tune GNN with pre-trained weights"
        )

        if success:
            # Load results
            results_path = MODELS_DIR / "gnn_results.json"
            if results_path.exists():
                with open(results_path) as f:
                    results['pretrained'] = json.load(f)
                print(f"\nPre-trained model AUC: {results['pretrained']['test_auc_mean']:.4f}")
    else:
        print("No pre-trained weights found. Skipping pre-trained fine-tuning.")

    # Step 4: Train baseline (no pre-training) for comparison
    if args.compare:
        print("\n" + "="*60)
        print("Training BASELINE (no pre-training) for comparison")
        print("="*60)

        # Temporarily rename the pretrained weights
        backup_path = MODELS_DIR / "pretrained_gnn_encoder.pt.backup"
        if pretrained_path.exists():
            pretrained_path.rename(backup_path)

        success = run_command(
            [sys.executable, "scripts/train_gnn.py",
             "--epochs", str(finetune_epochs)],
            "Train GNN baseline (no pre-training)"
        )

        # Restore pretrained weights
        if backup_path.exists():
            backup_path.rename(pretrained_path)

        if success:
            results_path = MODELS_DIR / "gnn_results.json"
            if results_path.exists():
                with open(results_path) as f:
                    results['baseline'] = json.load(f)
                print(f"\nBaseline model AUC: {results['baseline']['test_auc_mean']:.4f}")

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'#'*60}")
    print("# Pipeline Complete!")
    print(f"{'#'*60}")
    print(f"Duration: {duration}")

    if 'pretrained' in results:
        print(f"\nPre-trained GNN AUC: {results['pretrained']['test_auc_mean']:.4f}")

    if 'baseline' in results:
        print(f"Baseline GNN AUC:    {results['baseline']['test_auc_mean']:.4f}")

        if 'pretrained' in results:
            improvement = results['pretrained']['test_auc_mean'] - results['baseline']['test_auc_mean']
            print(f"\nImprovement: {improvement*100:+.2f}%")

    print(f"\nModels saved to: {MODELS_DIR}")
    print("\nNext steps:")
    print("  1. Run ensemble: python scripts/ensemble_all.py")
    print("  2. Try the dashboard: streamlit run app.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full pre-training pipeline')

    parser.add_argument('--quick', action='store_true',
                        help='Quick test run with reduced data and epochs')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip data download (use existing data)')
    parser.add_argument('--skip_pretrain', action='store_true',
                        help='Skip pre-training (use existing weights)')
    parser.add_argument('--compare', action='store_true', default=True,
                        help='Train baseline for comparison')
    parser.add_argument('--no_compare', action='store_false', dest='compare',
                        help='Skip baseline comparison')

    args = parser.parse_args()
    main(args)
