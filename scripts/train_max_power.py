"""
Maximum Power Pre-training Script for RTX 5090

This script runs the maximum training configuration optimized for
RTX 5090 with 32GB VRAM.

Configuration:
- 250K molecules from ZINC
- 200 epochs of pre-training
- Batch size 512
- Larger model (512 hidden channels, 6 layers)
- Fine-tuning on Tox21 with pre-trained weights

Estimated time: ~2 hours on RTX 5090

Usage:
    python scripts/train_max_power.py
    python scripts/train_max_power.py --molecules 500000  # Even more data
    python scripts/train_max_power.py --skip_download     # If data exists
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import argparse
import subprocess
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
    print("# MAXIMUM POWER Pre-training Pipeline (RTX 5090)")
    print(f"# Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Molecules: {args.molecules:,}")
    print(f"# Epochs: {args.epochs}")
    print(f"# Batch Size: {args.batch_size}")
    print(f"# Hidden Channels: {args.hidden_channels}")
    print(f"# Num Layers: {args.num_layers}")
    print(f"{'#'*60}")

    # Step 1: Download pre-training data
    if not args.skip_download:
        print("\n" + "="*60)
        print(f"Downloading {args.molecules:,} molecules...")
        print("="*60)

        success = run_command(
            [sys.executable, "scripts/download_pretrain_data.py",
             "--num_molecules", str(args.molecules)],
            f"Download {args.molecules:,} molecules"
        )
        if not success:
            print("Download failed!")
            return
    else:
        print("\nSkipping download (--skip_download flag)")

    # Step 2: Pre-train GNN encoder with maximum settings
    print("\n" + "="*60)
    print("Pre-training GNN with maximum settings...")
    print("="*60)

    success = run_command(
        [sys.executable, "scripts/pretrain_gnn.py",
         "--epochs", str(args.epochs),
         "--batch_size", str(args.batch_size),
         "--hidden_channels", str(args.hidden_channels),
         "--num_layers", str(args.num_layers)],
        "Pre-train GNN encoder (MAX POWER)"
    )

    if not success:
        print("Pre-training failed!")
        return

    # Step 3: Fine-tune on Tox21 with pre-trained weights
    print("\n" + "="*60)
    print("Fine-tuning on Tox21 with pre-trained weights...")
    print("="*60)

    success = run_command(
        [sys.executable, "scripts/train_gnn.py",
         "--epochs", "100",
         "--batch_size", str(args.batch_size),
         "--hidden_channels", str(args.hidden_channels),
         "--num_layers", str(args.num_layers),
         "--pretrained"],
        "Fine-tune GNN on Tox21"
    )

    if not success:
        print("Fine-tuning failed!")
        return

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'#'*60}")
    print("# MAXIMUM POWER Training Complete!")
    print(f"{'#'*60}")
    print(f"Duration: {duration}")
    print(f"\nModels saved to: {MODELS_DIR}")
    print("\nNext steps:")
    print("  1. Run ensemble: python scripts/ensemble_all.py")
    print("  2. Try the dashboard: streamlit run app.py")
    print("  3. Make predictions: python scripts/predict.py --smiles 'CCO'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Maximum power pre-training for RTX 5090')

    # Data settings
    parser.add_argument('--molecules', type=int, default=250000,
                        help='Number of molecules for pre-training (default: 250K)')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip data download (use existing data)')

    # Training settings
    parser.add_argument('--epochs', type=int, default=200,
                        help='Pre-training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (default: 512 for RTX 5090)')

    # Model settings
    parser.add_argument('--hidden_channels', type=int, default=512,
                        help='Hidden channels (default: 512)')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of GNN layers (default: 6)')

    args = parser.parse_args()
    main(args)
