"""
Train all models and run ensemble evaluation.
One script to train MLP, GCN, AttentiveFP, and compute ensemble results.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"


def run_script(script_name, args=None):
    """Run a training script and return success status."""
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    return result.returncode == 0


def main():
    print("="*60)
    print("TRAINING ALL MODELS")
    print("="*60)

    # Track results
    results = {}

    # 1. Train MLP
    print("\n[1/4] Training MLP model...")
    results['MLP'] = run_script("train.py")

    # 2. Train GCN
    print("\n[2/4] Training GCN model...")
    results['GCN'] = run_script("train_gnn.py")

    # 3. Train AttentiveFP
    print("\n[3/4] Training AttentiveFP model...")
    results['AttentiveFP'] = run_script("train_attentivefp.py")

    # 4. Run ensemble
    print("\n[4/4] Running ensemble evaluation...")
    results['Ensemble'] = run_script("ensemble_all.py")

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    for model, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {model}: {status}")

    if all(results.values()):
        print("\nAll models trained successfully!")
        print(f"Results saved in: {PROJECT_DIR / 'models'}")
    else:
        print("\nSome models failed to train. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
