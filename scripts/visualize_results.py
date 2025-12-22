"""
Visualize training results and model performance.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53"
]


def load_results():
    """Load training results."""
    with open(MODELS_DIR / 'results.json', 'r') as f:
        return json.load(f)


def plot_training_curves(results):
    """Plot training loss and validation AUC over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(results['history']['train_loss']) + 1)

    # Training Loss
    ax1.plot(epochs, results['history']['train_loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)

    # Validation AUC
    ax2.plot(epochs, results['history']['val_auc'], 'g-', linewidth=2)
    ax2.axhline(y=results['test_auc_mean'], color='r', linestyle='--',
                label=f'Test AUC: {results["test_auc_mean"]:.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC-ROC')
    ax2.set_title('Validation AUC Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'training_curves.png', dpi=150)
    print(f"Saved: {MODELS_DIR / 'training_curves.png'}")
    plt.show()


def plot_per_task_performance(results):
    """Plot AUC-ROC for each toxicity task."""
    tasks = list(results['test_aucs'].keys())
    aucs = list(results['test_aucs'].values())

    # Sort by AUC
    sorted_idx = np.argsort(aucs)[::-1]
    tasks = [tasks[i] for i in sorted_idx]
    aucs = [aucs[i] for i in sorted_idx]

    # Color code by performance
    colors = ['#2ecc71' if a > 0.85 else '#f39c12' if a > 0.75 else '#e74c3c' for a in aucs]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(tasks, aucs, color=colors)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.axvline(x=results['test_auc_mean'], color='red', linestyle='-',
               alpha=0.7, label=f'Mean: {results["test_auc_mean"]:.3f}')

    ax.set_xlabel('AUC-ROC Score')
    ax.set_title('Model Performance by Toxicity Endpoint')
    ax.set_xlim(0.5, 1.0)
    ax.legend()

    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.01, bar.get_y() + bar.get_height()/2,
                f'{auc:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'per_task_performance.png', dpi=150)
    print(f"Saved: {MODELS_DIR / 'per_task_performance.png'}")
    plt.show()


def print_summary(results):
    """Print results summary."""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)

    print(f"\nModel: {results['model'].upper()}")
    print(f"Epochs: {results['epochs']}")
    print(f"Best Validation AUC: {results['best_val_auc']:.4f}")
    print(f"Test AUC (Mean): {results['test_auc_mean']:.4f}")

    print(f"\n{'Task':<20} {'AUC-ROC':>10} {'Rating':>10}")
    print("-"*42)

    for task in TOX21_TASKS:
        auc = results['test_aucs'][task]
        if auc > 0.85:
            rating = "Excellent"
        elif auc > 0.80:
            rating = "Good"
        elif auc > 0.75:
            rating = "Fair"
        else:
            rating = "Needs Work"
        print(f"{task:<20} {auc:>10.4f} {rating:>10}")

    print("-"*42)
    print(f"{'OVERALL':<20} {results['test_auc_mean']:>10.4f}")

    # Comparison to benchmarks
    print("\n" + "="*60)
    print("COMPARISON TO PUBLISHED BENCHMARKS (Tox21)")
    print("="*60)
    print(f"Your Model:        {results['test_auc_mean']:.3f}")
    print(f"Random Forest:     ~0.750")
    print(f"Graph Neural Net:  ~0.830")
    print(f"State-of-the-Art:  ~0.850")

    if results['test_auc_mean'] > 0.83:
        print("\n🎉 Your model is competitive with published results!")
    elif results['test_auc_mean'] > 0.80:
        print("\n✓ Good performance! Room for improvement with advanced models.")


def main():
    results = load_results()
    print_summary(results)
    plot_training_curves(results)
    plot_per_task_performance(results)


if __name__ == "__main__":
    main()
