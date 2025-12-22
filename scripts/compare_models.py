"""
Compare all trained models: MLP, GNN, and optionally XGBoost/Ensemble.
Creates visualizations showing performance differences.
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


def load_all_results():
    """Load results from all available models."""
    results = {}

    # MLP results
    mlp_path = MODELS_DIR / 'results.json'
    if mlp_path.exists():
        with open(mlp_path, 'r') as f:
            results['MLP'] = json.load(f)

    # GNN results
    gnn_path = MODELS_DIR / 'gnn_results.json'
    if gnn_path.exists():
        with open(gnn_path, 'r') as f:
            results['GNN'] = json.load(f)

    # XGBoost results
    xgb_path = MODELS_DIR / 'xgboost_results.json'
    if xgb_path.exists():
        with open(xgb_path, 'r') as f:
            results['XGBoost'] = json.load(f)

    # Ensemble results
    ens_path = MODELS_DIR / 'ensemble_results.json'
    if ens_path.exists():
        with open(ens_path, 'r') as f:
            results['Ensemble'] = json.load(f)

    return results


def plot_model_comparison(results):
    """Bar chart comparing overall AUC across models."""
    models = list(results.keys())
    aucs = [results[m]['test_auc_mean'] for m in models]

    # Sort by AUC
    sorted_idx = np.argsort(aucs)[::-1]
    models = [models[i] for i in sorted_idx]
    aucs = [aucs[i] for i in sorted_idx]

    # Colors
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'][:len(models)]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(models, aucs, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Reference lines
    ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, label='Random Forest baseline (~0.75)')
    ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='State-of-the-art (~0.85)')

    ax.set_ylabel('Test AUC-ROC', fontsize=12)
    ax.set_title('Model Comparison - Tox21 Toxicity Prediction', fontsize=14, fontweight='bold')
    ax.set_ylim(0.7, 0.9)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'model_comparison.png', dpi=150)
    print(f"Saved: {MODELS_DIR / 'model_comparison.png'}")
    plt.show()


def plot_per_task_comparison(results):
    """Grouped bar chart comparing per-task performance across models."""
    models = list(results.keys())

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(TOX21_TASKS))
    width = 0.8 / len(models)
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'][:len(models)]

    for i, (model, color) in enumerate(zip(models, colors)):
        aucs = [results[model]['test_aucs'][task] for task in TOX21_TASKS]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, aucs, width, label=model, color=color, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Toxicity Endpoint', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Per-Task Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(TOX21_TASKS, rotation=45, ha='right')
    ax.set_ylim(0.6, 1.0)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Reference line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'per_task_comparison.png', dpi=150)
    print(f"Saved: {MODELS_DIR / 'per_task_comparison.png'}")
    plt.show()


def plot_training_curves_comparison(results):
    """Compare training curves if available."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'MLP': '#3498db', 'GNN': '#2ecc71', 'XGBoost': '#e74c3c', 'Ensemble': '#9b59b6'}

    for model, data in results.items():
        if 'history' in data:
            epochs = range(1, len(data['history']['train_loss']) + 1)
            color = colors.get(model, '#333333')

            # Training loss
            ax1.plot(epochs, data['history']['train_loss'], '-',
                    color=color, linewidth=2, label=model)

            # Validation AUC
            ax2.plot(epochs, data['history']['val_auc'], '-',
                    color=color, linewidth=2, label=model)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation AUC-ROC')
    ax2.set_title('Validation AUC Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'training_curves_comparison.png', dpi=150)
    print(f"Saved: {MODELS_DIR / 'training_curves_comparison.png'}")
    plt.show()


def print_summary_table(results):
    """Print a summary comparison table."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    # Overall comparison
    print(f"\n{'Model':<15} {'Test AUC':>12} {'Best Val AUC':>15} {'Epochs':>10}")
    print("-" * 55)

    for model, data in sorted(results.items(), key=lambda x: x[1]['test_auc_mean'], reverse=True):
        test_auc = data['test_auc_mean']
        val_auc = data.get('best_val_auc', data.get('test_auc_mean', 0))
        epochs = data.get('epochs', 'N/A')
        print(f"{model:<15} {test_auc:>12.4f} {val_auc:>15.4f} {str(epochs):>10}")

    print("-" * 55)

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_auc_mean'])
    print(f"\nBest Model: {best_model[0]} (AUC: {best_model[1]['test_auc_mean']:.4f})")

    # Per-task winners
    print(f"\n{'Task':<15}", end="")
    for model in results.keys():
        print(f"{model:>12}", end="")
    print(f"{'Best':>12}")
    print("-" * (15 + 12 * len(results) + 12))

    for task in TOX21_TASKS:
        print(f"{task:<15}", end="")
        task_aucs = {}
        for model, data in results.items():
            auc = data['test_aucs'][task]
            task_aucs[model] = auc
            print(f"{auc:>12.4f}", end="")

        best = max(task_aucs.items(), key=lambda x: x[1])
        print(f"{best[0]:>12}")

    # Improvement over MLP
    if 'MLP' in results and len(results) > 1:
        mlp_auc = results['MLP']['test_auc_mean']
        print(f"\n{'='*70}")
        print("IMPROVEMENT OVER MLP BASELINE")
        print("=" * 70)
        for model, data in results.items():
            if model != 'MLP':
                improvement = (data['test_auc_mean'] - mlp_auc) * 100
                print(f"{model}: {improvement:+.2f}% ({data['test_auc_mean']:.4f} vs {mlp_auc:.4f})")


def main():
    results = load_all_results()

    if not results:
        print("No model results found. Train at least one model first.")
        return

    print(f"Found results for: {', '.join(results.keys())}")

    print_summary_table(results)

    if len(results) > 1:
        plot_model_comparison(results)
        plot_per_task_comparison(results)
        plot_training_curves_comparison(results)
    else:
        print("\nTrain more models to see comparison visualizations.")


if __name__ == "__main__":
    main()
