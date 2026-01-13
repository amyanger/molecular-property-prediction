"""
Molecular Toxicity Prediction Dashboard
Interactive Streamlit app for visualizing model performance and making predictions.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from pathlib import Path
import torch

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Import from shared modules
from src.models import MolecularPropertyPredictor, GNN, AttentiveFPPredictor
from src.constants import TOX21_TASKS, TASK_DESCRIPTIONS, EXAMPLE_MOLECULES, MODELS_DIR
from src.utils import get_atom_features_gcn, get_atom_features_afp, get_bond_features


# ============== Model Loading ==============

@st.cache_resource
def load_models():
    """Load all models (cached)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}

    # MLP
    try:
        mlp = MolecularPropertyPredictor(input_size=2048, hidden_sizes=[1024, 512, 256], num_tasks=12, dropout=0.3)
        checkpoint = torch.load(MODELS_DIR / 'best_model.pt', weights_only=True, map_location=device)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        mlp.to(device).eval()
        models['mlp'] = mlp
    except:
        pass

    # GCN
    try:
        gcn = GNN(num_node_features=141, hidden_channels=256, num_layers=4, num_tasks=12, dropout=0.2)
        checkpoint = torch.load(MODELS_DIR / 'best_gnn_model.pt', weights_only=True, map_location=device)
        gcn.load_state_dict(checkpoint['model_state_dict'])
        gcn.to(device).eval()
        models['gcn'] = gcn
    except:
        pass

    # AttentiveFP
    try:
        afp = AttentiveFPPredictor(in_channels=148, hidden_channels=256, out_channels=12, edge_dim=12,
                                    num_layers=3, num_timesteps=3, dropout=0.2)
        checkpoint = torch.load(MODELS_DIR / 'best_attentivefp_model.pt', weights_only=True, map_location=device)
        afp.load_state_dict(checkpoint['model_state_dict'])
        afp.to(device).eval()
        models['attentivefp'] = afp
    except:
        pass

    return models, device


def load_results():
    """Load all results files."""
    results = {}

    files = {
        'MLP': 'results.json',
        'GCN': 'gnn_results.json',
        'AttentiveFP': 'attentivefp_results.json',
        'Ensemble': 'ensemble_results.json'
    }

    for name, filename in files.items():
        path = MODELS_DIR / filename
        if path.exists():
            with open(path, 'r') as f:
                results[name] = json.load(f)

    return results


# ============== Prediction ==============

def predict_molecule(smiles, models, device, weights=None):
    """Get ensemble prediction for a molecule."""
    if weights is None:
        weights = {'mlp': 0.1, 'gcn': 0.5, 'attentivefp': 0.4}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    predictions = {}

    # MLP
    if 'mlp' in models:
        fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
        fp = fp_gen.GetFingerprintAsNumPy(mol)
        x = torch.FloatTensor(fp).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = models['mlp'](x)
            predictions['mlp'] = torch.sigmoid(logits).cpu().numpy()[0]

    # GCN
    if 'gcn' in models:
        atom_features = [get_atom_features_gcn(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
        if len(edge_index) == 0:
            edge_index = [[0, 0]]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        batch = torch.zeros(x.size(0), dtype=torch.long)
        x, edge_index, batch = x.to(device), edge_index.to(device), batch.to(device)
        with torch.no_grad():
            logits = models['gcn'](x, edge_index, batch)
            predictions['gcn'] = torch.sigmoid(logits).cpu().numpy()[0]

    # AttentiveFP
    if 'attentivefp' in models:
        atom_features = [get_atom_features_afp(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = get_bond_features(bond)
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([bf, bf])
        if len(edge_index) == 0:
            edge_index = [[0, 0]]
            edge_attr = [[0] * 12]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        batch = torch.zeros(x.size(0), dtype=torch.long)
        x, edge_index, edge_attr, batch = x.to(device), edge_index.to(device), edge_attr.to(device), batch.to(device)
        with torch.no_grad():
            logits = models['attentivefp'](x, edge_index, edge_attr, batch)
            predictions['attentivefp'] = torch.sigmoid(logits).cpu().numpy()[0]

    # Ensemble
    if predictions:
        ensemble = np.zeros(12)
        total_weight = 0
        for model_name, preds in predictions.items():
            w = weights.get(model_name, 0.33)
            ensemble += w * preds
            total_weight += w
        ensemble /= total_weight
        return ensemble, predictions

    return None, None


# ============== Streamlit App ==============

st.set_page_config(
    page_title="Molecular Toxicity Predictor",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧪 Molecular Toxicity Predictor</div>', unsafe_allow_html=True)
st.markdown("**Deep learning models for predicting molecular toxicity across 12 biological endpoints**")

# What is this app? - Layman explanation
with st.expander("ℹ️ What does this app do? (Click to learn more)"):
    st.markdown("""
    ### Understanding Molecular Toxicity Prediction

    This app uses **artificial intelligence** to predict whether a chemical compound might be harmful to living cells or organisms.

    **How does it work?**
    - You give the app a molecule (using a text code called SMILES)
    - The AI models analyze the molecule's structure
    - The app predicts how likely the molecule is to cause harm in 12 different ways

    **Why is this useful?**
    - Drug companies can quickly screen chemicals before expensive lab tests
    - Researchers can identify potentially dangerous compounds early
    - It helps prioritize which chemicals need more safety testing

    **What is a SMILES string?**
    - SMILES is a way to write a molecule's structure as text
    - For example, `CCO` represents ethanol (drinking alcohol)
    - `CC(=O)OC1=CC=CC=C1C(=O)O` represents aspirin
    """)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["🏠 Overview", "🔬 Predict Toxicity", "📊 Model Comparison", "📈 Training History"])

# Load data
results = load_results()
models, device = load_models()


# ============== Pages ==============

if page == "🏠 Overview":
    st.header("Project Overview")

    st.info("""
    **What you're looking at:** This page shows how well our AI models perform at predicting molecular toxicity.
    The main number to look at is **AUC** (Area Under Curve) - a score from 0 to 1 where **higher is better**.
    An AUC of 0.5 means random guessing, while 1.0 would be perfect prediction. Our models achieve ~0.87, which is quite good!
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Best Model AUC", "0.867", "Ensemble",
                 help="Our best model correctly ranks toxic vs non-toxic molecules 87% of the time")
    with col2:
        st.metric("Models Trained", "3", "MLP, GCN, AttentiveFP",
                 help="Three different AI architectures combined for better accuracy")
    with col3:
        st.metric("Toxicity Endpoints", "12", "Tox21 Dataset",
                 help="12 different biological tests measuring various types of toxicity")
    with col4:
        st.metric("Dataset Size", "7,831", "molecules",
                 help="Number of molecules used to train and test the models")

    st.markdown("---")

    # Model summary
    st.subheader("Model Performance Summary")

    if results:
        model_data = []
        for name, data in results.items():
            if name != 'Ensemble':
                auc = data.get('test_auc_mean', data.get('ensemble_auc', 0))
            else:
                auc = data.get('ensemble_auc', 0)
            model_data.append({'Model': name, 'Test AUC-ROC': auc})

        df = pd.DataFrame(model_data).sort_values('Test AUC-ROC', ascending=False)

        fig = px.bar(df, x='Model', y='Test AUC-ROC',
                     color='Test AUC-ROC',
                     color_continuous_scale='viridis',
                     title='Model Performance Comparison')
        fig.update_layout(yaxis_range=[0.75, 0.9])
        fig.add_hline(y=0.85, line_dash="dash", line_color="red",
                      annotation_text="State-of-the-art (~0.85)")
        st.plotly_chart(fig, use_container_width=True)

    # Task descriptions
    st.subheader("Toxicity Endpoints")
    st.markdown("""
    **What are these endpoints?** Each endpoint represents a different way a chemical can harm biological systems.
    These are based on the **Tox21** dataset - a real-world dataset used by the EPA, FDA, and NIH to screen chemicals.
    Think of each endpoint as a different "test" the molecule goes through.
    """)
    task_df = pd.DataFrame([
        {'Endpoint': task, 'Description': desc}
        for task, desc in TASK_DESCRIPTIONS.items()
    ])
    st.dataframe(task_df, use_container_width=True, hide_index=True)


elif page == "🔬 Predict Toxicity":
    st.header("Predict Molecular Toxicity")

    st.info("""
    **How to use this page:**
    1. Select an example molecule from the dropdown OR enter your own SMILES string
    2. Click "Predict Toxicity" to run the AI analysis
    3. View the results showing the likelihood of toxicity for each biological test

    **Understanding the results:** Probabilities closer to 100% mean the molecule is more likely to be toxic for that endpoint.
    The red dashed line at 50% is the threshold - anything above this is flagged as potentially toxic.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Molecule")

        # Example selector
        example = st.selectbox("Select example molecule:", ["Custom"] + list(EXAMPLE_MOLECULES.keys()))

        if example == "Custom":
            smiles = st.text_input("Enter SMILES string:",
                                   placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O")
        else:
            smiles = EXAMPLE_MOLECULES[example]
            st.code(smiles)

        predict_btn = st.button("🔮 Predict Toxicity", type="primary", use_container_width=True)

    with col2:
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.subheader("Molecule Structure")
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img)
            else:
                st.error("Invalid SMILES string")

    if predict_btn and smiles:
        ensemble_pred, individual_preds = predict_molecule(smiles, models, device)

        if ensemble_pred is not None:
            st.markdown("---")
            st.subheader("Prediction Results")

            # Overall risk
            avg_tox = np.mean(ensemble_pred)
            if avg_tox > 0.5:
                risk_class = "risk-high"
                risk_label = "HIGH RISK"
                risk_explanation = "This molecule shows concerning toxicity signals across multiple tests. Further investigation recommended."
            elif avg_tox > 0.3:
                risk_class = "risk-medium"
                risk_label = "MODERATE RISK"
                risk_explanation = "This molecule shows some potential toxicity signals. May warrant additional testing."
            else:
                risk_class = "risk-low"
                risk_label = "LOW RISK"
                risk_explanation = "This molecule shows low toxicity signals across most tests. Generally favorable safety profile."

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Toxicity Score", f"{avg_tox:.1%}",
                         help="Average probability of toxicity across all 12 endpoints. Lower is safer.")
            with col2:
                st.markdown(f"**Risk Level:** <span class='{risk_class}'>{risk_label}</span>",
                           unsafe_allow_html=True)
            with col3:
                st.metric("Endpoints Flagged", f"{(ensemble_pred > 0.5).sum()}/12",
                         help="Number of toxicity tests where the molecule scored above 50% probability.")

            st.caption(f"💡 {risk_explanation}")

            # Per-task predictions
            st.subheader("Per-Endpoint Predictions")
            st.markdown("""
            Each bar shows the probability (0-100%) that the molecule is toxic for that specific biological test.
            **Green = safer**, **Red = more concerning**. The dashed red line marks the 50% threshold.
            """)

            pred_df = pd.DataFrame({
                'Endpoint': TOX21_TASKS,
                'Probability': ensemble_pred,
                'Description': [TASK_DESCRIPTIONS[t] for t in TOX21_TASKS]
            }).sort_values('Probability', ascending=True)

            fig = px.bar(pred_df, y='Endpoint', x='Probability',
                        orientation='h',
                        color='Probability',
                        color_continuous_scale='RdYlGn_r',
                        title='Toxicity Probability by Endpoint')
            fig.add_vline(x=0.5, line_dash="dash", line_color="red")
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # Model agreement
            if len(individual_preds) > 1:
                st.subheader("Model Agreement")
                st.markdown("""
                This table shows how each AI model voted. When models **agree** (similar percentages),
                we have more confidence in the prediction. Large disagreements suggest uncertainty.
                The **Ensemble** column is the weighted average of all models - our best overall prediction.
                """)
                agreement_data = []
                for task_idx, task in enumerate(TOX21_TASKS):
                    row = {'Endpoint': task}
                    for model_name, preds in individual_preds.items():
                        row[model_name.upper()] = preds[task_idx]
                    row['Ensemble'] = ensemble_pred[task_idx]
                    agreement_data.append(row)

                agreement_df = pd.DataFrame(agreement_data)
                st.dataframe(agreement_df.style.format({
                    col: '{:.1%}' for col in agreement_df.columns if col != 'Endpoint'
                }), use_container_width=True, hide_index=True)
        else:
            st.error("Could not generate prediction. Please check the SMILES string.")


elif page == "📊 Model Comparison":
    st.header("Model Comparison")

    st.info("""
    **What you're looking at:** This page compares the performance of our different AI models.

    **The models explained:**
    - **MLP (Multi-Layer Perceptron):** A simple neural network that looks at molecular "fingerprints" - numeric codes representing the molecule's structure
    - **GCN (Graph Convolutional Network):** Treats the molecule as a network of connected atoms and learns patterns from the connections
    - **AttentiveFP:** An advanced model that "pays attention" to the most important parts of the molecule
    - **Ensemble:** Combines all three models' predictions for better accuracy (like asking multiple experts)

    **What is AUC-ROC?** A score from 0.5 to 1.0 measuring how well the model distinguishes toxic from non-toxic molecules.
    Higher is better. 0.5 = random guessing, 1.0 = perfect.
    """)

    if not results:
        st.warning("No results found. Train models first.")
    else:
        # Per-task comparison
        st.subheader("Per-Task Performance")
        st.markdown("This chart shows how well each model performs on each of the 12 toxicity tests. Taller bars = better performance.")

        task_data = []
        for task in TOX21_TASKS:
            row = {'Task': task}
            for model_name, data in results.items():
                if 'test_aucs' in data:
                    row[model_name] = data['test_aucs'].get(task, 0)
                elif 'ensemble_aucs' in data:
                    row[model_name] = data['ensemble_aucs'].get(task, 0)
            task_data.append(row)

        task_df = pd.DataFrame(task_data)

        fig = go.Figure()
        colors = {'MLP': '#3498db', 'GCN': '#2ecc71', 'AttentiveFP': '#e74c3c', 'Ensemble': '#9b59b6'}

        for model in ['MLP', 'GCN', 'AttentiveFP', 'Ensemble']:
            if model in task_df.columns:
                fig.add_trace(go.Bar(
                    name=model,
                    x=task_df['Task'],
                    y=task_df[model],
                    marker_color=colors.get(model, '#333')
                ))

        fig.update_layout(
            barmode='group',
            title='AUC-ROC by Task and Model',
            yaxis_range=[0.6, 1.0],
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.subheader("Summary Statistics")
        st.markdown("""
        This table ranks models by overall performance. **Best Task** shows where each model excels,
        while **Worst Task** shows its weakest area. Some toxicity tests are harder to predict than others.
        """)
        summary_data = []
        for model_name, data in results.items():
            if model_name == 'Ensemble':
                auc = data.get('ensemble_auc', 0)
            else:
                auc = data.get('test_auc_mean', 0)

            summary_data.append({
                'Model': model_name,
                'Mean AUC': auc,
                'Best Task': max(data.get('test_aucs', data.get('ensemble_aucs', {})).items(),
                                key=lambda x: x[1], default=('N/A', 0))[0] if data.get('test_aucs') or data.get('ensemble_aucs') else 'N/A',
                'Worst Task': min(data.get('test_aucs', data.get('ensemble_aucs', {})).items(),
                                 key=lambda x: x[1], default=('N/A', 1))[0] if data.get('test_aucs') or data.get('ensemble_aucs') else 'N/A'
            })

        summary_df = pd.DataFrame(summary_data).sort_values('Mean AUC', ascending=False)
        st.dataframe(summary_df.style.format({'Mean AUC': '{:.4f}'}),
                    use_container_width=True, hide_index=True)

        # Ensemble weights
        if 'Ensemble' in results and 'weights' in results['Ensemble']:
            st.subheader("Ensemble Weights")
            st.markdown("""
            The ensemble combines predictions from all models, but not equally. Better-performing models
            get more "voting power". This pie chart shows how much each model contributes to the final prediction.
            """)
            weights = results['Ensemble']['weights']
            fig = px.pie(
                values=list(weights.values()),
                names=[k.upper() for k in weights.keys()],
                title='Optimal Ensemble Weights'
            )
            st.plotly_chart(fig, use_container_width=True)


elif page == "📈 Training History":
    st.header("Training History")

    st.info("""
    **What you're looking at:** This page shows how the models learned during training.

    **Training Loss:** How wrong the model's predictions are. Lower is better.
    A decreasing line means the model is learning and improving.

    **Validation AUC:** How well the model performs on data it hasn't seen before.
    An increasing line means the model is getting better at generalizing.
    If this starts dropping while training loss keeps decreasing, the model is "overfitting"
    (memorizing training data instead of learning general patterns).
    """)

    # Training curves
    models_with_history = {name: data for name, data in results.items()
                          if 'history' in data}

    if not models_with_history:
        st.warning("No training history found.")
    else:
        st.subheader("Training Loss")
        st.caption("Each 'epoch' is one complete pass through the training data. Watch how the loss decreases as models learn.")
        fig = go.Figure()
        for model_name, data in models_with_history.items():
            if 'train_loss' in data['history']:
                fig.add_trace(go.Scatter(
                    y=data['history']['train_loss'],
                    mode='lines',
                    name=model_name
                ))
        fig.update_layout(xaxis_title='Epoch', yaxis_title='Loss')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Validation AUC")
        st.caption("This shows model performance on held-out test data. Higher and more stable is better.")
        fig = go.Figure()
        for model_name, data in models_with_history.items():
            if 'val_auc' in data['history']:
                fig.add_trace(go.Scatter(
                    y=data['history']['val_auc'],
                    mode='lines',
                    name=model_name
                ))
        fig.update_layout(xaxis_title='Epoch', yaxis_title='AUC-ROC')
        st.plotly_chart(fig, use_container_width=True)

        # Training details table
        st.subheader("Training Details")
        details = []
        for model_name, data in results.items():
            if 'config' in data:
                details.append({
                    'Model': model_name,
                    'Epochs': data.get('epochs', 'N/A'),
                    'Best Val AUC': data.get('best_val_auc', 'N/A'),
                    'Test AUC': data.get('test_auc_mean', 'N/A')
                })
        if details:
            st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built with Streamlit | Models trained on Tox21 Dataset |
    <a href='https://github.com/amyanger/molecular-property-prediction'>GitHub</a>
</div>
""", unsafe_allow_html=True)
