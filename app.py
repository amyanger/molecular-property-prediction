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
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn.models import AttentiveFP

# Paths
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"

# Task info
TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53"
]

TASK_DESCRIPTIONS = {
    "NR-AR": "Androgen Receptor - hormone disruption",
    "NR-AR-LBD": "Androgen Receptor Binding Domain",
    "NR-AhR": "Aryl Hydrocarbon Receptor - dioxin-like toxicity",
    "NR-Aromatase": "Aromatase Enzyme - estrogen synthesis disruption",
    "NR-ER": "Estrogen Receptor - hormone disruption",
    "NR-ER-LBD": "Estrogen Receptor Binding Domain",
    "NR-PPAR-gamma": "PPAR-gamma - metabolic effects",
    "SR-ARE": "Antioxidant Response Element - oxidative stress",
    "SR-ATAD5": "ATAD5 - DNA damage response",
    "SR-HSE": "Heat Shock Response - cellular stress",
    "SR-MMP": "Mitochondrial Membrane Potential - mitochondrial toxicity",
    "SR-p53": "p53 Tumor Suppressor - cancer risk"
}

EXAMPLE_MOLECULES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Acetaminophen (Tylenol)": "CC(=O)NC1=CC=C(C=C1)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Nicotine": "CN1CCCC1C2=CN=CC=C2",
    "Ethanol": "CCO",
    "Benzene (toxic)": "C1=CC=CC=C1",
    "DDT (pesticide)": "ClC(Cl)=C(c1ccc(Cl)cc1)c2ccc(Cl)cc2",
    "Formaldehyde (toxic)": "C=O",
    "Vitamin C": "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O",
}


# ============== Model Definitions ==============

class MolecularPropertyPredictor(nn.Module):
    def __init__(self, input_size=2048, hidden_sizes=[1024, 512, 256], num_tasks=12, dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        self.encoder = nn.Sequential(*layers)
        self.predictor = nn.Linear(hidden_sizes[-1], num_tasks)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.predictor(hidden)


class GNN(nn.Module):
    def __init__(self, num_node_features=140, hidden_channels=256, num_layers=4, num_tasks=12, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_proj = nn.Linear(num_node_features, hidden_channels)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        self.output = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_tasks)
        )

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        x = torch.relu(x)
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
            x = x + x_res
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        return self.output(x)


class AttentiveFPPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, out_channels=12, edge_dim=12,
                 num_layers=3, num_timesteps=3, dropout=0.2):
        super().__init__()
        self.attentive_fp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        embedding = self.attentive_fp(x, edge_index, edge_attr, batch)
        return self.predictor(embedding)


# ============== Feature Extraction ==============

ATOM_FEATURES_GCN = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
    'is_aromatic': [False, True],
    'num_hs': [0, 1, 2, 3, 4],
}

ATOM_FEATURES_AFP = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-2, -1, 0, 1, 2, 3],
    'chiral_tag': [0, 1, 2, 3],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [0, 1, 2, 3, 4, 5],
}


def one_hot(value, choices):
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


def get_atom_features_gcn(atom):
    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_FEATURES_GCN['atomic_num']))
    features.extend(one_hot(atom.GetDegree(), ATOM_FEATURES_GCN['degree']))
    features.extend(one_hot(atom.GetFormalCharge(), ATOM_FEATURES_GCN['formal_charge']))
    features.extend(one_hot(atom.GetHybridization(), ATOM_FEATURES_GCN['hybridization']))
    features.append(1 if atom.GetIsAromatic() else 0)
    features.extend(one_hot(atom.GetTotalNumHs(), ATOM_FEATURES_GCN['num_hs']))
    return features


def get_atom_features_afp(atom):
    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_FEATURES_AFP['atomic_num']))
    features.extend(one_hot(atom.GetDegree(), ATOM_FEATURES_AFP['degree']))
    features.extend(one_hot(atom.GetFormalCharge(), ATOM_FEATURES_AFP['formal_charge']))
    features.extend(one_hot(int(atom.GetChiralTag()), ATOM_FEATURES_AFP['chiral_tag']))
    features.extend(one_hot(atom.GetTotalNumHs(), ATOM_FEATURES_AFP['num_hs']))
    features.extend(one_hot(atom.GetHybridization(), ATOM_FEATURES_AFP['hybridization']))
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    return features


def get_bond_features(bond):
    features = []
    features.extend(one_hot(bond.GetBondType(), BOND_FEATURES['bond_type']))
    features.extend(one_hot(int(bond.GetStereo()), BOND_FEATURES['stereo']))
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    return features


# ============== Model Loading ==============

@st.cache_resource
def load_models():
    """Load all models (cached)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}

    # MLP
    try:
        mlp = MolecularPropertyPredictor(input_size=2048, hidden_sizes=[1024, 512, 256], num_tasks=12, dropout=0.3)
        checkpoint = torch.load(MODELS_DIR / 'best_model.pt', weights_only=False, map_location=device)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        mlp.to(device).eval()
        models['mlp'] = mlp
    except:
        pass

    # GCN
    try:
        gcn = GNN(num_node_features=140, hidden_channels=256, num_layers=4, num_tasks=12, dropout=0.2)
        checkpoint = torch.load(MODELS_DIR / 'best_gnn_model.pt', weights_only=False, map_location=device)
        gcn.load_state_dict(checkpoint['model_state_dict'])
        gcn.to(device).eval()
        models['gcn'] = gcn
    except:
        pass

    # AttentiveFP
    try:
        afp = AttentiveFPPredictor(in_channels=148, hidden_channels=256, out_channels=12, edge_dim=12,
                                    num_layers=3, num_timesteps=3, dropout=0.2)
        checkpoint = torch.load(MODELS_DIR / 'best_attentivefp_model.pt', weights_only=False, map_location=device)
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

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["🏠 Overview", "🔬 Predict Toxicity", "📊 Model Comparison", "📈 Training History"])

# Load data
results = load_results()
models, device = load_models()


# ============== Pages ==============

if page == "🏠 Overview":
    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Best Model AUC", "0.857", "Ensemble")
    with col2:
        st.metric("Models Trained", "3", "MLP, GCN, AttentiveFP")
    with col3:
        st.metric("Toxicity Endpoints", "12", "Tox21 Dataset")
    with col4:
        st.metric("Dataset Size", "7,831", "molecules")

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
    task_df = pd.DataFrame([
        {'Endpoint': task, 'Description': desc}
        for task, desc in TASK_DESCRIPTIONS.items()
    ])
    st.dataframe(task_df, use_container_width=True, hide_index=True)


elif page == "🔬 Predict Toxicity":
    st.header("Predict Molecular Toxicity")

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
            elif avg_tox > 0.3:
                risk_class = "risk-medium"
                risk_label = "MODERATE RISK"
            else:
                risk_class = "risk-low"
                risk_label = "LOW RISK"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Toxicity Score", f"{avg_tox:.1%}")
            with col2:
                st.markdown(f"**Risk Level:** <span class='{risk_class}'>{risk_label}</span>",
                           unsafe_allow_html=True)
            with col3:
                st.metric("Endpoints Flagged", f"{(ensemble_pred > 0.5).sum()}/12")

            # Per-task predictions
            st.subheader("Per-Endpoint Predictions")

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

    if not results:
        st.warning("No results found. Train models first.")
    else:
        # Per-task comparison
        st.subheader("Per-Task Performance")

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
            weights = results['Ensemble']['weights']
            fig = px.pie(
                values=list(weights.values()),
                names=[k.upper() for k in weights.keys()],
                title='Optimal Ensemble Weights'
            )
            st.plotly_chart(fig, use_container_width=True)


elif page == "📈 Training History":
    st.header("Training History")

    # Training curves
    models_with_history = {name: data for name, data in results.items()
                          if 'history' in data}

    if not models_with_history:
        st.warning("No training history found.")
    else:
        st.subheader("Training Loss")
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
