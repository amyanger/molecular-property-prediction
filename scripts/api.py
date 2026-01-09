"""
FastAPI REST API for molecular toxicity prediction.

Run with: uvicorn scripts.api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.models import MolecularPropertyPredictor, GNN, AttentiveFPPredictor
from src.constants import TOX21_TASKS, TASK_DESCRIPTIONS, MODELS_DIR, DEFAULT_ENSEMBLE_WEIGHTS
from src.utils.features import get_atom_features_gcn, get_atom_features_afp, get_bond_features

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Molecular Toxicity Prediction API",
    description="Predict toxicity endpoints for molecules using ensemble deep learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Request/Response Models
# =============================================================================


class PredictionRequest(BaseModel):
    """Request model for toxicity prediction."""

    smiles: str = Field(..., description="SMILES string of the molecule")
    model: Optional[str] = Field(
        default="ensemble",
        description="Model to use: 'mlp', 'gnn', 'attentivefp', or 'ensemble'"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "model": "ensemble"
            }
        }


class ToxicityPrediction(BaseModel):
    """Individual toxicity endpoint prediction."""

    endpoint: str
    description: str
    probability: float
    risk_level: str


class PredictionResponse(BaseModel):
    """Response model for toxicity prediction."""

    smiles: str
    valid: bool
    model_used: str
    overall_toxicity_score: float
    overall_risk_level: str
    predictions: list[ToxicityPrediction]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: dict


# =============================================================================
# Model Loading
# =============================================================================

# Global model storage
models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_models():
    """Load all trained models."""
    global models

    # MLP Model
    try:
        mlp = MolecularPropertyPredictor(input_size=2048, num_tasks=12)
        checkpoint = torch.load(MODELS_DIR / 'best_model.pt', map_location=device, weights_only=False)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        mlp.to(device)
        mlp.eval()
        models['mlp'] = mlp
    except Exception as e:
        print(f"Warning: Could not load MLP model: {e}")

    # GNN Model
    try:
        gnn = GNN(num_node_features=141, num_tasks=12)
        checkpoint = torch.load(MODELS_DIR / 'best_gnn_model.pt', map_location=device, weights_only=False)
        gnn.load_state_dict(checkpoint['model_state_dict'])
        gnn.to(device)
        gnn.eval()
        models['gnn'] = gnn
    except Exception as e:
        print(f"Warning: Could not load GNN model: {e}")

    # AttentiveFP Model
    try:
        afp = AttentiveFPPredictor(in_channels=148, out_channels=12, edge_dim=12)
        checkpoint = torch.load(MODELS_DIR / 'best_attentivefp_model.pt', map_location=device, weights_only=False)
        afp.load_state_dict(checkpoint['model_state_dict'])
        afp.to(device)
        afp.eval()
        models['attentivefp'] = afp
    except Exception as e:
        print(f"Warning: Could not load AttentiveFP model: {e}")


# =============================================================================
# Prediction Functions
# =============================================================================


def smiles_to_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
    return fp_gen.GetFingerprintAsNumPy(mol)


def smiles_to_graph(smiles: str):
    """Convert SMILES to graph representation for GNN."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features_gcn(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edges
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])

    if not edge_index:
        edge_index = [[0, 0]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    batch = torch.zeros(x.size(0), dtype=torch.long)

    return x, edge_index, batch


def smiles_to_graph_afp(smiles: str):
    """Convert SMILES to graph with edge features for AttentiveFP."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None

    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features_afp(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edges with features
    edge_index = []
    edge_features = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = get_bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_features.extend([feat, feat])

    if not edge_index:
        edge_index = [[0, 0]]
        edge_features = [[0] * 12]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    return x, edge_index, edge_attr, batch


def get_risk_level(probability: float) -> str:
    """Classify risk level based on probability."""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MODERATE"
    else:
        return "HIGH"


def predict_mlp(smiles: str) -> Optional[np.ndarray]:
    """Get prediction from MLP model."""
    if 'mlp' not in models:
        return None

    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return None

    with torch.no_grad():
        x = torch.tensor(fp, dtype=torch.float).unsqueeze(0).to(device)
        logits = models['mlp'](x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return probs


def predict_gnn(smiles: str) -> Optional[np.ndarray]:
    """Get prediction from GNN model."""
    if 'gnn' not in models:
        return None

    result = smiles_to_graph(smiles)
    if result[0] is None:
        return None

    x, edge_index, batch = result

    with torch.no_grad():
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        logits = models['gnn'](x, edge_index, batch)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return probs


def predict_attentivefp(smiles: str) -> Optional[np.ndarray]:
    """Get prediction from AttentiveFP model."""
    if 'attentivefp' not in models:
        return None

    result = smiles_to_graph_afp(smiles)
    if result[0] is None:
        return None

    x, edge_index, edge_attr, batch = result

    with torch.no_grad():
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        batch = batch.to(device)
        logits = models['attentivefp'](x, edge_index, edge_attr, batch)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return probs


def predict_ensemble(smiles: str) -> Optional[np.ndarray]:
    """Get ensemble prediction from all models."""
    predictions = []
    weights = []

    mlp_pred = predict_mlp(smiles)
    if mlp_pred is not None:
        predictions.append(mlp_pred)
        weights.append(DEFAULT_ENSEMBLE_WEIGHTS['mlp'])

    gnn_pred = predict_gnn(smiles)
    if gnn_pred is not None:
        predictions.append(gnn_pred)
        weights.append(DEFAULT_ENSEMBLE_WEIGHTS['gcn'])

    afp_pred = predict_attentivefp(smiles)
    if afp_pred is not None:
        predictions.append(afp_pred)
        weights.append(DEFAULT_ENSEMBLE_WEIGHTS['attentivefp'])

    if not predictions:
        return None

    # Weighted average
    weights = np.array(weights) / sum(weights)
    ensemble_pred = sum(w * p for w, p in zip(weights, predictions))

    return ensemble_pred


# =============================================================================
# API Endpoints
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()
    print(f"Loaded models: {list(models.keys())}")
    print(f"Device: {device}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Molecular Toxicity Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "mlp": "mlp" in models,
            "gnn": "gnn" in models,
            "attentivefp": "attentivefp" in models
        }
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_toxicity(request: PredictionRequest):
    """
    Predict toxicity endpoints for a molecule.

    Args:
        request: PredictionRequest with SMILES string and optional model choice

    Returns:
        PredictionResponse with toxicity predictions for all 12 Tox21 endpoints
    """
    smiles = request.smiles
    model_choice = request.model.lower()

    # Validate SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")

    # Get predictions based on model choice
    if model_choice == "mlp":
        probs = predict_mlp(smiles)
        model_used = "mlp"
    elif model_choice == "gnn":
        probs = predict_gnn(smiles)
        model_used = "gnn"
    elif model_choice == "attentivefp":
        probs = predict_attentivefp(smiles)
        model_used = "attentivefp"
    else:  # ensemble
        probs = predict_ensemble(smiles)
        model_used = "ensemble"

    if probs is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_choice}' not available. Check /health endpoint."
        )

    # Build predictions list
    predictions = []
    for i, task in enumerate(TOX21_TASKS):
        prob = float(probs[i])
        predictions.append(ToxicityPrediction(
            endpoint=task,
            description=TASK_DESCRIPTIONS.get(task, ""),
            probability=round(prob, 4),
            risk_level=get_risk_level(prob)
        ))

    # Calculate overall score
    overall_score = float(np.mean(probs))

    return PredictionResponse(
        smiles=smiles,
        valid=True,
        model_used=model_used,
        overall_toxicity_score=round(overall_score, 4),
        overall_risk_level=get_risk_level(overall_score),
        predictions=predictions
    )


@app.get("/tasks", response_model=dict)
async def list_tasks():
    """List all toxicity endpoints with descriptions."""
    return {
        "tasks": [
            {"name": task, "description": TASK_DESCRIPTIONS.get(task, "")}
            for task in TOX21_TASKS
        ]
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
