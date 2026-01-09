# Molecular Property Prediction - Docker Container
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Build stage with all dependencies
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# =============================================================================
# Stage 2: Runtime stage
# =============================================================================
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies for RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app.py .
COPY config.yaml .

# Create directories for data and models
RUN mkdir -p data models

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: Run Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# =============================================================================
# Alternative entrypoints (use with docker run --entrypoint)
# =============================================================================
# Training: docker run --entrypoint python molprop scripts/train.py
# Prediction: docker run --entrypoint python molprop scripts/predict.py --smiles "CCO"
# Tests: docker run --entrypoint pytest molprop tests/
