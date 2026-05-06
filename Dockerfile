FROM python:3.11-slim

LABEL maintainer="Akhil Tripathi <240216005@hbtu.ac.in>"
LABEL description="Stacked Ensemble for CVD Prediction — Reproducible Environment"
LABEL org.opencontainers.image.source="https://doi.org/10.5281/zenodo.19606370"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY data/ data/
COPY README.md LICENSE CITATION.cff ./

# Create output directories
RUN mkdir -p results figures

# Default: run external validation
CMD ["python", "src/external_validation.py"]
