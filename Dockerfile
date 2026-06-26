FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent/       ./agent/
COPY synthetic/   ./synthetic/
COPY data/        ./data/
COPY app.py       .
COPY run_agent.py .
COPY run_checks.py .
COPY run_eval.py  .

# Generate initial data (already in data/ but ensure consistency)
RUN python -m synthetic.generate --output-dir data

# Populate the decision ledger on startup, then serve
# PORT is injected by Cloud Run (default 8080)
ENV DB_PATH=/app/controls_agent.db
ENV DATA_PATH=/app/data/transactions.csv

EXPOSE 8080

# Run agent to pre-populate the ledger, then start Streamlit
CMD python run_agent.py && \
    streamlit run app.py \
      --server.port=${PORT:-8080} \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --browser.gatherUsageStats=false
