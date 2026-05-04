#!/bin/bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p data/raw data/processed results

echo ""
echo "Setup complete."
echo "Activate the environment: source .venv/bin/activate"
echo "Then run the pipeline:    python src/main.py"
