#!/bin/bash

echo "Setting up Credit Card Fraud Detection System..."

# Create project structure
echo "Creating project directories..."
mkdir -p fraud-detection-system/{data/{raw,processed},notebooks,src,config,tests,results}

cd fraud-detection-system

# Initialize git repository
echo "Initializing git repository..."
git init

# Create .gitignore file
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# Environment variables
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data files
data/raw/*.csv
!data/raw/.gitkeep
data/processed/*.csv
!data/processed/.gitkeep

# Model files
*.joblib
*.pkl
*.pickle

# Results
results/*.png
results/*.html
results/*.json
!results/.gitkeep

# OS
.DS_Store
Thumbs.db
EOF

# Create empty .gitkeep files to preserve directory structure
echo "Creating .gitkeep files..."
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch results/.gitkeep

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv fraud_env

# Activation message based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    ACTIVATE_CMD="fraud_env\\Scripts\\activate"
else
    ACTIVATE_CMD="source fraud_env/bin/activate"
fi

echo "Virtual environment created!"
echo "To activate it, run: $ACTIVATE_CMD"

# Instructions for manual steps
echo ""
echo "=== SETUP COMPLETE ==="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: $ACTIVATE_CMD"
echo "2. Install requirements: pip install -r requirements.txt"
echo "3. Download dataset from Kaggle:"
echo "   URL: https://www.kaggle.com/mlg-ulb/creditcardfraud"
echo "   Save as: data/raw/creditcard.csv"
echo "4. Run the pipeline: python src/main.py"
echo "5. Open Jupyter notebooks: jupyter notebook notebooks/"
echo ""
echo "Alternative: Run with sample data (automatically generated):"
echo "python src/main.py"
echo ""
echo "Happy coding! 🚀"