"""Pytest configuration for LLM-CLP tests.

Note: Running tests requires the full dependency stack:
    pip install pandas scikit-learn transformers torch tqdm

If dependencies are missing, tests will fail to import.
Run: pip install -e ".[dev]" or pip install pandas scikit-learn transformers torch tqdm
"""
import sys
from pathlib import Path

# Add project root to path for src/ imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
