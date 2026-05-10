"""
Alternative interface to collect_dataset.py for backward compatibility.
This script runs the same image collection process from params.yaml.

Use: python src/download_data.py
Or use DVC: dvc repro
"""

from __future__ import annotations

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the main collection function
from collect_dataset import main


if __name__ == "__main__":
    # Ensure we're in the project root
    if not Path("params.yaml").exists():
        raise FileNotFoundError(
            "params.yaml not found. Please run from project root directory."
        )
    
    print("Collecting dataset using params.yaml configuration...")
    print("16 Pakistani political figures configured in params.yaml")
    main()