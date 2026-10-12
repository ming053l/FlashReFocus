import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # tests/regression_cases.py
warnings.filterwarnings("ignore", category=FutureWarning)
