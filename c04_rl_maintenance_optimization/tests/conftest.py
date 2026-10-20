"""
Pytest configuration and environment fixtures for Component 04.
Ensures repository root is registered on sys.path across all platforms.
"""

import sys
from pathlib import Path

# Resolve repository root: c04_rl_maintenance_optimization/tests/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
