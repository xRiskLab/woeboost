"""
__init__.py.

This module is the entry point for the package. It imports the main classes and functions
from the package and sets the version number.
"""

import sys
from pathlib import Path

# Add the src directory to the system path
src_dir = Path(__file__).resolve().parent.parent / "woeboost"
sys.path.append(str(src_dir))
