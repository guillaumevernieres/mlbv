#!/usr/bin/env python3
"""
CLI script for training IceNet models.
This provides a simple command-line interface to the icenet package.
"""

import sys
from pathlib import Path

# Add the parent directory to Python path to import icenet package
sys.path.insert(0, str(Path(__file__).parent.parent))

from icenet.training import main  # noqa: E402


if __name__ == "__main__":
    main()
