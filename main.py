#!/usr/bin/env python3
"""
BoardBench: LLM & Agent Board Game Benchmarking Framework

This script provides a command-line interface to the BoardBench framework,
allowing users to run matches between different agent types.
"""

import os
import sys

# Ensure the package is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the CLI
from boardbench.cli import app

if __name__ == "__main__":
    app()
