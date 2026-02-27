"""
conftest.py — pytest configuration for eco-material-predictor.
Adds src/ to sys.path so all modules are importable from tests.
"""
import sys
import os

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
