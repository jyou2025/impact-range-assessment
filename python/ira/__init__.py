"""
Impact Range Assessment (IRA)

A model-based sensitivity interpretability measure for regression modelling.
"""

from .irapy import single_ira, repeated_ira

__all__ = ["single_ira", "repeated_ira"]
__version__ = "0.1.0"
