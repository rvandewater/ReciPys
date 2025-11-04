"""
ReciPies: A modular preprocessing package for Pandas and Polars DataFrames
"""

from .recipe import Recipe
from .ingredients import Ingredients
from .step import Step
from .selector import (
    Selector,
    all_predictors,
    all_numeric_predictors,
    select_groups,
    select_sequence,
)
from .constants import Backend

__all__ = [
    "Recipe",
    "Ingredients",
    "Step",
    "Selector",
    "all_predictors",
    "all_numeric_predictors",
    "select_groups",
    "select_sequence",
    "Backend",
]

__version__ = "1.0.0"
