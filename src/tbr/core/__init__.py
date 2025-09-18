"""TBR Core Modules.

This package provides the core modular components for TBR analysis,
organizing the functionality into clean, focused modules while maintaining
full compatibility with the functional implementation.

The core modules include:
- regression: Linear regression model fitting and variance calculations
"""

from .regression import (
    calculate_sum_squared_deviations,
    calculate_variances,
    convert_to_integer,
    extract_sum_squared_deviations_from_model,
    fit_regression_model,
)

__all__ = [
    "fit_regression_model",
    "calculate_variances",
    "calculate_sum_squared_deviations",
    "extract_sum_squared_deviations_from_model",
    "convert_to_integer",
]
