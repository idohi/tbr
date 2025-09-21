"""TBR Core Modules.

This package provides the core modular components for TBR analysis,
organizing the functionality into clean, focused modules while maintaining
full compatibility with the functional implementation.

The core modules include:
- regression: Linear regression model fitting and variance calculations
- prediction: Counterfactual predictions and uncertainty quantification
"""

from .prediction import (
    calculate_cumulative_standard_deviation,
    compute_interval_estimate_and_ci,
    generate_counterfactual_predictions,
)
from .regression import (
    calculate_model_variance,
    calculate_prediction_variance,
    calculate_sum_squared_deviations,
    calculate_variances,
    convert_to_integer,
    extract_sum_squared_deviations_from_model,
    fit_regression_model,
)

__all__ = [
    # Regression functions
    "fit_regression_model",
    "calculate_model_variance",
    "calculate_prediction_variance",
    "calculate_variances",
    "calculate_sum_squared_deviations",
    "extract_sum_squared_deviations_from_model",
    "convert_to_integer",
    # Prediction functions
    "generate_counterfactual_predictions",
    "calculate_cumulative_standard_deviation",
    "compute_interval_estimate_and_ci",
]
