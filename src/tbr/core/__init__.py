"""TBR Core Modules.

This package provides the core modular components for TBR analysis,
organizing the functionality into clean, focused modules while maintaining
full compatibility with the functional implementation.

The core modules include:
- regression: Linear regression model fitting and variance calculations
- prediction: Counterfactual predictions and uncertainty quantification
- diagnostics: Model diagnostics and assumption testing

This module uses lazy loading (SPEC-1) to optimize memory usage and import times.
Heavy scientific modules (scipy, statsmodels) are only loaded when their
functions are actually accessed, providing significant performance benefits
for users who don't need all functionality.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],  # Don't expose submodules in __all__
    submod_attrs={
        "regression": [
            "fit_regression_model",
            "calculate_model_variance",
            "calculate_prediction_variance",
            "calculate_variances",
            "calculate_sum_squared_deviations",
            "extract_sum_squared_deviations_from_model",
            "convert_to_integer",
        ],
        "prediction": [
            "generate_counterfactual_predictions",
            "calculate_cumulative_standard_deviation",
            "compute_interval_estimate_and_ci",
        ],
        "diagnostics": [
            "calculate_residuals",
            "calculate_standardized_residuals",
            "calculate_studentized_residuals",
            "calculate_goodness_of_fit",
            "calculate_information_criteria",
            "check_normality",
            "check_homoscedasticity",
            "check_independence",
            "create_diagnostic_summary",
            "validate_model_assumptions",
        ],
    },
)
