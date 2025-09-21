"""
Integration tests for the complete regression pipeline.

This module tests the integration of all regression components working
together in realistic TBR analysis scenarios, ensuring the complete
pipeline produces consistent and mathematically valid results.
"""

import numpy as np
import pandas as pd

from tbr.core.regression import (
    calculate_sum_squared_deviations,
    calculate_variances,
    extract_sum_squared_deviations_from_model,
    fit_regression_model,
)
from tbr.functional.tbr_functions import (
    calculate_model_variance as func_calculate_model_variance,
)
from tbr.functional.tbr_functions import (
    calculate_prediction_variance as func_calculate_prediction_variance,
)
from tbr.functional.tbr_functions import (
    calculate_sum_x_squared_deviations as func_calculate_sum_x_squared_deviations,
)
from tbr.functional.tbr_functions import (
    extract_sum_x_squared_deviations as func_extract_sum_x_squared_deviations,
)
from tbr.functional.tbr_functions import (
    fit_tbr_regression_model as func_fit_tbr_regression_model,
)


class TestRegressionPipelineIntegration:
    """Test complete regression pipeline integration."""

    def test_end_to_end_pipeline_validation(self):
        """Test complete end-to-end regression pipeline validation."""
        # Generate realistic TBR analysis scenario
        np.random.seed(42)
        n_samples = 100

        # Create learning data with realistic TBR characteristics
        control_values = np.random.normal(1000, 80, n_samples)
        # Test values with treatment effect + noise
        test_values = 50 + 0.95 * control_values + np.random.normal(0, 25, n_samples)

        learning_data = pd.DataFrame({"control": control_values, "test": test_values})

        # Step 1: Fit regression models
        core_params = fit_regression_model(learning_data, "control", "test")
        func_params = func_fit_tbr_regression_model(learning_data, "control", "test")

        # Step 2: Calculate sum squared deviations
        core_sum_sq_dev = calculate_sum_squared_deviations(control_values)
        func_sum_sq_dev = func_calculate_sum_x_squared_deviations(control_values)

        # Step 3: Calculate variances for prediction
        test_x_values = np.array([950, 1000, 1050, 1100])
        core_model_vars, core_pred_vars = calculate_variances(
            test_x_values,
            core_params["x_mean"],
            core_params["sigma"],
            core_params["n_pretest"],
            core_sum_sq_dev,
        )

        func_model_vars = func_calculate_model_variance(
            test_x_values,
            func_params["x_mean"],
            func_params["sigma"],
            func_params["n_pretest"],
            func_sum_sq_dev,
        )
        func_pred_vars = func_calculate_prediction_variance(
            func_model_vars, func_params["sigma"]
        )

        # Comprehensive validation of entire pipeline
        # 1. Regression parameters should match exactly
        for key in core_params:
            assert (
                abs(core_params[key] - func_params[key]) < 1e-12
            ), f"Pipeline validation failed for parameter {key}"

        # 2. Sum squared deviations should match
        assert abs(core_sum_sq_dev - func_sum_sq_dev) < 1e-15

        # 3. Variance calculations should match
        np.testing.assert_allclose(core_model_vars, func_model_vars, rtol=1e-15)
        np.testing.assert_allclose(core_pred_vars, func_pred_vars, rtol=1e-15)

        # 4. Mathematical relationships should hold
        assert np.all(core_pred_vars > core_model_vars)  # Prediction > model variance
        assert np.all(func_pred_vars > func_model_vars)

        # 5. Statistical properties should be valid
        assert core_params["sigma"] > 0
        assert core_params["var_alpha"] > 0
        assert core_params["var_beta"] > 0
        assert core_params["degrees_freedom"] == n_samples - 2

    def test_mathematical_consistency_validation(self):
        """Test mathematical consistency across all regression functions."""
        # Create test scenario
        np.random.seed(789)
        learning_data = pd.DataFrame(
            {
                "control": np.random.normal(500, 50, 50),
                "test": np.random.normal(520, 55, 50),
            }
        )

        # Get regression parameters from both implementations
        core_params = fit_regression_model(learning_data, "control", "test")
        func_params = func_fit_tbr_regression_model(learning_data, "control", "test")

        # Test mathematical relationship: var_beta * sigma^2 = sum_x_squared_deviations
        core_extracted = extract_sum_squared_deviations_from_model(
            core_params["var_beta"], core_params["sigma"]
        )
        func_extracted = func_extract_sum_x_squared_deviations(
            func_params["var_beta"], func_params["sigma"]
        )

        # Direct calculation
        x_values = learning_data["control"].values
        core_direct = calculate_sum_squared_deviations(x_values)
        func_direct = func_calculate_sum_x_squared_deviations(x_values)

        # All methods should give consistent results
        assert abs(core_extracted - func_extracted) < 1e-15
        assert abs(core_direct - func_direct) < 1e-15
        assert (
            abs(core_extracted - core_direct) < 1e-10
        )  # Allow small numerical differences
        assert abs(func_extracted - func_direct) < 1e-10
