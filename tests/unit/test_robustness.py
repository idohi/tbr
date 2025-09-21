"""
Robustness testing for regression implementations.

This module tests the robustness of regression implementations across
extreme values, edge cases, and various data scenarios to ensure
stable behavior under challenging conditions.
"""

import numpy as np
import pandas as pd

from tbr.core.regression import calculate_sum_squared_deviations, fit_regression_model
from tbr.functional.tbr_functions import (
    calculate_sum_x_squared_deviations as func_calculate_sum_x_squared_deviations,
)
from tbr.functional.tbr_functions import (
    fit_tbr_regression_model as func_fit_tbr_regression_model,
)


class TestRobustnessValidation:
    """Test robustness across extreme values and edge cases."""

    def test_extreme_value_robustness(self):
        """Test robustness with extreme values."""
        extreme_scenarios = [
            # Very small values
            pd.DataFrame(
                {
                    "control": [1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
                    "test": [1.1e-6, 2.1e-6, 3.1e-6, 4.1e-6, 5.1e-6],
                }
            ),
            # Very large values
            pd.DataFrame(
                {
                    "control": [1e6, 2e6, 3e6, 4e6, 5e6],
                    "test": [1.1e6, 2.1e6, 3.1e6, 4.1e6, 5.1e6],
                }
            ),
            # Mixed positive and negative
            pd.DataFrame(
                {"control": [-100, -50, 0, 50, 100], "test": [-95, -45, 5, 55, 105]}
            ),
        ]

        for i, data in enumerate(extreme_scenarios):
            # Both implementations should handle extreme values identically
            core_params = fit_regression_model(data, "control", "test")
            func_params = func_fit_tbr_regression_model(data, "control", "test")

            # Validate parameters match
            for key in core_params:
                relative_error = abs(core_params[key] - func_params[key]) / max(
                    abs(func_params[key]), 1e-10
                )
                assert relative_error < 1e-10, (
                    f"Extreme value scenario {i} failed for {key}: "
                    f"core={core_params[key]}, func={func_params[key]}"
                )

    def test_statistical_edge_cases(self):
        """Test statistical edge cases and boundary conditions."""
        # Perfect correlation case
        perfect_corr_data = pd.DataFrame(
            {
                "control": [100, 200, 300, 400, 500],
                "test": [200, 400, 600, 800, 1000],  # test = 2 * control
            }
        )

        core_params = fit_regression_model(perfect_corr_data, "control", "test")
        func_params = func_fit_tbr_regression_model(
            perfect_corr_data, "control", "test"
        )

        # Should get identical results even for perfect correlation
        for key in core_params:
            assert (
                abs(core_params[key] - func_params[key]) < 1e-12
            ), f"Perfect correlation case failed for {key}"

        # Beta should be approximately 2.0
        assert abs(core_params["beta"] - 2.0) < 1e-10
        assert abs(func_params["beta"] - 2.0) < 1e-10

    def test_numerical_precision_validation(self):
        """Test numerical precision across different data types and ranges."""
        # Test with different floating point precisions
        base_data = np.array([1000.0, 1001.0, 1002.0, 1003.0, 1004.0])

        # Test with float32 precision
        float32_data = base_data.astype(np.float32)
        core_result_32 = calculate_sum_squared_deviations(float32_data)
        func_result_32 = func_calculate_sum_x_squared_deviations(float32_data)

        # Test with float64 precision
        float64_data = base_data.astype(np.float64)
        core_result_64 = calculate_sum_squared_deviations(float64_data)
        func_result_64 = func_calculate_sum_x_squared_deviations(float64_data)

        # Both implementations should handle precision consistently
        assert abs(core_result_32 - func_result_32) < 1e-6  # float32 precision
        assert abs(core_result_64 - func_result_64) < 1e-15  # float64 precision

    def test_data_type_compatibility(self):
        """Test compatibility with various pandas data types."""
        # Test with different pandas dtypes
        data_scenarios = [
            # Standard float64
            pd.DataFrame(
                {
                    "control": pd.Series([100, 200, 300, 400, 500], dtype="float64"),
                    "test": pd.Series([110, 220, 330, 440, 550], dtype="float64"),
                }
            ),
            # Float32
            pd.DataFrame(
                {
                    "control": pd.Series([100, 200, 300, 400, 500], dtype="float32"),
                    "test": pd.Series([110, 220, 330, 440, 550], dtype="float32"),
                }
            ),
            # Integer converted to float
            pd.DataFrame(
                {
                    "control": pd.Series(
                        [100, 200, 300, 400, 500], dtype="int64"
                    ).astype(float),
                    "test": pd.Series([110, 220, 330, 440, 550], dtype="int64").astype(
                        float
                    ),
                }
            ),
        ]

        for i, data in enumerate(data_scenarios):
            core_params = fit_regression_model(data, "control", "test")
            func_params = func_fit_tbr_regression_model(data, "control", "test")

            # Results should be consistent across data types
            for key in core_params:
                relative_error = abs(core_params[key] - func_params[key]) / max(
                    abs(func_params[key]), 1e-10
                )
                assert relative_error < 1e-10, (
                    f"Data type scenario {i} failed for {key}: "
                    f"relative_error={relative_error:.2e}"
                )
