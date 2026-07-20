"""
Cross-implementation validation tests.

This module validates that the core regression module produces identical
mathematical results to the functional implementation, ensuring backward
compatibility and mathematical accuracy.
"""

import time
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

# Core module imports
from tbr.core.regression import (
    calculate_model_variance,
    calculate_prediction_variance,
    calculate_sum_squared_deviations,
    convert_to_integer,
    extract_sum_squared_deviations_from_model,
    fit_regression_model,
)

# Functional module imports
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
from tbr.functional.tbr_functions import safe_int_conversion as func_safe_int_conversion


class TestCrossImplementationValidation:
    """Validate mathematical equivalence between core and functional implementations."""

    @pytest.mark.parametrize(
        "n_samples,mean_control,std_control,mean_test,std_test,seed",
        [
            (20, 1000, 50, 1020, 55, 42),  # Small dataset
            (50, 500, 100, 520, 110, 123),  # Medium dataset
            (100, 2000, 200, 2100, 220, 456),  # Large dataset
            (30, 100, 10, 105, 12, 789),  # Small values
            (40, 10000, 1000, 10500, 1100, 101112),  # Large values
        ],
    )
    def test_regression_fitting_cross_validation(
        self, n_samples, mean_control, std_control, mean_test, std_test, seed
    ):
        """Test regression fitting across various data scenarios."""
        np.random.seed(seed)

        # Generate test data
        learning_data = pd.DataFrame(
            {
                "control": np.random.normal(mean_control, std_control, n_samples),
                "test": np.random.normal(mean_test, std_test, n_samples),
            }
        )

        # Fit models with both implementations
        core_params = fit_regression_model(learning_data, "control", "test")
        func_params = func_fit_tbr_regression_model(learning_data, "control", "test")

        # Validate all parameters match exactly
        for key in core_params:
            assert (
                key in func_params
            ), f"Missing parameter {key} in functional implementation"

            # Use high precision comparison for mathematical accuracy
            relative_error = abs(core_params[key] - func_params[key]) / max(
                abs(func_params[key]), 1e-10
            )
            assert relative_error < 1e-12, (
                f"Parameter {key} differs: core={core_params[key]:.15f}, "
                f"func={func_params[key]:.15f}, rel_error={relative_error:.2e}"
            )

    @pytest.mark.parametrize(
        "array_size,distribution,seed",
        [
            (10, "normal", 42),
            (50, "normal", 123),
            (100, "normal", 456),
            (25, "uniform", 789),
            (75, "exponential", 101112),
        ],
    )
    def test_sum_squared_deviations_cross_validation(
        self, array_size, distribution, seed
    ):
        """Test sum of squared deviations calculation across various scenarios."""
        np.random.seed(seed)

        # Generate test arrays with different distributions
        if distribution == "normal":
            x = np.random.normal(1000, 100, array_size)
        elif distribution == "uniform":
            x = np.random.uniform(500, 1500, array_size)
        elif distribution == "exponential":
            x = np.random.exponential(100, array_size) + 900

        # Calculate with both implementations
        core_result = calculate_sum_squared_deviations(x)
        func_result = func_calculate_sum_x_squared_deviations(x)

        # Validate exact match
        relative_error = abs(core_result - func_result) / max(abs(func_result), 1e-10)
        assert relative_error < 1e-15, (
            f"Sum squared deviations differ: core={core_result:.15f}, "
            f"func={func_result:.15f}, rel_error={relative_error:.2e}"
        )

    def test_variance_calculations_cross_validation(self):
        """Test variance calculations with comprehensive parameter combinations."""
        test_scenarios = [
            # (x_values, x_mean, sigma, n_pretest, sum_x_squared_deviations)
            ([950, 1000, 1050], 1000.0, 25.0, 30, 2500.0),
            ([100, 200, 300], 200.0, 15.0, 50, 20000.0),
            ([999, 1000, 1001], 1000.0, 5.0, 100, 2.0),  # Small deviations
        ]

        for x_values, x_mean, sigma, n_pretest, sum_x_squared_dev in test_scenarios:
            x_array = np.array(x_values)

            # Test model variance
            core_model_var = calculate_model_variance(
                x_array, x_mean, sigma, n_pretest, sum_x_squared_dev
            )
            func_model_var = func_calculate_model_variance(
                x_array, x_mean, sigma, n_pretest, sum_x_squared_dev
            )

            np.testing.assert_allclose(
                core_model_var,
                func_model_var,
                rtol=1e-15,
                err_msg=f"Model variance mismatch for scenario: {x_values}",
            )

            # Test prediction variance
            core_pred_var = calculate_prediction_variance(core_model_var, sigma)
            func_pred_var = func_calculate_prediction_variance(func_model_var, sigma)

            np.testing.assert_allclose(
                core_pred_var,
                func_pred_var,
                rtol=1e-15,
                err_msg=f"Prediction variance mismatch for scenario: {x_values}",
            )

    def test_integer_conversion_cross_validation(self):
        """Test integer conversion across various edge cases."""
        test_cases = [
            (42.0, "degrees_freedom"),
            (100.0000001, "n_pretest"),
            (25.9999999, "sample_size"),
            (1.0, "min_value"),
            (999.0000000001, "large_value"),
        ]

        for value, param_name in test_cases:
            core_result = convert_to_integer(value, param_name)
            func_result = func_safe_int_conversion(value, param_name)

            assert core_result == func_result, (
                f"Integer conversion differs for {value}: "
                f"core={core_result}, func={func_result}"
            )

    def test_extract_sum_squared_deviations_cross_validation(self):
        """Test extraction of sum squared deviations from model parameters."""
        test_scenarios = [
            (0.001, 25.0),  # Typical variance and sigma
            (0.0001, 10.0),  # Small variance
            (0.01, 50.0),  # Large variance
            (0.000001, 5.0),  # Very small variance
        ]

        for var_beta, sigma in test_scenarios:
            core_result = extract_sum_squared_deviations_from_model(var_beta, sigma)
            func_result = func_extract_sum_x_squared_deviations(var_beta, sigma)

            relative_error = abs(core_result - func_result) / max(
                abs(func_result), 1e-10
            )
            assert relative_error < 1e-15, (
                f"Extracted sum squared deviations differ: "
                f"core={core_result:.15f}, func={func_result:.15f}, "
                f"rel_error={relative_error:.2e}"
            )

    def test_sum_squared_deviations_numerical_stability(self):
        """Test numerical stability with challenging data.

        Validates that both implementations maintain accuracy with:
        - Very small values (near machine epsilon)
        - Very large values
        - Mixed ranges spanning many orders of magnitude
        - Near-zero values with small perturbations
        """
        # Test scenarios with different numerical challenges
        test_scenarios = [
            ("small_values", np.random.uniform(1e-6, 1e-5, 1000)),
            ("large_values", np.random.uniform(1e6, 1e7, 1000)),
            (
                "mixed_range",
                np.concatenate(
                    [
                        np.random.uniform(1e-6, 1e-5, 500),
                        np.random.uniform(1e6, 1e7, 500),
                    ]
                ),
            ),
            ("near_zero", np.random.uniform(-1e-10, 1e-10, 1000) + 1000),
        ]

        for scenario_name, test_data in test_scenarios:
            # Calculate with both implementations
            core_result = calculate_sum_squared_deviations(test_data)
            func_result = func_calculate_sum_x_squared_deviations(test_data)

            # Validate numerical accuracy
            relative_error = abs(core_result - func_result) / max(
                abs(func_result), 1e-15
            )
            assert relative_error < 1e-12, (
                f"Numerical accuracy issue in {scenario_name}: "
                f"core={core_result:.6e}, func={func_result:.6e}, "
                f"relative_error={relative_error:.2e}"
            )


class TestPerformanceParityValidation:
    """Validate performance parity between core and functional implementations."""

    def _benchmark_function(self, func, *args, **kwargs) -> Tuple[float, any]:
        """Benchmark a function and return execution time and result."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result

    @pytest.mark.performance
    def test_regression_fitting_performance_parity(self):
        """Test that core regression fitting has comparable performance to functional."""
        # Generate realistic dataset
        np.random.seed(42)
        n_samples = 1000
        learning_data = pd.DataFrame(
            {
                "control": np.random.normal(1000, 100, n_samples),
                "test": np.random.normal(1050, 110, n_samples),
            }
        )

        # Benchmark both implementations
        core_time, core_result = self._benchmark_function(
            fit_regression_model, learning_data, "control", "test"
        )
        func_time, func_result = self._benchmark_function(
            func_fit_tbr_regression_model, learning_data, "control", "test"
        )

        # Performance should be comparable, but ratio checks are noisy for
        # very fast operations where fixed wrapper overhead dominates timing.
        performance_ratio = core_time / func_time
        min_time = min(core_time, func_time)

        if min_time < 0.01:
            assert core_time < 0.05 and func_time < 0.05, (
                f"Regression fitting fast-path exceeded absolute runtime budget: "
                f"core_time={core_time:.6f}s, func_time={func_time:.6f}s"
            )
        else:
            assert 0.5 <= performance_ratio <= 2.0, (
                f"Performance regression detected: core_time={core_time:.4f}s, "
                f"func_time={func_time:.4f}s, ratio={performance_ratio:.2f}"
            )

        # Results should be identical
        for key in core_result:
            assert abs(core_result[key] - func_result[key]) < 1e-12

    @pytest.mark.performance
    def test_sum_squared_deviations_performance_parity(self):
        """Test sum squared deviations performance parity."""
        # Generate large array for performance testing
        np.random.seed(123)
        large_array = np.random.normal(1000, 100, 10000)

        # Benchmark both implementations
        core_time, core_result = self._benchmark_function(
            calculate_sum_squared_deviations, large_array
        )
        func_time, func_result = self._benchmark_function(
            func_calculate_sum_x_squared_deviations, large_array
        )

        # Performance should be comparable, but ratio checks are noisy for
        # sub-millisecond operations where scheduler jitter dominates timing.
        performance_ratio = core_time / func_time if func_time > 0 else 1.0
        min_time = min(core_time, func_time)

        if min_time < 1e-3:
            assert core_time < 0.01 and func_time < 0.01, (
                f"Sub-millisecond operation exceeded absolute runtime budget: "
                f"core_time={core_time:.6f}s, func_time={func_time:.6f}s"
            )
        else:
            assert 0.2 <= performance_ratio <= 3.0, (
                f"Performance regression in sum squared deviations: "
                f"core_time={core_time:.6f}s, func_time={func_time:.6f}s, "
                f"ratio={performance_ratio:.2f}"
            )

        # Results should be identical
        assert abs(core_result - func_result) < 1e-15

    @pytest.mark.performance
    def test_variance_calculations_performance_parity(self):
        """Test variance calculations performance parity."""
        # Setup test data
        np.random.seed(456)
        x_values = np.random.normal(1000, 100, 1000)
        x_mean = 1000.0
        sigma = 25.0
        n_pretest = 500
        sum_x_squared_dev = np.sum((x_values - x_mean) ** 2)

        # Benchmark model variance calculations
        core_time, core_result = self._benchmark_function(
            calculate_model_variance,
            x_values,
            x_mean,
            sigma,
            n_pretest,
            sum_x_squared_dev,
        )
        func_time, func_result = self._benchmark_function(
            func_calculate_model_variance,
            x_values,
            x_mean,
            sigma,
            n_pretest,
            sum_x_squared_dev,
        )

        # Performance should be comparable, but ratio checks are noisy for
        # sub-millisecond operations where scheduler jitter dominates timing.
        performance_ratio = core_time / func_time
        min_time = min(core_time, func_time)

        if min_time < 1e-3:
            assert core_time < 0.01 and func_time < 0.01, (
                f"Model variance sub-millisecond operation exceeded absolute "
                f"runtime budget: core_time={core_time:.6f}s, "
                f"func_time={func_time:.6f}s"
            )
        else:
            assert 0.2 <= performance_ratio <= 3.0, (
                f"Performance regression in model variance: "
                f"core_time={core_time:.4f}s, func_time={func_time:.4f}s, "
                f"ratio={performance_ratio:.2f}"
            )

        # Results should be identical
        np.testing.assert_allclose(core_result, func_result, rtol=1e-15)
