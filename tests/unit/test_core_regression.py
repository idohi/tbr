"""
Tests for Core Regression Module.

This module tests the core regression interface that wraps the functional
TBR regression implementation, ensuring clean modular access to regression
functionality while maintaining backward compatibility.

The tests verify that the core regression module provides the same
functionality as the functional implementation with improved organization.
"""

import numpy as np
import pandas as pd
import pytest

from tbr.core.regression import (
    calculate_model_variance,
    calculate_prediction_variance,
    calculate_sum_squared_deviations,
    calculate_variances,
    convert_to_integer,
    extract_sum_squared_deviations_from_model,
    fit_regression_model,
)


class TestFitRegressionModel:
    """Test the fit_regression_model function."""

    def test_basic_regression_fitting(self):
        """Test basic regression model fitting with valid data."""
        # Create test data with known relationship
        np.random.seed(42)
        learning_data = pd.DataFrame(
            {
                "control": np.random.normal(1000, 50, 30),
                "test": np.random.normal(1020, 55, 30),
            }
        )

        # Fit regression model
        params = fit_regression_model(learning_data, "control", "test")

        # Verify all required parameters are present
        required_keys = [
            "alpha",
            "beta",
            "sigma",
            "var_alpha",
            "var_beta",
            "cov_alpha_beta",
            "degrees_freedom",
            "n_pretest",
            "x_mean",
        ]
        for key in required_keys:
            assert key in params, f"Missing parameter: {key}"

        # Verify parameter types and basic constraints
        assert isinstance(params["alpha"], float)
        assert isinstance(params["beta"], float)
        assert isinstance(params["sigma"], float)
        assert params["sigma"] > 0, "Sigma should be positive"
        assert params["var_alpha"] > 0, "Variance of alpha should be positive"
        assert params["var_beta"] > 0, "Variance of beta should be positive"
        assert isinstance(params["degrees_freedom"], int)
        assert params["degrees_freedom"] > 0, "Degrees of freedom should be positive"
        assert isinstance(params["n_pretest"], int)
        assert params["n_pretest"] == 30, "Should match input data length"

    def test_perfect_correlation(self):
        """Test regression with perfect correlation."""
        # Create data with perfect linear relationship
        control_vals = np.array([100, 200, 300, 400, 500])
        test_vals = 50 + 1.5 * control_vals  # Perfect linear relationship

        learning_data = pd.DataFrame({"control": control_vals, "test": test_vals})

        params = fit_regression_model(learning_data, "control", "test")

        # With perfect correlation, should get exact parameters
        assert abs(params["alpha"] - 50.0) < 1e-10, "Alpha should be exactly 50"
        assert abs(params["beta"] - 1.5) < 1e-10, "Beta should be exactly 1.5"
        assert params["sigma"] < 1e-10, "Sigma should be near zero for perfect fit"

    def test_statistical_properties(self):
        """Test that regression parameters have correct statistical properties."""
        # Create larger dataset for better statistical properties
        np.random.seed(123)
        n_points = 100
        control_vals = np.random.normal(1000, 100, n_points)
        test_vals = 200 + 0.8 * control_vals + np.random.normal(0, 50, n_points)

        learning_data = pd.DataFrame({"control": control_vals, "test": test_vals})

        params = fit_regression_model(learning_data, "control", "test")

        # Beta should be close to true value (0.8)
        assert 0.6 < params["beta"] < 1.0, f"Beta {params['beta']} should be near 0.8"

        # Alpha should be close to true value (200)
        assert (
            100 < params["alpha"] < 300
        ), f"Alpha {params['alpha']} should be near 200"

        # Degrees of freedom should be n - 2
        assert params["degrees_freedom"] == n_points - 2

        # X mean should match actual mean
        expected_x_mean = np.mean(control_vals)
        assert abs(params["x_mean"] - expected_x_mean) < 1e-10


class TestCalculateModelVariance:
    """Test the calculate_model_variance function."""

    def test_basic_model_variance_calculation(self):
        """Test basic model variance calculation."""
        x_values = np.array([900, 1000, 1100])
        x_mean = 1000.0
        sigma = 25.0
        n_pretest = 30
        sum_x_squared_deviations = 15000.0

        model_vars = calculate_model_variance(
            x_values, x_mean, sigma, n_pretest, sum_x_squared_deviations
        )

        # Verify output shape
        assert len(model_vars) == len(x_values)

        # Verify all variances are positive
        assert np.all(model_vars > 0), "All model variances should be positive"

        # Verify minimum variance at x_mean
        center_idx = 1  # x_values[1] = x_mean
        assert (
            model_vars[center_idx] <= model_vars[0]
        ), "Variance should be minimum at x_mean"
        assert (
            model_vars[center_idx] <= model_vars[2]
        ), "Variance should be minimum at x_mean"

    def test_model_variance_symmetry(self):
        """Test that model variances are symmetric around x_mean."""
        x_mean = 1000.0
        deviation = 100.0
        x_values = np.array([x_mean - deviation, x_mean, x_mean + deviation])

        model_vars = calculate_model_variance(
            x_values,
            pretest_x_mean=x_mean,
            sigma=20.0,
            n_pretest=50,
            pretest_sum_x_squared_deviations=10000.0,
        )

        # Variances should be symmetric around x_mean
        assert (
            abs(model_vars[0] - model_vars[2]) < 1e-10
        ), "Model variances should be symmetric"

    def test_model_variance_formula_validation(self):
        """Test that model variance formula is implemented correctly."""
        # Simple test case with known values
        x_values = np.array([10.0])  # Single point
        x_mean = 10.0  # Same as x_values
        sigma = 2.0
        n_pretest = 20
        sum_x_squared_deviations = 100.0

        model_vars = calculate_model_variance(
            x_values, x_mean, sigma, n_pretest, sum_x_squared_deviations
        )

        # At x_mean, deviation squared = 0, so formula becomes: σ² * (1/n + 0)
        expected = sigma**2 * (1.0 / n_pretest)  # 4 * (1/20) = 0.2
        assert (
            abs(model_vars[0] - expected) < 1e-10
        ), f"Expected {expected}, got {model_vars[0]}"


class TestCalculatePredictionVariance:
    """Test the calculate_prediction_variance function."""

    def test_basic_prediction_variance_calculation(self):
        """Test basic prediction variance calculation."""
        model_variances = np.array([10.0, 15.0, 20.0])
        sigma = 25.0

        pred_vars = calculate_prediction_variance(model_variances, sigma)

        # Verify output shape
        assert len(pred_vars) == len(model_variances)

        # Verify prediction variance = model variance + sigma²
        expected = model_variances + sigma**2
        np.testing.assert_array_almost_equal(pred_vars, expected, decimal=10)

        # Verify prediction variance > model variance
        assert np.all(
            pred_vars > model_variances
        ), "Prediction variance should exceed model variance"

    def test_prediction_variance_formula(self):
        """Test that prediction variance formula is implemented correctly."""
        # Test with single value
        model_vars = np.array([100.0])
        sigma = 10.0

        pred_vars = calculate_prediction_variance(model_vars, sigma)

        # Formula: V[y*] = σ² + V[ŷ*]
        expected = sigma**2 + model_vars[0]  # 100 + 100 = 200
        assert (
            abs(pred_vars[0] - expected) < 1e-10
        ), f"Expected {expected}, got {pred_vars[0]}"

    def test_prediction_variance_with_zero_model_variance(self):
        """Test prediction variance when model variance is zero."""
        model_vars = np.array([0.0, 0.0])
        sigma = 5.0

        pred_vars = calculate_prediction_variance(model_vars, sigma)

        # Should equal sigma² when model variance is zero
        expected = np.full_like(model_vars, sigma**2)
        np.testing.assert_array_almost_equal(pred_vars, expected, decimal=10)


class TestCalculateVariances:
    """Test the calculate_variances function."""

    def test_variance_calculation(self):
        """Test variance calculation for multiple x values."""
        # Test parameters
        x_values = np.array([900, 1000, 1100])
        x_mean = 1000.0
        sigma = 25.0
        n_pretest = 30
        sum_x_squared_deviations = 15000.0

        model_vars, pred_vars = calculate_variances(
            x_values, x_mean, sigma, n_pretest, sum_x_squared_deviations
        )

        # Verify output shapes
        assert len(model_vars) == len(x_values)
        assert len(pred_vars) == len(x_values)

        # Verify prediction variance > model variance (includes residual noise)
        assert np.all(
            pred_vars > model_vars
        ), "Prediction variance should exceed model variance"

        # Verify minimum variance at x_mean
        center_idx = 1  # x_values[1] = x_mean
        assert (
            model_vars[center_idx] <= model_vars[0]
        ), "Variance should be minimum at x_mean"
        assert (
            model_vars[center_idx] <= model_vars[2]
        ), "Variance should be minimum at x_mean"

    def test_variance_symmetry(self):
        """Test that variances are symmetric around x_mean."""
        x_mean = 1000.0
        deviation = 100.0
        x_values = np.array([x_mean - deviation, x_mean, x_mean + deviation])

        model_vars, pred_vars = calculate_variances(
            x_values,
            pretest_x_mean=x_mean,
            sigma=20.0,
            n_pretest=50,
            pretest_sum_x_squared_deviations=10000.0,
        )

        # Variances should be symmetric around x_mean
        assert (
            abs(model_vars[0] - model_vars[2]) < 1e-10
        ), "Model variances should be symmetric"
        assert (
            abs(pred_vars[0] - pred_vars[2]) < 1e-10
        ), "Prediction variances should be symmetric"

    def test_combined_function_uses_utilities(self):
        """Test that calculate_variances uses our new utility functions."""
        x_values = np.array([1000, 1010, 1020])
        x_mean = 1005.0
        sigma = 25.0
        n_pretest = 30
        sum_x_squared_deviations = 15000.0

        # Combined function result
        model_vars_combined, pred_vars_combined = calculate_variances(
            x_values, x_mean, sigma, n_pretest, sum_x_squared_deviations
        )

        # Individual utility function results
        model_vars_utility = calculate_model_variance(
            x_values, x_mean, sigma, n_pretest, sum_x_squared_deviations
        )
        pred_vars_utility = calculate_prediction_variance(model_vars_utility, sigma)

        # Should get identical results
        np.testing.assert_array_almost_equal(
            model_vars_combined, model_vars_utility, decimal=12
        )
        np.testing.assert_array_almost_equal(
            pred_vars_combined, pred_vars_utility, decimal=12
        )


class TestCalculateSumSquaredDeviations:
    """Test the calculate_sum_squared_deviations function."""

    def test_known_values(self):
        """Test with known mathematical values."""
        # Simple case: [1, 2, 3, 4, 5], mean = 3
        # Deviations: [-2, -1, 0, 1, 2]
        # Squared deviations: [4, 1, 0, 1, 4] = 10
        x = np.array([1, 2, 3, 4, 5])
        result = calculate_sum_squared_deviations(x)
        assert abs(result - 10.0) < 1e-10, f"Expected 10.0, got {result}"

    def test_single_value(self):
        """Test with single value (should be zero)."""
        x = np.array([42.0])
        result = calculate_sum_squared_deviations(x)
        assert abs(result - 0.0) < 1e-10, "Single value should have zero deviation"

    def test_constant_values(self):
        """Test with constant values (should be zero)."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        result = calculate_sum_squared_deviations(x)
        assert abs(result - 0.0) < 1e-10, "Constant values should have zero deviation"

    def test_mathematical_property(self):
        """Test mathematical property: sum of squared deviations."""
        np.random.seed(456)
        x = np.random.normal(100, 20, 50)

        # Direct calculation
        result = calculate_sum_squared_deviations(x)

        # Manual calculation for verification
        x_mean = np.mean(x)
        expected = np.sum((x - x_mean) ** 2)

        assert abs(result - expected) < 1e-10, "Should match manual calculation"


class TestExtractSumSquaredDeviationsFromModel:
    """Test the extract_sum_squared_deviations_from_model function."""

    def test_basic_extraction(self):
        """Test basic parameter extraction."""
        var_beta = 0.001
        sigma = 25.0

        result = extract_sum_squared_deviations_from_model(var_beta, sigma)
        expected = sigma**2 / var_beta  # 625 / 0.001 = 625000

        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_consistency_with_direct_calculation(self):
        """Test consistency between direct calculation and model extraction."""
        # Create test data and fit model
        np.random.seed(789)
        x_values = np.random.normal(1000, 100, 40)
        learning_data = pd.DataFrame(
            {
                "control": x_values,
                "test": 50 + 0.9 * x_values + np.random.normal(0, 20, 40),
            }
        )

        # Get model parameters
        params = fit_regression_model(learning_data, "control", "test")

        # Direct calculation
        direct_result = calculate_sum_squared_deviations(x_values)

        # Model extraction
        extracted_result = extract_sum_squared_deviations_from_model(
            params["var_beta"], params["sigma"]
        )

        # Should be approximately equal (within numerical precision)
        relative_error = abs(direct_result - extracted_result) / direct_result
        assert (
            relative_error < 0.01
        ), f"Results should be close: {direct_result} vs {extracted_result}"


class TestConvertToInteger:
    """Test the convert_to_integer function."""

    def test_exact_integer(self):
        """Test conversion of exact integer values."""
        assert convert_to_integer(42.0, "test_param") == 42
        assert convert_to_integer(0.0, "test_param") == 0
        assert convert_to_integer(-5.0, "test_param") == -5

    def test_near_integer(self):
        """Test conversion of near-integer values."""
        assert convert_to_integer(42.0000001, "test_param") == 42
        assert convert_to_integer(41.9999999, "test_param") == 42
        assert convert_to_integer(42.005, "test_param") == 42  # Within 1% tolerance

    def test_invalid_conversion(self):
        """Test that non-integer values raise ValueError."""
        with pytest.raises(ValueError, match="should be an integer"):
            convert_to_integer(42.5, "test_param")

        with pytest.raises(ValueError, match="should be an integer"):
            convert_to_integer(42.02, "test_param")  # Beyond 1% tolerance

    def test_error_message_includes_param_name(self):
        """Test that error message includes parameter name."""
        with pytest.raises(ValueError, match="degrees_freedom"):
            convert_to_integer(43.5, "degrees_freedom")


class TestCoreRegressionIntegration:
    """Integration tests for core regression module."""

    def test_end_to_end_workflow(self):
        """Test complete workflow using core regression module."""
        # Create realistic test data
        np.random.seed(101112)
        n_points = 50
        control_vals = np.random.normal(1000, 80, n_points)
        test_vals = 100 + 0.95 * control_vals + np.random.normal(0, 30, n_points)

        learning_data = pd.DataFrame({"control": control_vals, "test": test_vals})

        # Step 1: Fit regression model
        params = fit_regression_model(learning_data, "control", "test")

        # Step 2: Calculate sum of squared deviations
        sum_sq_dev_direct = calculate_sum_squared_deviations(control_vals)
        sum_sq_dev_extracted = extract_sum_squared_deviations_from_model(
            params["var_beta"], params["sigma"]
        )

        # Step 3: Calculate variances for prediction
        test_x_values = np.array([950, 1000, 1050])
        model_vars, pred_vars = calculate_variances(
            test_x_values,
            params["x_mean"],
            params["sigma"],
            params["n_pretest"],
            sum_sq_dev_direct,
        )

        # Step 4: Convert degrees of freedom safely
        dof_int = convert_to_integer(params["degrees_freedom"], "degrees_freedom")

        # Verify workflow results
        assert isinstance(dof_int, int)
        assert dof_int == n_points - 2
        assert len(model_vars) == len(test_x_values)
        assert len(pred_vars) == len(test_x_values)
        assert np.all(pred_vars > model_vars)

        # Verify consistency between direct and extracted calculations
        relative_error = (
            abs(sum_sq_dev_direct - sum_sq_dev_extracted) / sum_sq_dev_direct
        )
        assert (
            relative_error < 0.05
        ), "Direct and extracted calculations should be consistent"

    def test_backward_compatibility(self):
        """Test that core module provides same results as functional module."""
        from tbr.functional.tbr_functions import (
            calculate_sum_x_squared_deviations,
            fit_tbr_regression_model,
            safe_int_conversion,
        )

        # Create test data
        np.random.seed(131415)
        learning_data = pd.DataFrame(
            {
                "control": np.random.normal(500, 40, 25),
                "test": np.random.normal(520, 45, 25),
            }
        )

        # Compare core module vs functional module
        core_params = fit_regression_model(learning_data, "control", "test")
        functional_params = fit_tbr_regression_model(learning_data, "control", "test")

        # Should get identical results
        for key in core_params:
            assert (
                abs(core_params[key] - functional_params[key]) < 1e-12
            ), f"Parameter {key} differs: {core_params[key]} vs {functional_params[key]}"

        # Test sum squared deviations
        x_vals = learning_data["control"].values
        core_result = calculate_sum_squared_deviations(x_vals)
        functional_result = calculate_sum_x_squared_deviations(x_vals)
        assert (
            abs(core_result - functional_result) < 1e-12
        ), "Sum squared deviations should match"

        # Test integer conversion
        test_value = 42.0000001
        core_int = convert_to_integer(test_value, "test")
        functional_int = safe_int_conversion(test_value, "test")
        assert core_int == functional_int, "Integer conversion should match"
