"""
Test suite for core TBR mathematical functions.

This module tests the fundamental mathematical operations and calculations
that form the core of Time-Based Regression analysis, including:
- Sum of squared deviations calculations
- Model and prediction variance calculations
- Counterfactual prediction generation
- Cumulative standard deviation calculations
- Interval estimation and confidence intervals

These tests ensure mathematical accuracy and numerical stability of the
core TBR statistical computations.
"""


import numpy as np
import pandas as pd
import pytest

from tbr.functional.tbr_functions import (
    calculate_cumulative_standard_deviation,
    calculate_model_variance,
    calculate_prediction_variance,
    calculate_sum_x_squared_deviations,
    compute_interval_estimate_and_ci,
    extract_sum_x_squared_deviations,
    generate_counterfactual_predictions,
    safe_int_conversion,
)


class TestSumSquaredDeviations:
    """Test sum of squared deviations calculations."""

    def test_basic_calculation(self):
        """Test basic sum of squared deviations calculation."""
        x = np.array([1, 2, 3, 4, 5])
        result = calculate_sum_x_squared_deviations(x)
        expected = np.sum((x - np.mean(x)) ** 2)
        assert result == expected
        assert result == 10.0  # Known result for [1,2,3,4,5]

    def test_single_value(self):
        """Test with single value array."""
        x = np.array([42])
        result = calculate_sum_x_squared_deviations(x)
        assert result == 0.0

    def test_identical_values(self):
        """Test with array of identical values."""
        x = np.array([5, 5, 5, 5])
        result = calculate_sum_x_squared_deviations(x)
        assert result == 0.0

    def test_known_values(self):
        """Test with known mathematical values."""
        x = np.array([10, 20, 30])
        result = calculate_sum_x_squared_deviations(x)
        # Mean = 20, deviations = [-10, 0, 10], squared = [100, 0, 100], sum = 200
        assert result == 200.0

    def test_negative_values(self):
        """Test with negative values."""
        x = np.array([-2, -1, 0, 1, 2])
        result = calculate_sum_x_squared_deviations(x)
        expected = np.sum((x - np.mean(x)) ** 2)
        assert result == expected
        assert result == 10.0  # Mean = 0, so sum of squares = 4+1+0+1+4 = 10

    def test_mathematical_property(self):
        """Test mathematical property: always non-negative."""
        x = np.random.normal(0, 1, 100)
        result = calculate_sum_x_squared_deviations(x)
        assert result >= 0

    def test_zero_variance_case(self):
        """Test edge case with zero variance."""
        x = np.array([3.14159, 3.14159, 3.14159])
        result = calculate_sum_x_squared_deviations(x)
        assert result == pytest.approx(0.0, abs=1e-10)


class TestExtractSumSquaredDeviations:
    """Test extraction of sum squared deviations from regression parameters."""

    def test_basic_extraction(self):
        """Test basic extraction using var_beta and sigma."""
        var_beta = 0.01
        sigma = 2.0
        result = extract_sum_x_squared_deviations(var_beta, sigma)
        expected = sigma**2 / var_beta
        assert result == expected
        assert result == 400.0  # 4 / 0.01 = 400

    def test_unit_variance(self):
        """Test with unit variance case."""
        var_beta = 1.0
        sigma = 1.0
        result = extract_sum_x_squared_deviations(var_beta, sigma)
        assert result == 1.0

    def test_inverse_relationship(self):
        """Test inverse relationship with calculate_sum_x_squared_deviations."""
        # Create data where we know the sum of squared deviations
        x = np.array([1, 2, 3, 4, 5])
        actual_sum_sq_dev = calculate_sum_x_squared_deviations(x)

        # Simulate regression parameters that would give this result
        sigma = 2.0
        var_beta = sigma**2 / actual_sum_sq_dev

        # Extract should give us back the original
        extracted = extract_sum_x_squared_deviations(var_beta, sigma)
        assert extracted == pytest.approx(actual_sum_sq_dev, rel=1e-10)


class TestSafeIntConversion:
    """Test safe integer conversion for statistical parameters."""

    def test_valid_integer_conversion(self):
        """Test conversion of exact integer values."""
        assert safe_int_conversion(42.0, "test_param") == 42
        assert safe_int_conversion(0.0, "test_param") == 0
        assert safe_int_conversion(-5.0, "test_param") == -5

    def test_small_decimal_tolerance(self):
        """Test conversion with small floating point errors."""
        assert safe_int_conversion(42.0000001, "test_param") == 42
        assert safe_int_conversion(41.9999999, "test_param") == 42
        assert safe_int_conversion(42.005, "test_param") == 42  # Within 1% tolerance

    def test_invalid_conversion_raises_error(self):
        """Test that non-integer values raise appropriate error."""
        with pytest.raises(ValueError, match="test_param should be an integer"):
            safe_int_conversion(42.5, "test_param")

        with pytest.raises(ValueError, match="degrees_freedom should be an integer"):
            safe_int_conversion(43.2, "degrees_freedom")


class TestModelVariance:
    """Test model variance calculations."""

    def test_basic_model_variance(self):
        """Test basic model variance calculation."""
        x_values = np.array([100, 110, 120])
        x_mean = 110.0
        sigma = 10.0
        n_pretest = 30
        sum_x_squared_deviations = calculate_sum_x_squared_deviations(x_values)

        result = calculate_model_variance(
            x_values, x_mean, sigma, n_pretest, sum_x_squared_deviations
        )

        # Check shape and positivity
        assert result.shape == x_values.shape
        assert np.all(result > 0)

        # Check formula: σ² * (1/n + (x* - x̄)²/Σ(xi - x̄)²)
        expected = sigma**2 * (
            1.0 / n_pretest + (x_values - x_mean) ** 2 / sum_x_squared_deviations
        )
        np.testing.assert_array_almost_equal(result, expected)


class TestPredictionVariance:
    """Test prediction variance calculations."""

    def test_basic_prediction_variance(self):
        """Test basic prediction variance calculation."""
        model_variances = np.array([1.0, 2.0, 3.0])
        sigma = 5.0

        result = calculate_prediction_variance(model_variances, sigma)

        # Check formula: σ² + V[ŷ*]
        expected = sigma**2 + model_variances
        np.testing.assert_array_almost_equal(result, expected)

        # Prediction variance should always be larger than model variance
        assert np.all(result > model_variances)


class TestCounterfactualPredictions:
    """Test counterfactual prediction generation."""

    def test_basic_counterfactual_predictions(self):
        """Test basic counterfactual prediction generation."""
        alpha = 50.0
        beta = 0.95
        sigma = 25.0
        x_mean = 1000.0
        n_pretest = 45

        test_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-02-15", periods=14),
                "control": np.random.normal(1000, 50, 14),
            }
        )

        result = generate_counterfactual_predictions(
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            pretest_x_mean=x_mean,
            n_pretest=n_pretest,
            pretest_sum_x_squared_deviations=2500.0,  # Add missing parameter
            test_period_data=test_data,
            control_col="control",
            time_col="date",
        )

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert "pred" in result.columns
        assert "predsd" in result.columns
        assert len(result) == len(test_data)

        # Check predictions formula: ŷ* = α + β * x*
        expected_pred = alpha + beta * test_data["control"].values
        np.testing.assert_array_almost_equal(result["pred"].values, expected_pred)

        # Check that prediction std devs are positive
        assert np.all(result["predsd"] > 0)


class TestCumulativeStandardDeviation:
    """Test cumulative standard deviation calculations."""

    def test_basic_cumulative_calculation(self):
        """Test basic cumulative standard deviation calculation."""
        test_x_values = np.array([1000, 1020, 1010, 1030])
        sigma = 25.0
        var_alpha = 100.0
        var_beta = 0.001
        cov_alpha_beta = -0.05

        result = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Check shape and positivity
        assert result.shape == test_x_values.shape
        assert np.all(result > 0)

        # Check that values are increasing (cumulative effect)
        assert np.all(np.diff(result) > 0)


class TestIntervalEstimation:
    """Test interval estimation and confidence interval calculations."""

    def test_interval_estimation_basic(self):
        """Test basic interval estimation calculation."""
        # Create mock TBR dataframe
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 0, 1, 1, 1],
                "y": [10, 15, 20, 25, 30, 35],
                "pred": [12, 16, 18, 22, 28, 32],
                "estsd": [1, 1, 1, 2, 2, 2],
            }
        )

        # Create mock summary
        tbr_summary = pd.DataFrame({"sigma": [5.0], "t_dist_df": [10.0]})

        result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, start_day=1, end_day=2, ci_level=0.80
        )

        # Check structure
        assert isinstance(result, dict)
        required_keys = ["estimate", "precision", "lower", "upper"]
        assert all(key in result for key in required_keys)

        # Check that lower < estimate < upper
        assert result["lower"] < result["estimate"] < result["upper"]

        # Check that precision is positive
        assert result["precision"] > 0

    def test_interval_estimation_edge_cases(self):
        """Test interval estimation with edge cases."""
        # Single day interval
        tbr_df = pd.DataFrame({"period": [1], "y": [100], "pred": [95], "estsd": [2]})

        tbr_summary = pd.DataFrame({"sigma": [3.0], "t_dist_df": [20.0]})

        result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, start_day=1, end_day=1, ci_level=0.95
        )

        # Check that single day works
        assert result["estimate"] == 5.0  # 100 - 95
        assert result["precision"] > 0

    def test_multiple_interval_estimations(self):
        """Test multiple interval estimations with different parameters."""
        # Create comprehensive test data
        tbr_df = pd.DataFrame(
            {
                "period": [1] * 10,
                "y": np.random.normal(100, 10, 10),
                "pred": np.random.normal(95, 8, 10),
                "estsd": np.random.uniform(1, 3, 10),
            }
        )

        tbr_summary = pd.DataFrame({"sigma": [4.0], "t_dist_df": [15.0]})

        # Test different intervals
        intervals = [(1, 3, 0.80), (4, 6, 0.90), (7, 10, 0.95)]

        for start, end, level in intervals:
            result = compute_interval_estimate_and_ci(
                tbr_df, tbr_summary, start, end, level
            )

            # All results should be valid
            assert isinstance(result, dict)
            assert result["precision"] > 0
            assert result["lower"] < result["upper"]


class TestMathematicalConsistency:
    """Test mathematical consistency across functions."""

    def test_sum_squared_deviations_consistency(self):
        """Test consistency between direct and extracted calculations."""
        # Generate random data
        x = np.random.normal(50, 10, 20)

        # Calculate directly
        direct_result = calculate_sum_x_squared_deviations(x)

        # Simulate extraction parameters
        sigma = 3.0
        var_beta = sigma**2 / direct_result
        extracted_result = extract_sum_x_squared_deviations(var_beta, sigma)

        # Should be consistent
        assert extracted_result == pytest.approx(direct_result, rel=1e-10)

    def test_variance_relationship(self):
        """Test relationship between model and prediction variance."""
        x_values = np.array([10, 20, 30])
        model_vars = calculate_model_variance(
            x_values,
            pretest_x_mean=20,
            sigma=5,
            n_pretest=25,
            pretest_sum_x_squared_deviations=calculate_sum_x_squared_deviations(
                x_values
            ),
        )

        pred_vars = calculate_prediction_variance(model_vars, sigma=5)

        # Prediction variance should equal model variance + σ²
        expected_diff = np.full_like(model_vars, 25.0)  # σ² = 5² = 25
        actual_diff = pred_vars - model_vars

        np.testing.assert_array_almost_equal(actual_diff, expected_diff)

    def test_mathematical_properties_preservation(self):
        """Test that mathematical properties are preserved."""
        # Test with various data sets
        test_datasets = [
            np.array([1, 2, 3, 4, 5]),
            np.array([-10, -5, 0, 5, 10]),
            np.array([100, 200, 300]),
            np.random.normal(0, 1, 50),
        ]

        for data in test_datasets:
            sum_sq_dev = calculate_sum_x_squared_deviations(data)

            # Should always be non-negative
            assert sum_sq_dev >= 0

            # Should be zero only for constant data
            if len(np.unique(data)) == 1:
                assert sum_sq_dev == pytest.approx(0, abs=1e-10)
            else:
                assert sum_sq_dev > 0
