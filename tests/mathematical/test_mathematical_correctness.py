"""
Mathematical Validation Tests for TBR Implementation.

This module provides comprehensive mathematical validation tests that verify
our functional implementation against mathematical definitions and properties
to ensure mathematical accuracy, numerical consistency, and statistical correctness.

The tests validate:
1. Core mathematical functions against mathematical definitions
2. Regression model parameters and statistical properties
3. End-to-end TBR analysis pipeline accuracy
4. Statistical inference and credible interval calculations

These tests serve as the mathematical foundation ensuring our implementation
maintains mathematical rigor and statistical correctness.
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
    create_tbr_summary,
    extract_sum_x_squared_deviations,
    fit_tbr_regression_model,
    perform_tbr_analysis,
    safe_int_conversion,
)


class TestMathematicalFunctionValidation:
    """Test core mathematical functions against reference implementation."""

    def test_sum_squared_deviations_mathematical_validation(self):
        """Test calculate_sum_x_squared_deviations against mathematical definition."""
        # Test data with known mathematical properties
        test_cases = [
            np.array([1, 2, 3, 4, 5]),
            np.array([10, 20, 30, 40, 50]),
            np.array([-5, -2, 0, 3, 7]),
            np.array([100.5, 200.7, 300.2, 400.9]),
            np.random.normal(1000, 50, 100),  # Larger random dataset
        ]

        for test_data in test_cases:
            # Our implementation
            our_result = calculate_sum_x_squared_deviations(test_data)

            # Mathematical property validation: sum of squared deviations should equal sum((x - mean(x))^2)
            expected = np.sum((test_data - np.mean(test_data)) ** 2)
            np.testing.assert_allclose(
                our_result,
                expected,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"Mathematical property validation failed for data: {test_data[:5] if len(test_data) > 5 else test_data}",
            )

            # Additional validation: result should be non-negative
            assert (
                our_result >= 0
            ), f"Sum squared deviations must be non-negative, got {our_result}"

            # For constant arrays, sum of squared deviations should be zero
            if np.allclose(test_data, test_data[0]):
                assert np.allclose(
                    our_result, 0
                ), f"Sum squared deviations for constant array should be 0, got {our_result}"

    def test_extract_sum_squared_deviations_mathematical_consistency(self):
        """Test extract_sum_x_squared_deviations mathematical relationship."""
        # Test the mathematical relationship: sum_sq_dev = sigma^2 / var_beta
        test_cases = [
            {"var_beta": 0.001, "sigma": 25.0},
            {"var_beta": 0.0001, "sigma": 10.0},
            {"var_beta": 0.01, "sigma": 50.0},
            {"var_beta": 1e-6, "sigma": 5.0},
        ]

        for case in test_cases:
            result = extract_sum_x_squared_deviations(case["var_beta"], case["sigma"])
            expected = (case["sigma"] ** 2) / case["var_beta"]

            np.testing.assert_allclose(
                result,
                expected,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"Mathematical relationship failed for case: {case}",
            )

    def test_variance_calculations_mathematical_properties(self):
        """Test variance calculation functions against mathematical properties."""
        # Generate test data with known statistical properties
        np.random.seed(42)  # For reproducible tests
        x_values = np.random.normal(1000, 50, 100)
        x_mean = np.mean(x_values)
        sigma = 25.0
        n_pretest = len(x_values)
        sum_x_squared_deviations = calculate_sum_x_squared_deviations(x_values)

        # Test model variance calculation
        model_variances = calculate_model_variance(
            x_values, x_mean, sigma, n_pretest, sum_x_squared_deviations
        )

        # Mathematical property: model variance should be positive
        assert np.all(model_variances >= 0), "Model variances must be non-negative"

        # Mathematical property: variance at mean should be minimal
        variance_at_mean = calculate_model_variance(
            np.array([x_mean]), x_mean, sigma, n_pretest, sum_x_squared_deviations
        )[0]

        # At x = x_mean, variance should be sigma^2 / n
        expected_variance_at_mean = (sigma**2) / n_pretest
        np.testing.assert_allclose(
            variance_at_mean,
            expected_variance_at_mean,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Variance at mean calculation incorrect",
        )

        # Test prediction variance calculation
        prediction_variances = calculate_prediction_variance(model_variances, sigma)

        # Mathematical property: prediction variance = model variance + sigma^2
        expected_prediction_variances = model_variances + sigma**2
        np.testing.assert_allclose(
            prediction_variances,
            expected_prediction_variances,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Prediction variance calculation incorrect",
        )


class TestRegressionModelValidation:
    """Test regression model fitting against reference implementation and mathematical properties."""

    def test_regression_model_statistical_properties(self):
        """Test that regression model produces statistically valid results."""
        # Create synthetic data with known relationship
        np.random.seed(123)
        n = 50
        true_alpha = 100.0
        true_beta = 0.8
        true_sigma = 15.0

        x = np.random.normal(1000, 100, n)
        epsilon = np.random.normal(0, true_sigma, n)
        y = true_alpha + true_beta * x + epsilon

        learning_data = pd.DataFrame({"control": x, "test": y})

        # Fit model using our implementation
        model_params = fit_tbr_regression_model(learning_data, "control", "test")

        # Statistical validation tests
        assert isinstance(model_params, dict), "Model parameters should be a dictionary"

        required_keys = [
            "alpha",
            "beta",
            "sigma",
            "var_alpha",
            "var_beta",
            "cov_alpha_beta",
            "degrees_freedom",
            "n_pretest",
            "pretest_x_mean",
        ]
        for key in required_keys:
            assert key in model_params, f"Missing required parameter: {key}"

        # Mathematical property validation
        assert model_params["n_pretest"] == n, "Sample size should match input data"
        assert (
            model_params["degrees_freedom"] == n - 2
        ), "Degrees of freedom should be n-2"
        assert model_params["sigma"] > 0, "Residual standard deviation must be positive"
        assert model_params["var_alpha"] > 0, "Alpha variance must be positive"
        assert model_params["var_beta"] > 0, "Beta variance must be positive"

        # Estimate should be reasonably close to true values (with statistical tolerance)
        # Using 3-sigma rule for statistical significance
        alpha_se = np.sqrt(model_params["var_alpha"])
        beta_se = np.sqrt(model_params["var_beta"])

        # Alpha estimate should be within reasonable range of true value
        assert (
            abs(model_params["alpha"] - true_alpha) < 3 * alpha_se
        ), f"Alpha estimate {model_params['alpha']} too far from true value {true_alpha}"

        # Beta estimate should be within reasonable range of true value
        assert (
            abs(model_params["beta"] - true_beta) < 3 * beta_se
        ), f"Beta estimate {model_params['beta']} too far from true value {true_beta}"

    def test_regression_model_perfect_correlation(self):
        """Test regression model with perfect correlation (no noise)."""
        # Perfect linear relationship
        x = np.array([10, 20, 30, 40, 50], dtype=float)
        y = 5 + 2 * x  # Perfect relationship: y = 5 + 2*x

        learning_data = pd.DataFrame({"control": x, "test": y})
        model_params = fit_tbr_regression_model(learning_data, "control", "test")

        # With perfect correlation, estimates should be exact (within numerical precision)
        np.testing.assert_allclose(model_params["alpha"], 5.0, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(model_params["beta"], 2.0, rtol=1e-10, atol=1e-10)

        # Residual standard deviation should be very small (numerical precision)
        assert (
            model_params["sigma"] < 1e-10
        ), "Sigma should be near zero for perfect correlation"

    def test_regression_numerical_stability(self):
        """Test regression model numerical stability with extreme values."""
        test_cases = [
            # Very small values
            {
                "x": np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6]),
                "y": np.array([1e-5, 2e-5, 3e-5, 4e-5, 5e-5]),
            },
            # Very large values
            {
                "x": np.array([1e6, 2e6, 3e6, 4e6, 5e6]),
                "y": np.array([1e7, 2e7, 3e7, 4e7, 5e7]),
            },
        ]

        for case in test_cases:
            learning_data = pd.DataFrame({"control": case["x"], "test": case["y"]})
            model_params = fit_tbr_regression_model(learning_data, "control", "test")

            # All parameters should be finite
            for key, value in model_params.items():
                assert np.isfinite(value), f"Parameter {key} is not finite: {value}"

            # Variances should be positive
            assert model_params["var_alpha"] > 0, "Alpha variance must be positive"
            assert model_params["var_beta"] > 0, "Beta variance must be positive"
            assert model_params["sigma"] > 0, "Sigma must be positive"


class TestEndToEndTBRValidation:
    """Test end-to-end TBR analysis pipeline mathematical accuracy."""

    def test_complete_tbr_analysis_mathematical_consistency(self):
        """Test complete TBR analysis for mathematical consistency."""
        # Create synthetic dataset with known properties
        np.random.seed(456)
        dates = pd.date_range("2023-01-01", periods=90)

        # Pre-treatment period: no effect
        control_pre = np.random.normal(1000, 50, 60)
        test_pre = 100 + 0.9 * control_pre + np.random.normal(0, 25, 60)

        # Treatment period: positive effect
        control_test = np.random.normal(1000, 50, 30)
        test_test = (
            100 + 0.9 * control_test + 50 + np.random.normal(0, 25, 30)
        )  # +50 effect

        control = np.concatenate([control_pre, control_test])
        test = np.concatenate([test_pre, test_test])

        data = pd.DataFrame({"date": dates, "control": control, "test": test})

        # Run TBR analysis
        tbr_results, daily_summaries = perform_tbr_analysis(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-03-02"),
            test_end=pd.Timestamp("2023-04-01"),
            level=0.80,
            threshold=0.0,
        )

        # Mathematical consistency tests
        assert isinstance(
            tbr_results, pd.DataFrame
        ), "TBR results should be a DataFrame"
        assert isinstance(
            daily_summaries, pd.DataFrame
        ), "Daily summaries should be a DataFrame"

        # Check required columns exist
        required_tbr_cols = ["period", "y", "x", "pred", "cumdif", "cumsd"]
        for col in required_tbr_cols:
            assert col in tbr_results.columns, f"Missing required TBR column: {col}"

        # Mathematical property: cumulative differences should be cumulative
        test_period_data = tbr_results[tbr_results["period"] == 1]
        if len(test_period_data) > 1:
            diffs = test_period_data["y"] - test_period_data["pred"]
            expected_cumdif = np.cumsum(diffs)
            np.testing.assert_allclose(
                test_period_data["cumdif"].values,
                expected_cumdif,
                rtol=1e-10,
                atol=1e-10,
                err_msg="Cumulative differences calculation error",
            )

        # Mathematical property: final estimate should match final cumulative difference
        final_summary = daily_summaries.iloc[-1]
        final_cumdif = test_period_data["cumdif"].iloc[-1]

        np.testing.assert_allclose(
            final_summary["estimate"],
            final_cumdif,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Final estimate should match final cumulative difference",
        )

        # Statistical property: effect should be detectable (positive with reasonable confidence)
        # Given we added +50 effect, it should be statistically significant
        assert final_summary["estimate"] > 0, "Should detect positive treatment effect"

        # Confidence interval should be mathematically consistent
        assert (
            final_summary["lower"] < final_summary["estimate"] < final_summary["upper"]
        ), "Estimate should be within confidence interval"

        # Precision should be half the interval width
        interval_width = final_summary["upper"] - final_summary["lower"]
        expected_precision = interval_width / 2
        np.testing.assert_allclose(
            final_summary["precision"],
            expected_precision,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Precision should be half the interval width",
        )

    def test_tbr_summary_mathematical_properties(self):
        """Test TBR summary creation mathematical properties."""
        # Create minimal valid TBR dataframe
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 1, 1, 1],
                "cumdif": [np.nan, np.nan, 10, 25, 40],
                "cumsd": [0, 0, 5, 8, 10],
            }
        )

        # Model parameters for summary
        alpha, beta = 50.0, 0.95
        sigma = 25.0
        var_alpha, var_beta = 100.0, 0.001
        cov_alpha_beta = -0.05
        degrees_freedom = 43
        level, threshold = 0.80, 0.0

        # Create summary
        summary = create_tbr_summary(
            tbr_dataframe=tbr_df,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            var_alpha=var_alpha,
            var_beta=var_beta,
            cov_alpha_beta=cov_alpha_beta,
            degrees_freedom=degrees_freedom,
            level=level,
            threshold=threshold,
        )

        # Mathematical consistency tests
        assert len(summary) == 1, "Summary should have exactly one row"

        # Estimate should match final cumulative difference
        expected_estimate = tbr_df[tbr_df["period"] == 1]["cumdif"].iloc[-1]
        np.testing.assert_allclose(
            summary["estimate"].iloc[0], expected_estimate, rtol=1e-12, atol=1e-12
        )

        # SE should match final cumulative standard deviation
        expected_se = tbr_df[tbr_df["period"] == 1]["cumsd"].iloc[-1]
        np.testing.assert_allclose(
            summary["se"].iloc[0], expected_se, rtol=1e-12, atol=1e-12
        )

        # Confidence interval mathematical consistency
        estimate = summary["estimate"].iloc[0]
        precision = summary["precision"].iloc[0]
        lower = summary["lower"].iloc[0]
        upper = summary["upper"].iloc[0]

        # Lower = estimate - precision, Upper = estimate + precision
        np.testing.assert_allclose(lower, estimate - precision, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(upper, estimate + precision, rtol=1e-12, atol=1e-12)

        # Probability should be between 0 and 1
        prob = summary["prob"].iloc[0]
        assert 0 <= prob <= 1, f"Probability must be between 0 and 1, got {prob}"


class TestStatisticalInferenceValidation:
    """Test statistical inference calculations for mathematical accuracy."""

    def test_safe_int_conversion_mathematical_properties(self):
        """Test safe integer conversion mathematical properties."""
        # Test cases that should succeed
        valid_cases = [
            (43.0, 43),
            (43.999999999999, 44),  # Very close to integer
            (42.000000000001, 42),  # Very close to integer
            (100.0, 100),
        ]

        for input_val, expected in valid_cases:
            result = safe_int_conversion(input_val, "test_param")
            assert result == expected, f"Expected {expected}, got {result}"
            assert isinstance(result, int), "Result should be integer type"

        # Test cases that should fail (not close to integer)
        invalid_cases = [43.5, 42.1, 100.99]

        for input_val in invalid_cases:
            with pytest.raises(ValueError, match="should be an integer"):
                safe_int_conversion(input_val, "test_param")

    def test_cumulative_standard_deviation_mathematical_properties(self):
        """Test cumulative standard deviation calculation mathematical properties."""
        # Test data
        test_x_values = np.array([1000, 1020, 1010, 1030, 1015])
        sigma = 25.0
        var_alpha = 100.0
        var_beta = 0.001
        cov_alpha_beta = -0.05

        # Calculate cumulative standard deviations
        cumsd = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Mathematical properties
        assert len(cumsd) == len(test_x_values), "Length should match input"
        assert np.all(cumsd > 0), "All standard deviations should be positive"
        assert np.all(
            cumsd[1:] >= cumsd[:-1]
        ), "Should be non-decreasing (cumulative uncertainty)"

        # First value should be smallest (single observation)
        assert cumsd[0] <= np.min(cumsd), "First value should be minimal"

        # Mathematical relationship: variance grows with time
        # V[Δr(T)] = T·σ² + T²·v where v depends on x_mean_cumulative
        # The growth pattern depends on the specific x values and their cumulative means
        # We can't assume simple convexity due to the x_mean_cumulative term
        # Instead, verify that the final value is largest (cumulative uncertainty increases)
        assert cumsd[-1] == np.max(cumsd), "Final cumulative std dev should be largest"

    def test_interval_estimation_mathematical_consistency(self):
        """Test interval estimation mathematical consistency."""
        # Create test TBR dataframe
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 1, 1, 1, 1],
                "y": [100, 110, 120, 130, 125, 135],
                "pred": [105, 108, 118, 128, 127, 133],
                "estsd": [2, 2, 3, 3, 3, 3],
            }
        )

        # Create test summary
        tbr_summary = pd.DataFrame({"sigma": [10.0], "t_dist_df": [40.0]})

        # Test interval estimation
        result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, start_day=1, end_day=3, ci_level=0.80
        )

        # Mathematical consistency tests
        required_keys = ["estimate", "precision", "lower", "upper"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Mathematical relationships
        estimate = result["estimate"]
        precision = result["precision"]
        lower = result["lower"]
        upper = result["upper"]

        # Lower = estimate - precision, Upper = estimate + precision
        np.testing.assert_allclose(lower, estimate - precision, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(upper, estimate + precision, rtol=1e-12, atol=1e-12)

        # Precision should be positive
        assert precision > 0, "Precision should be positive"

        # Estimate should be within interval
        assert (
            lower <= estimate <= upper
        ), "Estimate should be within confidence interval"

        # Estimate should match sum of differences for the interval
        test_data = tbr_df[tbr_df["period"] == 1].iloc[0:3]  # Days 1-3
        expected_estimate = (test_data["y"] - test_data["pred"]).sum()
        np.testing.assert_allclose(
            estimate,
            expected_estimate,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Estimate should match sum of differences",
        )


class TestNumericalPrecisionValidation:
    """Test numerical precision and stability of mathematical operations."""

    def test_numerical_precision_extreme_values(self):
        """Test numerical precision with extreme values."""
        # Test with very small values
        small_values = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        result_small = calculate_sum_x_squared_deviations(small_values)
        assert np.isfinite(result_small), "Should handle very small values"
        assert result_small >= 0, "Result should be non-negative"

        # Test with very large values
        large_values = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
        result_large = calculate_sum_x_squared_deviations(large_values)
        assert np.isfinite(result_large), "Should handle very large values"
        assert result_large >= 0, "Result should be non-negative"

        # Test with mixed scales
        mixed_values = np.array([1e-6, 1e-3, 1.0, 1e3, 1e6])
        result_mixed = calculate_sum_x_squared_deviations(mixed_values)
        assert np.isfinite(result_mixed), "Should handle mixed scale values"
        assert result_mixed >= 0, "Result should be non-negative"

    def test_numerical_stability_regression(self):
        """Test numerical stability of regression calculations."""
        # Test with highly correlated data (near-singular case)
        np.random.seed(789)
        x_base = np.random.normal(1000, 100, 50)
        # Add tiny amount of noise to create near-perfect correlation
        y = 10 + 0.5 * x_base + np.random.normal(0, 0.001, 50)

        learning_data = pd.DataFrame({"control": x_base, "test": y})
        model_params = fit_tbr_regression_model(learning_data, "control", "test")

        # All parameters should be finite despite near-singularity
        for key, value in model_params.items():
            assert np.isfinite(value), f"Parameter {key} should be finite, got {value}"

        # Variances should still be positive (numerical stability)
        assert model_params["var_alpha"] > 0, "Alpha variance should be positive"
        assert model_params["var_beta"] > 0, "Beta variance should be positive"
        assert model_params["sigma"] >= 0, "Sigma should be non-negative"


# Mark all tests as mathematical validation tests
pytestmark = pytest.mark.mathematical
