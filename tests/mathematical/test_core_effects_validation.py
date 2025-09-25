"""
Comprehensive mathematical validation tests for TBR core effects module.

This module provides rigorous mathematical validation for the core effects functions
implemented in Task 4.1. Tests ensure mathematical correctness, cross-validation
with functional implementation, and proper statistical properties.

Test Categories
---------------
1. Cumulative standard deviation mathematical validation
2. Cumulative variance mathematical validation
3. TBR summary mathematical properties
4. Incremental TBR summaries mathematical consistency
5. Cross-validation with functional implementation
6. Statistical property verification
7. Edge case mathematical behavior
8. Numerical precision validation

Mathematical Validation
-----------------------
All tests validate against known mathematical formulas and cross-check with
the functional implementation to ensure identical results. Tests verify:

- Mathematical formula correctness
- Statistical property preservation
- Numerical stability and precision
- Edge case behavior
- Cross-implementation consistency
"""

import numpy as np
import pandas as pd

from tbr.core.effects import (
    calculate_cumulative_standard_deviation,
    calculate_cumulative_variance,
    compute_interval_estimate_and_ci,
    create_incremental_tbr_summaries,
    create_tbr_summary,
)
from tbr.functional.tbr_functions import (
    calculate_cumulative_standard_deviation as func_cumulative_std,
)
from tbr.functional.tbr_functions import (
    compute_interval_estimate_and_ci as func_compute_interval,
)
from tbr.functional.tbr_functions import (
    create_incremental_tbr_summaries as func_create_incremental,
)
from tbr.functional.tbr_functions import create_tbr_summary as func_create_summary


class TestCumulativeStandardDeviationMathematical:
    """Mathematical validation tests for cumulative standard deviation calculation."""

    def test_mathematical_formula_correctness(self):
        """Test that the mathematical formula is implemented correctly."""
        # Known test case with hand-calculated expected values
        test_x_values = np.array([1000.0, 1020.0, 1010.0, 1030.0])
        sigma = 25.0
        var_alpha = 100.0
        var_beta = 0.001
        cov_alpha_beta = -0.05

        # Calculate using core function
        result = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Manual calculation using TBR formula:
        # σ_cumulative[T] = sqrt(T * σ² + T² * v)
        # where v = var_alpha + 2 * x_mean_T * cov_alpha_beta + x_mean_T² * var_beta

        n = len(test_x_values)
        expected = np.zeros(n)

        for t in range(1, n + 1):
            x_mean_t = np.mean(test_x_values[:t])
            v_t = var_alpha + 2 * x_mean_t * cov_alpha_beta + (x_mean_t**2) * var_beta
            expected[t - 1] = np.sqrt(t * (sigma**2) + (t**2) * v_t)

        # Verify mathematical correctness with high precision
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_statistical_properties(self):
        """Test that statistical properties are preserved."""
        test_x_values = np.array([950.0, 1000.0, 1050.0, 1100.0, 1150.0])
        sigma = 30.0
        var_alpha = 80.0
        var_beta = 0.002
        cov_alpha_beta = -0.03

        result = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Property 1: All values should be positive
        assert np.all(result > 0), "All cumulative standard deviations must be positive"

        # Property 2: Generally increasing (uncertainty grows with time)
        # Note: This is generally true but not strictly monotonic due to v_t dependence on x_mean_t
        assert (
            result[-1] > result[0]
        ), "Final uncertainty should be greater than initial"

        # Property 3: Values should be finite and reasonable
        assert np.all(np.isfinite(result)), "All values must be finite"
        assert np.all(result < 1000), "Values should be reasonable in magnitude"

    def test_cross_validation_with_functional(self):
        """Cross-validate with functional implementation for identical results."""
        test_x_values = np.array([800.0, 850.0, 900.0, 950.0, 1000.0, 1050.0])
        sigma = 40.0
        var_alpha = 120.0
        var_beta = 0.0015
        cov_alpha_beta = -0.08

        # Calculate with both implementations
        core_result = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )
        func_result = func_cumulative_std(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Must be identical to machine precision
        np.testing.assert_allclose(core_result, func_result, rtol=1e-14)

    def test_edge_case_single_value(self):
        """Test mathematical correctness for single value case."""
        test_x_values = np.array([1000.0])
        sigma = 20.0
        var_alpha = 50.0
        var_beta = 0.001
        cov_alpha_beta = -0.02

        result = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # For T=1, x_mean_1 = x[0] = 1000.0
        x_mean = 1000.0
        v_1 = var_alpha + 2 * x_mean * cov_alpha_beta + (x_mean**2) * var_beta
        expected = np.sqrt(1 * (sigma**2) + (1**2) * v_1)

        assert len(result) == 1
        np.testing.assert_allclose(result, [expected], rtol=1e-12)

    def test_numerical_precision_extreme_values(self):
        """Test numerical precision with extreme parameter values."""
        # Test with very small values
        test_x_values = np.array([1e-6, 2e-6, 3e-6])
        sigma = 1e-8
        var_alpha = 1e-12
        var_beta = 1e-10
        cov_alpha_beta = -1e-11

        result_small = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Should handle small values without underflow
        assert np.all(result_small > 0)
        assert np.all(np.isfinite(result_small))

        # Test with large values
        test_x_values = np.array([1e6, 2e6, 3e6])
        sigma = 1e4
        var_alpha = 1e8
        var_beta = 1e-2
        cov_alpha_beta = -1e3

        result_large = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Should handle large values without overflow
        assert np.all(result_large > 0)
        assert np.all(np.isfinite(result_large))


class TestCumulativeVarianceMathematical:
    """Mathematical validation tests for cumulative variance calculation."""

    def test_mathematical_formula_correctness(self):
        """Test that cumulative variance formula is implemented correctly."""
        test_x_values = np.array([1000.0, 1020.0, 1010.0, 1030.0])
        sigma = 25.0
        var_alpha = 100.0
        var_beta = 0.001
        cov_alpha_beta = -0.05

        # Calculate using core function
        variance_result = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Also calculate standard deviation
        std_result = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Mathematical relationship: variance = std_dev²
        expected_variance = std_result**2

        # Verify mathematical relationship with high precision
        np.testing.assert_allclose(variance_result, expected_variance, rtol=1e-12)

    def test_direct_formula_implementation(self):
        """Test direct implementation of variance formula V[Δr(T)] = T·σ² + T²·v."""
        test_x_values = np.array([950.0, 1000.0, 1050.0])
        sigma = 30.0
        var_alpha = 80.0
        var_beta = 0.002
        cov_alpha_beta = -0.03

        result = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Manual calculation using direct formula
        n = len(test_x_values)
        expected = np.zeros(n)

        for t in range(1, n + 1):
            # Calculate cumulative mean up to time t
            x_mean_t = np.mean(test_x_values[:t])
            # Calculate v_t = Var(α̂) + 2·x̄_T·Cov(α̂,β̂) + x̄_T²·Var(β̂)
            v_t = var_alpha + 2 * x_mean_t * cov_alpha_beta + (x_mean_t**2) * var_beta
            # Apply formula: V[Δr(T)] = T·σ² + T²·v
            expected[t - 1] = t * (sigma**2) + (t**2) * v_t

        # Verify direct formula implementation
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_statistical_properties_variance(self):
        """Test statistical properties specific to variance."""
        test_x_values = np.array([900.0, 950.0, 1000.0, 1050.0, 1100.0])
        sigma = 35.0
        var_alpha = 90.0
        var_beta = 0.0025
        cov_alpha_beta = -0.04

        result = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Property 1: All variances must be positive
        assert np.all(result > 0), "All cumulative variances must be positive"

        # Property 2: Values should be finite
        assert np.all(np.isfinite(result)), "All variance values must be finite"

        # Property 3: Final variance should be larger than initial (generally true)
        assert result[-1] > result[0], "Final cumulative variance should exceed initial"

    def test_cross_validation_variance_std_relationship(self):
        """Cross-validate variance-standard deviation mathematical relationship."""
        test_x_values = np.array([800.0, 850.0, 900.0, 950.0])
        sigma = 28.0
        var_alpha = 75.0
        var_beta = 0.0018
        cov_alpha_beta = -0.035

        variance_result = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )
        std_result = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Mathematical relationship must hold exactly
        np.testing.assert_allclose(variance_result, std_result**2, rtol=1e-14)
        np.testing.assert_allclose(np.sqrt(variance_result), std_result, rtol=1e-14)


class TestTBRSummaryMathematical:
    """Mathematical validation tests for TBR summary creation."""

    def test_cross_validation_with_functional(self):
        """Cross-validate TBR summary creation with functional implementation."""
        # Create test TBR dataframe
        test_data = pd.DataFrame(
            {
                "period": [1, 1, 1, 1],
                "y": [1020.0, 1040.0, 1030.0, 1050.0],
                "x": [1000.0, 1020.0, 1010.0, 1030.0],
                "pred": [1010.0, 1030.0, 1020.0, 1040.0],
                "dif": [10.0, 10.0, 10.0, 10.0],
                "cumdif": [10.0, 20.0, 30.0, 40.0],
                "cumsd": [25.0, 35.0, 43.0, 50.0],
            }
        )

        # Model parameters
        alpha = 50.0
        beta = 0.95
        sigma = 25.0
        var_alpha = 100.0
        var_beta = 0.001
        cov_alpha_beta = -0.05
        degrees_freedom = 43
        level = 0.80
        threshold = 0.0

        # Calculate with both implementations
        core_result = create_tbr_summary(
            test_data,
            alpha,
            beta,
            sigma,
            var_alpha,
            var_beta,
            cov_alpha_beta,
            degrees_freedom,
            level,
            threshold,
        )

        func_result = func_create_summary(
            test_data,
            alpha,
            beta,
            sigma,
            var_alpha,
            var_beta,
            cov_alpha_beta,
            degrees_freedom,
            level,
            threshold,
        )

        # Compare all numerical columns
        numerical_cols = ["estimate", "precision", "lower", "upper", "prob"]
        for col in numerical_cols:
            if col in core_result.columns and col in func_result.columns:
                np.testing.assert_allclose(
                    core_result[col].values, func_result[col].values, rtol=1e-12
                )

    def test_statistical_inference_mathematical_properties(self):
        """Test mathematical properties of statistical inference in summary."""
        # Create test data with known final effect
        test_data = pd.DataFrame(
            {
                "period": [1, 1, 1],
                "cumdif": [5.0, 15.0, 25.0],  # Final cumulative effect = 25.0
                "cumsd": [10.0, 14.0, 18.0],  # Final cumulative std = 18.0
            }
        )

        alpha = 100.0
        beta = 1.0
        sigma = 20.0
        var_alpha = 50.0
        var_beta = 0.0005
        cov_alpha_beta = -0.02
        degrees_freedom = 40
        level = 0.95
        threshold = 0.0

        result = create_tbr_summary(
            test_data,
            alpha,
            beta,
            sigma,
            var_alpha,
            var_beta,
            cov_alpha_beta,
            degrees_freedom,
            level,
            threshold,
        )

        # Extract key values
        estimate = result["estimate"].iloc[0]
        precision = result["precision"].iloc[0]
        lower = result["lower"].iloc[0]
        upper = result["upper"].iloc[0]
        prob = result["prob"].iloc[0]

        # Mathematical property 1: estimate should match final cumdif
        assert (
            abs(estimate - 25.0) < 1e-10
        ), "Estimate should match final cumulative effect"

        # Mathematical property 2: credible interval properties
        assert lower < estimate < upper, "Estimate should be within credible interval"

        # Mathematical property 3: precision is half-width of interval
        interval_width = upper - lower
        expected_precision = interval_width / 2
        assert (
            abs(precision - expected_precision) < 1e-10
        ), "Precision should be half interval width"

        # Mathematical property 4: probability should be reasonable for positive effect
        assert 0.0 <= prob <= 1.0, "Probability must be between 0 and 1"
        if estimate > 0:
            assert (
                prob > 0.5
            ), "Positive estimate should have probability > 0.5 for threshold=0"


class TestIncrementalSummariesMathematical:
    """Mathematical validation tests for incremental TBR summaries."""

    def test_cross_validation_with_functional(self):
        """Cross-validate incremental summaries with functional implementation."""
        # Create test TBR dataframe with multiple test period rows
        test_data = pd.DataFrame(
            {
                "period": [1, 1, 1, 1],
                "y": [1020.0, 1040.0, 1030.0, 1050.0],
                "x": [1000.0, 1020.0, 1010.0, 1030.0],
                "pred": [1010.0, 1030.0, 1020.0, 1040.0],
                "dif": [10.0, 10.0, 10.0, 10.0],
                "cumdif": [10.0, 20.0, 30.0, 40.0],
                "cumsd": [15.0, 21.0, 26.0, 30.0],
            }
        )

        # Model parameters
        alpha = 60.0
        beta = 0.98
        sigma = 22.0
        var_alpha = 80.0
        var_beta = 0.0008
        cov_alpha_beta = -0.04
        degrees_freedom = 38
        level = 0.90
        threshold = 5.0

        # Calculate with both implementations
        core_result = create_incremental_tbr_summaries(
            test_data,
            alpha,
            beta,
            sigma,
            var_alpha,
            var_beta,
            cov_alpha_beta,
            degrees_freedom,
            level,
            threshold,
        )

        func_result = func_create_incremental(
            test_data,
            alpha,
            beta,
            sigma,
            var_alpha,
            var_beta,
            cov_alpha_beta,
            degrees_freedom,
            level,
            threshold,
        )

        # Compare all numerical columns
        numerical_cols = ["estimate", "precision", "lower", "upper", "prob"]
        for col in numerical_cols:
            if col in core_result.columns and col in func_result.columns:
                np.testing.assert_allclose(
                    core_result[col].values, func_result[col].values, rtol=1e-12
                )

    def test_incremental_mathematical_consistency(self):
        """Test mathematical consistency of incremental progression."""
        # Create test data with clear progression
        test_data = pd.DataFrame(
            {
                "period": [1, 1, 1],
                "cumdif": [10.0, 25.0, 45.0],
                "cumsd": [12.0, 18.0, 22.0],
            }
        )

        alpha = 50.0
        beta = 1.0
        sigma = 15.0
        var_alpha = 60.0
        var_beta = 0.0006
        cov_alpha_beta = -0.03
        degrees_freedom = 35
        level = 0.80
        threshold = 0.0

        result = create_incremental_tbr_summaries(
            test_data,
            alpha,
            beta,
            sigma,
            var_alpha,
            var_beta,
            cov_alpha_beta,
            degrees_freedom,
            level,
            threshold,
        )

        # Mathematical property: estimates should match cumulative differences
        expected_estimates = [10.0, 25.0, 45.0]
        actual_estimates = result["estimate"].values

        np.testing.assert_allclose(actual_estimates, expected_estimates, rtol=1e-10)

        # Mathematical property: progression should be monotonic for this case
        assert np.all(
            np.diff(actual_estimates) > 0
        ), "Estimates should increase monotonically"

        # Mathematical property: all probabilities should be valid
        probs = result["prob"].values
        assert np.all((probs >= 0) & (probs <= 1)), "All probabilities must be in [0,1]"


class TestIntervalEstimationMathematical:
    """Mathematical validation tests for interval estimation and CI."""

    def test_cross_validation_with_functional(self):
        """Cross-validate interval estimation with functional implementation."""
        # Create test TBR dataframe with required columns
        tbr_df = pd.DataFrame(
            {
                "period": [1, 1, 1, 1, 1],
                "y": [1020.0, 1035.0, 1045.0, 1055.0, 1070.0],
                "pred": [1015.0, 1023.0, 1025.0, 1027.0, 1035.0],
                "cumdif": [5.0, 12.0, 20.0, 28.0, 35.0],
                "cumsd": [8.0, 11.0, 14.0, 16.0, 18.0],
                "estsd": [2.5, 2.8, 3.1, 3.4, 3.7],
            }
        )

        # Create corresponding summary with required columns
        tbr_summary = pd.DataFrame(
            {
                "estimate": [35.0],
                "precision": [9.2],
                "lower": [25.8],
                "upper": [44.2],
                "sigma": [20.0],
                "t_dist_df": [40],
            }
        )

        start_day = 2  # Index 1 (0-based)
        end_day = 4  # Index 3 (0-based)
        ci_level = 0.90

        # Calculate with both implementations
        core_result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, start_day, end_day, ci_level
        )

        func_result = func_compute_interval(
            tbr_df, tbr_summary, start_day, end_day, ci_level
        )

        # Compare all numerical values
        for key in ["estimate", "precision", "lower", "upper"]:
            if key in core_result and key in func_result:
                np.testing.assert_allclose(
                    core_result[key], func_result[key], rtol=1e-12
                )

    def test_subinterval_mathematical_properties(self):
        """Test mathematical properties of subinterval estimation."""
        # Create test data with known subinterval
        tbr_df = pd.DataFrame(
            {
                "period": [1, 1, 1, 1],
                "y": [1030.0, 1050.0, 1065.0, 1080.0],
                "pred": [1020.0, 1025.0, 1025.0, 1030.0],
                "cumdif": [10.0, 25.0, 40.0, 50.0],
                "cumsd": [12.0, 18.0, 22.0, 25.0],
                "estsd": [3.0, 3.5, 4.0, 4.5],
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "estimate": [50.0],
                "precision": [12.5],
                "sigma": [15.0],
                "t_dist_df": [35],
            }
        )

        # Test subinterval from day 2 to day 3 (indices 1 to 2)
        start_day = 2
        end_day = 3
        ci_level = 0.95

        result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, start_day, end_day, ci_level
        )

        # Mathematical property: subinterval effect should be sum of (y - pred) for the interval
        # For days 2-3 (indices 1-2): (1050-1025) + (1065-1025) = 25 + 40 = 65
        expected_estimate = (1050.0 - 1025.0) + (1065.0 - 1025.0)

        assert (
            abs(result["estimate"] - expected_estimate) < 1e-10
        ), "Subinterval estimate should be sum of (y - pred) for the interval"

        # Mathematical property: interval should be properly formed
        assert (
            result["lower"] < result["estimate"] < result["upper"]
        ), "Estimate should be within confidence interval"

        # Mathematical property: precision should be half interval width
        interval_width = result["upper"] - result["lower"]
        expected_precision = interval_width / 2
        assert (
            abs(result["precision"] - expected_precision) < 1e-10
        ), "Precision should be half the interval width"


class TestEffectsModuleIntegrationMathematical:
    """Integration tests for mathematical consistency across effects module."""

    def test_variance_standard_deviation_consistency(self):
        """Test mathematical consistency between variance and standard deviation functions."""
        test_x_values = np.array([950.0, 1000.0, 1050.0, 1100.0])
        sigma = 32.0
        var_alpha = 95.0
        var_beta = 0.0022
        cov_alpha_beta = -0.045

        # Calculate both variance and standard deviation
        variance_result = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )
        std_result = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Mathematical relationship must hold exactly
        np.testing.assert_allclose(variance_result, std_result**2, rtol=1e-14)
        np.testing.assert_allclose(np.sqrt(variance_result), std_result, rtol=1e-14)

    def test_effects_module_numerical_stability(self):
        """Test numerical stability across all effects functions."""
        # Test with challenging numerical conditions
        test_x_values = np.array([1e-3, 2e-3, 3e-3, 4e-3])
        sigma = 1e-5
        var_alpha = 1e-10
        var_beta = 1e-8
        cov_alpha_beta = -1e-9

        # All functions should handle small values without numerical issues
        std_result = calculate_cumulative_standard_deviation(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )
        var_result = calculate_cumulative_variance(
            test_x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Results should be finite and positive
        assert np.all(np.isfinite(std_result)), "Standard deviations must be finite"
        assert np.all(np.isfinite(var_result)), "Variances must be finite"
        assert np.all(std_result > 0), "Standard deviations must be positive"
        assert np.all(var_result > 0), "Variances must be positive"

        # Mathematical relationship should still hold
        np.testing.assert_allclose(var_result, std_result**2, rtol=1e-12)
