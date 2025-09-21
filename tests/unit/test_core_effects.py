"""
Unit tests for the TBR core effects module.

This module provides comprehensive testing for the core effects module,
covering treatment effect calculations, lift measurement, summary statistics,
and backward compatibility with the functional implementation.
"""

import numpy as np
import pandas as pd

from tbr.core.effects import (
    calculate_cumulative_standard_deviation,
    compute_interval_estimate_and_ci,
    create_incremental_tbr_summaries,
    create_tbr_summary,
)


class TestCalculateCumulativeStandardDeviation:
    """Tests for cumulative standard deviation calculation."""

    def test_basic_cumulative_standard_deviation(self):
        """Test basic cumulative standard deviation calculation."""
        # Test data
        x_values = np.array([1000, 1020, 1010, 1030])
        sigma = 25.0
        var_alpha = 100.0
        var_beta = 0.001
        cov_alpha_beta = -0.05

        # Calculate cumulative standard deviations
        result = calculate_cumulative_standard_deviation(
            x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Verify result properties
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_values)
        assert all(result > 0)  # All standard deviations should be positive
        assert np.all(np.diff(result) > 0)  # Should be increasing

    def test_cumulative_standard_deviation_mathematical_properties(self):
        """Test mathematical properties of cumulative standard deviation."""
        x_values = np.array([100, 110, 105, 115, 120])
        sigma = 10.0
        var_alpha = 25.0
        var_beta = 0.0001
        cov_alpha_beta = -0.01

        result = calculate_cumulative_standard_deviation(
            x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Verify increasing property (uncertainty grows with time)
        for i in range(1, len(result)):
            assert (
                result[i] > result[i - 1]
            ), f"Standard deviation should increase: {result[i]} <= {result[i-1]}"

        # Verify reasonable magnitude
        assert all(5 < val < 200 for val in result), f"Unexpected magnitudes: {result}"

    def test_cumulative_standard_deviation_backward_compatibility(self):
        """Test backward compatibility with functional implementation."""
        from tbr.functional.tbr_functions import (
            calculate_cumulative_standard_deviation as func_calc,
        )

        # Test parameters
        x_values = np.array([500, 520, 510, 530, 525])
        sigma = 15.0
        var_alpha = 50.0
        var_beta = 0.0005
        cov_alpha_beta = -0.02

        # Calculate using both implementations
        core_result = calculate_cumulative_standard_deviation(
            x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )
        func_result = func_calc(x_values, sigma, var_alpha, var_beta, cov_alpha_beta)

        # Verify identical results
        np.testing.assert_array_almost_equal(core_result, func_result, decimal=10)


class TestComputeIntervalEstimateAndCI:
    """Tests for interval estimate and credible interval computation."""

    def test_basic_interval_estimation(self):
        """Test basic interval estimation functionality."""
        # Create test TBR dataframe
        tbr_df = pd.DataFrame(
            {
                "period": [1, 1, 1, 1, 1],
                "y": [105, 110, 108, 112, 115],
                "pred": [100, 105, 103, 107, 110],
                "estsd": [2.5, 2.6, 2.4, 2.7, 2.8],
            }
        )

        # Create test summary
        tbr_summary = pd.DataFrame({"sigma": [5.0], "t_dist_df": [20]})

        # Compute interval estimate
        result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, start_day=2, end_day=4, ci_level=0.80
        )

        # Verify result structure
        expected_keys = {"estimate", "precision", "lower", "upper"}
        assert set(result.keys()) == expected_keys

        # Verify result properties
        assert isinstance(result["estimate"], (int, float, np.integer, np.floating))
        assert isinstance(result["precision"], (int, float, np.integer, np.floating))
        assert result["precision"] > 0
        assert result["lower"] < result["upper"]
        assert result["lower"] < result["estimate"] < result["upper"]

    def test_interval_estimation_full_period(self):
        """Test interval estimation for full test period."""
        # Create test data
        tbr_df = pd.DataFrame(
            {
                "period": [1, 1, 1],
                "y": [102, 105, 108],
                "pred": [100, 103, 105],
                "estsd": [1.5, 1.6, 1.7],
            }
        )

        tbr_summary = pd.DataFrame({"sigma": [3.0], "t_dist_df": [15]})

        # Test full period
        result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, start_day=1, end_day=3, ci_level=0.95
        )

        # Verify estimate is sum of differences
        expected_estimate = (102 - 100) + (105 - 103) + (108 - 105)
        assert abs(result["estimate"] - expected_estimate) < 1e-10

    def test_interval_estimation_backward_compatibility(self):
        """Test backward compatibility with functional implementation."""
        from tbr.functional.tbr_functions import (
            compute_interval_estimate_and_ci as func_compute,
        )

        # Create test data
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 1, 1, 1, 1],
                "y": [98, 99, 105, 110, 108, 112],
                "pred": [98, 99, 100, 105, 103, 107],
                "estsd": [0, 0, 2.5, 2.6, 2.4, 2.7],
            }
        )

        tbr_summary = pd.DataFrame({"sigma": [4.0], "t_dist_df": [25]})

        # Test parameters
        start_day, end_day, ci_level = 2, 4, 0.90

        # Calculate using both implementations
        core_result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, start_day, end_day, ci_level
        )
        func_result = func_compute(tbr_df, tbr_summary, start_day, end_day, ci_level)

        # Verify identical results
        for key in ["estimate", "precision", "lower", "upper"]:
            assert abs(core_result[key] - func_result[key]) < 1e-10


class TestCreateTbrSummary:
    """Tests for TBR summary creation."""

    def test_basic_summary_creation(self):
        """Test basic TBR summary creation."""
        # Create test TBR dataframe
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 1, 1, 1],
                "cumdif": [np.nan, np.nan, 5, 12, 18],
                "cumsd": [0, 0, 3.5, 5.2, 6.8],
            }
        )

        # Test parameters
        alpha, beta, sigma = 50.0, 0.95, 25.0
        var_alpha, var_beta, cov_alpha_beta = 100.0, 0.001, -0.05
        degrees_freedom, level, threshold = 43, 0.80, 0.0

        # Create summary
        summary = create_tbr_summary(
            tbr_df,
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

        # Verify result structure
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1  # Single row summary

        # Verify required columns
        required_cols = ["estimate", "precision", "lower", "upper", "prob"]
        for col in required_cols:
            assert col in summary.columns

        # Verify result properties
        assert summary["estimate"].iloc[0] == 18  # Final cumdif
        assert summary["precision"].iloc[0] > 0
        assert summary["lower"].iloc[0] < summary["upper"].iloc[0]
        assert 0 <= summary["prob"].iloc[0] <= 1

    def test_summary_mathematical_properties(self):
        """Test mathematical properties of summary statistics."""
        # Create test data with known properties
        tbr_df = pd.DataFrame(
            {
                "period": [0, 1, 1, 1, 1],
                "cumdif": [np.nan, 2, 5, 8, 10],
                "cumsd": [0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        # Parameters
        alpha, beta, sigma = 100.0, 1.0, 20.0
        var_alpha, var_beta, cov_alpha_beta = 50.0, 0.0001, -0.01
        degrees_freedom, level, threshold = 30, 0.95, 5.0

        summary = create_tbr_summary(
            tbr_df,
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

        # Verify mathematical relationships
        estimate = summary["estimate"].iloc[0]
        precision = summary["precision"].iloc[0]
        lower = summary["lower"].iloc[0]
        upper = summary["upper"].iloc[0]

        assert abs((upper - lower) / 2 - precision) < 1e-10  # Precision is half-width
        assert abs((upper + lower) / 2 - estimate) < 1e-10  # Estimate is center

    def test_summary_backward_compatibility(self):
        """Test backward compatibility with functional implementation."""
        from tbr.functional.tbr_functions import create_tbr_summary as func_create

        # Create test data
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 1, 1, 1, 1],
                "cumdif": [np.nan, np.nan, 3, 7, 11, 15],
                "cumsd": [0, 0, 2.5, 3.8, 4.9, 6.0],
            }
        )

        # Parameters
        alpha, beta, sigma = 75.0, 0.85, 30.0
        var_alpha, var_beta, cov_alpha_beta = 80.0, 0.0008, -0.03
        degrees_freedom, level, threshold = 35, 0.85, 2.0

        # Create summaries using both implementations
        core_summary = create_tbr_summary(
            tbr_df,
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
        func_summary = func_create(
            tbr_df,
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

        # Verify identical results for key columns
        key_cols = ["estimate", "precision", "lower", "upper", "prob"]
        for col in key_cols:
            core_val = core_summary[col].iloc[0]
            func_val = func_summary[col].iloc[0]
            assert (
                abs(core_val - func_val) < 1e-10
            ), f"Mismatch in {col}: {core_val} vs {func_val}"


class TestCreateIncrementalTbrSummaries:
    """Tests for incremental TBR summaries creation."""

    def test_basic_incremental_summaries(self):
        """Test basic incremental summaries creation."""
        # Create test TBR dataframe
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 1, 1, 1, 1],
                "cumdif": [np.nan, np.nan, 2, 5, 8, 12],
                "cumsd": [0, 0, 1.5, 2.8, 4.0, 5.2],
            }
        )

        # Parameters
        alpha, beta, sigma = 60.0, 0.90, 22.0
        var_alpha, var_beta, cov_alpha_beta = 70.0, 0.0006, -0.025
        degrees_freedom, level, threshold = 40, 0.80, 1.0

        # Create incremental summaries
        summaries = create_incremental_tbr_summaries(
            tbr_df,
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

        # Verify result structure
        assert isinstance(summaries, pd.DataFrame)
        assert len(summaries) == 4  # Four test period days
        assert "test_day" in summaries.columns

        # Verify incremental property
        estimates = summaries["estimate"].values
        assert np.array_equal(estimates, [2, 5, 8, 12])  # Should match cumdif values

        # Verify test_day column
        assert np.array_equal(summaries["test_day"].values, [1, 2, 3, 4])

    def test_incremental_summaries_mathematical_properties(self):
        """Test mathematical properties of incremental summaries."""
        # Create test data
        tbr_df = pd.DataFrame(
            {
                "period": [0, 1, 1, 1],
                "cumdif": [np.nan, 3, 7, 10],
                "cumsd": [0, 2.0, 3.5, 4.8],
            }
        )

        # Parameters
        alpha, beta, sigma = 80.0, 1.1, 18.0
        var_alpha, var_beta, cov_alpha_beta = 60.0, 0.0004, -0.02
        degrees_freedom, level, threshold = 25, 0.90, 0.0

        summaries = create_incremental_tbr_summaries(
            tbr_df,
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

        # Verify increasing estimates (cumulative property)
        estimates = summaries["estimate"].values
        assert all(estimates[i] <= estimates[i + 1] for i in range(len(estimates) - 1))

        # Verify increasing precision (uncertainty grows)
        precisions = summaries["precision"].values
        assert all(
            precisions[i] <= precisions[i + 1] for i in range(len(precisions) - 1)
        )

    def test_incremental_summaries_backward_compatibility(self):
        """Test backward compatibility with functional implementation."""
        from tbr.functional.tbr_functions import (
            create_incremental_tbr_summaries as func_create,
        )

        # Create test data
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 1, 1, 1],
                "cumdif": [np.nan, np.nan, 4, 9, 13],
                "cumsd": [0, 0, 2.2, 3.6, 4.7],
            }
        )

        # Parameters
        alpha, beta, sigma = 65.0, 0.88, 26.0
        var_alpha, var_beta, cov_alpha_beta = 75.0, 0.0007, -0.028
        degrees_freedom, level, threshold = 32, 0.85, 1.5

        # Create summaries using both implementations
        core_summaries = create_incremental_tbr_summaries(
            tbr_df,
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
        func_summaries = func_create(
            tbr_df,
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

        # Verify identical results
        key_cols = ["estimate", "precision", "lower", "upper", "prob", "test_day"]
        for col in key_cols:
            core_vals = core_summaries[col].values
            func_vals = func_summaries[col].values
            np.testing.assert_array_almost_equal(core_vals, func_vals, decimal=10)


class TestEffectsModuleIntegration:
    """Tests for effects module integration and workflow."""

    def test_effects_workflow_integration(self):
        """Test integrated workflow using effects functions."""
        # Create realistic test data
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 0, 1, 1, 1, 1, 1],
                "cumdif": [np.nan, np.nan, np.nan, 2.5, 6.2, 9.8, 14.1, 18.5],
                "cumsd": [0, 0, 0, 2.1, 3.4, 4.5, 5.4, 6.2],
            }
        )

        # Model parameters
        alpha, beta, sigma = 100.0, 0.95, 20.0
        var_alpha, var_beta, cov_alpha_beta = 50.0, 0.0005, -0.02
        degrees_freedom, level, threshold = 45, 0.80, 5.0

        # Create both summary types
        summary = create_tbr_summary(
            tbr_df,
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

        incremental = create_incremental_tbr_summaries(
            tbr_df,
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

        # Verify consistency between summary types
        final_estimate_summary = summary["estimate"].iloc[0]
        final_estimate_incremental = incremental["estimate"].iloc[-1]
        assert abs(final_estimate_summary - final_estimate_incremental) < 1e-10

        # Verify reasonable results
        assert 15 < final_estimate_summary < 25  # Should be around 18.5
        assert summary["precision"].iloc[0] > 0
        assert len(incremental) == 5  # Five test period days

    def test_effects_module_imports(self):
        """Test that effects functions can be imported from core module."""
        from tbr.core import create_incremental_tbr_summaries, create_tbr_summary

        # Verify functions are callable
        assert callable(create_tbr_summary)
        assert callable(create_incremental_tbr_summaries)

        # Verify they're the same functions
        from tbr.core.effects import (
            create_incremental_tbr_summaries as direct_incremental,
        )
        from tbr.core.effects import create_tbr_summary as direct_summary

        assert create_tbr_summary is direct_summary
        assert create_incremental_tbr_summaries is direct_incremental


class TestEffectsModuleEdgeCases:
    """Tests for edge cases and error handling in effects module."""

    def test_cumulative_standard_deviation_single_value(self):
        """Test cumulative standard deviation with single value."""
        x_values = np.array([1000])
        sigma = 25.0
        var_alpha = 100.0
        var_beta = 0.001
        cov_alpha_beta = -0.05

        result = calculate_cumulative_standard_deviation(
            x_values, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        assert len(result) == 1
        assert result[0] > 0

    def test_effects_functions_mathematical_consistency(self):
        """Test mathematical consistency across effects functions."""
        # Create test data
        tbr_df = pd.DataFrame(
            {
                "period": [0, 1, 1, 1],
                "cumdif": [np.nan, 5, 12, 20],
                "cumsd": [0, 3.0, 4.5, 6.0],
            }
        )

        # Parameters
        alpha, beta, sigma = 50.0, 1.0, 15.0
        var_alpha, var_beta, cov_alpha_beta = 40.0, 0.0003, -0.015
        degrees_freedom, level, threshold = 30, 0.80, 0.0

        # Create both summary types
        summary = create_tbr_summary(
            tbr_df,
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

        incremental = create_incremental_tbr_summaries(
            tbr_df,
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

        # Verify final estimates match
        final_summary = summary["estimate"].iloc[0]
        final_incremental = incremental["estimate"].iloc[-1]
        assert abs(final_summary - final_incremental) < 1e-10

        # Verify final standard errors match
        final_se_summary = (
            summary["precision"].iloc[0] / 1.282
        )  # Approximate for 80% CI
        final_se_incremental = incremental["precision"].iloc[-1] / 1.282
        assert (
            abs(final_se_summary - final_se_incremental) < 0.1
        )  # Allow small numerical differences
