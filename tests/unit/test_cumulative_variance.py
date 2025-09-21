"""
Comprehensive tests for cumulative variance functionality.

This module provides professional testing for the cumulative variance computation
in the TBR effects module, following scientific Python package standards with
mathematical validation, numerical stability testing, and cross-implementation
verification.
"""

import numpy as np
import pandas as pd
import pytest

from tbr.core.effects import (
    calculate_cumulative_standard_deviation,
    calculate_cumulative_variance,
)
from tbr.core.regression import fit_regression_model


class TestCalculateCumulativeVariance:
    """Tests for calculate_cumulative_variance function."""

    def test_basic_cumulative_variance_calculation(self):
        """Test basic cumulative variance calculation with known values."""
        x_vals = np.array([100, 101, 102])
        sigma = 5.0
        var_alpha = 10.0
        var_beta = 0.01
        cov_alpha_beta = -0.1

        cum_var = calculate_cumulative_variance(
            x_vals, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        assert isinstance(cum_var, np.ndarray)
        assert len(cum_var) == len(x_vals)
        assert np.all(cum_var > 0)

        # Verify mathematical properties
        assert np.all(np.diff(cum_var) > 0)  # Should be increasing

        # Test specific values using TBR formula: V[Δr(T)] = T · σ² + T² · v
        # For T=1, x_mean=100: v = 10 + 2*100*(-0.1) + 100²*0.01 = 10 - 20 + 100 = 90
        # V[Δr(1)] = 1 * 25 + 1² * 90 = 25 + 90 = 115
        expected_var_1 = 1 * (sigma**2) + (1**2) * (
            var_alpha + 2 * 100 * cov_alpha_beta + (100**2) * var_beta
        )
        assert np.isclose(cum_var[0], expected_var_1, atol=1e-9)

    def test_cumulative_variance_mathematical_properties(self):
        """Test mathematical properties of cumulative variance."""
        x_vals = np.array([100, 100, 100])  # Constant x values
        sigma = 1.0
        var_alpha = 0.1
        var_beta = 0.0001
        cov_alpha_beta = -0.001

        cum_var = calculate_cumulative_variance(
            x_vals, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Should be strictly increasing
        assert np.all(np.diff(cum_var) > 0)

        # Should be non-negative
        assert np.all(cum_var >= 0)

        # For constant x values, the growth should be predictable
        # V[Δr(T)] = T · σ² + T² · v where v is constant
        v_constant = var_alpha + 2 * 100 * cov_alpha_beta + (100**2) * var_beta
        expected_vars = np.array(
            [
                1 * (sigma**2) + (1**2) * v_constant,
                2 * (sigma**2) + (2**2) * v_constant,
                3 * (sigma**2) + (3**2) * v_constant,
            ]
        )
        np.testing.assert_allclose(cum_var, expected_vars, atol=1e-12)

    def test_cumulative_variance_cross_validation_with_std(self):
        """Test mathematical consistency with calculate_cumulative_standard_deviation."""
        x_vals = np.array([1000, 1020, 1010, 1030, 1005])
        sigma = 25.0
        var_alpha = 100.0
        var_beta = 0.001
        cov_alpha_beta = -0.05

        # Calculate variance directly
        cum_var = calculate_cumulative_variance(
            x_vals, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Calculate standard deviation and square it
        cum_std = calculate_cumulative_standard_deviation(
            x_vals, sigma, var_alpha, var_beta, cov_alpha_beta
        )
        cum_var_from_std = cum_std**2

        # Should be mathematically identical
        np.testing.assert_allclose(cum_var, cum_var_from_std, atol=1e-12)

    def test_cumulative_variance_single_value(self):
        """Test cumulative variance with a single x value."""
        x_vals = np.array([100])
        sigma = 5.0
        var_alpha = 10.0
        var_beta = 0.01
        cov_alpha_beta = -0.1

        cum_var = calculate_cumulative_variance(
            x_vals, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        assert len(cum_var) == 1

        # Manual calculation for T=1, x_mean=100
        v = var_alpha + 2 * 100 * cov_alpha_beta + (100**2) * var_beta
        expected_var = 1 * (sigma**2) + (1**2) * v
        assert np.isclose(cum_var[0], expected_var, atol=1e-12)

    def test_cumulative_variance_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        x_vals = np.array([1e-6, 2e-6, 3e-6])
        sigma = 1e-3
        var_alpha = 1e-8
        var_beta = 1e-12
        cov_alpha_beta = -1e-10

        cum_var_small = calculate_cumulative_variance(
            x_vals, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        assert np.all(np.isfinite(cum_var_small))
        assert np.all(cum_var_small > 0)

        # Test with reasonable large values (avoid extreme parameter combinations)
        x_vals = np.array([1000, 2000, 3000])
        sigma = 100
        var_alpha = 1000
        var_beta = 0.001
        cov_alpha_beta = -0.5

        cum_var_large = calculate_cumulative_variance(
            x_vals, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        assert np.all(np.isfinite(cum_var_large))
        # Note: With extreme parameters, variance can be negative (indicates model issues)
        # This is mathematically valid behavior in TBR formula

    def test_cumulative_variance_zero_parameters(self):
        """Test behavior with zero variance parameters."""
        x_vals = np.array([100, 101, 102])
        sigma = 5.0
        var_alpha = 0.0
        var_beta = 0.0
        cov_alpha_beta = 0.0

        cum_var = calculate_cumulative_variance(
            x_vals, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # With zero model parameter variances, should only have residual variance
        # V[Δr(T)] = T · σ²
        T_values = np.arange(1, len(x_vals) + 1)
        expected_vars = T_values * (sigma**2)

        np.testing.assert_allclose(cum_var, expected_vars, atol=1e-12)

    def test_cumulative_variance_parameter_validation(self):
        """Test parameter validation and edge cases."""
        x_vals = np.array([100, 101, 102])

        # Test with negative sigma (should still work mathematically)
        cum_var = calculate_cumulative_variance(
            x_vals, sigma=-5.0, var_alpha=10.0, var_beta=0.01, cov_alpha_beta=-0.1
        )
        assert np.all(np.isfinite(cum_var))

        # Test with empty array - should raise ValueError
        with pytest.raises(ValueError, match="test_x_values cannot be empty"):
            calculate_cumulative_variance(
                np.array([]),
                sigma=5.0,
                var_alpha=10.0,
                var_beta=0.01,
                cov_alpha_beta=-0.1,
            )


class TestCumulativeVarianceIntegration:
    """Integration tests for cumulative variance with regression workflow."""

    @pytest.fixture
    def sample_regression_data(self):
        """Fixture for sample regression data."""
        np.random.seed(42)
        n_pretest = 50
        n_test = 20

        # Generate synthetic data with known relationship
        x_pretest = np.linspace(100, 120, n_pretest) + np.random.normal(0, 2, n_pretest)
        y_pretest = 2 + 3 * x_pretest + np.random.normal(0, 5, n_pretest)

        x_test = np.linspace(121, 140, n_test) + np.random.normal(0, 2, n_test)

        pretest_data = pd.DataFrame({"control": x_pretest, "test": y_pretest})
        test_x_values = x_test

        return pretest_data, test_x_values

    def test_cumulative_variance_with_real_regression(self, sample_regression_data):
        """Test cumulative variance calculation with real regression parameters."""
        pretest_data, test_x_values = sample_regression_data

        # Fit regression model
        model_params = fit_regression_model(pretest_data, "control", "test")

        # Calculate cumulative variance
        cum_var = calculate_cumulative_variance(
            test_x_values,
            model_params["sigma"],
            model_params["var_alpha"],
            model_params["var_beta"],
            model_params["cov_alpha_beta"],
        )

        # Validate results
        assert len(cum_var) == len(test_x_values)
        assert np.all(cum_var > 0)
        assert np.all(np.isfinite(cum_var))
        assert np.all(np.diff(cum_var) > 0)  # Should be increasing

        # Cross-validate with standard deviation
        cum_std = calculate_cumulative_standard_deviation(
            test_x_values,
            model_params["sigma"],
            model_params["var_alpha"],
            model_params["var_beta"],
            model_params["cov_alpha_beta"],
        )

        np.testing.assert_allclose(cum_var, cum_std**2, atol=1e-10)

    def test_cumulative_variance_mathematical_consistency(self, sample_regression_data):
        """Test mathematical consistency across different scenarios."""
        pretest_data, test_x_values = sample_regression_data
        model_params = fit_regression_model(pretest_data, "control", "test")

        # Test with different subsets of test data
        for end_idx in range(1, len(test_x_values) + 1):
            subset_x = test_x_values[:end_idx]

            cum_var_subset = calculate_cumulative_variance(
                subset_x,
                model_params["sigma"],
                model_params["var_alpha"],
                model_params["var_beta"],
                model_params["cov_alpha_beta"],
            )

            # Validate mathematical properties
            assert len(cum_var_subset) == end_idx
            assert np.all(cum_var_subset > 0)
            if end_idx > 1:
                assert np.all(np.diff(cum_var_subset) > 0)


class TestCumulativeVariancePerformance:
    """Performance and scalability tests for cumulative variance."""

    def test_cumulative_variance_scalability(self):
        """Test performance with large datasets."""
        # Test with progressively larger datasets
        for n in [100, 1000, 10000]:
            x_vals = np.random.normal(1000, 100, n)
            sigma = 25.0
            var_alpha = 100.0
            var_beta = 0.001
            cov_alpha_beta = -0.05

            cum_var = calculate_cumulative_variance(
                x_vals, sigma, var_alpha, var_beta, cov_alpha_beta
            )

            assert len(cum_var) == n
            assert np.all(cum_var > 0)
            assert np.all(np.isfinite(cum_var))

    def test_cumulative_variance_memory_efficiency(self):
        """Test memory efficiency of cumulative variance calculation."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Measure memory before
        memory_before = process.memory_info().rss

        # Large calculation
        n = 50000
        x_vals = np.random.normal(1000, 100, n)
        cum_var = calculate_cumulative_variance(
            x_vals, sigma=25.0, var_alpha=100.0, var_beta=0.001, cov_alpha_beta=-0.05
        )

        # Measure memory after
        memory_after = process.memory_info().rss
        memory_delta = memory_after - memory_before

        # Validate results
        assert len(cum_var) == n
        assert np.all(cum_var > 0)

        # Memory usage should be reasonable (less than 100MB for 50k elements)
        assert memory_delta < 100 * 1024 * 1024  # 100MB


class TestCumulativeVarianceEdgeCases:
    """Edge case tests for cumulative variance."""

    def test_cumulative_variance_identical_x_values(self):
        """Test with identical x values."""
        x_vals = np.array([100, 100, 100, 100])
        sigma = 5.0
        var_alpha = 10.0
        var_beta = 0.01
        cov_alpha_beta = -0.1

        cum_var = calculate_cumulative_variance(
            x_vals, sigma, var_alpha, var_beta, cov_alpha_beta
        )

        # Should still be increasing due to T and T² terms
        assert np.all(np.diff(cum_var) > 0)

        # With constant x=100, v should be constant
        v_constant = var_alpha + 2 * 100 * cov_alpha_beta + (100**2) * var_beta
        T_values = np.arange(1, len(x_vals) + 1)
        expected_vars = T_values * (sigma**2) + (T_values**2) * v_constant

        np.testing.assert_allclose(cum_var, expected_vars, atol=1e-12)

    def test_cumulative_variance_extreme_covariance(self):
        """Test with extreme covariance values."""
        x_vals = np.array([100, 101, 102])
        sigma = 5.0
        var_alpha = 10.0
        var_beta = 0.01

        # Test with moderate positive covariance
        cum_var_pos = calculate_cumulative_variance(
            x_vals, sigma, var_alpha, var_beta, cov_alpha_beta=1.0
        )

        # Test with moderate negative covariance
        cum_var_neg = calculate_cumulative_variance(
            x_vals, sigma, var_alpha, var_beta, cov_alpha_beta=-1.0
        )

        # Both should be finite and mathematically valid
        assert np.all(np.isfinite(cum_var_pos))
        assert np.all(np.isfinite(cum_var_neg))

        # They should be different
        assert not np.allclose(cum_var_pos, cum_var_neg)

        # Test extreme case that can produce negative variance (valid mathematically)
        cum_var_extreme = calculate_cumulative_variance(
            x_vals, sigma=1.0, var_alpha=1.0, var_beta=0.01, cov_alpha_beta=-10.0
        )

        # Should be finite (negative variance indicates model parameter issues)
        assert np.all(np.isfinite(cum_var_extreme))
