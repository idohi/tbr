"""
Test suite for regression model functions.

This module tests all regression model fitting, parameter validation,
and defensive programming aspects of the TBR regression implementation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tbr.functional.tbr_functions import fit_tbr_regression_model


class TestRegressionModelFitting:
    """Test regression model fitting functionality."""

    def test_basic_regression_fit(self):
        """Test basic regression model fitting."""
        learning_data = pd.DataFrame(
            {"control": [100, 110, 120, 130, 140], "test": [200, 220, 240, 260, 280]}
        )

        result = fit_tbr_regression_model(learning_data, "control", "test")

        # Check structure
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
        assert all(key in result for key in required_keys)

        # Check reasonable values
        assert isinstance(result["alpha"], float)
        assert isinstance(result["beta"], float)
        assert result["sigma"] > 0
        assert result["var_alpha"] > 0
        assert result["var_beta"] > 0
        assert result["degrees_freedom"] == 3  # n - 2 = 5 - 2 = 3
        assert result["n_pretest"] == 5

    def test_perfect_correlation(self):
        """Test regression with perfect correlation."""
        learning_data = pd.DataFrame(
            {
                "control": [1, 2, 3, 4, 5],
                "test": [2, 4, 6, 8, 10],  # Perfect 2x relationship
            }
        )

        result = fit_tbr_regression_model(learning_data, "control", "test")

        # Should have beta close to 2, alpha close to 0
        assert result["beta"] == pytest.approx(2.0, abs=1e-10)
        assert result["alpha"] == pytest.approx(0.0, abs=1e-10)
        assert result["sigma"] == pytest.approx(0.0, abs=1e-10)

    def test_constant_control_values(self):
        """Test error when control values are constant."""
        learning_data = pd.DataFrame(
            {
                "control": [100, 100, 100, 100, 100],  # All same
                "test": [200, 220, 240, 260, 280],
            }
        )

        with pytest.raises(ValueError, match="Control group values are constant"):
            fit_tbr_regression_model(learning_data, "control", "test")

    def test_insufficient_data(self):
        """Test error with insufficient data points."""
        learning_data = pd.DataFrame({"control": [100, 110], "test": [200, 220]})

        with pytest.raises(ValueError, match="Insufficient learning data"):
            fit_tbr_regression_model(learning_data, "control", "test")

    def test_minimum_data_points(self):
        """Test with minimum required data points (3)."""
        learning_data = pd.DataFrame(
            {
                "control": [100, 110, 120],
                "test": [200, 220, 241],  # Slight noise to avoid sigma=0
            }
        )

        result = fit_tbr_regression_model(learning_data, "control", "test")

        # Should work with minimum data
        assert result["degrees_freedom"] == 1  # 3 - 2 = 1
        assert result["n_pretest"] == 3

    def test_large_dataset(self):
        """Test with larger dataset."""
        np.random.seed(42)  # For reproducibility
        n = 100
        control_vals = np.random.normal(1000, 100, n)
        test_vals = 50 + 1.2 * control_vals + np.random.normal(0, 20, n)

        learning_data = pd.DataFrame({"control": control_vals, "test": test_vals})

        result = fit_tbr_regression_model(learning_data, "control", "test")

        # Check that we get reasonable estimates (allowing for random variation)
        assert result["beta"] == pytest.approx(
            1.2, abs=0.2
        )  # Should be close to true value
        assert result["alpha"] == pytest.approx(
            50, abs=50
        )  # Should be close to true value (wider tolerance)
        assert result["degrees_freedom"] == 98  # 100 - 2
        assert result["n_pretest"] == 100


class TestRegressionParameterValidation:
    """Test regression parameter validation and defensive programming."""

    def test_non_finite_parameters_validation(self):
        """Test validation of non-finite regression parameters."""
        df_valid = pd.DataFrame(
            {"control": [1, 2, 3, 4, 5], "test": [10, 20, 30, 40, 50]}
        )

        # Mock statsmodels to return NaN parameters
        with patch("tbr.core.regression.sm.OLS") as mock_ols:
            mock_model = MagicMock()
            mock_model.params = [np.nan, 1.5]  # alpha=NaN
            mock_model.bse = [0.1, 0.05]
            mock_model.cov_params.return_value = np.array(
                [[0.01, -0.002], [-0.002, 0.0025]]
            )
            mock_model.scale = 4.0
            mock_model.df_resid = 3

            mock_ols.return_value.fit.return_value = mock_model

            with pytest.raises(
                ValueError,
                match="Computed regression parameters contain invalid values",
            ):
                fit_tbr_regression_model(df_valid, "control", "test")

    def test_infinite_parameters_validation(self):
        """Test validation of infinite regression parameters."""
        df_valid = pd.DataFrame(
            {"control": [1, 2, 3, 4, 5], "test": [10, 20, 30, 40, 50]}
        )

        # Mock statsmodels to return infinite beta
        with patch("tbr.core.regression.sm.OLS") as mock_ols:
            mock_model = MagicMock()
            mock_model.params = [5.0, np.inf]  # beta=inf
            mock_model.bse = [0.1, 0.05]
            mock_model.cov_params.return_value = np.array(
                [[0.01, -0.002], [-0.002, 0.0025]]
            )
            mock_model.scale = 4.0
            mock_model.df_resid = 3

            mock_ols.return_value.fit.return_value = mock_model

            with pytest.raises(
                ValueError,
                match="Computed regression parameters contain invalid values",
            ):
                fit_tbr_regression_model(df_valid, "control", "test")

    def test_invalid_sigma_validation(self):
        """Test validation of invalid sigma values."""
        df_valid = pd.DataFrame(
            {"control": [1, 2, 3, 4, 5], "test": [10, 20, 30, 40, 50]}
        )

        # Mock statsmodels to return zero scale (sigma = 0)
        with patch("tbr.core.regression.sm.OLS") as mock_ols:
            mock_model = MagicMock()
            mock_model.params = [5.0, 2.0]
            mock_model.bse = [0.1, 0.05]
            mock_model.cov_params.return_value = np.array(
                [[0.01, -0.002], [-0.002, 0.0025]]
            )
            mock_model.scale = 0.0  # Zero scale -> sigma = 0
            mock_model.df_resid = 3

            mock_ols.return_value.fit.return_value = mock_model

            with pytest.raises(
                ValueError, match="Invalid residual standard deviation: 0"
            ):
                fit_tbr_regression_model(df_valid, "control", "test")

    def test_non_positive_variances_validation(self):
        """Test validation of non-positive coefficient variances."""
        df_valid = pd.DataFrame(
            {"control": [1, 2, 3, 4, 5], "test": [10, 20, 30, 40, 50]}
        )

        # Mock statsmodels to return zero standard error (zero variance)
        with patch("tbr.core.regression.sm.OLS") as mock_ols:
            mock_model = MagicMock()
            mock_model.params = [5.0, 2.0]
            mock_model.bse = [0.0, 0.05]  # Zero std error for alpha
            mock_model.cov_params.return_value = np.array(
                [[0.01, -0.002], [-0.002, 0.0025]]
            )
            mock_model.scale = 4.0
            mock_model.df_resid = 3

            mock_ols.return_value.fit.return_value = mock_model

            with pytest.raises(
                ValueError, match="Computed coefficient variances are non-positive"
            ):
                fit_tbr_regression_model(df_valid, "control", "test")

    def test_comprehensive_parameter_validation(self):
        """Test comprehensive parameter validation scenarios."""
        df_base = pd.DataFrame(
            {"control": [2, 4, 6, 8, 10], "test": [20, 40, 60, 80, 100]}
        )

        # Test multiple invalid parameter scenarios
        invalid_scenarios = [
            # (params, bse, scale, expected_error_pattern)
            ([np.nan, 2.0], [0.1, 0.05], 4.0, "invalid values"),
            ([1.0, np.inf], [0.1, 0.05], 4.0, "invalid values"),
            ([1.0, 2.0], [0.1, 0.05], 0.0, "Invalid residual standard deviation"),
            ([1.0, 2.0], [0.0, 0.05], 4.0, "non-positive"),
            ([1.0, 2.0], [0.1, 0.0], 4.0, "non-positive"),
        ]

        for params, bse, scale, error_pattern in invalid_scenarios:
            with patch("tbr.core.regression.sm.OLS") as mock_ols:
                mock_model = MagicMock()
                mock_model.params = params
                mock_model.bse = bse
                mock_model.cov_params.return_value = np.array(
                    [[0.01, -0.002], [-0.002, 0.0025]]
                )
                mock_model.scale = scale
                mock_model.df_resid = 3

                mock_ols.return_value.fit.return_value = mock_model

                with pytest.raises(ValueError, match=error_pattern):
                    fit_tbr_regression_model(df_base, "control", "test")


class TestRegressionEdgeCases:
    """Test regression model edge cases and numerical stability."""

    def test_very_small_values(self):
        """Test regression with very small values."""
        learning_data = pd.DataFrame(
            {
                "control": [1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
                "test": [2e-6, 4e-6, 6e-6, 8e-6, 10e-6],
            }
        )

        result = fit_tbr_regression_model(learning_data, "control", "test")

        # Should handle small values correctly
        assert result["beta"] == pytest.approx(2.0, abs=1e-10)
        assert result["alpha"] == pytest.approx(0.0, abs=1e-12)
        assert np.isfinite(result["sigma"])
        assert result["sigma"] >= 0

    def test_very_large_values(self):
        """Test regression with very large values."""
        learning_data = pd.DataFrame(
            {"control": [1e6, 2e6, 3e6, 4e6, 5e6], "test": [2e6, 4e6, 6e6, 8e6, 10e6]}
        )

        result = fit_tbr_regression_model(learning_data, "control", "test")

        # Should handle large values correctly
        assert result["beta"] == pytest.approx(2.0, rel=1e-10)
        assert result["alpha"] == pytest.approx(0.0, abs=1e-6)
        assert np.isfinite(result["sigma"])
        assert result["sigma"] >= 0

    def test_negative_values(self):
        """Test regression with negative values."""
        learning_data = pd.DataFrame(
            {"control": [-5, -3, -1, 1, 3], "test": [-10, -6, -2, 2, 6]}
        )

        result = fit_tbr_regression_model(learning_data, "control", "test")

        # Should handle negative values correctly
        assert result["beta"] == pytest.approx(2.0, abs=1e-10)
        assert result["alpha"] == pytest.approx(0.0, abs=1e-10)
        assert np.isfinite(result["sigma"])

    def test_mixed_positive_negative(self):
        """Test regression with mixed positive and negative values."""
        learning_data = pd.DataFrame(
            {"control": [-2, -1, 0, 1, 2], "test": [1, 3, 5, 7, 9]}  # y = 5 + 2x
        )

        result = fit_tbr_regression_model(learning_data, "control", "test")

        # Should correctly estimate intercept and slope
        assert result["beta"] == pytest.approx(2.0, abs=1e-10)
        assert result["alpha"] == pytest.approx(5.0, abs=1e-10)
        assert result["pretest_x_mean"] == pytest.approx(0.0, abs=1e-10)

    def test_noisy_data(self):
        """Test regression with noisy data."""
        np.random.seed(123)  # For reproducibility
        n = 50
        control_vals = np.linspace(10, 100, n)
        # True relationship: test = 20 + 1.5 * control + noise
        noise = np.random.normal(0, 5, n)
        test_vals = 20 + 1.5 * control_vals + noise

        learning_data = pd.DataFrame({"control": control_vals, "test": test_vals})

        result = fit_tbr_regression_model(learning_data, "control", "test")

        # Should recover approximate true parameters
        assert result["beta"] == pytest.approx(1.5, abs=0.1)
        assert result["alpha"] == pytest.approx(20, abs=2)
        assert result["sigma"] > 0  # Should capture noise
        assert result["degrees_freedom"] == 48  # n - 2

    def test_extreme_outliers(self):
        """Test regression robustness with extreme outliers."""
        learning_data = pd.DataFrame(
            {
                "control": [1, 2, 3, 4, 1000],  # Extreme outlier
                "test": [10, 20, 30, 40, 50],  # Normal progression
            }
        )

        result = fit_tbr_regression_model(learning_data, "control", "test")

        # Should still produce finite results
        assert np.isfinite(result["alpha"])
        assert np.isfinite(result["beta"])
        assert np.isfinite(result["sigma"])
        assert result["sigma"] > 0
        assert result["var_alpha"] > 0
        assert result["var_beta"] > 0


class TestRegressionStatisticalProperties:
    """Test statistical properties of regression results."""

    def test_degrees_of_freedom_calculation(self):
        """Test correct degrees of freedom calculation."""
        for n in [3, 5, 10, 20, 100]:
            learning_data = pd.DataFrame(
                {
                    "control": range(1, n + 1),
                    "test": [2 * x + 1 for x in range(1, n + 1)],
                }
            )

            result = fit_tbr_regression_model(learning_data, "control", "test")

            assert result["degrees_freedom"] == n - 2
            assert result["n_pretest"] == n

    def test_variance_positivity(self):
        """Test that all variances are positive."""
        learning_data = pd.DataFrame(
            {"control": [1, 3, 5, 7, 9, 11], "test": [2, 7, 12, 17, 22, 27]}
        )

        result = fit_tbr_regression_model(learning_data, "control", "test")

        # All variances must be positive
        assert result["var_alpha"] > 0
        assert result["var_beta"] > 0
        assert result["sigma"] > 0

    def test_covariance_structure(self):
        """Test covariance structure properties."""
        learning_data = pd.DataFrame(
            {"control": [10, 20, 30, 40, 50], "test": [15, 25, 35, 45, 55]}
        )

        result = fit_tbr_regression_model(learning_data, "control", "test")

        # Covariance should be finite
        assert np.isfinite(result["cov_alpha_beta"])

        # Check that correlation coefficient is within [-1, 1]
        corr = result["cov_alpha_beta"] / np.sqrt(
            result["var_alpha"] * result["var_beta"]
        )
        assert -1 <= corr <= 1

    def test_mean_calculation(self):
        """Test x_mean calculation accuracy."""
        control_values = [5, 15, 25, 35, 45]
        learning_data = pd.DataFrame(
            {
                "control": control_values,
                "test": [10, 30, 51, 70, 90],  # Slight noise to avoid sigma=0
            }
        )

        result = fit_tbr_regression_model(learning_data, "control", "test")

        expected_mean = np.mean(control_values)
        assert result["pretest_x_mean"] == pytest.approx(expected_mean, rel=1e-10)
        assert result["pretest_x_mean"] == 25.0  # Known result


class TestRegressionDefensiveProgramming:
    """Test comprehensive defensive programming scenarios that achieve 100% coverage."""

    def test_all_non_finite_parameter_scenarios(self):
        """Comprehensive test for all non-finite parameter scenarios."""
        df_base = pd.DataFrame(
            {"control": [2, 4, 6, 8, 10], "test": [20, 40, 60, 80, 100]}
        )

        # Test scenarios for each parameter that could be non-finite
        non_finite_scenarios = [
            # (alpha, beta, var_alpha, var_beta, cov_alpha_beta, scale)
            (np.nan, 2.0, 0.01, 0.0025, -0.002, 4.0),  # NaN alpha
            (1.0, np.inf, 0.01, 0.0025, -0.002, 4.0),  # Infinite beta
            (1.0, 2.0, 0.01, 0.0025, -0.002, np.inf),  # Infinite scale
            (-np.inf, 2.0, 0.01, 0.0025, -0.002, 4.0),  # Negative infinite alpha
        ]

        for (
            alpha,
            beta,
            var_alpha,
            var_beta,
            cov_ab,
            scale,
        ) in non_finite_scenarios:
            with patch("tbr.core.regression.sm.OLS") as mock_ols:
                mock_model = MagicMock()
                mock_model.params = [alpha, beta]
                mock_model.bse = [np.sqrt(var_alpha), np.sqrt(var_beta)]
                mock_model.cov_params.return_value = np.array(
                    [[var_alpha, cov_ab], [cov_ab, var_beta]]
                )
                mock_model.scale = scale
                mock_model.df_resid = 3

                mock_ols.return_value.fit.return_value = mock_model

                with pytest.raises(
                    ValueError,
                    match="Computed regression parameters contain invalid values",
                ):
                    fit_tbr_regression_model(df_base, "control", "test")

    def test_all_invalid_sigma_scenarios(self):
        """Comprehensive test for all invalid sigma scenarios."""
        df_base = pd.DataFrame(
            {"control": [4, 8, 12, 16, 20], "test": [40, 80, 120, 160, 200]}
        )

        # Test scenarios that should trigger sigma validation
        invalid_sigma_scenarios = [
            (0.0, 0.0),  # Zero scale -> zero sigma
        ]

        for scale, expected_sigma in invalid_sigma_scenarios:
            with patch("tbr.core.regression.sm.OLS") as mock_ols:
                mock_model = MagicMock()
                mock_model.params = [20.0, 5.0]
                mock_model.bse = [0.1, 0.05]
                mock_model.cov_params.return_value = np.array(
                    [[0.01, -0.002], [-0.002, 0.0025]]
                )
                mock_model.scale = scale
                mock_model.df_resid = 3

                mock_ols.return_value.fit.return_value = mock_model

                with pytest.raises(
                    ValueError,
                    match=f"Invalid residual standard deviation: {expected_sigma}",
                ):
                    fit_tbr_regression_model(df_base, "control", "test")

    def test_all_non_positive_variance_scenarios(self):
        """Comprehensive test for all non-positive variance scenarios."""
        df_base = pd.DataFrame(
            {"control": [5, 10, 15, 20, 25], "test": [50, 100, 150, 200, 250]}
        )

        # Test scenarios that should trigger variance validation
        non_positive_variance_scenarios = [
            (0.0, 0.05),  # Zero var_alpha
            (0.1, 0.0),  # Zero var_beta
            (0.0, 0.0),  # Both variances zero
        ]

        for bse_alpha, bse_beta in non_positive_variance_scenarios:
            with patch("tbr.core.regression.sm.OLS") as mock_ols:
                mock_model = MagicMock()
                mock_model.params = [25.0, 6.0]
                mock_model.bse = [bse_alpha, bse_beta]
                mock_model.cov_params.return_value = np.array(
                    [[0.01, -0.002], [-0.002, 0.0025]]
                )
                mock_model.scale = 4.0
                mock_model.df_resid = 3

                mock_ols.return_value.fit.return_value = mock_model

                with pytest.raises(
                    ValueError, match="Computed coefficient variances are non-positive"
                ):
                    fit_tbr_regression_model(df_base, "control", "test")

    def test_comprehensive_edge_case_coverage(self):
        """Comprehensive test for remaining edge cases and defensive programming."""
        import warnings

        df_base = pd.DataFrame(
            {"control": [1, 2, 3, 4, 5], "test": [10, 20, 30, 40, 50]}
        )

        # Test scenario 1: Mock to trigger line 682 (non-finite parameters)
        with patch("tbr.core.regression.sm.OLS") as mock_ols:
            mock_model = MagicMock()
            mock_model.params = [np.inf, 2.0]  # Infinite alpha
            mock_model.bse = [0.1, 0.05]
            mock_model.cov_params.return_value = np.array(
                [[0.01, -0.002], [-0.002, 0.0025]]
            )
            mock_model.scale = 4.0
            mock_model.df_resid = 3

            mock_ols.return_value.fit.return_value = mock_model

            with pytest.raises(
                ValueError,
                match="Computed regression parameters contain invalid values",
            ):
                fit_tbr_regression_model(df_base, "control", "test")

        # Test scenario 2: Mock to trigger line 685 (sigma <= 0)
        with patch("tbr.core.regression.sm.OLS") as mock_ols:
            mock_model = MagicMock()
            mock_model.params = [5.0, 2.0]
            mock_model.bse = [0.1, 0.05]
            mock_model.cov_params.return_value = np.array(
                [[0.01, -0.002], [-0.002, 0.0025]]
            )
            mock_model.scale = 0.0  # Zero scale -> sigma = 0
            mock_model.df_resid = 3

            mock_ols.return_value.fit.return_value = mock_model

            with pytest.raises(
                ValueError, match="Invalid residual standard deviation: 0"
            ):
                fit_tbr_regression_model(df_base, "control", "test")

        # Test scenario 3: Mock to trigger line 688 (non-positive variances)
        with patch("tbr.core.regression.sm.OLS") as mock_ols:
            mock_model = MagicMock()
            mock_model.params = [5.0, 2.0]
            mock_model.bse = [0.0, 0.05]  # Zero std error for alpha
            mock_model.cov_params.return_value = np.array(
                [[0.01, -0.002], [-0.002, 0.0025]]
            )
            mock_model.scale = 4.0
            mock_model.df_resid = 3

            mock_ols.return_value.fit.return_value = mock_model

            with pytest.raises(
                ValueError, match="Computed coefficient variances are non-positive"
            ):
                fit_tbr_regression_model(df_base, "control", "test")

        # Test scenario 4: Mock negative scale to trigger sqrt warning (line 685 alternative)
        with patch("tbr.core.regression.sm.OLS") as mock_ols:
            mock_model = MagicMock()
            mock_model.params = [5.0, 2.0]
            mock_model.bse = [0.1, 0.05]
            mock_model.cov_params.return_value = np.array(
                [[0.01, -0.002], [-0.002, 0.0025]]
            )
            mock_model.scale = -1.0  # Negative scale
            mock_model.df_resid = 3

            mock_ols.return_value.fit.return_value = mock_model

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with pytest.raises(
                    ValueError,
                    match="Computed regression parameters contain invalid values",
                ):
                    fit_tbr_regression_model(df_base, "control", "test")
