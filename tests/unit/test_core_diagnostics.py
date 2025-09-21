"""
Tests for Core Diagnostics Module.

This module tests the comprehensive diagnostic functionality for TBR regression
models, including goodness-of-fit metrics, residual analysis, and statistical
assumption testing.

The tests verify mathematical accuracy, statistical validity, and proper
handling of edge cases for all diagnostic functions.
"""

import numpy as np
import pandas as pd
import pytest

from tbr.core.diagnostics import (
    calculate_goodness_of_fit,
    calculate_information_criteria,
    calculate_residuals,
    calculate_standardized_residuals,
    calculate_studentized_residuals,
    check_homoscedasticity,
    check_independence,
    check_normality,
    create_diagnostic_summary,
    validate_model_assumptions,
)
from tbr.core.regression import fit_regression_model


class TestCalculateResiduals:
    """Test residual calculation functions."""

    def test_basic_residual_calculation(self):
        """Test basic residual calculation with known data."""
        # Create test data with known relationship: y = 5 + 2*x
        data = pd.DataFrame(
            {
                "control": [1, 2, 3, 4, 5],
                "test": [7, 9, 11, 13, 15],  # Perfect linear relationship
            }
        )

        # Fit model
        model_params = fit_regression_model(data, "control", "test")

        # Calculate residuals
        residuals = calculate_residuals(data, model_params, "control", "test")

        # With perfect linear relationship, residuals should be very small
        assert len(residuals) == 5
        assert np.allclose(
            residuals, 0, atol=1e-10
        ), "Residuals should be near zero for perfect fit"
        assert (
            abs(np.mean(residuals)) < 1e-10
        ), "Mean residual should be approximately zero"

    def test_residual_calculation_with_noise(self):
        """Test residual calculation with noisy data."""
        np.random.seed(42)
        n = 30
        x = np.random.normal(100, 10, n)
        y = 5 + 2 * x + np.random.normal(0, 1, n)  # y = 5 + 2*x + noise

        data = pd.DataFrame({"control": x, "test": y})
        model_params = fit_regression_model(data, "control", "test")

        residuals = calculate_residuals(data, model_params, "control", "test")

        assert len(residuals) == n
        assert abs(np.mean(residuals)) < 0.1, "Mean residual should be close to zero"
        assert (
            np.std(residuals) > 0
        ), "Residuals should have non-zero variance with noise"

    def test_standardized_residuals(self):
        """Test standardized residual calculation."""
        np.random.seed(123)
        data = pd.DataFrame(
            {
                "control": np.random.normal(1000, 50, 25),
                "test": np.random.normal(1020, 55, 25),
            }
        )

        model_params = fit_regression_model(data, "control", "test")
        std_residuals = calculate_standardized_residuals(
            data, model_params, "control", "test"
        )

        # Standardized residuals should have approximately unit variance
        assert len(std_residuals) == 25
        assert (
            0.8 < np.std(std_residuals) < 1.2
        ), "Standardized residuals should have ~unit variance"

    def test_studentized_residuals(self):
        """Test studentized residual calculation."""
        np.random.seed(456)
        data = pd.DataFrame(
            {
                "control": np.random.normal(500, 25, 20),
                "test": np.random.normal(510, 30, 20),
            }
        )

        model_params = fit_regression_model(data, "control", "test")
        student_residuals = calculate_studentized_residuals(
            data, model_params, "control", "test"
        )

        assert len(student_residuals) == 20
        assert np.all(
            np.isfinite(student_residuals)
        ), "All studentized residuals should be finite"
        # Studentized residuals should generally be larger than standardized residuals
        std_residuals = calculate_standardized_residuals(
            data, model_params, "control", "test"
        )
        assert np.mean(np.abs(student_residuals)) >= np.mean(np.abs(std_residuals))


class TestGoodnessOfFit:
    """Test goodness-of-fit metrics calculation."""

    def test_perfect_fit_metrics(self):
        """Test goodness-of-fit metrics with perfect linear relationship."""
        # Perfect linear relationship
        data = pd.DataFrame(
            {
                "control": [1, 2, 3, 4, 5, 6],
                "test": [10, 12, 14, 16, 18, 20],  # y = 8 + 2*x
            }
        )

        model_params = fit_regression_model(data, "control", "test")
        gof = calculate_goodness_of_fit(data, model_params, "control", "test")

        # Perfect fit should have R² ≈ 1
        assert (
            gof["r_squared"] > 0.99
        ), "R-squared should be very close to 1 for perfect fit"
        assert (
            gof["adj_r_squared"] > 0.99
        ), "Adjusted R-squared should be very close to 1"
        assert (
            gof["f_statistic"] > 100
        ), "F-statistic should be very large for perfect fit"
        assert gof["f_p_value"] < 0.001, "F-test p-value should be very small"
        assert gof["mse"] < 1e-10, "MSE should be very small for perfect fit"
        assert gof["rmse"] < 1e-5, "RMSE should be very small for perfect fit"

    def test_noisy_data_metrics(self):
        """Test goodness-of-fit metrics with realistic noisy data."""
        np.random.seed(789)
        n = 50
        x = np.random.normal(100, 15, n)
        y = 20 + 0.5 * x + np.random.normal(0, 5, n)

        data = pd.DataFrame({"control": x, "test": y})
        model_params = fit_regression_model(data, "control", "test")
        gof = calculate_goodness_of_fit(data, model_params, "control", "test")

        # Verify reasonable ranges for noisy data
        assert 0 <= gof["r_squared"] <= 1, "R-squared should be between 0 and 1"
        assert (
            gof["adj_r_squared"] <= gof["r_squared"]
        ), "Adjusted R-squared should be ≤ R-squared"
        assert gof["f_statistic"] > 0, "F-statistic should be positive"
        assert 0 <= gof["f_p_value"] <= 1, "F-test p-value should be between 0 and 1"
        assert gof["mse"] > 0, "MSE should be positive"
        assert gof["rmse"] == np.sqrt(gof["mse"]), "RMSE should equal sqrt(MSE)"

    def test_goodness_of_fit_mathematical_properties(self):
        """Test mathematical properties of goodness-of-fit metrics."""
        np.random.seed(101)
        data = pd.DataFrame(
            {
                "control": np.random.normal(200, 20, 30),
                "test": np.random.normal(205, 25, 30),
            }
        )

        model_params = fit_regression_model(data, "control", "test")
        gof = calculate_goodness_of_fit(data, model_params, "control", "test")

        # Mathematical relationships
        assert gof["adj_r_squared"] <= gof["r_squared"], "Adjusted R² ≤ R²"
        assert gof["rmse"] ** 2 == pytest.approx(gof["mse"], rel=1e-10), "RMSE² = MSE"
        assert 0 <= gof["r_squared"] <= 1, "R² ∈ [0,1]"


class TestInformationCriteria:
    """Test information criteria calculation."""

    def test_information_criteria_calculation(self):
        """Test AIC and BIC calculation."""
        np.random.seed(202)
        data = pd.DataFrame(
            {
                "control": np.random.normal(1000, 100, 40),
                "test": np.random.normal(1050, 110, 40),
            }
        )

        model_params = fit_regression_model(data, "control", "test")
        ic = calculate_information_criteria(data, model_params, "control", "test")

        # Verify all metrics are present and finite
        assert "aic" in ic
        assert "bic" in ic
        assert "log_likelihood" in ic
        assert np.isfinite(ic["aic"])
        assert np.isfinite(ic["bic"])
        assert np.isfinite(ic["log_likelihood"])

        # BIC penalizes complexity more than AIC
        assert (
            ic["bic"] > ic["aic"]
        ), "BIC should be larger than AIC for this sample size"

    def test_information_criteria_mathematical_properties(self):
        """Test mathematical properties of information criteria."""
        np.random.seed(303)
        data = pd.DataFrame(
            {
                "control": np.random.normal(50, 5, 25),
                "test": np.random.normal(52, 6, 25),
            }
        )

        model_params = fit_regression_model(data, "control", "test")
        ic = calculate_information_criteria(data, model_params, "control", "test")

        # Log-likelihood should be negative (for typical regression)
        assert (
            ic["log_likelihood"] < 0
        ), "Log-likelihood typically negative for regression"

        # AIC and BIC should be positive (typical for this formulation)
        assert ic["aic"] > 0, "AIC should be positive"
        assert ic["bic"] > 0, "BIC should be positive"


class TestAssumptionTesting:
    """Test statistical assumption testing functions."""

    def test_normality_test_normal_data(self):
        """Test normality test with normal residuals."""
        np.random.seed(404)
        normal_residuals = np.random.normal(0, 1, 50)

        result = check_normality(normal_residuals)

        assert "statistic" in result
        assert "p_value" in result
        assert "is_normal" in result
        assert "test_name" in result
        assert result["test_name"] == "Shapiro-Wilk"
        assert 0 <= result["p_value"] <= 1, "P-value should be between 0 and 1"
        # With normal data, we expect high p-value (though not guaranteed)
        assert isinstance(result["is_normal"], bool)

    def test_normality_test_non_normal_data(self):
        """Test normality test with clearly non-normal residuals."""
        # Create clearly non-normal data (uniform distribution)
        np.random.seed(505)
        non_normal_residuals = np.random.uniform(-3, 3, 50)

        result = check_normality(non_normal_residuals)

        assert result["test_name"] == "Shapiro-Wilk"
        assert 0 <= result["p_value"] <= 1
        # Uniform data should often fail normality test
        assert isinstance(result["is_normal"], bool)

    def test_homoscedasticity_test(self):
        """Test homoscedasticity (Breusch-Pagan) test."""
        np.random.seed(606)
        # Create data with constant variance (homoscedastic)
        x = np.random.normal(100, 10, 30)
        y = 5 + 2 * x + np.random.normal(0, 2, 30)  # Constant error variance

        data = pd.DataFrame({"control": x, "test": y})
        model_params = fit_regression_model(data, "control", "test")

        result = check_homoscedasticity(data, model_params, "control", "test")

        assert "statistic" in result
        assert "p_value" in result
        assert "is_homoscedastic" in result
        assert "test_name" in result
        assert result["test_name"] == "Breusch-Pagan"
        assert 0 <= result["p_value"] <= 1
        assert isinstance(result["is_homoscedastic"], bool)

    def test_independence_test(self):
        """Test independence (Durbin-Watson) test."""
        np.random.seed(707)
        # Create independent residuals
        independent_residuals = np.random.normal(0, 1, 40)

        result = check_independence(independent_residuals)

        assert "statistic" in result
        assert "interpretation" in result
        assert "is_independent" in result
        assert "test_name" in result
        assert result["test_name"] == "Durbin-Watson"
        assert (
            0 <= result["statistic"] <= 4
        ), "Durbin-Watson statistic should be between 0 and 4"
        assert isinstance(result["is_independent"], bool)
        assert isinstance(result["interpretation"], str)

    def test_independence_test_autocorrelated_data(self):
        """Test independence test with autocorrelated residuals."""
        np.random.seed(808)
        # Create autocorrelated residuals
        n = 50
        residuals = np.zeros(n)
        residuals[0] = np.random.normal(0, 1)
        for i in range(1, n):
            residuals[i] = 0.7 * residuals[i - 1] + np.random.normal(0, 1)

        result = check_independence(residuals)

        assert result["test_name"] == "Durbin-Watson"
        # Autocorrelated data should often show dependence
        assert isinstance(result["is_independent"], bool)
        assert "autocorrelation" in result["interpretation"].lower()


class TestDiagnosticSummary:
    """Test comprehensive diagnostic summary functions."""

    def test_diagnostic_summary_creation(self):
        """Test creation of comprehensive diagnostic summary."""
        np.random.seed(909)
        data = pd.DataFrame(
            {
                "control": np.random.normal(1000, 50, 35),
                "test": np.random.normal(1020, 55, 35),
            }
        )

        model_params = fit_regression_model(data, "control", "test")
        summary = create_diagnostic_summary(data, model_params, "control", "test")

        # Verify all components are present
        assert "goodness_of_fit" in summary
        assert "information_criteria" in summary
        assert "normality_test" in summary
        assert "homoscedasticity_test" in summary
        assert "independence_test" in summary
        assert "overall_validity" in summary
        assert "warnings" in summary

        # Verify types
        assert isinstance(summary["overall_validity"], bool)
        assert isinstance(summary["warnings"], list)

        # Verify nested dictionaries have expected keys
        gof = summary["goodness_of_fit"]
        assert "r_squared" in gof
        assert "f_statistic" in gof

        ic = summary["information_criteria"]
        assert "aic" in ic
        assert "bic" in ic

    def test_model_assumption_validation(self):
        """Test comprehensive model assumption validation."""
        np.random.seed(1010)
        # Create well-behaved data
        x = np.random.normal(100, 15, 40)
        y = 10 + 0.8 * x + np.random.normal(0, 3, 40)

        data = pd.DataFrame({"control": x, "test": y})
        model_params = fit_regression_model(data, "control", "test")

        validation = validate_model_assumptions(data, model_params, "control", "test")

        # Verify all assumption checks are present
        assert "linearity_valid" in validation
        assert "normality_valid" in validation
        assert "homoscedasticity_valid" in validation
        assert "independence_valid" in validation
        assert "all_assumptions_valid" in validation
        assert "significance_level" in validation
        assert "validation_summary" in validation

        # Verify types
        assert isinstance(validation["linearity_valid"], bool)
        assert isinstance(validation["normality_valid"], bool)
        assert isinstance(validation["homoscedasticity_valid"], bool)
        assert isinstance(validation["independence_valid"], bool)
        assert isinstance(validation["all_assumptions_valid"], bool)
        assert isinstance(validation["significance_level"], float)
        assert isinstance(validation["validation_summary"], str)

        # Verify significance level
        assert validation["significance_level"] == 0.05  # Default value

    def test_assumption_validation_custom_alpha(self):
        """Test assumption validation with custom significance level."""
        np.random.seed(1111)
        data = pd.DataFrame(
            {
                "control": np.random.normal(200, 20, 30),
                "test": np.random.normal(210, 25, 30),
            }
        )

        model_params = fit_regression_model(data, "control", "test")
        validation = validate_model_assumptions(
            data, model_params, "control", "test", alpha=0.01
        )

        assert validation["significance_level"] == 0.01
        assert isinstance(validation["all_assumptions_valid"], bool)


class TestDiagnosticsEdgeCases:
    """Test diagnostic functions with edge cases and error conditions."""

    def test_residuals_with_minimum_data(self):
        """Test residual calculation with minimum required data."""
        # Minimum data for regression (3 points)
        data = pd.DataFrame({"control": [1, 2, 3], "test": [2, 4, 6]})

        model_params = fit_regression_model(data, "control", "test")
        residuals = calculate_residuals(data, model_params, "control", "test")

        assert len(residuals) == 3
        assert np.all(np.isfinite(residuals))

    def test_diagnostics_with_perfect_correlation(self):
        """Test diagnostics with perfect linear correlation."""
        data = pd.DataFrame(
            {
                "control": [10, 20, 30, 40, 50],
                "test": [25, 45, 65, 85, 105],  # y = 5 + 2*x (perfect correlation)
            }
        )

        model_params = fit_regression_model(data, "control", "test")

        # Test all diagnostic functions
        residuals = calculate_residuals(data, model_params, "control", "test")
        gof = calculate_goodness_of_fit(data, model_params, "control", "test")
        ic = calculate_information_criteria(data, model_params, "control", "test")

        # Perfect fit should have near-zero residuals and R² ≈ 1
        assert np.allclose(residuals, 0, atol=1e-10)
        assert gof["r_squared"] > 0.999
        assert np.isfinite(ic["aic"])
        assert np.isfinite(ic["bic"])

    def test_diagnostics_integration_workflow(self):
        """Test complete diagnostic workflow integration."""
        np.random.seed(1212)
        # Create realistic dataset
        n = 45
        x = np.random.normal(500, 50, n)
        y = 100 + 0.6 * x + np.random.normal(0, 10, n)

        data = pd.DataFrame({"control": x, "test": y})

        # Complete workflow
        model_params = fit_regression_model(data, "control", "test")

        # Calculate all diagnostic metrics
        residuals = calculate_residuals(data, model_params, "control", "test")
        std_residuals = calculate_standardized_residuals(
            data, model_params, "control", "test"
        )
        student_residuals = calculate_studentized_residuals(
            data, model_params, "control", "test"
        )
        gof = calculate_goodness_of_fit(data, model_params, "control", "test")
        ic = calculate_information_criteria(data, model_params, "control", "test")

        # Test assumption tests
        normality = check_normality(residuals)
        homoscedasticity = check_homoscedasticity(data, model_params, "control", "test")
        independence = check_independence(residuals)

        # Verify all tests return proper structures
        assert "test_name" in normality
        assert "test_name" in homoscedasticity
        assert "test_name" in independence

        # Create comprehensive summary
        summary = create_diagnostic_summary(data, model_params, "control", "test")
        validation = validate_model_assumptions(data, model_params, "control", "test")

        # Verify all components work together
        assert len(residuals) == n
        assert len(std_residuals) == n
        assert len(student_residuals) == n
        assert all(np.isfinite([gof["r_squared"], ic["aic"], normality["p_value"]]))
        assert isinstance(summary["overall_validity"], bool)
        assert isinstance(validation["all_assumptions_valid"], bool)

        # Verify mathematical consistency
        assert np.abs(np.mean(residuals)) < 0.5, "Mean residual should be close to zero"
        assert 0 <= gof["r_squared"] <= 1, "R² should be in valid range"
        assert (
            ic["bic"] > ic["aic"]
        ), "BIC should be larger than AIC for this sample size"


class TestDiagnosticsErrorHandling:
    """Test error handling in diagnostic functions."""

    def test_empty_data_error(self):
        """Test error handling with empty data."""
        empty_data = pd.DataFrame({"control": [], "test": []})

        with pytest.raises(
            ValueError, match="Insufficient learning data.*0 observations"
        ):
            # This should fail at the model fitting stage
            fit_regression_model(empty_data, "control", "test")

    def test_insufficient_data_error(self):
        """Test error handling with insufficient data."""
        insufficient_data = pd.DataFrame({"control": [1, 2], "test": [3, 4]})

        with pytest.raises(
            ValueError, match="Insufficient learning data.*2 observations"
        ):
            fit_regression_model(insufficient_data, "control", "test")

    def test_normality_test_empty_residuals(self):
        """Test normality test error handling with empty residuals."""
        empty_residuals = np.array([])

        with pytest.raises(ValueError, match="residuals.*empty"):
            check_normality(empty_residuals)

    def test_independence_test_insufficient_data(self):
        """Test independence test with insufficient data."""
        insufficient_residuals = np.array([1.0, 2.0])  # Only 2 points

        with pytest.raises(ValueError, match="residuals.*at least 3"):
            check_independence(insufficient_residuals)


class TestDiagnosticsCoverageCompletion:
    """Tests to achieve 100% coverage of diagnostic functions."""

    def test_diagnostic_summary_with_assumption_violations(self):
        """Test diagnostic summary with violated assumptions to cover warning branches."""
        # Create data that will violate assumptions
        np.random.seed(999)

        # Create non-normal, heteroscedastic, and autocorrelated data
        n = 50
        x = np.linspace(1, 10, n)
        # Non-normal errors (exponential distribution)
        errors = np.random.exponential(1, n) - 1  # Mean-centered exponential
        # Heteroscedastic (variance increases with x)
        errors = errors * x / 5
        # Add autocorrelation
        for i in range(1, n):
            errors[i] += 0.5 * errors[i - 1]

        y = 2 + 3 * x + errors

        data = pd.DataFrame({"control": x, "test": y})
        model_params = fit_regression_model(data, "control", "test")

        # Create diagnostic summary - this should trigger warning branches
        summary = create_diagnostic_summary(data, model_params, "control", "test")

        # Verify warnings are present (covering lines 637, 640, 643)
        assert "warnings" in summary
        assert len(summary["warnings"]) > 0

        # Check that specific warning types can be triggered
        warnings_text = " ".join(summary["warnings"])
        # At least one of these should be present given our data construction
        assert any(
            keyword in warnings_text.lower()
            for keyword in ["normality", "homoscedasticity", "independence"]
        )

    def test_validate_model_assumptions_with_failures(self):
        """Test assumption validation with failed assumptions to cover failure branches."""
        # Create data that will fail assumptions
        np.random.seed(888)

        # Create clearly non-normal data (bimodal distribution)
        n = 40
        x = np.linspace(1, 5, n)
        # Bimodal errors (mix of two normals)
        errors1 = np.random.normal(-2, 0.5, n // 2)
        errors2 = np.random.normal(2, 0.5, n // 2)
        errors = np.concatenate([errors1, errors2])
        np.random.shuffle(errors)

        # Add heteroscedasticity
        errors = errors * (1 + x / 10)

        # Add strong autocorrelation
        for i in range(1, n):
            errors[i] += 0.8 * errors[i - 1]

        y = 1 + 2 * x + errors

        data = pd.DataFrame({"control": x, "test": y})
        model_params = fit_regression_model(data, "control", "test")

        # Validate assumptions - this should trigger failure branches
        validation = validate_model_assumptions(data, model_params, "control", "test")

        # Verify that some assumptions fail (covering lines 736, 738, 740)
        assert "all_assumptions_valid" in validation
        assert not validation["all_assumptions_valid"]  # Should be False

        # Check that the validation summary reflects failures
        assert "validation_summary" in validation
        assert "failed assumptions" in validation["validation_summary"].lower()

        # Verify individual assumption results
        assert "normality_valid" in validation
        assert "homoscedasticity_valid" in validation
        assert "independence_valid" in validation

        # At least one should be False (this covers the missing lines)
        assumption_results = [
            validation["normality_valid"],
            validation["homoscedasticity_valid"],
            validation["independence_valid"],
        ]
        assert not all(assumption_results), "At least one assumption should fail"

    def test_extreme_assumption_violations(self):
        """Test with extreme violations to ensure all warning branches are covered."""
        # Create data with extreme violations
        np.random.seed(777)
        n = 30

        # Extremely non-normal data (uniform distribution)
        x = np.linspace(1, 3, n)
        errors = np.random.uniform(-5, 5, n)  # Uniform errors

        # Extreme heteroscedasticity
        errors = errors * (x**2)  # Variance grows quadratically

        # Perfect autocorrelation (each error = previous error)
        for i in range(1, n):
            errors[i] = errors[i - 1] + np.random.normal(0, 0.1)

        y = 5 + x + errors

        data = pd.DataFrame({"control": x, "test": y})
        model_params = fit_regression_model(data, "control", "test")

        # Test both functions to ensure all branches are covered
        summary = create_diagnostic_summary(data, model_params, "control", "test")
        validation = validate_model_assumptions(data, model_params, "control", "test")

        # Verify comprehensive coverage
        assert len(summary["warnings"]) >= 1  # Should have multiple warnings
        assert not validation["all_assumptions_valid"]  # Should have failures

        # Verify overall validity is False
        assert not summary["overall_validity"]
        assert not validation["all_assumptions_valid"]

        # Verify the validation summary indicates failures
        assert "failed assumptions" in validation["validation_summary"].lower()
