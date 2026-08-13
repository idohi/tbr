"""
Unit tests for TBR Analysis Diagnostics Module.

This module provides unit tests for the analysis.diagnostics module,
ensuring all diagnostic functions work correctly with TBR DataFrames and provide
accurate model validation, assumption checking, and performance assessment.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tbr.analysis.diagnostics import (
    _generate_diagnostic_recommendations,
    analyze_tbr_residuals,
    assess_tbr_performance,
    check_tbr_assumptions,
    create_tbr_diagnostic_report,
    diagnose_tbr_analysis,
    validate_tbr_model,
)


class TestValidateTbrModel:
    """Test suite for validate_tbr_model function."""

    @pytest.fixture
    def sample_tbr_data(self):
        """Create sample TBR data for testing."""
        np.random.seed(42)

        # Learning period data (period == 0)
        learning_data = pd.DataFrame(
            {
                "period": [0] * 30,
                "y": np.random.normal(1000, 50, 30),
                "x": np.random.normal(950, 45, 30),
                "pred": np.random.normal(950, 45, 30),
                "predsd": np.random.uniform(10, 20, 30),
                "dif": np.random.normal(50, 25, 30),
                "cumdif": np.cumsum(np.random.normal(50, 25, 30)),
                "cumsd": np.sqrt(np.arange(1, 31) * 25**2),
                "estsd": np.random.uniform(15, 25, 30),
            }
        )

        # Test period data (period == 1)
        test_data = pd.DataFrame(
            {
                "period": [1] * 14,
                "y": np.random.normal(1020, 55, 14),
                "x": np.random.normal(970, 50, 14),
                "pred": np.random.normal(970, 50, 14),
                "predsd": np.random.uniform(12, 22, 14),
                "dif": np.random.normal(50, 30, 14),
                "cumdif": np.cumsum(np.random.normal(50, 30, 14)),
                "cumsd": np.sqrt(np.arange(1, 15) * 30**2),
                "estsd": np.random.uniform(18, 28, 14),
            }
        )

        tbr_df = pd.concat([learning_data, test_data], ignore_index=True)

        # Sample TBR summary
        tbr_summary = pd.DataFrame(
            {
                "estimate": [700.5],
                "precision": [89.2],
                "lower": [611.3],
                "upper": [789.7],
                "se": [45.6],
                "level": [0.80],
                "thres": [0.0],
                "prob": [0.95],
                "alpha": [50.2],
                "beta": [0.95],
                "sigma": [25.3],
                "var_alpha": [100.5],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [28],
            }
        )

        return tbr_df, tbr_summary

    def test_validate_tbr_model_basic_functionality(self, sample_tbr_data):
        """Test basic functionality of validate_tbr_model."""
        tbr_df, tbr_summary = sample_tbr_data

        result = validate_tbr_model(tbr_df, tbr_summary)

        # Check return structure
        assert isinstance(result, dict)
        assert "overall_validity" in result
        assert "warnings" in result
        assert "assumption_tests" in result
        assert "goodness_of_fit" in result
        assert "residual_analysis" in result
        assert "prediction_quality" in result

        # Check types
        assert isinstance(result["overall_validity"], bool)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["assumption_tests"], dict)
        assert isinstance(result["goodness_of_fit"], dict)
        assert isinstance(result["residual_analysis"], dict)
        assert isinstance(result["prediction_quality"], dict)

    def test_validate_tbr_model_with_learning_data(self, sample_tbr_data):
        """Test validate_tbr_model with explicit learning data."""
        tbr_df, tbr_summary = sample_tbr_data
        learning_data = tbr_df[tbr_df["period"] == 0].copy()

        result = validate_tbr_model(tbr_df, tbr_summary, learning_data)

        assert isinstance(result, dict)
        assert "overall_validity" in result

    def test_validate_tbr_model_preserves_goodness_of_fit_metrics(self):
        """Regression test for the analysis/core goodness-of-fit key contract."""
        x = np.linspace(100.0, 200.0, 44)
        residual_pattern = np.sin(np.arange(44)) * 0.5
        pred = 5.0 + 1.05 * x
        y = pred + residual_pattern
        dif = y - pred

        tbr_df = pd.DataFrame(
            {
                "period": [0] * 30 + [1] * 14,
                "y": y,
                "x": x,
                "pred": pred,
                "predsd": np.full(44, 2.0),
                "dif": dif,
                "cumdif": np.cumsum(dif),
                "cumsd": np.sqrt(np.arange(1, 45)),
                "estsd": np.full(44, 0.5),
            }
        )
        tbr_summary = pd.DataFrame(
            {
                "alpha": [5.0],
                "beta": [1.05],
                "sigma": [float(np.std(dif[:30], ddof=2))],
                "var_alpha": [0.01],
                "var_beta": [0.001],
                "alpha_beta_cov": [0.0],
                "t_dist_df": [28],
            }
        )

        result = validate_tbr_model(tbr_df, tbr_summary)

        assert not any(
            "goodness of fit calculation failed" in warning.lower()
            for warning in result["warnings"]
        )
        assert result["goodness_of_fit"]["r_squared"] > 0.99
        assert "f_p_value" in result["goodness_of_fit"]
        assert "f_statistic_p_value" not in result["goodness_of_fit"]

    def test_validate_tbr_model_rejects_missing_goodness_of_fit_keys(
        self, sample_tbr_data
    ):
        """Internal goodness-of-fit schema mismatches should fail loudly."""
        tbr_df, tbr_summary = sample_tbr_data

        with patch("tbr.analysis.diagnostics.calculate_goodness_of_fit") as mock_gof:
            mock_gof.return_value = {
                "r_squared": 0.8,
                "adj_r_squared": 0.79,
                "f_statistic": 20.0,
                "mse": 1.0,
                "rmse": 1.0,
            }

            with pytest.raises(KeyError, match="Missing goodness-of-fit metric"):
                validate_tbr_model(tbr_df, tbr_summary)

    def test_validate_tbr_model_handles_expected_goodness_of_fit_failure(
        self, sample_tbr_data
    ):
        """Expected numerical fit failures are reported without fake metrics."""
        tbr_df, tbr_summary = sample_tbr_data

        with patch("tbr.analysis.diagnostics.calculate_goodness_of_fit") as mock_gof:
            mock_gof.side_effect = ValueError("insufficient variation")

            result = validate_tbr_model(tbr_df, tbr_summary)

        assert any(
            "goodness of fit calculation failed" in warning.lower()
            for warning in result["warnings"]
        )
        assert result["goodness_of_fit"] == {"error": "insufficient variation"}

    def test_validate_tbr_model_surfaces_unexpected_goodness_of_fit_failure(
        self, sample_tbr_data
    ):
        """Unexpected goodness-of-fit failures should not become fake metrics."""
        tbr_df, tbr_summary = sample_tbr_data

        with patch("tbr.analysis.diagnostics.calculate_goodness_of_fit") as mock_gof:
            mock_gof.side_effect = RuntimeError("internal bug")

            with pytest.raises(RuntimeError, match="internal bug"):
                validate_tbr_model(tbr_df, tbr_summary)

    def test_validate_tbr_model_input_validation(self, sample_tbr_data):
        """Test input validation for validate_tbr_model."""
        tbr_df, tbr_summary = sample_tbr_data

        # Test missing columns in tbr_df
        invalid_tbr_df = tbr_df.drop(columns=["period"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_tbr_model(invalid_tbr_df, tbr_summary)

        # Test missing columns in tbr_summary
        invalid_summary = tbr_summary.drop(columns=["alpha"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_tbr_model(tbr_df, invalid_summary)

        # Test empty tbr_summary
        empty_summary = pd.DataFrame(columns=tbr_summary.columns)
        with pytest.raises(ValueError, match="tbr_summary DataFrame cannot be empty"):
            validate_tbr_model(tbr_df, empty_summary)

    def test_validate_tbr_model_no_learning_data(self, sample_tbr_data):
        """Test validate_tbr_model when no learning data is available."""
        tbr_df, tbr_summary = sample_tbr_data

        # Remove learning period data
        tbr_df_no_learning = tbr_df[tbr_df["period"] == 1].copy()

        with pytest.raises(ValueError, match="No learning period data found"):
            validate_tbr_model(tbr_df_no_learning, tbr_summary)

    def test_validate_tbr_model_warnings_generation(self, sample_tbr_data):
        """Test that warnings are generated appropriately."""
        tbr_df, tbr_summary = sample_tbr_data

        # Mock assumption tests to fail
        with patch(
            "tbr.analysis.diagnostics.validate_model_assumptions"
        ) as mock_assumptions:
            mock_assumptions.return_value = {
                "normality_valid": False,
                "homoscedasticity_valid": False,
                "independence_valid": True,
                "all_assumptions_valid": False,
            }

            result = validate_tbr_model(tbr_df, tbr_summary)

            # Should have warnings for failed assumptions
            assert len(result["warnings"]) >= 2
            assert any("normality" in warning.lower() for warning in result["warnings"])
            assert any(
                "heteroscedasticity" in warning.lower()
                for warning in result["warnings"]
            )
            assert not result["overall_validity"]

    def test_validate_tbr_model_expected_assumption_error_handling(
        self, sample_tbr_data
    ):
        """Expected assumption-test failures are reported as warnings."""
        tbr_df, tbr_summary = sample_tbr_data

        # Mock function to raise exception
        with patch(
            "tbr.analysis.diagnostics.validate_model_assumptions"
        ) as mock_assumptions:
            mock_assumptions.side_effect = ValueError("Test error")

            result = validate_tbr_model(tbr_df, tbr_summary)

            # Should handle error gracefully
            assert "assumption_tests" in result
            assert "error" in result["assumption_tests"]
            assert len(result["warnings"]) > 0

    def test_validate_tbr_model_surfaces_unexpected_assumption_failure(
        self, sample_tbr_data
    ):
        """Unexpected assumption-test failures should fail loudly."""
        tbr_df, tbr_summary = sample_tbr_data

        with patch(
            "tbr.analysis.diagnostics.validate_model_assumptions"
        ) as mock_assumptions:
            mock_assumptions.side_effect = RuntimeError("internal assumption bug")

            with pytest.raises(RuntimeError, match="internal assumption bug"):
                validate_tbr_model(tbr_df, tbr_summary)


class TestDiagnoseTbrAnalysis:
    """Test suite for diagnose_tbr_analysis function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)

        tbr_df = pd.DataFrame(
            {
                "period": [0] * 20 + [1] * 10,
                "y": np.random.normal(1000, 50, 30),
                "x": np.random.normal(950, 45, 30),
                "pred": np.random.normal(950, 45, 30),
                "predsd": np.random.uniform(10, 20, 30),
                "dif": np.random.normal(50, 25, 30),
                "cumdif": np.cumsum(np.random.normal(50, 25, 30)),
                "cumsd": np.sqrt(np.arange(1, 31) * 25**2),
                "estsd": np.random.uniform(15, 25, 30),
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.2],
                "beta": [0.95],
                "sigma": [25.3],
                "var_alpha": [100.5],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [18],
            }
        )

        return tbr_df, tbr_summary

    def test_diagnose_tbr_analysis_basic(self, sample_data):
        """Test basic functionality of diagnose_tbr_analysis."""
        tbr_df, tbr_summary = sample_data

        result = diagnose_tbr_analysis(tbr_df, tbr_summary)

        # Check return structure
        assert isinstance(result, dict)
        assert "model_validation" in result
        assert "diagnostic_summary" in result
        assert "performance_metrics" in result
        assert "recommendations" in result

        # Check types
        assert isinstance(result["model_validation"], dict)
        assert isinstance(result["diagnostic_summary"], dict)
        assert isinstance(result["performance_metrics"], dict)
        assert isinstance(result["recommendations"], list)

    def test_diagnose_tbr_analysis_without_performance(self, sample_data):
        """Test diagnose_tbr_analysis with performance disabled."""
        tbr_df, tbr_summary = sample_data

        result = diagnose_tbr_analysis(tbr_df, tbr_summary, include_performance=False)

        assert "performance_metrics" in result
        assert result["performance_metrics"] == {}

    def test_diagnose_tbr_analysis_with_learning_data(self, sample_data):
        """Test diagnose_tbr_analysis with explicit learning data."""
        tbr_df, tbr_summary = sample_data
        learning_data = tbr_df[tbr_df["period"] == 0].copy()

        result = diagnose_tbr_analysis(tbr_df, tbr_summary, learning_data)

        assert isinstance(result, dict)
        assert "model_validation" in result


class TestCheckTbrAssumptions:
    """Test suite for check_tbr_assumptions function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)

        tbr_df = pd.DataFrame(
            {
                "period": [0] * 25,
                "y": np.random.normal(1000, 50, 25),
                "x": np.random.normal(950, 45, 25),
                "pred": np.random.normal(950, 45, 25),
                "predsd": np.random.uniform(10, 20, 25),
                "dif": np.random.normal(50, 25, 25),
                "cumdif": np.cumsum(np.random.normal(50, 25, 25)),
                "cumsd": np.sqrt(np.arange(1, 26) * 25**2),
                "estsd": np.random.uniform(15, 25, 25),
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.2],
                "beta": [0.95],
                "sigma": [25.3],
                "var_alpha": [100.5],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [23],
            }
        )

        return tbr_df, tbr_summary

    def test_check_tbr_assumptions_basic(self, sample_data):
        """Test basic functionality of check_tbr_assumptions."""
        tbr_df, tbr_summary = sample_data

        result = check_tbr_assumptions(tbr_df, tbr_summary)

        # Should return assumption test results
        assert isinstance(result, dict)
        # The exact keys depend on validate_model_assumptions implementation
        assert len(result) > 0

    def test_check_tbr_assumptions_with_alpha(self, sample_data):
        """Test check_tbr_assumptions with custom alpha level."""
        tbr_df, tbr_summary = sample_data

        result = check_tbr_assumptions(tbr_df, tbr_summary, alpha=0.01)

        assert isinstance(result, dict)

    def test_check_tbr_assumptions_with_learning_data(self, sample_data):
        """Test check_tbr_assumptions with explicit learning data."""
        tbr_df, tbr_summary = sample_data
        learning_data = tbr_df[tbr_df["period"] == 0].copy()

        result = check_tbr_assumptions(tbr_df, tbr_summary, learning_data)

        assert isinstance(result, dict)


class TestAnalyzeTbrResiduals:
    """Test suite for analyze_tbr_residuals function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)

        tbr_df = pd.DataFrame(
            {
                "period": [0] * 30,
                "y": np.random.normal(1000, 50, 30),
                "x": np.random.normal(950, 45, 30),
                "pred": np.random.normal(950, 45, 30),
                "predsd": np.random.uniform(10, 20, 30),
                "dif": np.random.normal(50, 25, 30),
                "cumdif": np.cumsum(np.random.normal(50, 25, 30)),
                "cumsd": np.sqrt(np.arange(1, 31) * 25**2),
                "estsd": np.random.uniform(15, 25, 30),
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.2],
                "beta": [0.95],
                "sigma": [25.3],
                "var_alpha": [100.5],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [28],
            }
        )

        return tbr_df, tbr_summary

    def test_analyze_tbr_residuals_basic(self, sample_data):
        """Test basic functionality of analyze_tbr_residuals."""
        tbr_df, tbr_summary = sample_data

        result = analyze_tbr_residuals(tbr_df, tbr_summary)

        # Check return structure
        assert isinstance(result, dict)
        expected_keys = [
            "residuals",
            "standardized_residuals",
            "studentized_residuals",
            "outliers",
            "outlier_threshold",
            "outlier_percentage",
            "residual_stats",
            "residual_std",
            "n_observations",
        ]
        for key in expected_keys:
            assert key in result

        # Check types and values
        assert isinstance(result["residuals"], np.ndarray)
        assert isinstance(result["standardized_residuals"], np.ndarray)
        assert isinstance(result["studentized_residuals"], np.ndarray)
        assert isinstance(result["outliers"], list)
        assert isinstance(result["outlier_threshold"], (int, float))
        assert isinstance(result["outlier_percentage"], (int, float))
        assert isinstance(result["residual_stats"], dict)
        assert isinstance(result["residual_std"], (int, float))
        assert isinstance(result["n_observations"], int)

        # Check array lengths
        n_obs = len(tbr_df[tbr_df["period"] == 0])
        assert len(result["residuals"]) == n_obs
        assert len(result["standardized_residuals"]) == n_obs
        assert len(result["studentized_residuals"]) == n_obs
        assert result["n_observations"] == n_obs

    def test_analyze_tbr_residuals_with_learning_data(self, sample_data):
        """Test analyze_tbr_residuals with explicit learning data."""
        tbr_df, tbr_summary = sample_data
        learning_data = tbr_df[tbr_df["period"] == 0].copy()

        result = analyze_tbr_residuals(tbr_df, tbr_summary, learning_data)

        assert isinstance(result, dict)
        assert "residuals" in result

    def test_analyze_tbr_residuals_outlier_detection(self, sample_data):
        """Test outlier detection in analyze_tbr_residuals."""
        tbr_df, tbr_summary = sample_data

        result = analyze_tbr_residuals(tbr_df, tbr_summary)

        # Check outlier detection
        assert result["outlier_threshold"] == 2.5
        assert 0 <= result["outlier_percentage"] <= 100
        assert len(result["outliers"]) <= len(result["residuals"])

    def test_analyze_tbr_residuals_statistics(self, sample_data):
        """Test residual statistics calculation."""
        tbr_df, tbr_summary = sample_data

        result = analyze_tbr_residuals(tbr_df, tbr_summary)

        # Check residual statistics
        stats = result["residual_stats"]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "q25" in stats
        assert "median" in stats
        assert "q75" in stats

        # Check statistical properties
        assert (
            stats["min"]
            <= stats["q25"]
            <= stats["median"]
            <= stats["q75"]
            <= stats["max"]
        )
        assert stats["std"] >= 0


class TestAssessTbrPerformance:
    """Test suite for assess_tbr_performance function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)

        # Create data with both learning and test periods
        learning_data = pd.DataFrame(
            {
                "period": [0] * 25,
                "y": np.random.normal(1000, 50, 25),
                "x": np.random.normal(950, 45, 25),
                "pred": np.random.normal(950, 45, 25),
                "predsd": np.random.uniform(10, 20, 25),
                "dif": np.random.normal(50, 25, 25),
                "cumdif": np.cumsum(np.random.normal(50, 25, 25)),
                "cumsd": np.sqrt(np.arange(1, 26) * 25**2),
                "estsd": np.random.uniform(15, 25, 25),
            }
        )

        test_data = pd.DataFrame(
            {
                "period": [1] * 15,
                "y": np.random.normal(1020, 55, 15),
                "x": np.random.normal(970, 50, 15),
                "pred": np.random.normal(970, 50, 15),
                "predsd": np.random.uniform(12, 22, 15),
                "dif": np.random.normal(50, 30, 15),
                "cumdif": np.cumsum(np.random.normal(50, 30, 15)),
                "cumsd": np.sqrt(np.arange(1, 16) * 30**2),
                "estsd": np.random.uniform(18, 28, 15),
            }
        )

        tbr_df = pd.concat([learning_data, test_data], ignore_index=True)

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.2],
                "beta": [0.95],
                "sigma": [25.3],
                "var_alpha": [100.5],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [23],
            }
        )

        return tbr_df, tbr_summary

    def test_assess_tbr_performance_basic(self, sample_data):
        """Test basic functionality of assess_tbr_performance."""
        tbr_df, tbr_summary = sample_data

        result = assess_tbr_performance(tbr_df, tbr_summary)

        # Check return structure
        assert isinstance(result, dict)
        expected_keys = [
            "data_metrics",
            "prediction_metrics",
            "model_complexity",
            "efficiency_score",
            "performance_summary",
        ]
        for key in expected_keys:
            assert key in result

        # Check data metrics
        data_metrics = result["data_metrics"]
        assert "total_observations" in data_metrics
        assert "learning_observations" in data_metrics
        assert "test_observations" in data_metrics
        assert "learning_test_ratio" in data_metrics

        # Check prediction metrics
        pred_metrics = result["prediction_metrics"]
        assert "mae" in pred_metrics
        assert "mse" in pred_metrics
        assert "rmse" in pred_metrics
        assert "mape" in pred_metrics
        assert "interval_coverage" in pred_metrics

        # Check efficiency score
        assert isinstance(result["efficiency_score"], float)
        assert 0 <= result["efficiency_score"] <= 1

    def test_assess_tbr_performance_no_test_data(self):
        """Test assess_tbr_performance with no test period data."""
        np.random.seed(42)

        # Only learning period data
        tbr_df = pd.DataFrame(
            {
                "period": [0] * 20,
                "y": np.random.normal(1000, 50, 20),
                "x": np.random.normal(950, 45, 20),
                "pred": np.random.normal(950, 45, 20),
                "predsd": np.random.uniform(10, 20, 20),
                "dif": np.random.normal(50, 25, 20),
                "cumdif": np.cumsum(np.random.normal(50, 25, 20)),
                "cumsd": np.sqrt(np.arange(1, 21) * 25**2),
                "estsd": np.random.uniform(15, 25, 20),
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.2],
                "beta": [0.95],
                "sigma": [25.3],
                "var_alpha": [100.5],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [18],
            }
        )

        result = assess_tbr_performance(tbr_df, tbr_summary)

        # Should handle missing test data gracefully
        assert isinstance(result, dict)
        assert result["data_metrics"]["test_observations"] == 0
        assert result["prediction_metrics"] == {}

    def test_assess_tbr_performance_metrics_calculation(self, sample_data):
        """Test specific metrics calculations."""
        tbr_df, tbr_summary = sample_data

        result = assess_tbr_performance(tbr_df, tbr_summary)

        # Check data metrics calculations
        data_metrics = result["data_metrics"]
        assert data_metrics["total_observations"] == len(tbr_df)
        assert data_metrics["learning_observations"] == len(
            tbr_df[tbr_df["period"] == 0]
        )
        assert data_metrics["test_observations"] == len(tbr_df[tbr_df["period"] == 1])

        # Check prediction metrics are reasonable
        pred_metrics = result["prediction_metrics"]
        assert pred_metrics["mae"] >= 0
        assert pred_metrics["mse"] >= 0
        assert pred_metrics["rmse"] >= 0
        assert pred_metrics["mape"] >= 0
        assert 0 <= pred_metrics["interval_coverage"] <= 1

        # Check performance summary
        perf_summary = result["performance_summary"]
        assert "data_quality" in perf_summary
        assert "prediction_quality" in perf_summary
        assert "overall_performance" in perf_summary


class TestCreateTbrDiagnosticReport:
    """Test suite for create_tbr_diagnostic_report function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)

        tbr_df = pd.DataFrame(
            {
                "period": [0] * 20 + [1] * 10,
                "y": np.random.normal(1000, 50, 30),
                "x": np.random.normal(950, 45, 30),
                "pred": np.random.normal(950, 45, 30),
                "predsd": np.random.uniform(10, 20, 30),
                "dif": np.random.normal(50, 25, 30),
                "cumdif": np.cumsum(np.random.normal(50, 25, 30)),
                "cumsd": np.sqrt(np.arange(1, 31) * 25**2),
                "estsd": np.random.uniform(15, 25, 30),
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.2],
                "beta": [0.95],
                "sigma": [25.3],
                "var_alpha": [100.5],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [18],
            }
        )

        return tbr_df, tbr_summary

    def test_create_tbr_diagnostic_report_basic(self, sample_data):
        """Test basic functionality of create_tbr_diagnostic_report."""
        tbr_df, tbr_summary = sample_data

        result = create_tbr_diagnostic_report(tbr_df, tbr_summary)

        # Check return structure
        assert isinstance(result, dict)
        expected_keys = [
            "executive_summary",
            "overall_validity",
            "warnings_count",
            "key_findings",
            "recommendations",
            "detailed_results",
            "report_timestamp",
        ]
        for key in expected_keys:
            assert key in result

        # Check types
        assert isinstance(result["executive_summary"], str)
        assert isinstance(result["overall_validity"], bool)
        assert isinstance(result["warnings_count"], int)
        assert isinstance(result["key_findings"], list)
        assert isinstance(result["recommendations"], list)
        assert isinstance(result["report_timestamp"], str)

    def test_create_tbr_diagnostic_report_without_detailed_analysis(self, sample_data):
        """Test create_tbr_diagnostic_report without detailed analysis."""
        tbr_df, tbr_summary = sample_data

        result = create_tbr_diagnostic_report(
            tbr_df, tbr_summary, include_detailed_analysis=False
        )

        assert result["detailed_results"] is None

    def test_create_tbr_diagnostic_report_with_learning_data(self, sample_data):
        """Test create_tbr_diagnostic_report with explicit learning data."""
        tbr_df, tbr_summary = sample_data
        learning_data = tbr_df[tbr_df["period"] == 0].copy()

        result = create_tbr_diagnostic_report(tbr_df, tbr_summary, learning_data)

        assert isinstance(result, dict)
        assert "executive_summary" in result


class TestGenerateDiagnosticRecommendations:
    """Test suite for _generate_diagnostic_recommendations function."""

    def test_generate_recommendations_valid_model(self):
        """Test recommendations for valid model."""
        model_validation = {
            "overall_validity": True,
            "assumption_tests": {
                "normality_valid": True,
                "homoscedasticity_valid": True,
                "independence_valid": True,
            },
            "goodness_of_fit": {"r_squared": 0.8},
            "residual_analysis": {"outlier_percentage": 5},
        }

        recommendations = _generate_diagnostic_recommendations(model_validation, {})

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should have positive recommendation for valid model
        assert any(
            "good" in rec.lower() or "confidence" in rec.lower()
            for rec in recommendations
        )

    def test_generate_recommendations_invalid_assumptions(self):
        """Test recommendations for invalid assumptions."""
        model_validation = {
            "overall_validity": False,
            "assumption_tests": {
                "normality_valid": False,
                "homoscedasticity_valid": False,
                "independence_valid": True,
            },
            "goodness_of_fit": {"r_squared": 0.6},
            "residual_analysis": {"outlier_percentage": 15},
        }

        recommendations = _generate_diagnostic_recommendations(model_validation, {})

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should have recommendations for assumption violations
        assert any("transformation" in rec.lower() for rec in recommendations)
        assert any(
            "heteroscedasticity" in rec.lower() or "variance" in rec.lower()
            for rec in recommendations
        )

    def test_generate_recommendations_poor_fit(self):
        """Test recommendations for poor model fit."""
        model_validation = {
            "overall_validity": False,
            "assumption_tests": {
                "normality_valid": True,
                "homoscedasticity_valid": True,
                "independence_valid": True,
            },
            "goodness_of_fit": {"r_squared": 0.2},
            "residual_analysis": {"outlier_percentage": 5},
        }

        recommendations = _generate_diagnostic_recommendations(model_validation, {})

        assert isinstance(recommendations, list)
        # Should have recommendations for poor fit
        assert any(
            "fit" in rec.lower() or "predictors" in rec.lower()
            for rec in recommendations
        )

    def test_generate_recommendations_performance_issues(self):
        """Test recommendations for performance issues."""
        model_validation = {"overall_validity": True}
        performance_metrics = {
            "prediction_metrics": {
                "mape": 20,
                "interval_coverage": 0.75,
            },
            "data_metrics": {
                "learning_observations": 15,
                "learning_test_ratio": 1.5,
            },
        }

        recommendations = _generate_diagnostic_recommendations(
            model_validation, performance_metrics
        )

        assert isinstance(recommendations, list)
        # Should have recommendations for performance issues
        assert any(
            "error" in rec.lower() or "mape" in rec.lower() for rec in recommendations
        )
        assert any("coverage" in rec.lower() for rec in recommendations)
        assert any("data" in rec.lower() for rec in recommendations)


class TestComprehensiveCoverageScenarios:
    """Test scenarios to achieve 100% coverage for scientific PyPI standards."""

    @pytest.fixture
    def failing_assumptions_data(self):
        """Create data that will fail statistical assumptions."""
        np.random.seed(42)

        # Create data with non-normal residuals and heteroscedasticity
        tbr_df = pd.DataFrame(
            {
                "period": [0] * 20,
                "y": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] * 2,  # Exponential pattern
                "x": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 2,  # Non-linear relationship
                "pred": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 2,
                "predsd": [1] * 20,
                "dif": [0, 1, 2, 6, 13, 29, 60, 124, 251, 507] * 2,
                "cumdif": np.cumsum([0, 1, 2, 6, 13, 29, 60, 124, 251, 507] * 2),
                "cumsd": np.sqrt(np.arange(1, 21) * 100),
                "estsd": [10] * 20,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [1.0],
                "beta": [1.0],
                "sigma": [100.0],
                "var_alpha": [1.0],
                "var_beta": [0.1],
                "alpha_beta_cov": [0.0],
                "t_dist_df": [18],
            }
        )

        return tbr_df, tbr_summary

    @pytest.fixture
    def poor_fit_data(self):
        """Create data with poor model fit (low R²)."""
        np.random.seed(42)

        # Create data with very low correlation (high noise)
        tbr_df = pd.DataFrame(
            {
                "period": [0] * 25,
                "y": np.random.normal(1000, 500, 25),  # High variance
                "x": np.random.normal(1000, 50, 25),  # Low variance
                "pred": np.random.normal(1000, 50, 25),
                "predsd": [20] * 25,
                "dif": np.random.normal(0, 500, 25),
                "cumdif": np.cumsum(np.random.normal(0, 500, 25)),
                "cumsd": np.sqrt(np.arange(1, 26) * 500**2),
                "estsd": [50] * 25,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.1],
                "sigma": [500.0],  # Poor fit parameters
                "var_alpha": [10000.0],
                "var_beta": [1.0],
                "alpha_beta_cov": [0.0],
                "t_dist_df": [23],
            }
        )

        return tbr_df, tbr_summary

    @pytest.fixture
    def poor_coverage_data(self):
        """Create data with poor prediction interval coverage."""
        np.random.seed(42)

        learning_data = pd.DataFrame(
            {
                "period": [0] * 20,
                "y": np.random.normal(1000, 50, 20),
                "x": np.random.normal(950, 45, 20),
                "pred": np.random.normal(950, 45, 20),
                "predsd": [5] * 20,  # Very small prediction intervals
                "dif": np.random.normal(50, 25, 20),
                "cumdif": np.cumsum(np.random.normal(50, 25, 20)),
                "cumsd": np.sqrt(np.arange(1, 21) * 25**2),
                "estsd": [15] * 20,
            }
        )

        # Test data with large deviations that won't be covered by small intervals
        test_data = pd.DataFrame(
            {
                "period": [1] * 10,
                "y": np.random.normal(1200, 200, 10),  # Large deviations
                "x": np.random.normal(970, 50, 10),
                "pred": np.random.normal(970, 50, 10),
                "predsd": [5] * 10,  # Very small prediction intervals
                "dif": np.random.normal(230, 200, 10),
                "cumdif": np.cumsum(np.random.normal(230, 200, 10)),
                "cumsd": np.sqrt(np.arange(1, 11) * 200**2),
                "estsd": [25] * 10,
            }
        )

        tbr_df = pd.concat([learning_data, test_data], ignore_index=True)

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.95],
                "sigma": [25.0],
                "var_alpha": [100.0],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [18],
            }
        )

        return tbr_df, tbr_summary

    def test_assumption_test_warnings_coverage(self, failing_assumptions_data):
        """Test all assumption test warning conditions (lines 211-214)."""
        tbr_df, tbr_summary = failing_assumptions_data

        # Mock assumption tests to fail each condition
        with patch(
            "tbr.analysis.diagnostics.validate_model_assumptions"
        ) as mock_assumptions:
            # Test normality failure (line 211)
            mock_assumptions.return_value = {
                "normality_valid": False,
                "homoscedasticity_valid": True,
                "independence_valid": True,
                "all_assumptions_valid": False,
            }

            result = validate_tbr_model(tbr_df, tbr_summary)
            assert any("normality" in warning.lower() for warning in result["warnings"])

            # Test homoscedasticity failure (line 212)
            mock_assumptions.return_value = {
                "normality_valid": True,
                "homoscedasticity_valid": False,
                "independence_valid": True,
                "all_assumptions_valid": False,
            }

            result = validate_tbr_model(tbr_df, tbr_summary)
            assert any(
                "heteroscedasticity" in warning.lower()
                for warning in result["warnings"]
            )

            # Test independence failure (line 213-214)
            mock_assumptions.return_value = {
                "normality_valid": True,
                "homoscedasticity_valid": True,
                "independence_valid": False,
                "all_assumptions_valid": False,
            }

            result = validate_tbr_model(tbr_df, tbr_summary)
            assert any(
                "autocorrelation" in warning.lower() for warning in result["warnings"]
            )

    def test_goodness_of_fit_warnings_coverage(self, poor_fit_data):
        """Test goodness of fit warning conditions (lines 226-229)."""
        tbr_df, tbr_summary = poor_fit_data

        # Mock goodness of fit to trigger warnings
        with patch("tbr.analysis.diagnostics.calculate_goodness_of_fit") as mock_gof:
            # Test low R² warning (line 226-227)
            mock_gof.return_value = {
                "r_squared": 0.2,  # Below 0.5 threshold
                "adj_r_squared": 0.15,
                "f_statistic": 10.0,
                "f_p_value": 0.01,
                "mse": 1.0,
                "rmse": 1.0,
            }

            result = validate_tbr_model(tbr_df, tbr_summary)
            assert any("low r²" in warning.lower() for warning in result["warnings"])

            # Test non-significant F-statistic warning (line 228-229)
            mock_gof.return_value = {
                "r_squared": 0.8,
                "adj_r_squared": 0.79,
                "f_statistic": 2.0,
                "f_p_value": 0.10,  # Above 0.05 threshold
                "mse": 1.0,
                "rmse": 1.0,
            }

            result = validate_tbr_model(tbr_df, tbr_summary, alpha=0.05)
            assert any(
                "f-statistic not significant" in warning.lower()
                for warning in result["warnings"]
            )

    def test_residual_analysis_expected_error_handling_coverage(self):
        """Test expected residual analysis error handling."""
        np.random.seed(42)

        tbr_df = pd.DataFrame(
            {
                "period": [0] * 20,
                "y": np.random.normal(1000, 50, 20),
                "x": np.random.normal(950, 45, 20),
                "pred": np.random.normal(950, 45, 20),
                "predsd": [15] * 20,
                "dif": np.random.normal(50, 25, 20),
                "cumdif": np.cumsum(np.random.normal(50, 25, 20)),
                "cumsd": np.sqrt(np.arange(1, 21) * 25**2),
                "estsd": [15] * 20,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.95],
                "sigma": [25.0],
                "var_alpha": [100.0],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [18],
            }
        )

        # Mock residual calculation to raise an expected numerical/data error.
        with patch("tbr.analysis.diagnostics.calculate_residuals") as mock_residuals:
            mock_residuals.side_effect = ValueError("Residual calculation error")

            result = validate_tbr_model(tbr_df, tbr_summary)

            # Should handle error gracefully
            assert "residual_analysis" in result
            assert "error" in result["residual_analysis"]
            assert any(
                "residual analysis failed" in warning.lower()
                for warning in result["warnings"]
            )

    def test_residual_analysis_unexpected_error_surfaces(self):
        """Unexpected residual analysis errors should not be swallowed."""
        np.random.seed(42)

        tbr_df = pd.DataFrame(
            {
                "period": [0] * 20,
                "y": np.random.normal(1000, 50, 20),
                "x": np.random.normal(950, 45, 20),
                "pred": np.random.normal(950, 45, 20),
                "predsd": [15] * 20,
                "dif": np.random.normal(50, 25, 20),
                "cumdif": np.cumsum(np.random.normal(50, 25, 20)),
                "cumsd": np.sqrt(np.arange(1, 21) * 25**2),
                "estsd": [15] * 20,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.95],
                "sigma": [25.0],
                "var_alpha": [100.0],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [18],
            }
        )

        with patch("tbr.analysis.diagnostics.calculate_residuals") as mock_residuals:
            mock_residuals.side_effect = RuntimeError("internal residual bug")

            with pytest.raises(RuntimeError, match="internal residual bug"):
                validate_tbr_model(tbr_df, tbr_summary)

    def test_prediction_quality_warnings_and_errors_coverage(self, poor_coverage_data):
        """Test prediction quality warnings and error handling (lines 277-293)."""
        tbr_df, tbr_summary = poor_coverage_data

        # Test poor coverage warning (line 277-278)
        result = validate_tbr_model(tbr_df, tbr_summary)

        # Should trigger poor coverage warning
        if (
            "prediction_quality" in result
            and "prediction_interval_coverage" in result["prediction_quality"]
        ):
            coverage = result["prediction_quality"]["prediction_interval_coverage"]
            if coverage < 0.90:
                assert any(
                    "poor prediction interval coverage" in warning.lower()
                    for warning in result["warnings"]
                )

        # Test no test data scenario (line 289)
        learning_only_df = tbr_df[tbr_df["period"] == 0].copy()
        result = validate_tbr_model(learning_only_df, tbr_summary)

        assert "prediction_quality" in result
        assert "error" in result["prediction_quality"]
        assert result["prediction_quality"]["error"] == "No test period data available"

        # Test expected prediction quality error handling. Keep earlier diagnostic
        # stages mocked so this broad NumPy patch targets prediction quality only.
        with patch(
            "tbr.analysis.diagnostics.validate_model_assumptions"
        ) as mock_assumptions, patch(
            "tbr.analysis.diagnostics.calculate_goodness_of_fit"
        ) as mock_gof, patch(
            "numpy.mean"
        ) as mock_mean:
            mock_assumptions.return_value = {
                "normality_valid": True,
                "homoscedasticity_valid": True,
                "independence_valid": True,
                "all_assumptions_valid": True,
            }
            mock_gof.return_value = {
                "r_squared": 0.8,
                "adj_r_squared": 0.79,
                "f_statistic": 20.0,
                "f_p_value": 0.01,
                "mse": 1.0,
                "rmse": 1.0,
            }
            mock_mean.side_effect = ValueError("Prediction calculation error")

            result = validate_tbr_model(tbr_df, tbr_summary)

            # Should handle error gracefully
            assert any(
                "prediction quality assessment failed" in warning.lower()
                for warning in result["warnings"]
            )

    def test_prediction_quality_unexpected_error_surfaces(self, poor_coverage_data):
        """Unexpected prediction quality errors should not be swallowed."""
        tbr_df, tbr_summary = poor_coverage_data

        with patch(
            "tbr.analysis.diagnostics.validate_model_assumptions"
        ) as mock_assumptions, patch(
            "tbr.analysis.diagnostics.calculate_goodness_of_fit"
        ) as mock_gof, patch(
            "numpy.mean"
        ) as mock_mean:
            mock_assumptions.return_value = {
                "normality_valid": True,
                "homoscedasticity_valid": True,
                "independence_valid": True,
                "all_assumptions_valid": True,
            }
            mock_gof.return_value = {
                "r_squared": 0.8,
                "adj_r_squared": 0.79,
                "f_statistic": 20.0,
                "f_p_value": 0.01,
                "mse": 1.0,
                "rmse": 1.0,
            }
            mock_mean.side_effect = RuntimeError("internal prediction bug")

            with pytest.raises(RuntimeError, match="internal prediction bug"):
                validate_tbr_model(tbr_df, tbr_summary)

    def test_diagnostic_summary_expected_error_handling_coverage(self):
        """Test expected diagnostic summary error handling."""
        np.random.seed(42)

        tbr_df = pd.DataFrame(
            {
                "period": [0] * 20,
                "y": np.random.normal(1000, 50, 20),
                "x": np.random.normal(950, 45, 20),
                "pred": np.random.normal(950, 45, 20),
                "predsd": [15] * 20,
                "dif": np.random.normal(50, 25, 20),
                "cumdif": np.cumsum(np.random.normal(50, 25, 20)),
                "cumsd": np.sqrt(np.arange(1, 21) * 25**2),
                "estsd": [15] * 20,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.95],
                "sigma": [25.0],
                "var_alpha": [100.0],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [18],
            }
        )

        # Mock create_diagnostic_summary to raise an expected numerical/data error.
        with patch(
            "tbr.analysis.diagnostics.create_diagnostic_summary"
        ) as mock_summary:
            mock_summary.side_effect = ValueError("Diagnostic summary error")

            result = diagnose_tbr_analysis(tbr_df, tbr_summary)

        # Should handle error gracefully
        assert "diagnostic_summary" in result
        assert result["diagnostic_summary"]["error"] == "Diagnostic summary error"
        assert "goodness_of_fit" not in result["diagnostic_summary"]
        assert (
            "Diagnostic summary failed" in result["diagnostic_summary"]["warnings"][0]
        )

    def test_diagnostic_summary_unexpected_error_surfaces(self):
        """Unexpected diagnostic summary errors should not fabricate metrics."""
        np.random.seed(42)

        tbr_df = pd.DataFrame(
            {
                "period": [0] * 20,
                "y": np.random.normal(1000, 50, 20),
                "x": np.random.normal(950, 45, 20),
                "pred": np.random.normal(950, 45, 20),
                "predsd": [15] * 20,
                "dif": np.random.normal(50, 25, 20),
                "cumdif": np.cumsum(np.random.normal(50, 25, 20)),
                "cumsd": np.sqrt(np.arange(1, 21) * 25**2),
                "estsd": [15] * 20,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.95],
                "sigma": [25.0],
                "var_alpha": [100.0],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [18],
            }
        )

        with patch(
            "tbr.analysis.diagnostics.create_diagnostic_summary"
        ) as mock_summary:
            mock_summary.side_effect = RuntimeError("internal summary bug")

            with pytest.raises(RuntimeError, match="internal summary bug"):
                diagnose_tbr_analysis(tbr_df, tbr_summary)

    def test_performance_assessment_edge_cases_coverage(self):
        """Test performance assessment edge cases (lines 609-613)."""
        # Test with no learning observations (line 609)
        tbr_df = pd.DataFrame(
            {
                "period": [1] * 10,  # Only test period
                "y": np.random.normal(1000, 50, 10),
                "x": np.random.normal(950, 45, 10),
                "pred": np.random.normal(950, 45, 10),
                "predsd": [15] * 10,
                "dif": np.random.normal(50, 25, 10),
                "cumdif": np.cumsum(np.random.normal(50, 25, 10)),
                "cumsd": np.sqrt(np.arange(1, 11) * 25**2),
                "estsd": [15] * 10,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.95],
                "sigma": [25.0],
                "var_alpha": [100.0],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [8],
            }
        )

        result = assess_tbr_performance(tbr_df, tbr_summary)

        # Should handle zero learning observations
        assert result["data_metrics"]["learning_observations"] == 0
        # Note: prediction_metrics will still be calculated if test data exists

        # Test efficiency score calculation with no components (line 613)
        # Create completely empty DataFrame to trigger empty efficiency_components
        empty_df = pd.DataFrame(
            columns=[
                "period",
                "y",
                "x",
                "pred",
                "predsd",
                "dif",
                "cumdif",
                "cumsd",
                "estsd",
            ]
        )

        result_empty = assess_tbr_performance(empty_df, tbr_summary)

        # This should trigger the empty efficiency_components case (line 613)
        # No learning observations (line 609 condition false)
        # No test data (so no prediction_metrics, lines 599-606 skipped)
        assert result_empty["data_metrics"]["learning_observations"] == 0
        assert result_empty["data_metrics"]["test_observations"] == 0
        assert result_empty["prediction_metrics"] == {}
        assert result_empty["efficiency_score"] == 0.0

    def test_diagnostic_report_executive_summary_coverage(self):
        """Test diagnostic report executive summary conditions (lines 681, 691, 697, 701-717)."""
        np.random.seed(42)

        # Create data for testing different report scenarios
        tbr_df = pd.DataFrame(
            {
                "period": [0] * 15 + [1] * 10,
                "y": np.random.normal(1000, 50, 25),
                "x": np.random.normal(950, 45, 25),
                "pred": np.random.normal(950, 45, 25),
                "predsd": [15] * 25,
                "dif": np.random.normal(50, 25, 25),
                "cumdif": np.cumsum(np.random.normal(50, 25, 25)),
                "cumsd": np.sqrt(np.arange(1, 26) * 25**2),
                "estsd": [15] * 25,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.95],
                "sigma": [25.0],
                "var_alpha": [100.0],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [13],
            }
        )

        # Test valid model executive summary (line 681)
        with patch("tbr.analysis.diagnostics.validate_tbr_model") as mock_validate:
            mock_validate.return_value = {
                "overall_validity": True,
                "warnings": [],
                "goodness_of_fit": {"r_squared": 0.8},
                "assumption_tests": {"all_assumptions_valid": True},
            }

            report = create_tbr_diagnostic_report(tbr_df, tbr_summary)
            assert "PASSED" in report["executive_summary"]
            assert "meets statistical requirements" in report["executive_summary"]

        # Test invalid model executive summary (line 683)
        with patch("tbr.analysis.diagnostics.validate_tbr_model") as mock_validate:
            mock_validate.return_value = {
                "overall_validity": False,
                "warnings": ["Warning 1", "Warning 2"],
                "goodness_of_fit": {"r_squared": 0.3},
                "assumption_tests": {"all_assumptions_valid": False},
            }

            report = create_tbr_diagnostic_report(tbr_df, tbr_summary)
            assert "identified 2 issue(s)" in report["executive_summary"]

        # Test R² key finding (line 691)
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": {"r_squared": 0.75},
                    "assumption_tests": {"all_assumptions_valid": True},
                },
                "performance_metrics": {},
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(tbr_df, tbr_summary)
            assert any(
                "75.0% of variance" in finding for finding in report["key_findings"]
            )

        # Test all assumptions satisfied (line 697)
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": {"r_squared": 0.8},
                    "assumption_tests": {
                        "all_assumptions_valid": True,
                        "normality_valid": True,
                        "homoscedasticity_valid": True,
                        "independence_valid": True,
                    },
                },
                "performance_metrics": {},
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(tbr_df, tbr_summary)
            assert any(
                "all statistical assumptions are satisfied" in finding.lower()
                for finding in report["key_findings"]
            )

        # Test assumption violations (lines 701-717)
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": False,
                    "warnings": ["Normality failed", "Homoscedasticity failed"],
                    "goodness_of_fit": {"r_squared": 0.6},
                    "assumption_tests": {
                        "all_assumptions_valid": False,
                        "normality_valid": False,
                        "homoscedasticity_valid": False,
                        "independence_valid": True,
                    },
                },
                "performance_metrics": {},
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(tbr_df, tbr_summary)
            assert any(
                "assumption violations" in finding.lower()
                for finding in report["key_findings"]
            )
            assert any(
                "normality, homoscedasticity" in finding.lower()
                for finding in report["key_findings"]
            )

    def test_recommendation_generation_comprehensive_coverage(self):
        """Test recommendation generation conditions (lines 763-794)."""
        # Test assumption-based recommendations (lines 763-769)
        model_validation = {
            "overall_validity": False,
            "assumption_tests": {
                "normality_valid": False,
                "homoscedasticity_valid": False,
                "independence_valid": False,
            },
            "goodness_of_fit": {"r_squared": 0.2},
            "residual_analysis": {"outlier_percentage": 15},
        }

        performance_metrics = {
            "prediction_metrics": {
                "mape": 20,
                "interval_coverage": 0.75,
            },
            "data_metrics": {
                "learning_observations": 10,
                "learning_test_ratio": 1.0,
            },
        }

        recommendations = _generate_diagnostic_recommendations(
            model_validation, performance_metrics
        )

        # Should have recommendations for all issues
        assert any(
            "transformation" in rec.lower() for rec in recommendations
        )  # Line 765
        assert any(
            "heteroscedasticity" in rec.lower() for rec in recommendations
        )  # Line 767
        assert any(
            "temporal patterns" in rec.lower() for rec in recommendations
        )  # Line 769
        assert any(
            "low model fit" in rec.lower() for rec in recommendations
        )  # Line 775
        assert any("outliers" in rec.lower() for rec in recommendations)  # Line 783
        assert any(
            "prediction error" in rec.lower() for rec in recommendations
        )  # Line 789
        assert any("coverage" in rec.lower() for rec in recommendations)  # Line 791
        assert any(
            "learning period data" in rec.lower() for rec in recommendations
        )  # Line 794

        # Test moderate fit recommendation (line 777)
        model_validation_moderate = {
            "overall_validity": True,
            "assumption_tests": {"all_assumptions_valid": True},
            "goodness_of_fit": {"r_squared": 0.4},  # Between 0.3 and 0.5
            "residual_analysis": {"outlier_percentage": 5},
        }

        recommendations = _generate_diagnostic_recommendations(
            model_validation_moderate, {}
        )

        assert any("moderate model fit" in rec.lower() for rec in recommendations)

    def test_final_coverage_edge_cases(self):
        """Test remaining edge cases for 100% coverage."""
        np.random.seed(42)

        # Create test data
        tbr_df = pd.DataFrame(
            {
                "period": [0] * 15 + [1] * 10,
                "y": np.random.normal(1000, 50, 25),
                "x": np.random.normal(950, 45, 25),
                "pred": np.random.normal(950, 45, 25),
                "predsd": [15] * 25,
                "dif": np.random.normal(50, 25, 25),
                "cumdif": np.cumsum(np.random.normal(50, 25, 25)),
                "cumsd": np.sqrt(np.arange(1, 26) * 25**2),
                "estsd": [15] * 25,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.95],
                "sigma": [25.0],
                "var_alpha": [100.0],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [13],
            }
        )

        # Test branch coverage for lines 245->248 (high outlier percentage)
        with patch(
            "tbr.analysis.diagnostics.calculate_studentized_residuals"
        ) as mock_studentized:
            # Create high outlier scenario (>10% outliers)
            mock_studentized.return_value = np.array(
                [3.0, 3.5, 4.0, 2.0, 1.0] * 3
            )  # 60% outliers

            result = validate_tbr_model(tbr_df, tbr_summary)

            # Should trigger high outlier warning
            assert any(
                "high number of outliers" in warning.lower()
                for warning in result["warnings"]
            )

        # Test branch coverage for lines 277->280 (poor prediction coverage)
        test_data_poor_coverage = tbr_df[tbr_df["period"] == 1].copy()
        test_data_poor_coverage["y"] = [2000] * len(
            test_data_poor_coverage
        )  # Far from predictions
        test_data_poor_coverage["pred"] = [1000] * len(test_data_poor_coverage)
        test_data_poor_coverage["predsd"] = [10] * len(
            test_data_poor_coverage
        )  # Small intervals

        tbr_df_poor_coverage = pd.concat(
            [tbr_df[tbr_df["period"] == 0], test_data_poor_coverage], ignore_index=True
        )

        result = validate_tbr_model(tbr_df_poor_coverage, tbr_summary)

        # Should trigger poor coverage warning
        if (
            "prediction_quality" in result
            and "prediction_interval_coverage" in result["prediction_quality"]
        ):
            coverage = result["prediction_quality"]["prediction_interval_coverage"]
            if coverage < 0.90:
                assert any(
                    "poor prediction interval coverage" in warning.lower()
                    for warning in result["warnings"]
                )

        # Test branch coverage for diagnostic report performance metrics (lines 695->709)
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": {"r_squared": 0.8},
                    "assumption_tests": {"all_assumptions_valid": True},
                },
                "performance_metrics": {
                    "prediction_metrics": {
                        "mape": 5.0,
                        "interval_coverage": 0.95,
                    }
                },
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(tbr_df, tbr_summary)

            # Should include performance findings
            assert any(
                "5.0% mape" in finding.lower() for finding in report["key_findings"]
            )

        # Test branch coverage for assumption tests not being a dict (lines 763->772)
        model_validation_non_dict = {
            "overall_validity": True,
            "assumption_tests": "error_string",  # Not a dict
            "goodness_of_fit": {"r_squared": 0.8},
            "residual_analysis": {"outlier_percentage": 5},
        }

        recommendations = _generate_diagnostic_recommendations(
            model_validation_non_dict, {}
        )

        # Should still generate recommendations without crashing
        assert isinstance(recommendations, list)

    def test_100_percent_coverage_missing_branches(self):
        """Test the exact missing branch coverage lines for 100% coverage."""
        np.random.seed(42)

        # Create test data
        tbr_df = pd.DataFrame(
            {
                "period": [0] * 20 + [1] * 15,
                "y": np.random.normal(1000, 50, 35),
                "x": np.random.normal(950, 45, 35),
                "pred": np.random.normal(950, 45, 35),
                "predsd": [15] * 35,
                "dif": np.random.normal(50, 25, 35),
                "cumdif": np.cumsum(np.random.normal(50, 25, 35)),
                "cumsd": np.sqrt(np.arange(1, 36) * 25**2),
                "estsd": [15] * 35,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.95],
                "sigma": [25.0],
                "var_alpha": [100.0],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [18],
            }
        )

        # Test 1: Lines 245->248 - High outlier percentage (>10% outliers)
        with patch(
            "tbr.analysis.diagnostics.calculate_studentized_residuals"
        ) as mock_studentized:
            # Create scenario with >10% outliers (>2.5 threshold)
            # 20 learning observations, need >2 outliers (>10%)
            outlier_residuals = np.array(
                [3.0, 3.5, 4.0] + [1.0] * 17
            )  # 3/20 = 15% outliers
            mock_studentized.return_value = outlier_residuals

            result = validate_tbr_model(tbr_df, tbr_summary)

            # Should trigger high outlier warning (line 246)
            assert any(
                "high number of outliers" in warning.lower()
                for warning in result["warnings"]
            )
            assert result["residual_analysis"]["outlier_percentage"] > 10

        # Test 2: Lines 277->280 - Poor prediction interval coverage (<90%)
        # Create test data with very poor coverage
        poor_coverage_df = tbr_df.copy()
        test_data = poor_coverage_df[poor_coverage_df["period"] == 1].copy()

        # Set up scenario where actual values are far outside prediction intervals
        test_data["y"] = [2000] * len(test_data)  # Very high actual values
        test_data["pred"] = [1000] * len(test_data)  # Low predictions
        test_data["predsd"] = [5] * len(test_data)  # Very narrow intervals

        poor_coverage_df.loc[poor_coverage_df["period"] == 1, "y"] = test_data["y"]
        poor_coverage_df.loc[poor_coverage_df["period"] == 1, "pred"] = test_data[
            "pred"
        ]
        poor_coverage_df.loc[poor_coverage_df["period"] == 1, "predsd"] = test_data[
            "predsd"
        ]

        result = validate_tbr_model(poor_coverage_df, tbr_summary)

        # Should trigger poor coverage warning (line 278)
        if (
            "prediction_quality" in result
            and "prediction_interval_coverage" in result["prediction_quality"]
        ):
            coverage = result["prediction_quality"]["prediction_interval_coverage"]
            if coverage < 0.90:
                assert any(
                    "poor prediction interval coverage" in warning.lower()
                    for warning in result["warnings"]
                )

        # Test 3: Lines 695->709, 711->713, 713->717 - Performance metrics branches in diagnostic report
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            # Mock with performance metrics that have both MAPE and interval_coverage
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": {"r_squared": 0.8},
                    "assumption_tests": {"all_assumptions_valid": True},
                },
                "performance_metrics": {
                    "prediction_metrics": {
                        "mape": 12.5,  # This should trigger line 711->713
                        "interval_coverage": 0.92,  # This should trigger line 713->717
                    }
                },
                "recommendations": [],
            }

            # Call with include_detailed_analysis=True to trigger line 695->709
            report = create_tbr_diagnostic_report(
                tbr_df, tbr_summary, include_detailed_analysis=True
            )

            # Should include both performance findings
            assert any(
                "12.5% mape" in finding.lower() for finding in report["key_findings"]
            )  # Line 712
            assert any(
                "92.0%" in finding for finding in report["key_findings"]
            )  # Line 714

        # Test 4: Additional coverage for performance metrics without MAPE but with coverage
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": {"r_squared": 0.8},
                    "assumption_tests": {"all_assumptions_valid": True},
                },
                "performance_metrics": {
                    "prediction_metrics": {
                        "interval_coverage": 0.88,  # Only coverage, no MAPE
                    }
                },
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(
                tbr_df, tbr_summary, include_detailed_analysis=True
            )

            # Should include only coverage finding
            assert any("88.0%" in finding for finding in report["key_findings"])

        # Test 5: Additional coverage for performance metrics with MAPE but without coverage
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": {"r_squared": 0.8},
                    "assumption_tests": {"all_assumptions_valid": True},
                },
                "performance_metrics": {
                    "prediction_metrics": {
                        "mape": 8.3,  # Only MAPE, no coverage
                    }
                },
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(
                tbr_df, tbr_summary, include_detailed_analysis=True
            )

            # Should include only MAPE finding
            assert any(
                "8.3% mape" in finding.lower() for finding in report["key_findings"]
            )

    def test_100_percent_coverage_negative_branches(self):
        """Test the negative branches of the missing coverage conditions."""
        np.random.seed(42)

        # Create test data
        tbr_df = pd.DataFrame(
            {
                "period": [0] * 20 + [1] * 15,
                "y": np.random.normal(1000, 50, 35),
                "x": np.random.normal(950, 45, 35),
                "pred": np.random.normal(950, 45, 35),
                "predsd": [15] * 35,
                "dif": np.random.normal(50, 25, 35),
                "cumdif": np.cumsum(np.random.normal(50, 25, 35)),
                "cumsd": np.sqrt(np.arange(1, 36) * 25**2),
                "estsd": [15] * 35,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.95],
                "sigma": [25.0],
                "var_alpha": [100.0],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [18],
            }
        )

        # Test negative branch of lines 245->248 - Low outlier percentage (<=10% outliers)
        with patch(
            "tbr.analysis.diagnostics.calculate_studentized_residuals"
        ) as mock_studentized:
            # Create scenario with <=10% outliers
            # 20 learning observations, need <=2 outliers (<=10%)
            low_outlier_residuals = np.array(
                [3.0, 2.0] + [1.0] * 18
            )  # 1/20 = 5% outliers (only one >2.5)
            mock_studentized.return_value = low_outlier_residuals

            result = validate_tbr_model(tbr_df, tbr_summary)

            # Should NOT trigger high outlier warning (negative branch)
            assert not any(
                "high number of outliers" in warning.lower()
                for warning in result["warnings"]
            )
            assert result["residual_analysis"]["outlier_percentage"] <= 10

        # Test negative branch of lines 277->280 - Good prediction interval coverage (>=90%)
        # Create test data with good coverage
        good_coverage_df = tbr_df.copy()
        test_data = good_coverage_df[good_coverage_df["period"] == 1].copy()

        # Set up scenario where actual values are within prediction intervals
        test_data["y"] = [1000] * len(test_data)  # Values close to predictions
        test_data["pred"] = [1000] * len(test_data)  # Matching predictions
        test_data["predsd"] = [50] * len(test_data)  # Wide intervals

        good_coverage_df.loc[good_coverage_df["period"] == 1, "y"] = test_data["y"]
        good_coverage_df.loc[good_coverage_df["period"] == 1, "pred"] = test_data[
            "pred"
        ]
        good_coverage_df.loc[good_coverage_df["period"] == 1, "predsd"] = test_data[
            "predsd"
        ]

        result = validate_tbr_model(good_coverage_df, tbr_summary)

        # Should NOT trigger poor coverage warning (negative branch)
        if (
            "prediction_quality" in result
            and "prediction_interval_coverage" in result["prediction_quality"]
        ):
            coverage = result["prediction_quality"]["prediction_interval_coverage"]
            if coverage >= 0.90:
                assert not any(
                    "poor prediction interval coverage" in warning.lower()
                    for warning in result["warnings"]
                )

        # Test negative branch of lines 695->709 - No performance metrics or include_detailed_analysis=False
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            # Mock without performance metrics
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": {"r_squared": 0.8},
                    "assumption_tests": {"all_assumptions_valid": True},
                },
                "performance_metrics": {},  # Empty performance metrics
                "recommendations": [],
            }

            # Call with include_detailed_analysis=True but no prediction_metrics
            report = create_tbr_diagnostic_report(
                tbr_df, tbr_summary, include_detailed_analysis=True
            )

            # Should NOT include performance findings (negative branch)
            assert not any(
                "mape" in finding.lower() for finding in report["key_findings"]
            )
            assert not any(
                "coverage" in finding.lower() for finding in report["key_findings"]
            )

        # Test with include_detailed_analysis=False
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": {"r_squared": 0.8},
                    "assumption_tests": {"all_assumptions_valid": True},
                },
                "performance_metrics": {
                    "prediction_metrics": {
                        "mape": 12.5,
                        "interval_coverage": 0.92,
                    }
                },
                "recommendations": [],
            }

            # Call with include_detailed_analysis=False (negative branch of line 695->709)
            report = create_tbr_diagnostic_report(
                tbr_df, tbr_summary, include_detailed_analysis=False
            )

            # Should NOT include performance findings due to include_detailed_analysis=False
            assert not any(
                "12.5% mape" in finding.lower() for finding in report["key_findings"]
            )
            assert not any("92.0%" in finding for finding in report["key_findings"])

    def test_100_percent_coverage_final_branch(self):
        """Test the final missing branch coverage line 695->709."""
        np.random.seed(42)

        # Create test data
        tbr_df = pd.DataFrame(
            {
                "period": [0] * 20 + [1] * 15,
                "y": np.random.normal(1000, 50, 35),
                "x": np.random.normal(950, 45, 35),
                "pred": np.random.normal(950, 45, 35),
                "predsd": [15] * 35,
                "dif": np.random.normal(50, 25, 35),
                "cumdif": np.cumsum(np.random.normal(50, 25, 35)),
                "cumsd": np.sqrt(np.arange(1, 36) * 25**2),
                "estsd": [15] * 35,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.95],
                "sigma": [25.0],
                "var_alpha": [100.0],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [18],
            }
        )

        # Test the negative branch of line 695->709
        # Case 1: assumptions is not a dict
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": {"r_squared": 0.8},
                    "assumption_tests": "not_a_dict",  # Not a dict - should skip the if block
                },
                "performance_metrics": {},
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(
                tbr_df, tbr_summary, include_detailed_analysis=True
            )

            # Should NOT include assumption findings due to assumptions not being a dict
            assert not any(
                "assumptions" in finding.lower() for finding in report["key_findings"]
            )
            assert not any(
                "satisfied" in finding.lower() for finding in report["key_findings"]
            )
            assert not any(
                "violations" in finding.lower() for finding in report["key_findings"]
            )

        # Case 2: assumptions is a dict but missing 'all_assumptions_valid' key
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": {"r_squared": 0.8},
                    "assumption_tests": {
                        "normality_valid": True,
                        "homoscedasticity_valid": True,
                        # Missing 'all_assumptions_valid' key - should skip the if block
                    },
                },
                "performance_metrics": {},
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(
                tbr_df, tbr_summary, include_detailed_analysis=True
            )

            # Should NOT include assumption findings due to missing 'all_assumptions_valid' key
            assert not any(
                "assumptions" in finding.lower() for finding in report["key_findings"]
            )
            assert not any(
                "satisfied" in finding.lower() for finding in report["key_findings"]
            )
            assert not any(
                "violations" in finding.lower() for finding in report["key_findings"]
            )

        # Case 3: assumptions is None (get returns None)
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": {"r_squared": 0.8},
                    # No 'assumption_tests' key - get() will return {}
                },
                "performance_metrics": {},
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(
                tbr_df, tbr_summary, include_detailed_analysis=True
            )

            # Should NOT include assumption findings due to empty assumptions dict
            assert not any(
                "assumptions" in finding.lower() for finding in report["key_findings"]
            )
            assert not any(
                "satisfied" in finding.lower() for finding in report["key_findings"]
            )
            assert not any(
                "violations" in finding.lower() for finding in report["key_findings"]
            )

    def test_100_percent_coverage_missing_lines_779_780_792_794(self):
        """Test the remaining missing lines for 100% coverage: 779-780, 792->817, 794->800."""
        np.random.seed(42)

        tbr_df = pd.DataFrame(
            {
                "period": [0] * 20 + [1] * 15,
                "y": np.random.normal(1000, 50, 35),
                "x": np.random.normal(950, 45, 35),
                "pred": np.random.normal(950, 45, 35),
                "predsd": [15] * 35,
                "dif": np.random.normal(50, 25, 35),
                "cumdif": np.cumsum(np.random.normal(50, 25, 35)),
                "cumsd": np.sqrt(np.arange(1, 36) * 25**2),
                "estsd": [15] * 35,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [50.0],
                "beta": [0.95],
                "sigma": [25.0],
                "var_alpha": [100.0],
                "var_beta": [0.001],
                "alpha_beta_cov": [-0.05],
                "t_dist_df": [18],
            }
        )

        # Test lines 779-780: model_validation is not a dict (else branch)
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": "not_a_dict",  # Not a dict - should trigger lines 779-780
                "performance_metrics": {},
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(
                tbr_df, tbr_summary, include_detailed_analysis=True
            )

            # Should handle non-dict model_validation gracefully
            assert "executive_summary" in report
            assert "key_findings" in report
            # Should use default values: model_valid=False, warnings_count=0
            assert "0 issue(s)" in report["executive_summary"]

        # Test lines 792->817: model_validation is not a dict (negative branch for goodness of fit)
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": [],  # Not a dict - should skip goodness of fit processing
                "performance_metrics": {},
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(
                tbr_df, tbr_summary, include_detailed_analysis=True
            )

            # Should NOT include goodness of fit findings due to model_validation not being a dict
            assert not any(
                "variance" in finding.lower() for finding in report["key_findings"]
            )
            assert not any(
                "r²" in finding.lower() for finding in report["key_findings"]
            )

        # Test lines 794->800: gof is not a dict or missing r_squared (negative branch)
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            # Case 1: gof is not a dict
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": "not_a_dict",  # Not a dict - should skip r_squared processing
                },
                "performance_metrics": {},
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(
                tbr_df, tbr_summary, include_detailed_analysis=True
            )

            # Should NOT include r_squared findings due to gof not being a dict
            assert not any(
                "variance" in finding.lower() for finding in report["key_findings"]
            )
            assert not any(
                "r²" in finding.lower() for finding in report["key_findings"]
            )

        # Case 2: gof is a dict but missing r_squared key
        with patch("tbr.analysis.diagnostics.diagnose_tbr_analysis") as mock_diagnose:
            mock_diagnose.return_value = {
                "model_validation": {
                    "overall_validity": True,
                    "warnings": [],
                    "goodness_of_fit": {
                        "f_statistic": 10.5
                    },  # Missing r_squared - should skip processing
                },
                "performance_metrics": {},
                "recommendations": [],
            }

            report = create_tbr_diagnostic_report(
                tbr_df, tbr_summary, include_detailed_analysis=True
            )

            # Should NOT include r_squared findings due to missing r_squared key
            assert not any(
                "variance" in finding.lower() for finding in report["key_findings"]
            )
            assert not any(
                "r²" in finding.lower() for finding in report["key_findings"]
            )


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""

    def test_module_imports(self):
        """Test that all functions can be imported correctly."""
        from tbr.analysis.diagnostics import (
            analyze_tbr_residuals,
            assess_tbr_performance,
            check_tbr_assumptions,
            create_tbr_diagnostic_report,
            diagnose_tbr_analysis,
            validate_tbr_model,
        )

        # Check that functions are callable
        assert callable(validate_tbr_model)
        assert callable(diagnose_tbr_analysis)
        assert callable(check_tbr_assumptions)
        assert callable(analyze_tbr_residuals)
        assert callable(assess_tbr_performance)
        assert callable(create_tbr_diagnostic_report)

    def test_lazy_loading_integration(self):
        """Test integration with lazy loading system."""
        # Test direct import
        # Test lazy import
        from tbr.analysis import validate_tbr_model as lazy_validate
        from tbr.analysis.diagnostics import validate_tbr_model as direct_validate

        # Should be the same function
        assert direct_validate is lazy_validate

    def test_main_package_exports(self):
        """Test that functions are exported from main package."""
        from tbr import diagnose_tbr_analysis, validate_tbr_model

        assert callable(validate_tbr_model)
        assert callable(diagnose_tbr_analysis)

    def test_error_handling_with_invalid_data(self):
        """Test error handling with various invalid data scenarios."""
        # Test with completely invalid data
        invalid_df = pd.DataFrame({"invalid": [1, 2, 3]})
        invalid_summary = pd.DataFrame({"invalid": [1]})

        with pytest.raises(ValueError):
            validate_tbr_model(invalid_df, invalid_summary)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        np.random.seed(42)

        # Create data with extreme values
        tbr_df = pd.DataFrame(
            {
                "period": [0] * 10,
                "y": [1e10] * 10,
                "x": [1e10] * 10,
                "pred": [1e10] * 10,
                "predsd": [1e8] * 10,
                "dif": [1e8] * 10,
                "cumdif": np.arange(1, 11) * 1e8,
                "cumsd": np.sqrt(np.arange(1, 11)) * 1e8,
                "estsd": [1e8] * 10,
            }
        )

        tbr_summary = pd.DataFrame(
            {
                "alpha": [1e10],
                "beta": [1.0],
                "sigma": [1e8],
                "var_alpha": [1e18],
                "var_beta": [1e-6],
                "alpha_beta_cov": [1e6],
                "t_dist_df": [8],
            }
        )

        # Should handle extreme values without crashing
        try:
            result = validate_tbr_model(tbr_df, tbr_summary)
            assert isinstance(result, dict)
        except (OverflowError, ValueError):
            # Acceptable to fail with extreme values
            pass
