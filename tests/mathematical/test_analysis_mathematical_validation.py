"""
Mathematical Validation Tests for Analysis Framework.

This module provides comprehensive mathematical validation for the analysis
framework implemented in Phase 3. Tests ensure mathematical accuracy,
consistency with theoretical foundations, and cross-validation with
functional implementations.

Test Categories
---------------
1. Mathematical Consistency - Core mathematical relationships and properties
2. Statistical Properties - T-distribution, confidence intervals, probabilities
3. Numerical Stability - Extreme values and edge case handling
4. Formula Validation - Direct validation against mathematical derivations
5. Edge Case Mathematics - Boundary conditions and special cases

All tests use proper TBR DataFrame format with required columns:
- period, y, x, pred, predsd, dif, cumdif, cumsd, estsd
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tbr.analysis.incremental import create_incremental_tbr_summaries
from tbr.analysis.subinterval import compute_interval_estimate_and_ci
from tbr.analysis.summary import create_tbr_summary


def create_proper_tbr_dataframe(n_pretest=3, n_test=5, seed=42):
    """Helper to create properly formatted TBR DataFrame."""
    np.random.seed(seed)

    # Pretest data
    pretest_y = np.random.normal(100, 5, n_pretest)
    pretest_x = np.random.normal(95, 4, n_pretest)
    pretest_pred = pretest_x * 1.05  # Simple relationship
    pretest_dif = pretest_y - pretest_pred

    # Test data
    test_y = np.random.normal(110, 6, n_test)
    test_x = np.random.normal(105, 5, n_test)
    test_pred = test_x * 1.05
    test_dif = test_y - test_pred
    test_cumdif = np.cumsum(test_dif)
    test_cumsd = np.sqrt(np.arange(1, n_test + 1) * 4.0)  # Cumulative SD

    return pd.DataFrame(
        {
            "period": [0] * n_pretest + [1] * n_test,
            "y": np.concatenate([pretest_y, test_y]),
            "x": np.concatenate([pretest_x, test_x]),
            "pred": np.concatenate([pretest_pred, test_pred]),
            "predsd": np.concatenate(
                [np.zeros(n_pretest), np.random.uniform(2, 3, n_test)]
            ),
            "dif": np.concatenate([pretest_dif, test_dif]),
            "cumdif": np.concatenate([[np.nan] * n_pretest, test_cumdif]),
            "cumsd": np.concatenate([np.zeros(n_pretest), test_cumsd]),
            "estsd": np.concatenate(
                [np.random.uniform(1, 2, n_pretest), [np.nan] * n_test]
            ),
        }
    )


class TestAnalysisMathematicalConsistency:
    """Mathematical consistency tests across analysis modules."""

    @pytest.fixture
    def mathematical_test_data(self):
        """Create precise mathematical test data."""
        tbr_df = create_proper_tbr_dataframe(n_pretest=4, n_test=6, seed=123)

        params = {
            "alpha": 100.0,
            "beta": 1.0,
            "sigma": 2.0,
            "var_alpha": 4.0,
            "var_beta": 0.01,
            "cov_alpha_beta": -0.02,
            "degrees_freedom": 5,
            "level": 0.90,
            "threshold": 0.0,
        }

        return tbr_df, params

    def test_mathematical_additivity_property(self, mathematical_test_data):
        """Test mathematical additivity property of analysis estimates."""
        tbr_df, params = mathematical_test_data
        tbr_summary = pd.DataFrame(
            {"sigma": [params["sigma"]], "t_dist_df": [params["degrees_freedom"]]}
        )

        # Test additivity: interval(1,3) + interval(4,6) should relate to interval(1,6)
        interval_1_3 = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, 1, 3, params["level"]
        )
        interval_4_6 = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, 4, 6, params["level"]
        )
        interval_1_6 = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, 1, 6, params["level"]
        )

        # Additive property for estimates
        combined_estimate = interval_1_3["estimate"] + interval_4_6["estimate"]
        np.testing.assert_allclose(
            combined_estimate,
            interval_1_6["estimate"],
            rtol=1e-12,
            err_msg="Subinterval estimates should be additive",
        )

    def test_mathematical_monotonicity_properties(self, mathematical_test_data):
        """Test monotonicity properties with different confidence levels."""
        tbr_df, params = mathematical_test_data

        confidence_levels = [0.80, 0.90, 0.95]
        interval_widths = []

        for level in confidence_levels:
            test_params = params.copy()
            test_params["level"] = level
            result = create_tbr_summary(tbr_df, **test_params)
            width = result.iloc[0]["upper"] - result.iloc[0]["lower"]
            interval_widths.append(width)

        # Validate monotonicity: higher confidence = wider intervals
        for i in range(1, len(interval_widths)):
            assert (
                interval_widths[i] > interval_widths[i - 1]
            ), "Confidence interval should widen with higher confidence level"

    def test_mathematical_incremental_consistency(self, mathematical_test_data):
        """Test mathematical consistency between incremental and summary analysis."""
        tbr_df, params = mathematical_test_data

        # Generate results
        summary_result = create_tbr_summary(tbr_df, **params)
        incremental_result = create_incremental_tbr_summaries(tbr_df, **params)

        # Final incremental should match summary
        final_incremental_estimate = incremental_result.iloc[-1]["estimate"]
        summary_estimate = summary_result.iloc[0]["estimate"]

        np.testing.assert_allclose(
            summary_estimate,
            final_incremental_estimate,
            rtol=1e-12,
            err_msg="Summary and final incremental estimates should match",
        )

    def test_mathematical_cross_module_relationships(self, mathematical_test_data):
        """Test mathematical relationships across analysis modules."""
        tbr_df, params = mathematical_test_data
        tbr_summary = pd.DataFrame(
            {"sigma": [params["sigma"]], "t_dist_df": [params["degrees_freedom"]]}
        )

        # Get results from different modules
        summary_result = create_tbr_summary(tbr_df, **params)
        # incremental_result = create_incremental_tbr_summaries(tbr_df, **params)  # Not used in current test

        # Test period length
        test_period_length = len(tbr_df[tbr_df["period"] == 1])
        subinterval_result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, 1, test_period_length, params["level"]
        )

        # All should give same estimate for full period
        np.testing.assert_allclose(
            summary_result.iloc[0]["estimate"],
            subinterval_result["estimate"],
            rtol=1e-12,
            err_msg="Cross-module estimates should be consistent",
        )


class TestAnalysisStatisticalProperties:
    """Statistical property validation for analysis framework."""

    @pytest.fixture
    def statistical_test_data(self):
        """Create data for statistical property testing."""
        tbr_df = create_proper_tbr_dataframe(n_pretest=5, n_test=8, seed=456)

        params = {
            "alpha": 105.0,
            "beta": 0.98,
            "sigma": 3.0,
            "var_alpha": 6.0,
            "var_beta": 0.005,
            "cov_alpha_beta": -0.01,
            "degrees_freedom": 8,
            "level": 0.85,
            "threshold": 1.0,
        }

        return tbr_df, params

    def test_statistical_t_distribution_properties(self, statistical_test_data):
        """Test t-distribution properties in analysis results."""
        tbr_df, params = statistical_test_data

        result = create_tbr_summary(tbr_df, **params)

        # Extract values - use actual column names from functional implementation
        estimate = result.iloc[0]["estimate"]
        se = result.iloc[0]["se"]  # Standard error
        lower = result.iloc[0]["lower"]
        upper = result.iloc[0]["upper"]
        level = result.iloc[0]["level"]
        df = result.iloc[0]["t_dist_df"]

        # Calculate expected t-critical value
        alpha_level = 1 - level
        expected_t_critical = stats.t.ppf(1 - alpha_level / 2, df=df)

        # Verify interval construction
        expected_margin = expected_t_critical * se
        expected_lower = estimate - expected_margin
        expected_upper = estimate + expected_margin

        np.testing.assert_allclose(lower, expected_lower, rtol=1e-10)
        np.testing.assert_allclose(upper, expected_upper, rtol=1e-10)

    def test_statistical_confidence_interval_properties(self, statistical_test_data):
        """Test confidence interval mathematical properties."""
        tbr_df, params = statistical_test_data

        # Test different confidence levels
        levels = [0.80, 0.90, 0.95]
        results = []

        for level in levels:
            test_params = params.copy()
            test_params["level"] = level
            result = create_tbr_summary(tbr_df, **test_params)
            results.append(result.iloc[0])

        # Verify interval width increases with confidence level
        for i in range(1, len(results)):
            width_prev = results[i - 1]["upper"] - results[i - 1]["lower"]
            width_curr = results[i]["upper"] - results[i]["lower"]
            assert (
                width_curr > width_prev
            ), "Higher confidence should give wider intervals"

    def test_statistical_posterior_probability_properties(self, statistical_test_data):
        """Test posterior probability mathematical properties."""
        tbr_df, params = statistical_test_data

        # Test different thresholds
        thresholds = [-2.0, 0.0, 2.0]
        probabilities = []

        for threshold in thresholds:
            test_params = params.copy()
            test_params["threshold"] = threshold
            result = create_tbr_summary(tbr_df, **test_params)
            probabilities.append(result.iloc[0]["prob"])

        # Verify probability decreases as threshold increases (for positive estimates)
        estimate = create_tbr_summary(tbr_df, **params).iloc[0]["estimate"]
        if estimate > 0:
            assert (
                probabilities[0] > probabilities[1] > probabilities[2]
            ), "Probability should decrease as threshold increases above estimate"

    def test_statistical_incremental_progression_properties(
        self, statistical_test_data
    ):
        """Test statistical properties of incremental progression."""
        tbr_df, params = statistical_test_data

        incremental_result = create_incremental_tbr_summaries(tbr_df, **params)

        # Verify incremental properties
        estimates = incremental_result["estimate"].values
        precisions = incremental_result["precision"].values

        # Generally, precision should increase (intervals get wider) as we add more days
        # This is because cumulative variance typically increases
        assert len(estimates) > 1, "Should have multiple incremental results"
        assert all(np.isfinite(estimates)), "All estimates should be finite"
        assert all(precisions > 0), "All precisions should be positive"


class TestAnalysisNumericalStability:
    """Numerical stability tests for analysis framework."""

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme parameter values."""
        # Test with very small values
        small_tbr_df = create_proper_tbr_dataframe(n_pretest=2, n_test=3, seed=789)
        # Scale down the values
        for col in ["y", "x", "pred", "dif", "cumdif"]:
            if col in small_tbr_df.columns:
                small_tbr_df[col] = small_tbr_df[col] * 1e-6
        small_tbr_df["cumsd"] = small_tbr_df["cumsd"] * 1e-6

        small_params = {
            "alpha": 1e-6,
            "beta": 1.0,
            "sigma": 1e-7,
            "var_alpha": 1e-12,
            "var_beta": 1e-15,
            "cov_alpha_beta": -1e-14,
            "degrees_freedom": 2,
            "level": 0.90,
            "threshold": 0.0,
        }

        # Should handle small values without numerical issues
        try:
            small_result = create_tbr_summary(small_tbr_df, **small_params)
            assert np.isfinite(
                small_result.iloc[0]["estimate"]
            ), "Small values should produce finite results"
            assert np.isfinite(
                small_result.iloc[0]["precision"]
            ), "Small values should produce finite precision"
        except (ValueError, RuntimeError) as e:
            # Acceptable if proper error handling for extreme values
            assert any(
                word in str(e).lower()
                for word in ["numerical", "precision", "stability", "finite"]
            ), f"Should have appropriate error message for extreme values, got: {e}"

    def test_numerical_precision_consistency(self):
        """Test numerical precision consistency across analysis modules."""
        tbr_df = create_proper_tbr_dataframe(n_pretest=3, n_test=4, seed=999)

        params = {
            "alpha": 100.0,
            "beta": 1.0,
            "sigma": 2.0,
            "var_alpha": 4.0,
            "var_beta": 0.0,
            "cov_alpha_beta": 0.0,
            "degrees_freedom": 3,
            "level": 0.90,
            "threshold": 0.0,
        }

        # Test precision across multiple runs (should be deterministic)
        results = []
        for _ in range(3):
            result = create_tbr_summary(tbr_df, **params)
            results.append(result.iloc[0]["estimate"])

        # All results should be identical (deterministic)
        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0],
                results[i],
                rtol=1e-15,
                err_msg="Results should be deterministic across runs",
            )

    def test_numerical_edge_case_handling(self):
        """Test numerical handling of mathematical edge cases."""
        # Test with zero variance scenario
        zero_var_df = create_proper_tbr_dataframe(n_pretest=2, n_test=2, seed=111)

        zero_var_params = {
            "alpha": 100.0,
            "beta": 1.0,
            "sigma": 0.1,
            "var_alpha": 0.01,
            "var_beta": 0.0,
            "cov_alpha_beta": 0.0,
            "degrees_freedom": 2,
            "level": 0.90,
            "threshold": 0.0,
        }

        # Should handle zero variance scenario
        zero_var_result = create_tbr_summary(zero_var_df, **zero_var_params)
        assert np.isfinite(
            zero_var_result.iloc[0]["estimate"]
        ), "Zero variance should produce finite estimate"
        assert (
            zero_var_result.iloc[0]["precision"] >= 0
        ), "Precision should be non-negative"


class TestAnalysisFormulaValidation:
    """Direct formula validation tests."""

    def test_tbr_summary_formula_validation(self):
        """Test TBR summary mathematical formula validation."""
        tbr_df = create_proper_tbr_dataframe(n_pretest=2, n_test=3, seed=222)

        params = {
            "alpha": 100.0,
            "beta": 1.0,
            "sigma": 2.0,
            "var_alpha": 4.0,
            "var_beta": 0.0,
            "cov_alpha_beta": 0.0,
            "degrees_freedom": 2,
            "level": 0.90,
            "threshold": 0.0,
        }

        result = create_tbr_summary(tbr_df, **params)

        # Verify basic mathematical relationships
        estimate = result.iloc[0]["estimate"]
        precision = result.iloc[0]["precision"]
        lower = result.iloc[0]["lower"]
        upper = result.iloc[0]["upper"]

        # Interval should be symmetric around estimate
        np.testing.assert_allclose(
            estimate - lower,
            upper - estimate,
            rtol=1e-10,
            err_msg="Confidence interval should be symmetric",
        )

        # Precision should be half the interval width
        expected_precision = (upper - lower) / 2
        np.testing.assert_allclose(
            precision,
            expected_precision,
            rtol=1e-10,
            err_msg="Precision should be half the interval width",
        )

    def test_incremental_formula_validation(self):
        """Test incremental analysis mathematical formula validation."""
        tbr_df = create_proper_tbr_dataframe(n_pretest=2, n_test=4, seed=333)

        params = {
            "alpha": 100.0,
            "beta": 1.0,
            "sigma": 2.0,
            "var_alpha": 4.0,
            "var_beta": 0.0,
            "cov_alpha_beta": 0.0,
            "degrees_freedom": 3,
            "level": 0.90,
            "threshold": 0.0,
        }

        result = create_incremental_tbr_summaries(tbr_df, **params)

        # Verify incremental progression properties
        estimates = result["estimate"].values

        # Should have one result per test day
        expected_length = len(tbr_df[tbr_df["period"] == 1])
        assert (
            len(result) == expected_length
        ), f"Should have {expected_length} incremental results"

        # All estimates should be finite
        assert all(np.isfinite(estimates)), "All incremental estimates should be finite"

    def test_subinterval_formula_validation(self):
        """Test subinterval analysis mathematical formula validation."""
        tbr_df = create_proper_tbr_dataframe(n_pretest=3, n_test=6, seed=444)
        tbr_summary = pd.DataFrame({"sigma": [2.0], "t_dist_df": [5]})

        # Test single day
        single_day_result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, 1, 1, 0.90
        )

        # Test multiple days
        multi_day_result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, 1, 3, 0.90
        )

        # Basic validation
        assert np.isfinite(
            single_day_result["estimate"]
        ), "Single day estimate should be finite"
        assert np.isfinite(
            multi_day_result["estimate"]
        ), "Multi-day estimate should be finite"
        assert (
            single_day_result["precision"] > 0
        ), "Single day precision should be positive"
        assert (
            multi_day_result["precision"] > 0
        ), "Multi-day precision should be positive"


class TestAnalysisEdgeCaseMathematics:
    """Edge case mathematical behavior tests."""

    def test_single_day_mathematical_behavior(self):
        """Test mathematical behavior with single day analysis."""
        tbr_df = create_proper_tbr_dataframe(n_pretest=2, n_test=1, seed=555)

        params = {
            "alpha": 100.0,
            "beta": 1.0,
            "sigma": 2.0,
            "var_alpha": 4.0,
            "var_beta": 0.0,
            "cov_alpha_beta": 0.0,
            "degrees_freedom": 1,
            "level": 0.90,
            "threshold": 0.0,
        }

        # All analysis methods should work with single day
        summary_result = create_tbr_summary(tbr_df, **params)
        incremental_result = create_incremental_tbr_summaries(tbr_df, **params)

        # Should have consistent results
        assert (
            len(incremental_result) == 1
        ), "Should have one incremental result for single day"
        np.testing.assert_allclose(
            summary_result.iloc[0]["estimate"],
            incremental_result.iloc[0]["estimate"],
            rtol=1e-12,
            err_msg="Single day results should be consistent",
        )

    def test_boundary_confidence_levels_mathematics(self):
        """Test mathematical behavior at boundary confidence levels."""
        tbr_df = create_proper_tbr_dataframe(n_pretest=3, n_test=4, seed=666)

        params = {
            "alpha": 100.0,
            "beta": 1.0,
            "sigma": 2.0,
            "var_alpha": 4.0,
            "var_beta": 0.0,
            "cov_alpha_beta": 0.0,
            "degrees_freedom": 3,
            "level": 0.50,
            "threshold": 0.0,
        }

        # Test very low confidence level (should have narrow intervals)
        low_conf_result = create_tbr_summary(tbr_df, **params)

        # Test higher confidence level
        params["level"] = 0.99
        high_conf_result = create_tbr_summary(tbr_df, **params)

        # Higher confidence should give wider intervals
        low_width = low_conf_result.iloc[0]["upper"] - low_conf_result.iloc[0]["lower"]
        high_width = (
            high_conf_result.iloc[0]["upper"] - high_conf_result.iloc[0]["lower"]
        )

        assert high_width > low_width, "Higher confidence should give wider intervals"

    def test_zero_threshold_mathematical_properties(self):
        """Test mathematical properties with zero threshold."""
        tbr_df = create_proper_tbr_dataframe(n_pretest=2, n_test=3, seed=777)

        params = {
            "alpha": 100.0,
            "beta": 1.0,
            "sigma": 2.0,
            "var_alpha": 4.0,
            "var_beta": 0.0,
            "cov_alpha_beta": 0.0,
            "degrees_freedom": 2,
            "level": 0.90,
            "threshold": 0.0,
        }

        result = create_tbr_summary(tbr_df, **params)

        # With zero threshold, probability should relate to whether estimate is positive/negative
        estimate = result.iloc[0]["estimate"]
        prob = result.iloc[0]["prob"]

        # Basic probability properties
        assert 0 <= prob <= 1, "Probability should be between 0 and 1"

        # If estimate is positive, probability should be > 0.5 (for zero threshold)
        if estimate > 0:
            assert (
                prob > 0.5
            ), "Positive estimate should have >50% probability of exceeding zero"
        elif estimate < 0:
            assert (
                prob < 0.5
            ), "Negative estimate should have <50% probability of exceeding zero"
