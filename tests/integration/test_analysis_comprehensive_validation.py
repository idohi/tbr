"""
Comprehensive Analysis Framework Validation Tests - Streamlined Version.

This module provides focused validation for the entire analysis framework
implemented in Phase 3 (Tasks 5.1-5.4). Tests ensure integration consistency,
mathematical accuracy, and professional standards compliance across all
analysis modules: summary, incremental, and subinterval.

Test Categories
---------------
1. Core Integration - Essential module interaction validation
2. Cross-Validation - Consistency with functional implementation
3. Mathematical Properties - Key mathematical relationships
4. Performance Validation - Scalability and efficiency testing

This streamlined version focuses on the most critical validation requirements
while maintaining comprehensive coverage of the analysis framework.
"""

import time

import numpy as np
import pandas as pd
import pytest

from tbr.analysis.incremental import create_incremental_tbr_summaries
from tbr.analysis.subinterval import (
    analyze_multiple_subintervals,
    compute_interval_estimate_and_ci,
    create_subinterval_summary,
)

# Import all analysis modules for comprehensive testing
from tbr.analysis.summary import create_tbr_summary

# Import core modules for integration testing
from tbr.core.effects import compute_interval_estimate_and_ci as core_compute_interval

# Import functional implementation for cross-validation
from tbr.functional.tbr_functions import (
    create_incremental_tbr_summaries as functional_create_incremental_tbr_summaries,
)
from tbr.functional.tbr_functions import (
    create_tbr_summary as functional_create_tbr_summary,
)


def create_test_tbr_dataframe(n_pretest=5, n_test=10, seed=42):
    """Helper function to create proper TBR DataFrame for testing."""
    np.random.seed(seed)

    # Create pretest data
    pretest_y = np.random.normal(100, 10, n_pretest)
    pretest_x = np.random.normal(95, 8, n_pretest)
    pretest_pred = pretest_x * 1.05  # Simple linear relationship
    pretest_dif = pretest_y - pretest_pred

    # Create test data
    test_y = np.random.normal(110, 12, n_test)
    test_x = np.random.normal(95, 8, n_test)
    test_pred = test_x * 1.05
    test_dif = test_y - test_pred
    test_cumdif = np.cumsum(test_dif)
    test_cumsd = np.sqrt(np.arange(1, n_test + 1) * 10.0**2)

    # Combine all data
    tbr_df = pd.DataFrame(
        {
            "period": [0] * n_pretest + [1] * n_test,
            "y": np.concatenate([pretest_y, test_y]),
            "x": np.concatenate([pretest_x, test_x]),
            "pred": np.concatenate([pretest_pred, test_pred]),
            "predsd": np.concatenate(
                [np.zeros(n_pretest), np.random.uniform(2, 4, n_test)]
            ),
            "dif": np.concatenate([pretest_dif, test_dif]),
            "cumdif": np.concatenate([[np.nan] * n_pretest, test_cumdif]),
            "cumsd": np.concatenate([np.zeros(n_pretest), test_cumsd]),
            "estsd": np.concatenate(
                [np.random.uniform(2, 4, n_pretest), [np.nan] * n_test]
            ),
        }
    )

    return tbr_df


class TestAnalysisFrameworkCoreIntegration:
    """Core integration tests for analysis framework."""

    @pytest.fixture
    def test_data(self):
        """Create test data and parameters."""
        tbr_df = create_test_tbr_dataframe(n_pretest=8, n_test=12, seed=123)

        params = {
            "alpha": 105.0,
            "beta": 1.05,
            "sigma": 10.0,
            "var_alpha": 20.0,
            "var_beta": 0.001,
            "cov_alpha_beta": -0.02,
            "degrees_freedom": 15,
            "level": 0.90,
            "threshold": 0.0,
        }

        tbr_summary = pd.DataFrame(
            {"sigma": [params["sigma"]], "t_dist_df": [params["degrees_freedom"]]}
        )

        return tbr_df, tbr_summary, params

    def test_analysis_modules_basic_integration(self, test_data):
        """Test basic integration between all analysis modules."""
        tbr_df, tbr_summary, params = test_data

        # Test that all modules work without errors
        summary_result = create_tbr_summary(tbr_df, **params)
        incremental_result = create_incremental_tbr_summaries(tbr_df, **params)
        subinterval_result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, start_day=1, end_day=5, ci_level=params["level"]
        )

        # Basic structure validation
        assert isinstance(summary_result, pd.DataFrame)
        assert len(summary_result) == 1
        assert isinstance(incremental_result, pd.DataFrame)
        assert len(incremental_result) == 12  # Number of test days
        assert isinstance(subinterval_result, dict)
        assert "estimate" in subinterval_result

    def test_analysis_mathematical_consistency(self, test_data):
        """Test mathematical consistency between modules."""
        tbr_df, tbr_summary, params = test_data

        # Generate results
        summary_result = create_tbr_summary(tbr_df, **params)
        incremental_result = create_incremental_tbr_summaries(tbr_df, **params)

        # Test mathematical consistency: final incremental should match summary
        final_incremental_estimate = incremental_result.iloc[-1]["estimate"]
        summary_estimate = summary_result.iloc[0]["estimate"]

        np.testing.assert_allclose(
            summary_estimate,
            final_incremental_estimate,
            rtol=1e-10,
            err_msg="Summary and final incremental estimates should be mathematically consistent",
        )

    def test_analysis_parameter_handling(self, test_data):
        """Test consistent parameter handling across modules."""
        tbr_df, tbr_summary, params = test_data

        # Test with different confidence levels
        for level in [0.80, 0.90, 0.95]:
            test_params = params.copy()
            test_params["level"] = level

            summary_result = create_tbr_summary(tbr_df, **test_params)
            incremental_result = create_incremental_tbr_summaries(tbr_df, **test_params)

            # Verify results are generated successfully
            assert len(summary_result) == 1
            assert len(incremental_result) == 12

            # Verify confidence level affects precision
            summary_precision = summary_result.iloc[0]["precision"]
            final_incremental_precision = incremental_result.iloc[-1]["precision"]

            # Both should be consistent
            np.testing.assert_allclose(
                summary_precision,
                final_incremental_precision,
                rtol=1e-10,
                err_msg=f"Precision should be consistent across modules for level={level}",
            )

    def test_analysis_error_handling_consistency(self, test_data):
        """Test consistent error handling across modules."""
        tbr_df, tbr_summary, params = test_data

        # Test invalid confidence level
        invalid_params = params.copy()
        invalid_params["level"] = 1.5

        # Both modules should handle invalid parameters consistently
        with pytest.raises(ValueError, match="Level must be between 0 and 1"):
            create_tbr_summary(tbr_df, **invalid_params)

        with pytest.raises(ValueError, match="Level must be between 0 and 1"):
            create_incremental_tbr_summaries(tbr_df, **invalid_params)


class TestAnalysisCrossValidation:
    """Cross-validation tests between analysis modules and functional implementation."""

    @pytest.fixture
    def cross_validation_data(self):
        """Create data for cross-validation testing."""
        tbr_df = create_test_tbr_dataframe(n_pretest=6, n_test=8, seed=456)

        params = {
            "alpha": 98.5,
            "beta": 1.02,
            "sigma": 8.5,
            "var_alpha": 15.0,
            "var_beta": 0.0008,
            "cov_alpha_beta": -0.015,
            "degrees_freedom": 12,
            "level": 0.85,
            "threshold": 1.0,
        }

        return tbr_df, params

    def test_analysis_summary_cross_validation(self, cross_validation_data):
        """Cross-validate analysis summary with functional implementation."""
        tbr_df, params = cross_validation_data

        # Analysis module result
        analysis_result = create_tbr_summary(tbr_df, **params)

        # Functional implementation result
        functional_result = functional_create_tbr_summary(tbr_df, **params)

        # Cross-validation with high precision
        for column in ["estimate", "precision", "lower", "upper", "prob"]:
            if (
                column in analysis_result.columns
                and column in functional_result.columns
            ):
                np.testing.assert_allclose(
                    analysis_result[column].iloc[0],
                    functional_result[column].iloc[0],
                    rtol=1e-14,
                    err_msg=f"Analysis and functional {column} should be identical",
                )

    def test_analysis_incremental_cross_validation(self, cross_validation_data):
        """Cross-validate analysis incremental with functional implementation."""
        tbr_df, params = cross_validation_data

        # Analysis module result
        analysis_result = create_incremental_tbr_summaries(tbr_df, **params)

        # Functional implementation result
        functional_result = functional_create_incremental_tbr_summaries(
            tbr_df, **params
        )

        # Cross-validation for key columns
        for column in ["estimate", "precision", "lower", "upper", "prob"]:
            if (
                column in analysis_result.columns
                and column in functional_result.columns
            ):
                np.testing.assert_allclose(
                    analysis_result[column].values,
                    functional_result[column].values,
                    rtol=1e-14,
                    err_msg=f"Analysis and functional incremental {column} should be identical",
                )

    def test_analysis_subinterval_cross_validation(self, cross_validation_data):
        """Cross-validate analysis subinterval with core implementation."""
        tbr_df, params = cross_validation_data

        # Create TBR summary for core function
        tbr_summary = pd.DataFrame(
            {"sigma": [params["sigma"]], "t_dist_df": [params["degrees_freedom"]]}
        )

        # Test specific subinterval
        start_day, end_day = 2, 5

        # Analysis module result
        analysis_result = compute_interval_estimate_and_ci(
            tbr_df, tbr_summary, start_day, end_day, params["level"]
        )

        # Core implementation result
        core_result = core_compute_interval(
            tbr_df, tbr_summary, start_day, end_day, params["level"]
        )

        # Cross-validation with machine precision
        for key in ["estimate", "precision", "lower", "upper"]:
            np.testing.assert_allclose(
                analysis_result[key],
                core_result[key],
                rtol=1e-15,
                err_msg=f"Analysis and core subinterval {key} should be identical",
            )


class TestAnalysisMathematicalProperties:
    """Mathematical property validation for analysis framework."""

    @pytest.fixture
    def mathematical_data(self):
        """Create data for mathematical property testing."""
        tbr_df = create_test_tbr_dataframe(n_pretest=4, n_test=8, seed=789)

        params = {
            "alpha": 100.0,
            "beta": 1.0,
            "sigma": 5.0,
            "var_alpha": 10.0,
            "var_beta": 0.001,
            "cov_alpha_beta": -0.01,
            "degrees_freedom": 10,
            "level": 0.90,
            "threshold": 0.0,
        }

        tbr_summary = pd.DataFrame(
            {"sigma": [params["sigma"]], "t_dist_df": [params["degrees_freedom"]]}
        )

        return tbr_df, tbr_summary, params

    def test_subinterval_additivity_property(self, mathematical_data):
        """Test additivity property of subinterval estimates."""
        tbr_df, tbr_summary, params = mathematical_data

        # Test additivity: interval(1,3) + interval(4,6) = interval(1,6)
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

    def test_confidence_level_monotonicity(self, mathematical_data):
        """Test that confidence intervals widen with higher confidence levels."""
        tbr_df, tbr_summary, params = mathematical_data

        confidence_levels = [0.80, 0.90, 0.95]
        interval_widths = []

        for level in confidence_levels:
            test_params = params.copy()
            test_params["level"] = level

            summary_result = create_tbr_summary(tbr_df, **test_params)
            width = summary_result.iloc[0]["upper"] - summary_result.iloc[0]["lower"]
            interval_widths.append(width)

        # Validate monotonicity: higher confidence = wider intervals
        for i in range(1, len(interval_widths)):
            assert (
                interval_widths[i] > interval_widths[i - 1]
            ), "Confidence interval should widen with higher confidence level"

    def test_incremental_consistency_property(self, mathematical_data):
        """Test consistency between incremental and subinterval analysis."""
        tbr_df, tbr_summary, params = mathematical_data

        # Generate incremental analysis
        incremental_result = create_incremental_tbr_summaries(tbr_df, **params)

        # Generate equivalent subinterval analysis for first 3 days
        subinterval_results = []
        for day in range(1, 4):
            result = compute_interval_estimate_and_ci(
                tbr_df, tbr_summary, 1, day, params["level"]
            )
            subinterval_results.append(result)

        # Validate consistency for first 3 days
        for i, sub_result in enumerate(subinterval_results):
            np.testing.assert_allclose(
                incremental_result.iloc[i]["estimate"],
                sub_result["estimate"],
                rtol=1e-14,
                err_msg=f"Incremental and subinterval should be consistent for day {i+1}",
            )


class TestAnalysisPerformanceValidation:
    """Performance validation tests for analysis modules."""

    @pytest.fixture
    def performance_data(self):
        """Create large dataset for performance testing."""
        tbr_df = create_test_tbr_dataframe(n_pretest=30, n_test=50, seed=999)

        params = {
            "alpha": 1000,
            "beta": 0.98,
            "sigma": 50,
            "var_alpha": 100,
            "var_beta": 0.001,
            "cov_alpha_beta": -0.05,
            "degrees_freedom": 75,
            "level": 0.90,
            "threshold": 10,
        }

        return tbr_df, params

    def test_analysis_modules_performance(self, performance_data):
        """Test performance of analysis modules."""
        tbr_df, params = performance_data

        # Measure performance for each module
        start_time = time.time()
        summary_result = create_tbr_summary(tbr_df, **params)
        summary_time = time.time() - start_time

        start_time = time.time()
        incremental_result = create_incremental_tbr_summaries(tbr_df, **params)
        incremental_time = time.time() - start_time

        # Performance validation
        assert summary_time < 1.0, "Summary analysis should complete within 1 second"
        assert (
            incremental_time < 3.0
        ), "Incremental analysis should complete within 3 seconds"

        # Results validation
        assert len(summary_result) == 1
        assert len(incremental_result) == 50  # Number of test days

        # Mathematical consistency under performance conditions
        np.testing.assert_allclose(
            summary_result.iloc[0]["estimate"],
            incremental_result.iloc[-1]["estimate"],
            rtol=1e-12,
            err_msg="Performance testing should maintain mathematical accuracy",
        )

    def test_subinterval_batch_performance(self, performance_data):
        """Test subinterval analysis performance with multiple intervals."""
        tbr_df, params = performance_data
        tbr_summary = pd.DataFrame(
            {"sigma": [params["sigma"]], "t_dist_df": [params["degrees_freedom"]]}
        )

        # Create multiple subintervals for performance testing
        intervals = [(i, i + 4) for i in range(1, 46, 5)]  # 5-day intervals

        start_time = time.time()

        # Test multiple subinterval analysis
        subinterval_results = analyze_multiple_subintervals(
            tbr_df, tbr_summary, intervals, ci_level=params["level"]
        )

        subinterval_time = time.time() - start_time

        # Performance validation
        assert (
            subinterval_time < 2.0
        ), "Multiple subinterval analysis should complete within 2 seconds"
        assert len(subinterval_results) == len(intervals)

        # Validate all results have required structure
        for result in subinterval_results:
            assert "estimate" in result
            assert "precision" in result
            assert "lower" in result
            assert "upper" in result


class TestAnalysisModuleIntegration:
    """Integration tests for analysis module system."""

    def test_lazy_loading_integration(self):
        """Test that lazy loading works correctly for analysis modules."""
        # Test that all import patterns work
        from tbr import compute_interval_estimate_and_ci as main_subinterval
        from tbr import create_incremental_tbr_summaries as main_incremental
        from tbr import create_tbr_summary as main_summary
        from tbr.analysis import (
            compute_interval_estimate_and_ci as analysis_subinterval,
        )
        from tbr.analysis import (
            create_incremental_tbr_summaries as analysis_incremental,
        )
        from tbr.analysis import create_tbr_summary as analysis_summary
        from tbr.analysis.incremental import (
            create_incremental_tbr_summaries as direct_incremental,
        )
        from tbr.analysis.subinterval import (
            compute_interval_estimate_and_ci as direct_subinterval,
        )
        from tbr.analysis.summary import create_tbr_summary as direct_summary

        # Verify they are the same functions (backward compatibility)
        assert main_summary is analysis_summary is direct_summary
        assert main_incremental is analysis_incremental is direct_incremental
        assert main_subinterval is analysis_subinterval is direct_subinterval

    def test_module_organization_standards(self):
        """Test professional module organization standards."""
        # Test module has proper docstring
        from tbr.analysis import incremental, subinterval, summary

        assert summary.__doc__ is not None
        assert incremental.__doc__ is not None
        assert subinterval.__doc__ is not None

        # Test that core functions are exposed
        assert hasattr(summary, "create_tbr_summary")
        assert hasattr(incremental, "create_incremental_tbr_summaries")
        assert hasattr(subinterval, "compute_interval_estimate_and_ci")
        assert hasattr(subinterval, "analyze_multiple_subintervals")
        assert hasattr(subinterval, "create_subinterval_summary")

    def test_comprehensive_workflow_integration(self):
        """Test comprehensive workflow using all analysis modules."""
        # Create test data
        tbr_df = create_test_tbr_dataframe(n_pretest=5, n_test=10, seed=12345)

        params = {
            "alpha": 102.0,
            "beta": 1.01,
            "sigma": 8.0,
            "var_alpha": 12.0,
            "var_beta": 0.0005,
            "cov_alpha_beta": -0.01,
            "degrees_freedom": 13,
            "level": 0.90,
            "threshold": 2.0,
        }

        # Step 1: Overall summary analysis
        summary = create_tbr_summary(tbr_df, **params)

        # Step 2: Day-by-day incremental analysis
        incremental = create_incremental_tbr_summaries(tbr_df, **params)

        # Step 3: Custom subinterval analysis
        tbr_summary = pd.DataFrame(
            {"sigma": [params["sigma"]], "t_dist_df": [params["degrees_freedom"]]}
        )
        intervals = [(1, 3), (4, 7), (8, 10)]
        subinterval_summary = create_subinterval_summary(
            tbr_df,
            tbr_summary,
            intervals,
            significance_threshold=params["threshold"],
            ci_level=params["level"],
        )

        # Workflow validation
        assert len(summary) == 1, "Summary should have one row"
        assert len(incremental) == 10, "Incremental should have 10 days"
        assert (
            len(subinterval_summary) == 3
        ), "Subinterval summary should have 3 intervals"

        # Mathematical consistency across workflow
        final_incremental_estimate = incremental.iloc[-1]["estimate"]
        summary_estimate = summary.iloc[0]["estimate"]

        np.testing.assert_allclose(
            summary_estimate,
            final_incremental_estimate,
            rtol=1e-12,
            err_msg="Workflow should maintain mathematical consistency",
        )
