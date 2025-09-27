"""
Analysis Framework Integration Validation Tests.

This module provides comprehensive integration validation for the analysis
framework implemented in Phase 3. Tests ensure proper module interaction,
cross-validation with functional implementation, and end-to-end workflow
validation across all analysis modules.

Test Categories
---------------
1. Framework Integration - Module interaction and consistency
2. Workflow Validation - End-to-end analysis workflows
3. Cross-Validation - Consistency with functional implementation
4. Consistency Validation - Mathematical relationships and properties
5. Performance Validation - Scalability and efficiency testing

All tests use proper TBR DataFrame format with required columns.
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


def create_comprehensive_tbr_data(scenario="default", seed=42):
    """Create comprehensive TBR data for integration testing."""
    np.random.seed(seed)

    if scenario == "high_variance":
        # High variance scenario: larger values, higher variability
        n_pretest, n_test = 10, 15
        base_control, base_test = 1000, 1050
        control_std, test_std = 50, 60
        # effect_size = 0.05  # Not used in current implementation
    elif scenario == "low_variance":
        # Low variance scenario: smaller values, lower variability
        n_pretest, n_test = 8, 12
        base_control, base_test = 100, 102
        control_std, test_std = 5, 6
        # effect_size = 0.02  # Not used in current implementation
    else:
        # Default scenario
        n_pretest, n_test = 6, 10
        base_control, base_test = 500, 520
        control_std, test_std = 25, 30
        # effect_size = 0.04  # Not used in current implementation

    # Create pretest data
    pretest_y = np.random.normal(base_control, control_std, n_pretest)
    pretest_x = np.random.normal(base_control * 0.95, control_std * 0.8, n_pretest)
    pretest_pred = pretest_x * 1.05
    pretest_dif = pretest_y - pretest_pred

    # Create test data with treatment effect
    test_y = np.random.normal(base_test, test_std, n_test)
    test_x = np.random.normal(base_control * 0.95, control_std * 0.8, n_test)
    test_pred = test_x * 1.05
    test_dif = test_y - test_pred
    test_cumdif = np.cumsum(test_dif)
    test_cumsd = np.sqrt(np.arange(1, n_test + 1) * (test_std**2))

    tbr_df = pd.DataFrame(
        {
            "period": [0] * n_pretest + [1] * n_test,
            "y": np.concatenate([pretest_y, test_y]),
            "x": np.concatenate([pretest_x, test_x]),
            "pred": np.concatenate([pretest_pred, test_pred]),
            "predsd": np.concatenate(
                [np.zeros(n_pretest), np.random.uniform(10, 20, n_test)]
            ),
            "dif": np.concatenate([pretest_dif, test_dif]),
            "cumdif": np.concatenate([[np.nan] * n_pretest, test_cumdif]),
            "cumsd": np.concatenate([np.zeros(n_pretest), test_cumsd]),
            "estsd": np.concatenate(
                [np.random.uniform(5, 15, n_pretest), [np.nan] * n_test]
            ),
        }
    )

    # TBR summary parameters
    tbr_summary = pd.DataFrame(
        {"sigma": [test_std], "t_dist_df": [n_pretest + n_test - 2]}
    )

    # Analysis parameters
    analysis_params = {
        "alpha": base_control,
        "beta": 1.05,
        "sigma": test_std,
        "var_alpha": (control_std**2) / n_pretest,
        "var_beta": 0.001,
        "cov_alpha_beta": -0.01,
        "degrees_freedom": n_pretest + n_test - 2,
        "level": 0.90,
        "threshold": 0.0,
    }

    return tbr_df, tbr_summary, analysis_params


class TestAnalysisFrameworkIntegration:
    """Core integration tests for analysis framework."""

    @pytest.fixture
    def comprehensive_tbr_data(self):
        """Create comprehensive TBR data for integration testing."""
        return create_comprehensive_tbr_data(scenario="default", seed=123)

    def test_analysis_modules_integration_consistency(self, comprehensive_tbr_data):
        """Test integration consistency between all analysis modules."""
        tbr_df, tbr_summary, params = comprehensive_tbr_data

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
        assert len(incremental_result) == len(tbr_df[tbr_df["period"] == 1])
        assert isinstance(subinterval_result, dict)
        assert "estimate" in subinterval_result

    def test_analysis_parameter_consistency_across_modules(
        self, comprehensive_tbr_data
    ):
        """Test consistent parameter handling across modules."""
        tbr_df, tbr_summary, params = comprehensive_tbr_data

        # Test with different confidence levels
        for level in [0.80, 0.90, 0.95]:
            test_params = params.copy()
            test_params["level"] = level

            summary_result = create_tbr_summary(tbr_df, **test_params)
            incremental_result = create_incremental_tbr_summaries(tbr_df, **test_params)

            # Verify results are generated successfully
            assert len(summary_result) == 1
            assert len(incremental_result) > 0

            # Verify confidence level affects precision (wider intervals for higher confidence)
            summary_precision = summary_result.iloc[0]["precision"]
            final_incremental_precision = incremental_result.iloc[-1]["precision"]

            # Both should be consistent
            np.testing.assert_allclose(
                summary_precision,
                final_incremental_precision,
                rtol=1e-10,
                err_msg=f"Precision should be consistent across modules for level={level}",
            )

    def test_analysis_data_flow_integration(self, comprehensive_tbr_data):
        """Test data flow integration between analysis modules."""
        tbr_df, tbr_summary, params = comprehensive_tbr_data

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

        # Test precision consistency (allowing for small numerical differences)
        final_incremental_precision = incremental_result.iloc[-1]["precision"]
        summary_precision = summary_result.iloc[0]["precision"]

        np.testing.assert_allclose(
            summary_precision,
            final_incremental_precision,
            rtol=1e-8,
            err_msg="Precision should be consistent between summary and incremental",
        )

    def test_analysis_modules_error_handling_integration(self, comprehensive_tbr_data):
        """Test consistent error handling across modules."""
        tbr_df, tbr_summary, params = comprehensive_tbr_data

        # Test invalid confidence level
        invalid_params = params.copy()
        invalid_params["level"] = 1.5

        # Both modules should handle invalid parameters consistently
        with pytest.raises(ValueError, match="Level must be between 0 and 1"):
            create_tbr_summary(tbr_df, **invalid_params)

        with pytest.raises(ValueError, match="Level must be between 0 and 1"):
            create_incremental_tbr_summaries(tbr_df, **invalid_params)


class TestAnalysisWorkflowValidation:
    """End-to-end workflow validation tests."""

    @pytest.fixture
    def multi_scenario_data(self):
        """Create data for multiple scenario testing."""
        scenarios = {}
        for scenario_name in ["high_variance", "low_variance"]:
            scenarios[scenario_name] = create_comprehensive_tbr_data(
                scenario=scenario_name, seed=456 + hash(scenario_name) % 1000
            )
        return scenarios

    def test_complete_analysis_workflow_high_variance(self, multi_scenario_data):
        """Test complete analysis workflow for high variance scenario."""
        tbr_df, tbr_summary, params = multi_scenario_data["high_variance"]

        # Step 1: Overall summary analysis
        summary = create_tbr_summary(tbr_df, **params)

        # Step 2: Day-by-day incremental analysis
        incremental = create_incremental_tbr_summaries(tbr_df, **params)

        # Step 3: Custom subinterval analysis
        intervals = [(1, 5), (6, 10), (11, 15)]
        subinterval_summary = create_subinterval_summary(
            tbr_df,
            tbr_summary,
            intervals,
            significance_threshold=params["threshold"],
            ci_level=params["level"],
        )

        # Workflow validation
        assert len(summary) == 1, "Summary should have one row"
        assert (
            len(incremental) == 15
        ), "Incremental should have 15 days for high variance scenario"
        assert (
            len(subinterval_summary) == 3
        ), "Subinterval summary should have 3 intervals"

        # Mathematical consistency across workflow
        final_incremental_estimate = incremental.iloc[-1]["estimate"]
        summary_estimate = summary.iloc[0]["estimate"]

        np.testing.assert_allclose(
            summary_estimate,
            final_incremental_estimate,
            rtol=1e-10,
            err_msg="Workflow should maintain mathematical consistency",
        )

    def test_complete_analysis_workflow_low_variance(self, multi_scenario_data):
        """Test complete analysis workflow for low variance scenario."""
        tbr_df, tbr_summary, params = multi_scenario_data["low_variance"]

        # Complete workflow
        summary = create_tbr_summary(tbr_df, **params)
        incremental = create_incremental_tbr_summaries(tbr_df, **params)

        # Workflow validation
        assert len(summary) == 1, "Summary should have one row"
        assert (
            len(incremental) == 12
        ), "Incremental should have 12 days for low variance scenario"

        # Verify all results are finite and reasonable
        assert np.isfinite(
            summary.iloc[0]["estimate"]
        ), "Summary estimate should be finite"
        assert all(
            np.isfinite(incremental["estimate"])
        ), "All incremental estimates should be finite"

    def test_cross_scenario_workflow_consistency(self, multi_scenario_data):
        """Test workflow consistency across different scenarios."""
        results = {}

        for scenario_name, (
            tbr_df,
            _tbr_summary,
            params,
        ) in multi_scenario_data.items():
            summary = create_tbr_summary(tbr_df, **params)
            incremental = create_incremental_tbr_summaries(tbr_df, **params)

            results[scenario_name] = {"summary": summary, "incremental": incremental}

        # Validate that all scenarios produce valid results
        for scenario_name, scenario_results in results.items():
            summary = scenario_results["summary"]
            incremental = scenario_results["incremental"]

            # Basic validation - use actual column names
            assert (
                "estimate" in summary.columns
            ), f"Summary should have estimate column for {scenario_name}"
            assert (
                "precision" in summary.columns
            ), f"Summary should have precision column for {scenario_name}"
            assert (
                "estimate" in incremental.columns
            ), f"Incremental should have estimate column for {scenario_name}"

            # Mathematical consistency within scenario
            np.testing.assert_allclose(
                summary.iloc[0]["estimate"],
                incremental.iloc[-1]["estimate"],
                rtol=1e-10,
                err_msg=f"Consistency should be maintained for {scenario_name}",
            )


class TestAnalysisCrossValidation:
    """Cross-validation tests between analysis modules and functional implementation."""

    @pytest.fixture
    def cross_validation_data(self):
        """Create data for cross-validation testing."""
        return create_comprehensive_tbr_data(scenario="default", seed=789)

    def test_analysis_summary_cross_validation(self, cross_validation_data):
        """Cross-validate analysis summary with functional implementation."""
        tbr_df, tbr_summary, params = cross_validation_data

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
        tbr_df, tbr_summary, params = cross_validation_data

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
        tbr_df, tbr_summary, params = cross_validation_data

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


class TestAnalysisConsistencyValidation:
    """Mathematical consistency validation tests."""

    @pytest.fixture
    def consistency_test_data(self):
        """Create data for consistency testing."""
        return create_comprehensive_tbr_data(scenario="default", seed=999)

    def test_additive_property_validation(self, consistency_test_data):
        """Test additive property of subinterval estimates."""
        tbr_df, tbr_summary, params = consistency_test_data

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

    def test_incremental_consistency_validation(self, consistency_test_data):
        """Test consistency between incremental and subinterval analysis."""
        tbr_df, tbr_summary, params = consistency_test_data

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

    def test_confidence_level_monotonicity(self, consistency_test_data):
        """Test that confidence intervals widen with higher confidence levels."""
        tbr_df, tbr_summary, params = consistency_test_data

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

    def test_statistical_relationship_preservation(self, consistency_test_data):
        """Test preservation of statistical relationships across modules."""
        tbr_df, tbr_summary, params = consistency_test_data

        # Generate results
        summary_result = create_tbr_summary(tbr_df, **params)
        incremental_result = create_incremental_tbr_summaries(tbr_df, **params)

        # Test statistical relationships - use actual column names
        summary_estimate = summary_result.iloc[0]["estimate"]
        # summary_se = summary_result.iloc[0]["se"]  # Standard error (not used in current test)
        summary_precision = summary_result.iloc[0]["precision"]

        final_incremental_estimate = incremental_result.iloc[-1]["estimate"]
        final_incremental_precision = incremental_result.iloc[-1]["precision"]

        # Mathematical consistency
        np.testing.assert_allclose(
            summary_estimate,
            final_incremental_estimate,
            rtol=1e-12,
            err_msg="Estimates should be consistent",
        )

        np.testing.assert_allclose(
            summary_precision,
            final_incremental_precision,
            rtol=1e-10,
            err_msg="Precision should be consistent",
        )


class TestAnalysisPerformanceValidation:
    """Performance validation tests for analysis modules."""

    @pytest.fixture
    def performance_test_data(self):
        """Create large dataset for performance testing."""
        return create_comprehensive_tbr_data(scenario="default", seed=1111)

    def test_analysis_modules_performance_scalability(self, performance_test_data):
        """Test performance scalability of analysis modules."""
        tbr_df, tbr_summary, params = performance_test_data

        # Measure performance for each module
        start_time = time.time()
        summary_result = create_tbr_summary(tbr_df, **params)
        summary_time = time.time() - start_time

        start_time = time.time()
        incremental_result = create_incremental_tbr_summaries(tbr_df, **params)
        incremental_time = time.time() - start_time

        # Performance validation
        assert summary_time < 2.0, "Summary analysis should complete within 2 seconds"
        assert (
            incremental_time < 5.0
        ), "Incremental analysis should complete within 5 seconds"

        # Results validation
        assert len(summary_result) == 1
        assert len(incremental_result) == len(tbr_df[tbr_df["period"] == 1])

        # Mathematical consistency under performance conditions
        np.testing.assert_allclose(
            summary_result.iloc[0]["estimate"],
            incremental_result.iloc[-1]["estimate"],
            rtol=1e-12,
            err_msg="Performance testing should maintain mathematical accuracy",
        )

    def test_subinterval_analysis_performance(self, performance_test_data):
        """Test subinterval analysis performance with multiple intervals."""
        tbr_df, tbr_summary, params = performance_test_data

        # Create multiple subintervals for performance testing
        test_period_length = len(tbr_df[tbr_df["period"] == 1])
        intervals = [
            (i, min(i + 2, test_period_length))
            for i in range(1, test_period_length - 1, 2)
        ]

        start_time = time.time()

        # Test multiple subinterval analysis
        subinterval_results = analyze_multiple_subintervals(
            tbr_df, tbr_summary, intervals, ci_level=params["level"]
        )

        subinterval_time = time.time() - start_time

        # Performance validation
        assert (
            subinterval_time < 3.0
        ), "Multiple subinterval analysis should complete within 3 seconds"
        assert len(subinterval_results) == len(intervals)

        # Validate all results have required structure
        for result in subinterval_results:
            assert "estimate" in result
            assert "precision" in result
            assert "lower" in result
            assert "upper" in result

    def test_analysis_memory_efficiency(self, performance_test_data):
        """Test memory efficiency of analysis modules."""
        tbr_df, tbr_summary, params = performance_test_data

        # Test that analysis doesn't create excessive memory overhead
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run analysis
        summary_result = create_tbr_summary(tbr_df, **params)
        incremental_result = create_incremental_tbr_summaries(tbr_df, **params)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for this dataset)
        assert memory_increase < 50 * 1024 * 1024, "Memory usage should be reasonable"

        # Results should still be valid
        assert len(summary_result) == 1
        assert len(incremental_result) > 0
