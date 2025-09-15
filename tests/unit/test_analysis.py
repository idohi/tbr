"""
Test suite for main TBR analysis workflow functions.

This module tests the complete TBR analysis pipeline, including:
- Period splitting functionality
- Summary creation and incremental summaries
- Main analysis workflow integration
- End-to-end analysis scenarios

These tests ensure the complete TBR methodology works correctly.
"""

import numpy as np
import pandas as pd
import pytest

from tbr.functional.tbr_functions import (
    create_incremental_tbr_summaries,
    create_tbr_summary,
    perform_tbr_analysis,
    split_by_periods,
)


class TestPeriodSplitting:
    """Test period splitting functionality."""

    def test_basic_period_splitting(self):
        """Test basic period splitting with datetime data."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=90),
                "control": np.random.normal(1000, 50, 90),
                "test": np.random.normal(1020, 55, 90),
            }
        )

        baseline, pretest, test, cooldown = split_by_periods(
            data,
            time_col="date",
            pretest_start=pd.Timestamp("2023-01-15"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Check that all periods have data
        assert len(baseline) > 0  # Jan 1-14
        assert len(pretest) > 0  # Jan 15 - Feb 14
        assert len(test) > 0  # Feb 15 - Feb 28
        assert len(cooldown) > 0  # Mar 1 onwards

        # Check total data preservation
        total_rows = len(baseline) + len(pretest) + len(test) + len(cooldown)
        assert total_rows == len(data)

    def test_period_splitting_with_inclusive_end(self):
        """Test period splitting with inclusive end boundary."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=30),
                "control": range(30),
                "test": range(30, 60),
            }
        )

        baseline, pretest, test, cooldown = split_by_periods(
            data,
            time_col="date",
            pretest_start=pd.Timestamp("2023-01-10"),
            test_start=pd.Timestamp("2023-01-20"),
            test_end=pd.Timestamp("2023-01-25"),
            test_end_inclusive=True,
        )

        # With inclusive end, test period should include Jan 25
        test_dates = test["date"].dt.day.tolist()
        assert 25 in test_dates

        # Cooldown should start after Jan 25
        if not cooldown.empty:
            cooldown_dates = cooldown["date"].dt.day.tolist()
            assert min(cooldown_dates) > 25

    def test_period_splitting_integer_time(self):
        """Test period splitting with integer time column."""
        data = pd.DataFrame(
            {
                "hour": range(1, 49),  # Hours 1-48
                "control": np.random.normal(500, 25, 48),
                "test": np.random.normal(520, 30, 48),
            }
        )

        baseline, pretest, test, cooldown = split_by_periods(
            data, time_col="hour", pretest_start=10, test_start=25, test_end=35
        )

        # Check period boundaries
        assert baseline["hour"].max() < 10
        assert pretest["hour"].min() >= 10
        assert pretest["hour"].max() < 25
        assert test["hour"].min() >= 25
        assert test["hour"].max() < 35
        if not cooldown.empty:
            assert cooldown["hour"].min() >= 35

    def test_empty_periods_handling(self):
        """Test handling when some periods are empty."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-02-10", periods=20),  # Only Feb 10-Mar 1
                "control": range(20),
                "test": range(20, 40),
            }
        )

        baseline, pretest, test, cooldown = split_by_periods(
            data,
            time_col="date",
            pretest_start=pd.Timestamp("2023-01-01"),  # Before data starts
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-02-25"),
        )

        # Baseline should be empty (no data before Feb 10)
        assert len(baseline) == 0

        # Other periods should have data
        assert len(pretest) > 0  # Feb 10-14
        assert len(test) > 0  # Feb 15-24
        assert len(cooldown) > 0  # Feb 25 onwards


class TestSummaryCreation:
    """Test TBR summary creation functions."""

    def test_basic_summary_creation(self):
        """Test basic TBR summary creation."""
        # Create mock TBR dataframe
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 0, 1, 1, 1],
                "cumdif": [np.nan, np.nan, np.nan, 5, 12, 20],
                "cumsd": [0, 0, 0, 2, 3, 4],
            }
        )

        summary = create_tbr_summary(
            tbr_dataframe=tbr_df,
            alpha=10.0,
            beta=1.5,
            sigma=5.0,
            var_alpha=1.0,
            var_beta=0.01,
            cov_alpha_beta=0.1,
            degrees_freedom=20,
            level=0.80,
            threshold=0.0,
        )

        # Check structure
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1

        required_cols = ["estimate", "precision", "lower", "upper", "prob"]
        assert all(col in summary.columns for col in required_cols)

        # Check values
        assert summary["estimate"].iloc[0] == 20  # Final cumulative effect
        assert summary["precision"].iloc[0] > 0
        assert summary["lower"].iloc[0] < summary["upper"].iloc[0]
        assert 0 <= summary["prob"].iloc[0] <= 1

    def test_incremental_summaries_creation(self):
        """Test incremental TBR summaries creation."""
        # Create mock TBR dataframe with multiple test days
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 1, 1, 1, 1],
                "cumdif": [np.nan, np.nan, 3, 7, 12, 18],
                "cumsd": [0, 0, 1.5, 2.1, 2.8, 3.2],
            }
        )

        summaries = create_incremental_tbr_summaries(
            tbr_dataframe=tbr_df,
            alpha=5.0,
            beta=2.0,
            sigma=3.0,
            var_alpha=0.5,
            var_beta=0.005,
            cov_alpha_beta=0.05,
            degrees_freedom=15,
            level=0.90,
            threshold=0.0,
        )

        # Check structure
        assert isinstance(summaries, pd.DataFrame)
        assert len(summaries) == 4  # 4 test days
        assert "test_day" in summaries.columns

        # Check incremental progression
        test_days = summaries["test_day"].tolist()
        assert test_days == [1, 2, 3, 4]

        # Effects should generally increase over time
        estimates = summaries["estimate"].tolist()
        assert estimates[0] < estimates[-1]  # First < last

    def test_summary_validation_errors(self):
        """Test summary creation validation errors."""
        # Test empty dataframe
        with pytest.raises(ValueError, match="TBR dataframe cannot be empty"):
            create_tbr_summary(
                tbr_dataframe=pd.DataFrame(),
                alpha=5.0,
                beta=1.5,
                sigma=2.0,
                var_alpha=1.0,
                var_beta=0.01,
                cov_alpha_beta=0.1,
                degrees_freedom=20,
                level=0.80,
                threshold=0.0,
            )

        # Test invalid level
        valid_df = pd.DataFrame({"period": [1], "cumdif": [5], "cumsd": [2]})
        with pytest.raises(ValueError, match="Level must be between 0 and 1"):
            create_tbr_summary(
                tbr_dataframe=valid_df,
                alpha=5.0,
                beta=1.5,
                sigma=2.0,
                var_alpha=1.0,
                var_beta=0.01,
                cov_alpha_beta=0.1,
                degrees_freedom=20,
                level=1.5,
                threshold=0.0,
            )

    def test_summary_comprehensive_validation_errors(self):
        """Test comprehensive validation error paths for summary creation."""
        valid_df = pd.DataFrame({"period": [1], "cumdif": [5], "cumsd": [2]})

        # Test missing columns error (lines 1115)
        invalid_df = pd.DataFrame({"period": [1]})  # Missing cumdif, cumsd
        with pytest.raises(
            ValueError, match="Missing required columns in TBR dataframe"
        ):
            create_tbr_summary(
                tbr_dataframe=invalid_df,
                alpha=5.0,
                beta=1.5,
                sigma=2.0,
                var_alpha=1.0,
                var_beta=0.01,
                cov_alpha_beta=0.1,
                degrees_freedom=20,
                level=0.80,
                threshold=0.0,
            )

        # Test negative degrees of freedom (line 1121)
        with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
            create_tbr_summary(
                tbr_dataframe=valid_df,
                alpha=5.0,
                beta=1.5,
                sigma=2.0,
                var_alpha=1.0,
                var_beta=0.01,
                cov_alpha_beta=0.1,
                degrees_freedom=-5,
                level=0.80,
                threshold=0.0,
            )

        # Test zero degrees of freedom (line 1121)
        with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
            create_tbr_summary(
                tbr_dataframe=valid_df,
                alpha=5.0,
                beta=1.5,
                sigma=2.0,
                var_alpha=1.0,
                var_beta=0.01,
                cov_alpha_beta=0.1,
                degrees_freedom=0,
                level=0.80,
                threshold=0.0,
            )

        # Test negative sigma (line 1124)
        with pytest.raises(ValueError, match="Sigma must be positive"):
            create_tbr_summary(
                tbr_dataframe=valid_df,
                alpha=5.0,
                beta=1.5,
                sigma=-2.0,
                var_alpha=1.0,
                var_beta=0.01,
                cov_alpha_beta=0.1,
                degrees_freedom=20,
                level=0.80,
                threshold=0.0,
            )

        # Test zero sigma (line 1124)
        with pytest.raises(ValueError, match="Sigma must be positive"):
            create_tbr_summary(
                tbr_dataframe=valid_df,
                alpha=5.0,
                beta=1.5,
                sigma=0.0,
                var_alpha=1.0,
                var_beta=0.01,
                cov_alpha_beta=0.1,
                degrees_freedom=20,
                level=0.80,
                threshold=0.0,
            )

        # Test no test period data (line 1130)
        no_test_df = pd.DataFrame(
            {"period": [0, 0], "cumdif": [np.nan, np.nan], "cumsd": [0, 0]}
        )
        with pytest.raises(ValueError, match="No test period data found"):
            create_tbr_summary(
                tbr_dataframe=no_test_df,
                alpha=5.0,
                beta=1.5,
                sigma=2.0,
                var_alpha=1.0,
                var_beta=0.01,
                cov_alpha_beta=0.1,
                degrees_freedom=20,
                level=0.80,
                threshold=0.0,
            )

    def test_summary_statistical_properties(self):
        """Test statistical properties of summary creation."""
        # Create test data with known properties
        tbr_df = pd.DataFrame(
            {"period": [1, 1, 1], "cumdif": [2, 5, 8], "cumsd": [1, 1.5, 2]}
        )

        # Test different confidence levels
        levels = [0.80, 0.90, 0.95, 0.99]
        precisions = []

        for level in levels:
            summary = create_tbr_summary(
                tbr_dataframe=tbr_df,
                alpha=0.0,
                beta=1.0,
                sigma=1.0,
                var_alpha=0.1,
                var_beta=0.001,
                cov_alpha_beta=0.0,
                degrees_freedom=30,
                level=level,
                threshold=0.0,
            )
            precisions.append(summary["precision"].iloc[0])

        # Higher confidence levels should give wider intervals (higher precision)
        assert precisions[0] < precisions[-1]  # 80% < 99%

    def test_incremental_summaries_validation_errors(self):
        """Test validation error paths for incremental summaries (lines 1274, 1279, 1282, 1285, 1288, 1294)."""
        valid_df = pd.DataFrame(
            {"period": [0, 1, 1], "cumdif": [np.nan, 3, 7], "cumsd": [0, 1.5, 2.1]}
        )

        # Test empty dataframe (line 1274)
        with pytest.raises(ValueError, match="TBR dataframe cannot be empty"):
            create_incremental_tbr_summaries(
                tbr_dataframe=pd.DataFrame(),
                alpha=5.0,
                beta=2.0,
                sigma=3.0,
                var_alpha=0.5,
                var_beta=0.005,
                cov_alpha_beta=0.05,
                degrees_freedom=15,
                level=0.90,
                threshold=0.0,
            )

        # Test missing columns (line 1279)
        invalid_df = pd.DataFrame({"period": [1]})
        with pytest.raises(
            ValueError, match="Missing required columns in TBR dataframe"
        ):
            create_incremental_tbr_summaries(
                tbr_dataframe=invalid_df,
                alpha=5.0,
                beta=2.0,
                sigma=3.0,
                var_alpha=0.5,
                var_beta=0.005,
                cov_alpha_beta=0.05,
                degrees_freedom=15,
                level=0.90,
                threshold=0.0,
            )

        # Test invalid level (line 1282)
        with pytest.raises(ValueError, match="Level must be between 0 and 1"):
            create_incremental_tbr_summaries(
                tbr_dataframe=valid_df,
                alpha=5.0,
                beta=2.0,
                sigma=3.0,
                var_alpha=0.5,
                var_beta=0.005,
                cov_alpha_beta=0.05,
                degrees_freedom=15,
                level=1.5,
                threshold=0.0,
            )

        # Test negative degrees of freedom (line 1285)
        with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
            create_incremental_tbr_summaries(
                tbr_dataframe=valid_df,
                alpha=5.0,
                beta=2.0,
                sigma=3.0,
                var_alpha=0.5,
                var_beta=0.005,
                cov_alpha_beta=0.05,
                degrees_freedom=-10,
                level=0.90,
                threshold=0.0,
            )

        # Test negative sigma (line 1288)
        with pytest.raises(ValueError, match="Sigma must be positive"):
            create_incremental_tbr_summaries(
                tbr_dataframe=valid_df,
                alpha=5.0,
                beta=2.0,
                sigma=-3.0,
                var_alpha=0.5,
                var_beta=0.005,
                cov_alpha_beta=0.05,
                degrees_freedom=15,
                level=0.90,
                threshold=0.0,
            )

        # Test no test period data (line 1294)
        no_test_df = pd.DataFrame(
            {"period": [0, 0], "cumdif": [np.nan, np.nan], "cumsd": [0, 0]}
        )
        with pytest.raises(ValueError, match="No test period data found"):
            create_incremental_tbr_summaries(
                tbr_dataframe=no_test_df,
                alpha=5.0,
                beta=2.0,
                sigma=3.0,
                var_alpha=0.5,
                var_beta=0.005,
                cov_alpha_beta=0.05,
                degrees_freedom=15,
                level=0.90,
                threshold=0.0,
            )


class TestMainAnalysisWorkflow:
    """Test complete TBR analysis workflow."""

    def test_complete_analysis_workflow(self):
        """Test complete TBR analysis from start to finish."""
        # Create realistic test data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=90)

        # Baseline effect with some treatment effect starting Feb 15
        control_vals = np.random.normal(1000, 50, 90)
        test_vals = np.random.normal(1020, 55, 90)

        # Add treatment effect for test period (Feb 15 - Mar 1)
        treatment_start_idx = 45  # Feb 15
        treatment_end_idx = 60  # Mar 1
        test_vals[treatment_start_idx:treatment_end_idx] += 50  # Treatment boost

        data = pd.DataFrame({"date": dates, "control": control_vals, "test": test_vals})

        # Run complete analysis
        tbr_results, daily_summaries = perform_tbr_analysis(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-15"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-03-01"),
            level=0.80,
            threshold=0.0,
        )

        # Check results structure
        assert isinstance(tbr_results, pd.DataFrame)
        assert isinstance(daily_summaries, pd.DataFrame)

        # Check TBR results columns
        expected_tbr_cols = [
            "date",
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
        assert all(col in tbr_results.columns for col in expected_tbr_cols)

        # Check daily summaries columns
        expected_summary_cols = ["test_day", "estimate", "precision", "lower", "upper"]
        assert all(col in daily_summaries.columns for col in expected_summary_cols)

        # Check that we have the right number of test days
        test_days = len(daily_summaries)
        assert test_days > 0

        # Check that final estimate captures treatment effect
        final_estimate = daily_summaries["estimate"].iloc[-1]
        assert final_estimate > 0  # Should detect positive treatment effect

    def test_analysis_with_integer_time(self):
        """Test analysis with integer time column."""
        # Create hourly data
        data = pd.DataFrame(
            {
                "hour": range(1, 73),  # 72 hours
                "control": np.random.normal(500, 25, 72),
                "test": np.random.normal(520, 30, 72),
            }
        )

        tbr_results, daily_summaries = perform_tbr_analysis(
            data=data,
            time_col="hour",
            control_col="control",
            test_col="test",
            pretest_start=10,
            test_start=40,
            test_end=60,
            level=0.90,
            threshold=0.0,
        )

        # Should handle integer time correctly
        assert isinstance(tbr_results, pd.DataFrame)
        assert isinstance(daily_summaries, pd.DataFrame)
        assert len(daily_summaries) > 0

    def test_analysis_with_minimal_data(self):
        """Test analysis with minimal valid data."""
        # Create minimal dataset
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=20),
                "control": np.random.normal(100, 10, 20),
                "test": np.random.normal(105, 12, 20),
            }
        )

        tbr_results, daily_summaries = perform_tbr_analysis(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-10"),
            test_end=pd.Timestamp("2023-01-15"),
            level=0.80,
            threshold=0.0,
        )

        # Should work with minimal data
        assert len(tbr_results) > 0
        assert len(daily_summaries) > 0

    def test_analysis_validation_errors(self):
        """Test analysis input validation errors."""
        # Test empty data
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            perform_tbr_analysis(
                data=pd.DataFrame(),
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-01-15"),
                test_end=pd.Timestamp("2023-01-30"),
                level=0.80,
                threshold=0.0,
            )

        # Test missing columns
        data = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=10)})
        with pytest.raises(ValueError, match="Missing required columns"):
            perform_tbr_analysis(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-01-05"),
                test_end=pd.Timestamp("2023-01-10"),
                level=0.80,
                threshold=0.0,
            )

    def test_analysis_with_baseline_data(self):
        """Test analysis that includes baseline period data."""
        # Create data that starts before pretest
        data = pd.DataFrame(
            {
                "date": pd.date_range("2022-12-01", periods=120),  # Start in December
                "control": np.random.normal(1000, 50, 120),
                "test": np.random.normal(1020, 55, 120),
            }
        )

        tbr_results, daily_summaries = perform_tbr_analysis(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),  # Pretest starts Jan 1
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-02-15"),
            level=0.80,
            threshold=0.0,
        )

        # Should include baseline period (Dec data)
        periods = tbr_results["period"].unique()
        assert -1 in periods  # Baseline period
        assert 0 in periods  # Pretest period
        assert 1 in periods  # Test period

        # Check baseline data properties
        baseline_data = tbr_results[tbr_results["period"] == -1]
        assert len(baseline_data) > 0
        assert pd.isna(baseline_data["pred"].iloc[0])  # No predictions for baseline


class TestAnalysisEdgeCases:
    """Test edge cases and boundary conditions in analysis."""

    def test_same_day_analysis(self):
        """Test analysis where test period is a single day."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=30),
                "control": np.random.normal(1000, 50, 30),
                "test": np.random.normal(1020, 55, 30),
            }
        )

        tbr_results, daily_summaries = perform_tbr_analysis(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-20"),
            test_end=pd.Timestamp("2023-01-20"),  # Same day
            test_end_inclusive=True,
            level=0.80,
            threshold=0.0,
        )

        # Should work with single day test period
        assert len(daily_summaries) == 1
        assert daily_summaries["test_day"].iloc[0] == 1

    def test_analysis_with_high_threshold(self):
        """Test analysis with high threshold for probability calculation."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=50),
                "control": np.random.normal(1000, 50, 50),
                "test": np.random.normal(1010, 55, 50),  # Small effect
            }
        )

        tbr_results, daily_summaries = perform_tbr_analysis(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-25"),
            test_end=pd.Timestamp("2023-02-05"),
            level=0.95,
            threshold=100.0,  # High threshold
        )

        # With high threshold, probability should be low
        final_prob = daily_summaries["prob"].iloc[-1]
        assert 0 <= final_prob <= 1

    def test_analysis_comprehensive_parameter_combinations(self):
        """Test analysis with various parameter combinations."""
        base_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=60),
                "control": np.random.normal(1000, 50, 60),
                "test": np.random.normal(1020, 55, 60),
            }
        )

        # Test different parameter combinations
        test_configs = [
            {"level": 0.80, "threshold": 0.0, "test_end_inclusive": False},
            {"level": 0.90, "threshold": 10.0, "test_end_inclusive": True},
            {"level": 0.95, "threshold": -5.0, "test_end_inclusive": False},
        ]

        for config in test_configs:
            tbr_results, daily_summaries = perform_tbr_analysis(
                data=base_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-01-30"),
                test_end=pd.Timestamp("2023-02-10"),
                **config,
            )

        # All configurations should work
        assert len(tbr_results) > 0
        assert len(daily_summaries) > 0
        assert daily_summaries["level"].iloc[0] == config["level"]

    def test_analysis_no_cooldown_data(self):
        """Test analysis with no cooldown data (line 1564)."""
        # Create data where test period ends exactly at data end (no cooldown)
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=30),
                "control": np.random.normal(1000, 50, 30),
                "test": np.random.normal(1020, 55, 30),
            }
        )

        tbr_results, daily_summaries = perform_tbr_analysis(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-25"),
            test_end=pd.Timestamp("2023-01-30"),  # Exactly at end, no cooldown
            test_end_inclusive=True,
            level=0.80,
            threshold=0.0,
        )

        # Should work without cooldown data (hits line 1564)
        assert len(tbr_results) > 0
        assert len(daily_summaries) > 0

        # Should only have periods -1 (baseline), 0 (pretest), 1 (test), no 3 (cooldown)
        periods = tbr_results["period"].unique()
        assert 0 in periods  # pretest
        assert 1 in periods  # test
        assert 3 not in periods  # no cooldown

    def test_analysis_no_pretest_data_error(self):
        """Test error when no pretest data is found (line 448)."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "control": np.random.normal(1000, 50, 10),
                "test": np.random.normal(1020, 55, 10),
            }
        )

        # Set pretest_start after all data dates
        with pytest.raises(
            ValueError, match="No pretest data found - check pretest period dates"
        ):
            perform_tbr_analysis(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-02-01"),  # After all data
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-02-28"),
                level=0.80,
                threshold=0.0,
            )

    def test_analysis_no_test_data_error(self):
        """Test error when no test data is found (line 451)."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "control": np.random.normal(1000, 50, 10),
                "test": np.random.normal(1020, 55, 10),
            }
        )

        # Set test period after all data dates
        with pytest.raises(
            ValueError, match="No test data found - check test period dates"
        ):
            perform_tbr_analysis(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-01"),  # After all data
                test_end=pd.Timestamp("2023-02-15"),
                level=0.80,
                threshold=0.0,
            )
