"""
Tests for TBR preprocessing utilities.

This module contains comprehensive tests for all preprocessing functions
extracted from the functional TBR implementation.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from tbr.utils.preprocessing import (
    assign_period_indicators,
    calculate_basic_statistics,
    extract_regression_arrays,
    prepare_regression_arrays,
    split_time_series_by_periods,
)


class TestSplitTimeSeriesByPeriods:
    """Test the split_time_series_by_periods function."""

    def test_datetime_split_exclusive_end(self):
        """Test period splitting with datetime columns and exclusive end."""
        # Create test data
        dates = pd.date_range("2023-01-01", periods=90)
        data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, 90),
                "test": np.random.normal(1020, 55, 90),
            }
        )

        # Define periods
        pretest_start = pd.Timestamp("2023-01-15")
        test_start = pd.Timestamp("2023-02-15")
        test_end = pd.Timestamp("2023-03-01")

        # Split data
        baseline, pretest, test, cooldown = split_time_series_by_periods(
            data, "date", pretest_start, test_start, test_end, test_end_inclusive=False
        )

        # Verify splits
        assert len(baseline) == 14  # Jan 1-14
        assert len(pretest) == 31  # Jan 15 - Feb 14
        assert len(test) == 14  # Feb 15 - Feb 28
        assert len(cooldown) == 31  # Mar 1 onwards

        # Verify boundaries
        assert baseline["date"].max() < pretest_start
        assert pretest["date"].min() >= pretest_start
        assert pretest["date"].max() < test_start
        assert test["date"].min() >= test_start
        assert test["date"].max() < test_end
        assert cooldown["date"].min() >= test_end

    def test_datetime_split_inclusive_end(self):
        """Test period splitting with datetime columns and inclusive end."""
        dates = pd.date_range("2023-01-01", periods=60)
        data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, 60),
                "test": np.random.normal(1020, 55, 60),
            }
        )

        pretest_start = pd.Timestamp("2023-01-15")
        test_start = pd.Timestamp("2023-02-01")
        test_end = pd.Timestamp("2023-02-14")  # Same day as last test day

        baseline, pretest, test, cooldown = split_time_series_by_periods(
            data, "date", pretest_start, test_start, test_end, test_end_inclusive=True
        )

        # Verify inclusive end
        assert test["date"].max() <= test_end
        if not cooldown.empty:
            assert cooldown["date"].min() > test_end

    def test_integer_time_column(self):
        """Test period splitting with integer time column."""
        data = pd.DataFrame(
            {
                "hour": range(1, 49),  # Hours 1-48
                "control": np.random.normal(500, 25, 48),
                "test": np.random.normal(520, 30, 48),
            }
        )

        baseline, pretest, test, cooldown = split_time_series_by_periods(
            data, "hour", 10, 25, 35, test_end_inclusive=False
        )

        # Verify splits
        assert len(baseline) == 9  # Hours 1-9
        assert len(pretest) == 15  # Hours 10-24
        assert len(test) == 10  # Hours 25-34
        assert len(cooldown) == 14  # Hours 35-48

    def test_float_time_column(self):
        """Test period splitting with float time column."""
        data = pd.DataFrame(
            {
                "time": np.arange(0.0, 10.0, 0.1),  # 0.0 to 9.9 in 0.1 increments
                "control": np.random.normal(100, 10, 100),
                "test": np.random.normal(105, 12, 100),
            }
        )

        baseline, pretest, test, cooldown = split_time_series_by_periods(
            data, "time", 2.0, 5.0, 7.5, test_end_inclusive=False
        )

        # Verify approximate splits (floating point precision)
        assert len(baseline) == 20  # 0.0-1.9
        assert len(pretest) == 30  # 2.0-4.9
        assert len(test) == 25  # 5.0-7.4
        assert len(cooldown) == 25  # 7.5-9.9

    def test_empty_periods(self):
        """Test handling of empty periods."""
        # Create data with no baseline period
        dates = pd.date_range("2023-01-01", periods=30)
        data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, 30),
                "test": np.random.normal(1020, 55, 30),
            }
        )

        # Start pretest immediately
        pretest_start = pd.Timestamp("2023-01-01")
        test_start = pd.Timestamp("2023-01-15")
        test_end = pd.Timestamp("2023-01-30")

        baseline, pretest, test, cooldown = split_time_series_by_periods(
            data, "date", pretest_start, test_start, test_end, test_end_inclusive=True
        )

        # Verify empty baseline and cooldown
        assert len(baseline) == 0
        assert len(cooldown) == 0
        assert len(pretest) == 14
        assert len(test) == 16

    def test_data_integrity(self):
        """Test that original data is preserved in splits."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=30),
                "control": range(30),
                "test": range(100, 130),
                "extra_col": ["A"] * 30,
            }
        )

        baseline, pretest, test, cooldown = split_time_series_by_periods(
            data,
            "date",
            pd.Timestamp("2023-01-10"),
            pd.Timestamp("2023-01-20"),
            pd.Timestamp("2023-01-25"),
        )

        # Verify all columns preserved
        for df in [baseline, pretest, test, cooldown]:
            if not df.empty:
                assert list(df.columns) == list(data.columns)

        # Verify data integrity
        assert pretest["control"].tolist() == list(range(9, 19))
        assert test["test"].tolist() == list(range(119, 124))


class TestExtractRegressionArrays:
    """Test the extract_regression_arrays function."""

    def test_basic_extraction(self):
        """Test basic array extraction."""
        data = pd.DataFrame(
            {
                "control": [1, 2, 3, 4, 5],
                "test": [10, 20, 30, 40, 50],
                "extra": [100, 200, 300, 400, 500],
            }
        )

        x, y = extract_regression_arrays(data, "control", "test")

        # Verify arrays
        np.testing.assert_array_equal(x, np.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(y, np.array([10, 20, 30, 40, 50]))
        assert x.dtype == y.dtype

    def test_float_data(self):
        """Test extraction with float data."""
        data = pd.DataFrame({"control": [1.1, 2.2, 3.3], "test": [10.5, 20.5, 30.5]})

        x, y = extract_regression_arrays(data, "control", "test")

        np.testing.assert_array_almost_equal(x, np.array([1.1, 2.2, 3.3]))
        np.testing.assert_array_almost_equal(y, np.array([10.5, 20.5, 30.5]))

    def test_empty_dataframe(self):
        """Test extraction from empty DataFrame."""
        data = pd.DataFrame(columns=["control", "test"])

        x, y = extract_regression_arrays(data, "control", "test")

        assert len(x) == 0
        assert len(y) == 0
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)


class TestAssignPeriodIndicators:
    """Test the assign_period_indicators function."""

    def test_basic_assignment(self):
        """Test basic period indicator assignment."""
        data = pd.DataFrame(
            {
                "control": [1, 2, 3],
                "test": [10, 20, 30],
                "date": pd.date_range("2023-01-01", periods=3),
            }
        )

        result = assign_period_indicators(data, "test", "control", 1)

        # Verify new columns
        assert "period" in result.columns
        assert "y" in result.columns
        assert "x" in result.columns

        # Verify values
        assert all(result["period"] == 1)
        np.testing.assert_array_equal(result["y"].values, data["test"].values)
        np.testing.assert_array_equal(result["x"].values, data["control"].values)

        # Verify original columns preserved
        assert "date" in result.columns

    def test_different_period_values(self):
        """Test assignment with different period values."""
        data = pd.DataFrame({"control": [1, 2], "test": [10, 20]})

        for period_val in [-1, 0, 1, 3]:
            result = assign_period_indicators(data, "test", "control", period_val)
            assert all(result["period"] == period_val)

    def test_data_independence(self):
        """Test that original data is not modified."""
        original = pd.DataFrame({"control": [1, 2, 3], "test": [10, 20, 30]})
        original_copy = original.copy()

        result = assign_period_indicators(original, "test", "control", 5)

        # Verify original unchanged
        pd.testing.assert_frame_equal(original, original_copy)

        # Verify result is different
        assert "period" not in original.columns
        assert "period" in result.columns


class TestPrepareRegressionArrays:
    """Test the prepare_regression_arrays function."""

    def test_add_constant_true(self):
        """Test adding constant term."""
        x = np.array([1, 2, 3, 4, 5])

        X = prepare_regression_arrays(x, add_constant=True)

        # Should have 2 columns: constant and original
        assert X.shape == (5, 2)

        # First column should be all ones (constant)
        np.testing.assert_array_equal(X[:, 0], np.ones(5))

        # Second column should be original data
        np.testing.assert_array_equal(X[:, 1], x)

    def test_add_constant_false(self):
        """Test without adding constant term."""
        x = np.array([1, 2, 3, 4, 5])

        X = prepare_regression_arrays(x, add_constant=False)

        # Should be same as input
        np.testing.assert_array_equal(X, x)
        assert X.shape == x.shape

    def test_2d_input(self):
        """Test with 2D input array."""
        x = np.array([[1, 2], [3, 4], [5, 6]])

        X = prepare_regression_arrays(x, add_constant=True)

        # Should add constant as first column
        assert X.shape == (3, 3)
        np.testing.assert_array_equal(X[:, 0], np.ones(3))
        np.testing.assert_array_equal(X[:, 1:], x)

    def test_statsmodels_compatibility(self):
        """Test that output works with statsmodels."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = 2 + 3 * x + np.random.normal(0, 0.1, 100)

        X = prepare_regression_arrays(x, add_constant=True)

        # Should work with statsmodels
        model = sm.OLS(y, X).fit()

        # Verify reasonable coefficients
        assert abs(model.params[0] - 2) < 0.1  # Intercept ~ 2
        assert abs(model.params[1] - 3) < 0.1  # Slope ~ 3


class TestCalculateBasicStatistics:
    """Test the calculate_basic_statistics function."""

    def test_simple_statistics(self):
        """Test with simple known values."""
        x = np.array([1, 2, 3, 4, 5])

        stats = calculate_basic_statistics(x)

        # Mean should be 3.0
        assert abs(stats["mean"] - 3.0) < 1e-10

        # Sum of squared deviations: (1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)² = 4 + 1 + 0 + 1 + 4 = 10
        assert abs(stats["sum_squared_deviations"] - 10.0) < 1e-10

    def test_constant_values(self):
        """Test with constant values."""
        x = np.array([5, 5, 5, 5])

        stats = calculate_basic_statistics(x)

        assert stats["mean"] == 5.0
        assert stats["sum_squared_deviations"] == 0.0

    def test_single_value(self):
        """Test with single value."""
        x = np.array([42])

        stats = calculate_basic_statistics(x)

        assert stats["mean"] == 42.0
        assert stats["sum_squared_deviations"] == 0.0

    def test_negative_values(self):
        """Test with negative values."""
        x = np.array([-2, -1, 0, 1, 2])

        stats = calculate_basic_statistics(x)

        assert abs(stats["mean"] - 0.0) < 1e-10
        # Sum of squared deviations: (-2)² + (-1)² + 0² + 1² + 2² = 4 + 1 + 0 + 1 + 4 = 10
        assert abs(stats["sum_squared_deviations"] - 10.0) < 1e-10

    def test_float_precision(self):
        """Test floating point precision."""
        x = np.array([1.1, 2.2, 3.3])

        stats = calculate_basic_statistics(x)

        expected_mean = (1.1 + 2.2 + 3.3) / 3
        assert abs(stats["mean"] - expected_mean) < 1e-10

        # Verify sum of squared deviations calculation
        expected_ssd = sum((val - expected_mean) ** 2 for val in x)
        assert abs(stats["sum_squared_deviations"] - expected_ssd) < 1e-10

    def test_return_types(self):
        """Test that return types are correct."""
        x = np.array([1, 2, 3])

        stats = calculate_basic_statistics(x)

        assert isinstance(stats, dict)
        assert isinstance(stats["mean"], float)
        assert isinstance(stats["sum_squared_deviations"], float)
        assert len(stats) == 2


class TestPreprocessingIntegration:
    """Integration tests for preprocessing functions."""

    def test_full_preprocessing_workflow(self):
        """Test complete preprocessing workflow."""
        # Create realistic time series data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=60)
        data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, 60),
                "test": np.random.normal(1020, 55, 60),
            }
        )

        # Step 1: Split periods
        baseline, pretest, test, cooldown = split_time_series_by_periods(
            data,
            "date",
            pd.Timestamp("2023-01-15"),  # pretest_start
            pd.Timestamp("2023-02-01"),  # test_start
            pd.Timestamp("2023-02-15"),  # test_end
        )

        # Step 2: Extract arrays for regression
        x, y = extract_regression_arrays(pretest, "control", "test")

        # Step 3: Prepare for regression
        X = prepare_regression_arrays(x, add_constant=True)

        # Step 4: Calculate statistics
        stats = calculate_basic_statistics(x)

        # Step 5: Assign period indicators
        test_with_period = assign_period_indicators(test, "test", "control", 1)

        # Verify workflow results
        assert len(pretest) > 0
        assert len(test) > 0
        assert X.shape[0] == len(pretest)
        assert X.shape[1] == 2  # constant + x
        assert "mean" in stats
        assert "sum_squared_deviations" in stats
        assert "period" in test_with_period.columns
        assert all(test_with_period["period"] == 1)

    def test_preprocessing_with_functional_code_compatibility(self):
        """Test that preprocessing functions work with functional code patterns."""
        # This test verifies the integration works as expected
        np.random.seed(123)

        # Create test data similar to functional code usage
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=90),
                "control": np.random.normal(1000, 50, 90),
                "test": np.random.normal(1050, 60, 90),  # Clear effect
            }
        )

        # Use preprocessing functions as in functional code
        baseline, pretest, test, cooldown = split_time_series_by_periods(
            data,
            "date",
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2023-02-15"),
            pd.Timestamp("2023-03-01"),
        )

        # Extract and prepare regression data
        x, y = extract_regression_arrays(pretest, "control", "test")
        X = prepare_regression_arrays(x, add_constant=True)

        # Verify we can fit a model
        model = sm.OLS(y, X).fit()

        # Basic sanity checks
        assert model.params.shape[0] == 2  # intercept + slope
        assert model.rsquared >= 0  # R-squared should be non-negative
        assert len(pretest) > 10  # Reasonable sample size
        assert len(test) > 10  # Reasonable test period
