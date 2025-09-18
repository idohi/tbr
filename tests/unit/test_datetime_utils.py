"""
Tests for date/time handling utilities.

This module contains comprehensive tests for the datetime_utils module,
covering sorting, processing, and time range operations across all
supported time column types (datetime64[ns], int64, float64).
"""

import numpy as np
import pandas as pd
import pytest

from tbr.utils.datetime_utils import (
    create_time_range_mask,
    process_time_column,
    sort_dataframe_by_time,
)


class TestSortDataFrameByTime:
    """Test sort_dataframe_by_time function."""

    def test_sort_datetime_column_basic(self):
        """Test sorting with datetime column."""
        # Create unsorted datetime data
        data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"]),
                "value": [30, 10, 20],
            }
        )

        result = sort_dataframe_by_time(data, "date")

        # Check sorting order
        expected_values = [10, 20, 30]
        assert result["value"].tolist() == expected_values

        # Check index is reset
        assert result.index.tolist() == [0, 1, 2]

    def test_sort_integer_column(self):
        """Test sorting with integer time column."""
        data = pd.DataFrame(
            {"hour": [3, 1, 2, 5, 4], "metric": [300, 100, 200, 500, 400]}
        )

        result = sort_dataframe_by_time(data, "hour")

        expected_metrics = [100, 200, 300, 400, 500]
        assert result["metric"].tolist() == expected_metrics

    def test_sort_float_column(self):
        """Test sorting with float time column."""
        data = pd.DataFrame(
            {"time": [2.5, 1.0, 3.7, 1.5], "measurement": [25, 10, 37, 15]}
        )

        result = sort_dataframe_by_time(data, "time")

        expected_measurements = [10, 15, 25, 37]
        assert result["measurement"].tolist() == expected_measurements

    def test_sort_already_sorted_data(self):
        """Test sorting already sorted data."""
        data = pd.DataFrame(
            {"date": pd.date_range("2023-01-01", periods=5), "value": [1, 2, 3, 4, 5]}
        )

        result = sort_dataframe_by_time(data, "date")

        # Should remain the same
        assert result["value"].tolist() == [1, 2, 3, 4, 5]

    def test_sort_with_duplicate_times(self):
        """Test sorting with duplicate time values."""
        data = pd.DataFrame({"hour": [2, 1, 2, 1, 3], "value": [20, 10, 21, 11, 30]})

        result = sort_dataframe_by_time(data, "hour")

        # First two should be hour=1, next two hour=2, last hour=3
        assert result["hour"].tolist() == [1, 1, 2, 2, 3]
        # Original order preserved within same time
        assert result.loc[result["hour"] == 1, "value"].tolist() == [10, 11]

    def test_sort_empty_dataframe(self):
        """Test sorting empty DataFrame."""
        data = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")})

        # Empty time column should raise validation error
        with pytest.raises(
            ValueError, match="Time column 'date' in input DataFrame is empty"
        ):
            sort_dataframe_by_time(data, "date")

        # But should work with validation disabled
        result = sort_dataframe_by_time(data, "date", validate_column=False)
        assert result.empty
        assert list(result.columns) == ["date"]

    def test_sort_single_row(self):
        """Test sorting DataFrame with single row."""
        data = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")], "value": [100]})

        result = sort_dataframe_by_time(data, "date")

        assert len(result) == 1
        assert result["value"].iloc[0] == 100

    def test_sort_without_validation(self):
        """Test sorting with validation disabled."""
        data = pd.DataFrame({"hour": [3, 1, 2], "value": [30, 10, 20]})

        result = sort_dataframe_by_time(data, "hour", validate_column=False)

        assert result["value"].tolist() == [10, 20, 30]

    def test_sort_with_invalid_column_validation_enabled(self):
        """Test sorting with invalid time column when validation is enabled."""
        data = pd.DataFrame(
            {"time": ["late", "early", "medium"], "value": [30, 10, 20]}  # Object dtype
        )

        with pytest.raises(ValueError, match="Unsupported dtype"):
            sort_dataframe_by_time(data, "time", validate_column=True)

    def test_sort_preserves_other_columns(self):
        """Test that sorting preserves all other columns."""
        data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"]),
                "control": [300, 100, 200],
                "test": [330, 110, 220],
                "category": ["C", "A", "B"],
            }
        )

        result = sort_dataframe_by_time(data, "date")

        # Check all columns preserved
        assert set(result.columns) == set(data.columns)

        # Check sorting affected all columns
        assert result["control"].tolist() == [100, 200, 300]
        assert result["test"].tolist() == [110, 220, 330]
        assert result["category"].tolist() == ["A", "B", "C"]


class TestProcessTimeColumn:
    """Test process_time_column function."""

    def test_process_datetime_column(self):
        """Test processing datetime column."""
        data = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-15", "2023-01-10", "2023-01-20"]),
                "control": [150, 100, 200],
                "test": [155, 105, 205],
            }
        )

        result = process_time_column(data, "timestamp")

        # Should be sorted
        assert result["control"].tolist() == [100, 150, 200]
        assert result["test"].tolist() == [105, 155, 205]

    def test_process_integer_column(self):
        """Test processing integer time column."""
        data = pd.DataFrame({"day": [5, 2, 8, 1], "metric": [50, 20, 80, 10]})

        result = process_time_column(data, "day")

        assert result["metric"].tolist() == [10, 20, 50, 80]

    def test_process_invalid_time_column(self):
        """Test processing with invalid time column."""
        data = pd.DataFrame(
            {"time": ["morning", "evening", "afternoon"], "value": [1, 3, 2]}
        )

        with pytest.raises(ValueError, match="Unsupported dtype"):
            process_time_column(data, "time")

    def test_process_missing_column(self):
        """Test processing with missing time column."""
        data = pd.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(ValueError, match="Time column 'missing' not found"):
            process_time_column(data, "missing")

    def test_process_empty_dataframe(self):
        """Test processing empty DataFrame."""
        data = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")})

        # Empty time column should raise validation error
        with pytest.raises(
            ValueError, match="Time column 'date' in input data is empty"
        ):
            process_time_column(data, "date")

    def test_process_null_values_in_time_column(self):
        """Test processing with null values in time column."""
        data = pd.DataFrame(
            {
                "date": [pd.Timestamp("2023-01-01"), None, pd.Timestamp("2023-01-03")],
                "value": [1, 2, 3],
            }
        )

        # Should not raise error - nulls are handled by sorting
        result = process_time_column(data, "date")

        # Nulls should be sorted to the end
        assert pd.isna(result["date"].iloc[-1])


class TestCreateTimeRangeMask:
    """Test create_time_range_mask function."""

    def test_datetime_range_exclusive_end(self):
        """Test datetime range with exclusive end."""
        dates = pd.date_range("2023-01-01", periods=10)

        mask = create_time_range_mask(
            dates,
            pd.Timestamp("2023-01-03"),
            pd.Timestamp("2023-01-07"),
            inclusive_end=False,
        )

        # Should include dates 3, 4, 5, 6 (indices 2, 3, 4, 5)
        expected_indices = [2, 3, 4, 5]
        assert mask.sum() == 4
        assert dates[mask].tolist() == [dates[i] for i in expected_indices]

    def test_datetime_range_inclusive_end(self):
        """Test datetime range with inclusive end."""
        dates = pd.date_range("2023-01-01", periods=10)

        mask = create_time_range_mask(
            dates,
            pd.Timestamp("2023-01-03"),
            pd.Timestamp("2023-01-07"),
            inclusive_end=True,
        )

        # Should include dates 3, 4, 5, 6, 7 (indices 2, 3, 4, 5, 6)
        expected_indices = [2, 3, 4, 5, 6]
        assert mask.sum() == 5
        assert dates[mask].tolist() == [dates[i] for i in expected_indices]

    def test_integer_range_exclusive_end(self):
        """Test integer range with exclusive end."""
        hours = pd.Series(range(1, 25))  # Hours 1-24

        mask = create_time_range_mask(hours, 8, 17, inclusive_end=False)

        # Should include 8, 9, 10, 11, 12, 13, 14, 15, 16
        expected = list(range(8, 17))
        assert hours[mask].tolist() == expected

    def test_integer_range_inclusive_end(self):
        """Test integer range with inclusive end."""
        hours = pd.Series(range(1, 25))  # Hours 1-24

        mask = create_time_range_mask(hours, 8, 17, inclusive_end=True)

        # Should include 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
        expected = list(range(8, 18))
        assert hours[mask].tolist() == expected

    def test_float_range(self):
        """Test float time range."""
        times = pd.Series([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

        mask = create_time_range_mask(times, 1.0, 3.0, inclusive_end=False)

        # Should include 1.0, 1.5, 2.0, 2.5
        expected = [1.0, 1.5, 2.0, 2.5]
        assert times[mask].tolist() == expected

    def test_single_point_range_exclusive(self):
        """Test single point range with exclusive end."""
        times = pd.Series([1, 2, 3, 4, 5])

        mask = create_time_range_mask(times, 3, 3, inclusive_end=False)

        # Should include nothing (3 >= 3 and 3 < 3 is False)
        assert mask.sum() == 0

    def test_single_point_range_inclusive(self):
        """Test single point range with inclusive end."""
        times = pd.Series([1, 2, 3, 4, 5])

        mask = create_time_range_mask(times, 3, 3, inclusive_end=True)

        # Should include just 3
        assert mask.sum() == 1
        assert times[mask].tolist() == [3]

    def test_empty_series(self):
        """Test with empty time series."""
        times = pd.Series([], dtype="datetime64[ns]")

        mask = create_time_range_mask(
            times, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")
        )

        assert len(mask) == 0
        assert mask.empty

    def test_no_matches_in_range(self):
        """Test range that matches no values."""
        times = pd.Series([1, 2, 3, 7, 8, 9])

        mask = create_time_range_mask(times, 4, 6, inclusive_end=True)

        assert mask.sum() == 0

    def test_all_values_in_range(self):
        """Test range that includes all values."""
        times = pd.Series([2, 3, 4, 5])

        mask = create_time_range_mask(times, 1, 6, inclusive_end=True)

        assert mask.sum() == 4
        assert mask.all()


class TestIntegrationWithValidation:
    """Test integration with existing validation utilities."""

    def test_sort_integrates_with_validation(self):
        """Test that sorting properly integrates with time column validation."""
        # Valid datetime column
        data = pd.DataFrame(
            {"date": pd.date_range("2023-01-01", periods=3), "value": [1, 2, 3]}
        )

        # Should work without error
        result = sort_dataframe_by_time(data, "date")
        assert len(result) == 3

    def test_process_column_comprehensive_validation(self):
        """Test that process_time_column performs comprehensive validation."""
        # Test with all null values
        data = pd.DataFrame({"date": [None, None, None], "value": [1, 2, 3]})

        with pytest.raises(ValueError, match="contains only null values"):
            process_time_column(data, "date")

    def test_datetime_utils_with_preprocessing_functions(self):
        """Test compatibility with existing preprocessing functions."""
        # Create data that would be used with split_time_series_by_periods
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=30),
                "control": np.random.normal(1000, 50, 30),
                "test": np.random.normal(1020, 55, 30),
            }
        )

        # Process with datetime utils
        processed = process_time_column(data, "date")

        # Should work with time range mask
        mask = create_time_range_mask(
            processed["date"], pd.Timestamp("2023-01-10"), pd.Timestamp("2023-01-20")
        )

        subset = processed[mask]
        assert len(subset) == 10  # 10 days in range


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_timezone_aware_datetime(self):
        """Test with timezone-aware datetime columns."""
        dates = pd.date_range("2023-01-01", periods=5, tz="UTC")
        data = pd.DataFrame({"date": dates, "value": [5, 3, 1, 4, 2]})

        result = sort_dataframe_by_time(data, "date")

        # Should sort properly
        assert result["value"].tolist() == [5, 3, 1, 4, 2]  # Already sorted

    def test_mixed_timezone_boundaries(self):
        """Test time range mask with timezone-aware data."""
        dates = pd.date_range("2023-01-01", periods=5, tz="UTC")

        mask = create_time_range_mask(
            dates,
            pd.Timestamp("2023-01-02", tz="UTC"),
            pd.Timestamp("2023-01-04", tz="UTC"),
        )

        assert mask.sum() == 2  # 2nd and 3rd dates

    def test_performance_with_large_dataset(self):
        """Test performance with reasonably large dataset."""
        # Create larger dataset
        n_points = 10000
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=n_points, freq="min"),
                "value": np.random.randn(n_points),
            }
        )

        # Shuffle to test sorting
        shuffled = data.sample(frac=1).reset_index(drop=True)

        # Should complete quickly
        result = sort_dataframe_by_time(shuffled, "timestamp")

        # Verify it's sorted
        assert result["timestamp"].is_monotonic_increasing
        assert len(result) == n_points
