"""
Test suite for input validation functions.

This module tests all input validation and data integrity functions
used throughout the TBR analysis pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from tbr.utils.validation import (
    validate_array_not_empty,
    validate_confidence_level,
    validate_dataframe_not_empty,
    validate_degrees_freedom,
    validate_learning_set,
    validate_metric_columns,
    validate_no_nulls,
    validate_period_data,
    validate_required_columns,
    validate_sample_size,
    validate_threshold_parameter,
    validate_time_boundaries_type,
    validate_time_column_type,
    validate_time_periods,
    validate_time_series_continuity,
    validate_variance_parameters,
)


class TestTimeColumnValidation:
    """Test time column validation functions."""

    def test_valid_datetime_column(self):
        """Test validation of valid datetime columns."""
        df = pd.DataFrame(
            {"date": pd.date_range("2023-01-01", periods=5), "values": range(5)}
        )
        validate_time_column_type(df, "date")

    def test_time_column_not_found(self):
        """Test error when time column is missing."""
        df = pd.DataFrame({"values": [1, 2, 3]})

        with pytest.raises(ValueError, match="Time column 'missing_col' not found"):
            validate_time_column_type(df, "missing_col")

    def test_all_null_time_column(self):
        """Test error when time column contains only null values."""
        df = pd.DataFrame({"date": [pd.NaT, pd.NaT, pd.NaT]})

        with pytest.raises(ValueError, match="contains only null values"):
            validate_time_column_type(df, "date")

    def test_empty_time_column(self):
        """Test error when time column is completely empty (line 196)."""
        df = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")})

        with pytest.raises(ValueError, match="Time column 'date' in data is empty"):
            validate_time_column_type(df, "date")

    def test_unsupported_dtype_error(self):
        """Test error for completely unsupported dtype (line 219)."""
        df = pd.DataFrame(
            {"date": pd.Categorical(["A", "B", "C"])}  # Categorical dtype
        )

        with pytest.raises(ValueError, match="Unsupported dtype.*Use pd.to_datetime"):
            validate_time_column_type(df, "date")


class TestRequiredColumnsValidation:
    """Test required columns validation."""

    def test_all_columns_present(self):
        """Test when all required columns are present."""
        df = pd.DataFrame(
            {"time": [1, 2, 3], "control": [10, 20, 30], "test": [15, 25, 35]}
        )

        validate_required_columns(df, ["time", "control", "test"], "test_data")

    def test_missing_columns(self):
        """Test error when columns are missing."""
        df = pd.DataFrame({"time": [1, 2, 3]})

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_required_columns(df, ["time", "control", "test"], "test_data")


class TestNullValidation:
    """Test null value validation."""

    def test_no_nulls(self):
        """Test when no null values are present."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [10, 20, 30]})

        validate_no_nulls(df, ["col1", "col2"], "test_data")

    def test_nulls_present(self):
        """Test error when nulls are present."""
        df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": [10, 20, 30]})

        with pytest.raises(ValueError, match="Null values found"):
            validate_no_nulls(df, ["col1", "col2"], "test_data")


class TestTimeBoundariesValidation:
    """Test time boundaries validation."""

    def test_valid_timestamp_boundaries(self):
        """Test valid timestamp boundaries."""
        pretest_start = pd.Timestamp("2023-01-01")
        test_start = pd.Timestamp("2023-02-01")
        test_end = pd.Timestamp("2023-02-15")
        time_dtype = pd.Series([pd.Timestamp("2023-01-01")]).dtype

        validate_time_boundaries_type(pretest_start, test_start, test_end, time_dtype)

    def test_mixed_boundary_types(self):
        """Test error with mixed boundary types."""
        pretest_start = pd.Timestamp("2023-01-01")
        test_start = 10  # Integer instead of timestamp
        test_end = pd.Timestamp("2023-02-15")
        time_dtype = pd.Series([pd.Timestamp("2023-01-01")]).dtype

        with pytest.raises(
            ValueError, match="All time boundaries must have the same type"
        ):
            validate_time_boundaries_type(
                pretest_start, test_start, test_end, time_dtype
            )

    def test_float_boundaries_with_float_column(self):
        """Test float boundaries with float64 column (lines 398-399)."""
        pretest_start = 1.5
        test_start = 10.7
        test_end = 20.3
        time_dtype = pd.Series([1.0, 2.0, 3.0]).dtype  # float64

        # Should not raise any exception
        validate_time_boundaries_type(pretest_start, test_start, test_end, time_dtype)

    def test_float_boundaries_with_wrong_type(self):
        """Test error when using wrong type with float64 column (line 399)."""
        pretest_start = 1  # Integer instead of float
        test_start = 10
        test_end = 20
        time_dtype = pd.Series([1.0, 2.0, 3.0]).dtype  # float64

        with pytest.raises(
            ValueError, match="Time column has dtype 'float64' but boundaries are int"
        ):
            validate_time_boundaries_type(
                pretest_start, test_start, test_end, time_dtype
            )

    def test_int64_boundaries_with_wrong_type(self):
        """Test error when using wrong type with int64 column (line 393)."""
        pretest_start = 1.5  # Float instead of int
        test_start = 10.5
        test_end = 20.5
        time_dtype = pd.Series([1, 2, 3]).dtype  # int64

        with pytest.raises(
            ValueError, match="Time column has dtype 'int64' but boundaries are float"
        ):
            validate_time_boundaries_type(
                pretest_start, test_start, test_end, time_dtype
            )

    def test_unsupported_boundary_dtype_combination(self):
        """Test unsupported boundary/dtype combination (line 403-407)."""
        pretest_start = "2023-01-01"  # String, not supported
        test_start = "2023-02-01"
        test_end = "2023-02-15"
        time_dtype = pd.Series([pd.Timestamp("2023-01-01")]).dtype

        with pytest.raises(
            ValueError, match="Time column has dtype.*but boundaries are str"
        ):
            validate_time_boundaries_type(
                pretest_start, test_start, test_end, time_dtype
            )

    def test_completely_unsupported_dtype_combination(self):
        """Test completely unsupported dtype combination (line 404)."""
        pretest_start = complex(1, 2)  # Complex number, completely unsupported
        test_start = complex(2, 3)
        test_end = complex(3, 4)
        # Create a Series with an unsupported dtype
        time_dtype = pd.Series([b"bytes1", b"bytes2"]).dtype  # bytes dtype

        with pytest.raises(
            ValueError, match="Boundary type complex does not match time column dtype"
        ):
            validate_time_boundaries_type(
                pretest_start, test_start, test_end, time_dtype
            )


class TestMetricColumnsValidation:
    """Test metric columns validation."""

    def test_valid_numeric_columns(self):
        """Test validation of valid numeric columns."""
        data = pd.DataFrame({"control": [1.0, 2.0, 3.0], "test": [10, 20, 30]})

        validate_metric_columns(data, "control", "test")

    def test_non_numeric_control_column(self):
        """Test error when control column is non-numeric."""
        data = pd.DataFrame({"control": ["a", "b", "c"], "test": [10, 20, 30]})

        with pytest.raises(
            ValueError, match="Control column 'control' must be numeric"
        ):
            validate_metric_columns(data, "control", "test")

    def test_non_numeric_test_column(self):
        """Test error when test column is non-numeric (line 430)."""
        data = pd.DataFrame(
            {"control": [1.0, 2.0, 3.0], "test": ["a", "b", "c"]}  # String values
        )

        with pytest.raises(ValueError, match="Test column 'test' must be numeric"):
            validate_metric_columns(data, "control", "test")


class TestLearningSetValidation:
    """Test learning set validation."""

    def test_sufficient_learning_data(self):
        """Test validation of sufficient learning data."""
        learning_df = pd.DataFrame(
            {"control": [100, 110, 120, 130, 140], "test": [200, 220, 240, 260, 280]}
        )

        validate_learning_set(learning_df, "control", "test")

    def test_insufficient_learning_data(self):
        """Test error with insufficient learning data."""
        learning_df = pd.DataFrame({"control": [100, 110], "test": [200, 220]})

        with pytest.raises(ValueError, match="Insufficient learning data"):
            validate_learning_set(learning_df, "control", "test")

    def test_learning_data_with_infinite_values(self):
        """Test error when learning data contains infinite values (line 483)."""
        learning_df = pd.DataFrame(
            {"control": [100, 110, np.inf, 130, 140], "test": [200, 220, 240, 260, 280]}
        )

        with pytest.raises(
            ValueError, match="Learning data contains infinite or NaN values"
        ):
            validate_learning_set(learning_df, "control", "test")

    def test_learning_data_with_null_values(self):
        """Test error when learning data contains null values (line 479)."""
        learning_df = pd.DataFrame(
            {
                "control": [100, 110, None, 130, 140],  # Null value
                "test": [200, 220, 240, 260, 280],
            }
        )

        with pytest.raises(ValueError, match="Learning data contains null values"):
            validate_learning_set(learning_df, "control", "test")


class TestTimePeriodsValidation:
    """Test time periods validation."""

    def test_valid_time_periods(self):
        """Test validation of valid time periods."""
        pretest_start = pd.Timestamp("2023-01-01")
        test_start = pd.Timestamp("2023-02-01")
        test_end = pd.Timestamp("2023-02-15")

        validate_time_periods(pretest_start, test_start, test_end)

    def test_invalid_time_order(self):
        """Test error when time periods are in wrong order."""
        pretest_start = pd.Timestamp("2023-02-01")
        test_start = pd.Timestamp("2023-01-01")  # Before pretest
        test_end = pd.Timestamp("2023-02-15")

        with pytest.raises(ValueError, match="pretest_start must be before test_start"):
            validate_time_periods(pretest_start, test_start, test_end)

    def test_invalid_inclusive_time_periods(self):
        """Test error when test_start > test_end with inclusive boundary (line 515)."""
        pretest_start = pd.Timestamp("2023-01-01")
        test_start = pd.Timestamp("2023-02-20")
        test_end = pd.Timestamp("2023-02-15")  # Before start

        with pytest.raises(
            ValueError,
            match="test_start must be <= test_end when test_end_inclusive=True",
        ):
            validate_time_periods(
                pretest_start, test_start, test_end, test_end_inclusive=True
            )

    def test_invalid_exclusive_time_periods(self):
        """Test error when test_start >= test_end with exclusive boundary (line 520)."""
        pretest_start = pd.Timestamp("2023-01-01")
        test_start = pd.Timestamp("2023-02-15")
        test_end = pd.Timestamp("2023-02-15")  # Same as start

        with pytest.raises(
            ValueError,
            match="test_start must be < test_end when test_end_inclusive=False",
        ):
            validate_time_periods(
                pretest_start, test_start, test_end, test_end_inclusive=False
            )


class TestUtilityValidation:
    """Test utility validation functions."""

    def test_validate_array_not_empty_valid(self):
        """Test array validation with valid arrays."""
        validate_array_not_empty(np.array([1, 2, 3]), "test_array")

    def test_validate_array_not_empty_invalid(self):
        """Test array validation with empty arrays."""
        with pytest.raises(ValueError, match="test_array cannot be empty"):
            validate_array_not_empty(np.array([]), "test_array")

    def test_validate_sample_size_valid(self):
        """Test sample size validation with valid sizes."""
        validate_sample_size(5, min_size=3)

    def test_validate_sample_size_invalid(self):
        """Test sample size validation with invalid sizes."""
        with pytest.raises(ValueError, match="sample size.*at least 3"):
            validate_sample_size(2, min_size=3)

    def test_validate_sample_size_with_custom_param_name(self):
        """Test sample size validation with custom parameter name (line 47)."""
        with pytest.raises(ValueError, match="custom_param.*at least 5"):
            validate_sample_size(3, min_size=5, param_name="custom_param")

    def test_validate_sample_size_negative(self):
        """Test sample size validation with negative value (line 47)."""
        with pytest.raises(ValueError, match="test_param cannot be negative"):
            validate_sample_size(-1, min_size=3, param_name="test_param")


class TestStatisticalParameterValidation:
    """Test statistical parameter validation functions."""

    def test_validate_confidence_level_valid(self):
        """Test validation of valid confidence levels."""
        validate_confidence_level(0.80)
        validate_confidence_level(0.95)
        validate_confidence_level(0.99, "credibility level")

    def test_validate_confidence_level_invalid_bounds(self):
        """Test error for confidence levels outside (0,1)."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_confidence_level(0.0)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_confidence_level(1.0)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_confidence_level(1.2)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_confidence_level(-0.1)

    def test_validate_threshold_parameter_valid(self):
        """Test validation of valid threshold parameters."""
        validate_threshold_parameter(0.0)
        validate_threshold_parameter(5.5)
        validate_threshold_parameter(-2.3, "effect threshold")

    def test_validate_threshold_parameter_invalid(self):
        """Test error for non-finite threshold values."""
        with pytest.raises(ValueError, match="must be finite"):
            validate_threshold_parameter(float("inf"))

        with pytest.raises(ValueError, match="must be finite"):
            validate_threshold_parameter(float("nan"))

    def test_validate_degrees_freedom_valid(self):
        """Test validation of valid degrees of freedom."""
        validate_degrees_freedom(1)
        validate_degrees_freedom(10)
        validate_degrees_freedom(100, "residual df")

    def test_validate_degrees_freedom_invalid(self):
        """Test error for non-positive degrees of freedom."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_degrees_freedom(0)

        with pytest.raises(ValueError, match="must be positive"):
            validate_degrees_freedom(-5)

    def test_validate_variance_parameters_valid(self):
        """Test validation of valid variance parameters."""
        validate_variance_parameters(var_alpha=0.01, var_beta=0.005)
        validate_variance_parameters(sigma_squared=100.0)
        validate_variance_parameters(variance=0.0)  # Zero variance is valid

    def test_validate_variance_parameters_negative(self):
        """Test error for negative variance parameters."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_variance_parameters(var_alpha=-0.01)

        with pytest.raises(ValueError, match="must be non-negative"):
            validate_variance_parameters(var_beta=-0.005)

    def test_validate_variance_parameters_non_finite(self):
        """Test error for non-finite variance parameters."""
        with pytest.raises(ValueError, match="must be finite"):
            validate_variance_parameters(var_alpha=float("inf"))

        with pytest.raises(ValueError, match="must be finite"):
            validate_variance_parameters(var_beta=float("nan"))


class TestEnhancedDataQualityValidation:
    """Test enhanced data quality validation functions."""

    def test_validate_dataframe_not_empty_valid(self):
        """Test validation of non-empty DataFrames."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        validate_dataframe_not_empty(df, "test_data")

    def test_validate_dataframe_not_empty_invalid(self):
        """Test error for empty DataFrames."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty_data cannot be empty"):
            validate_dataframe_not_empty(empty_df, "empty_data")

    def test_validate_time_series_continuity_valid(self):
        """Test validation of continuous time series."""
        df = pd.DataFrame(
            {"date": pd.date_range("2023-01-01", periods=5), "value": [1, 2, 3, 4, 5]}
        )
        validate_time_series_continuity(df, "date")

    def test_validate_time_series_continuity_empty(self):
        """Test validation handles empty DataFrames gracefully."""
        empty_df = pd.DataFrame()
        validate_time_series_continuity(empty_df, "date")  # Should not raise

    def test_validate_time_series_continuity_unsorted(self):
        """Test error for unsorted time series."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"]),
                "value": [3, 1, 2],
            }
        )
        with pytest.raises(ValueError, match="must be sorted in ascending order"):
            validate_time_series_continuity(df, "date")

    def test_validate_time_series_continuity_large_gaps(self):
        """Test detection of large gaps in time series."""
        # Create time series with a large gap
        dates = list(pd.date_range("2023-01-01", periods=3, freq="D"))
        dates.append(pd.Timestamp("2023-02-01"))  # Large 29-day gap
        df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4]})
        with pytest.raises(ValueError, match="has large gaps"):
            validate_time_series_continuity(df, "date")


class TestPeriodDataValidation:
    """Test period data validation functions."""

    def test_validate_period_data_valid(self):
        """Test validation of valid period data."""
        pretest = pd.DataFrame({"date": [1, 2], "control": [100, 110]})
        test = pd.DataFrame({"date": [3, 4], "control": [120, 130]})
        validate_period_data(pretest, test)

    def test_validate_period_data_empty_pretest(self):
        """Test error when pretest data is empty."""
        empty_df = pd.DataFrame()
        test = pd.DataFrame({"date": [3, 4], "control": [120, 130]})
        with pytest.raises(ValueError, match="No pretest data found"):
            validate_period_data(empty_df, test)

    def test_validate_period_data_empty_test(self):
        """Test error when test data is empty."""
        pretest = pd.DataFrame({"date": [1, 2], "control": [100, 110]})
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="No test data found"):
            validate_period_data(pretest, empty_df)
