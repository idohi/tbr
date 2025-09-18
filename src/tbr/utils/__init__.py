"""TBR utilities module."""

from .constants import CONTROL_VAL, TEST_VAL
from .datetime_utils import (
    create_time_range_mask,
    process_time_column,
    sort_dataframe_by_time,
)
from .exceptions import (
    ConvergenceError,
    InsufficientDataError,
    NumericalInstabilityError,
    TBRError,
)
from .preprocessing import (
    assign_period_indicators,
    calculate_basic_statistics,
    extract_regression_arrays,
    prepare_regression_arrays,
    split_time_series_by_periods,
)
from .validation import (
    validate_array_not_empty,
    validate_column_types,
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

__all__ = [
    # Constants
    "CONTROL_VAL",
    "TEST_VAL",
    # Exceptions
    "TBRError",
    "ConvergenceError",
    "NumericalInstabilityError",
    "InsufficientDataError",
    # Array & Sample Validation
    "validate_array_not_empty",
    "validate_sample_size",
    # Core DataFrame Validation
    "validate_required_columns",
    "validate_no_nulls",
    "validate_metric_columns",
    # Time-Related Validation
    "validate_time_column_type",
    "validate_time_boundaries_type",
    "validate_time_periods",
    # Data Quality Validation
    "validate_period_data",
    "validate_learning_set",
    # Statistical Parameter Validation
    "validate_confidence_level",
    "validate_threshold_parameter",
    "validate_degrees_freedom",
    "validate_variance_parameters",
    # Enhanced Data Quality Validation
    "validate_dataframe_not_empty",
    "validate_column_types",
    "validate_time_series_continuity",
    # Data Preprocessing Functions
    "split_time_series_by_periods",
    "extract_regression_arrays",
    "assign_period_indicators",
    "prepare_regression_arrays",
    "calculate_basic_statistics",
    # Date/Time Handling Functions
    "sort_dataframe_by_time",
    "process_time_column",
    "create_time_range_mask",
]
