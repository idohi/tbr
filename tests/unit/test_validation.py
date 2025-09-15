"""Unit tests for TBR validation utilities."""

import numpy as np
import pytest

from tbr.utils.validation import validate_array_not_empty, validate_sample_size


class TestValidateArrayNotEmpty:
    """Test cases for validate_array_not_empty function."""

    def test_valid_non_empty_array(self) -> None:
        """Test validation passes for non-empty arrays."""
        arr = np.array([1, 2, 3, 4, 5])
        # Should not raise any exception
        validate_array_not_empty(arr, "test_array")

    def test_empty_array_raises_error(self) -> None:
        """Test validation raises ValueError for empty arrays."""
        arr = np.array([])

        with pytest.raises(ValueError, match="test_array cannot be empty"):
            validate_array_not_empty(arr, "test_array")

    def test_custom_parameter_name_in_error(self) -> None:
        """Test error message includes custom parameter name."""
        arr = np.array([])

        with pytest.raises(ValueError, match="custom_param cannot be empty"):
            validate_array_not_empty(arr, "custom_param")

    def test_different_array_types(self) -> None:
        """Test validation works with different array types."""
        # Integer array
        int_arr = np.array([1, 2, 3])
        validate_array_not_empty(int_arr, "int_array")

        # Float array
        float_arr = np.array([1.0, 2.5, 3.7])
        validate_array_not_empty(float_arr, "float_array")

        # String array
        str_arr = np.array(["a", "b", "c"])
        validate_array_not_empty(str_arr, "str_array")

    def test_multidimensional_arrays(self) -> None:
        """Test validation works with multidimensional arrays."""
        # 2D array
        arr_2d = np.array([[1, 2], [3, 4]])
        validate_array_not_empty(arr_2d, "2d_array")

        # Empty 2D array
        empty_2d = np.array([]).reshape(0, 2)
        with pytest.raises(ValueError):
            validate_array_not_empty(empty_2d, "empty_2d")


class TestValidateSampleSize:
    """Test cases for validate_sample_size function."""

    def test_valid_sample_size(self) -> None:
        """Test validation passes for sufficient sample size."""
        # Should not raise any exception
        validate_sample_size(50, 30, "sample_size")
        validate_sample_size(100, 10, "observations")

    def test_exact_minimum_sample_size(self) -> None:
        """Test validation passes when sample size equals minimum."""
        validate_sample_size(30, 30, "exact_minimum")

    def test_insufficient_sample_size_raises_error(self) -> None:
        """Test validation raises ValueError for insufficient sample size."""
        with pytest.raises(
            ValueError,
            match="Insufficient sample_size: 20 observations. Need at least 30",
        ):
            validate_sample_size(20, 30, "sample_size")

    def test_negative_sample_size_raises_error(self) -> None:
        """Test validation raises ValueError for negative sample size."""
        with pytest.raises(ValueError, match="sample_size cannot be negative, got -5"):
            validate_sample_size(-5, 10, "sample_size")

    def test_zero_sample_size_raises_error(self) -> None:
        """Test validation raises ValueError for zero sample size."""
        with pytest.raises(
            ValueError,
            match="Insufficient sample_size: 0 observations. Need at least 10",
        ):
            validate_sample_size(0, 10, "sample_size")

    def test_custom_parameter_name_in_error(self) -> None:
        """Test error message includes custom parameter name."""
        with pytest.raises(
            ValueError,
            match="Insufficient observations: 5 observations. Need at least 10",
        ):
            validate_sample_size(5, 10, "observations")

    def test_default_parameter_name(self) -> None:
        """Test default parameter name is used when not specified."""
        with pytest.raises(
            ValueError,
            match="Insufficient sample size: 5 observations. Need at least 10",
        ):
            validate_sample_size(5, 10)

    def test_edge_cases(self) -> None:
        """Test edge cases for sample size validation."""
        # Very large numbers
        validate_sample_size(1000000, 100000, "large_sample")

        # Minimum requirement of 1
        validate_sample_size(1, 1, "minimal")

        # Zero minimum requirement
        validate_sample_size(5, 0, "no_minimum")


class TestValidationIntegration:
    """Integration tests for validation utilities."""

    def test_validation_functions_work_together(self) -> None:
        """Test validation functions can be used in sequence."""
        # Create valid data
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        n = len(arr)

        # Both validations should pass
        validate_array_not_empty(arr, "data_array")
        validate_sample_size(n, 5, "sample_count")

    def test_typical_usage_pattern(self) -> None:
        """Test typical usage pattern for TBR validation."""
        # Simulate typical TBR data validation
        control_data = np.array([100, 105, 98, 102, 107, 99, 104, 101])
        test_data = np.array([102, 108, 101, 105, 110, 102, 107, 104])

        # Validate arrays are not empty
        validate_array_not_empty(control_data, "control_data")
        validate_array_not_empty(test_data, "test_data")

        # Validate sufficient sample sizes
        validate_sample_size(len(control_data), 5, "control_sample_size")
        validate_sample_size(len(test_data), 5, "test_sample_size")
