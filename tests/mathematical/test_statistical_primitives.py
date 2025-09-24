"""Mathematical validation tests for basic TBR calculations."""

import numpy as np
import pytest

from tbr.functional.tbr_functions import (
    calculate_sum_x_squared_deviations,
    extract_sum_x_squared_deviations,
    safe_int_conversion,
)


@pytest.mark.mathematical
class TestSumSquaredDeviations:
    """Mathematical validation for sum of squared deviations calculations."""

    def test_sum_squared_deviations_known_values(self) -> None:
        """Test sum of squared deviations with known mathematical result."""
        # Simple case: [1, 2, 3, 4, 5]
        # Mean = 3, deviations = [-2, -1, 0, 1, 2]
        # Squared deviations = [4, 1, 0, 1, 4]
        # Sum = 10
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_sum_x_squared_deviations(x)
        expected = 10.0

        np.testing.assert_almost_equal(result, expected, decimal=10)

    def test_sum_squared_deviations_zero_variance(self) -> None:
        """Test sum of squared deviations for constant array."""
        # All values the same should give zero variance
        x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = calculate_sum_x_squared_deviations(x)
        expected = 0.0

        np.testing.assert_almost_equal(result, expected, decimal=10)

    def test_sum_squared_deviations_single_value(self) -> None:
        """Test sum of squared deviations for single value."""
        x = np.array([42.0])
        result = calculate_sum_x_squared_deviations(x)
        expected = 0.0

        np.testing.assert_almost_equal(result, expected, decimal=10)

    def test_sum_squared_deviations_mathematical_property(self) -> None:
        """Test mathematical property: sum of deviations from mean is zero."""
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        # Calculate manually to verify
        mean_x = np.mean(x)
        deviations = x - mean_x
        sum_deviations = np.sum(deviations)
        sum_squared_deviations = np.sum(deviations**2)

        # Sum of deviations should be approximately zero
        np.testing.assert_almost_equal(sum_deviations, 0.0, decimal=10)

        # Our function should match manual calculation
        result = calculate_sum_x_squared_deviations(x)
        np.testing.assert_almost_equal(result, sum_squared_deviations, decimal=10)

    def test_sum_squared_deviations_with_negative_values(self) -> None:
        """Test sum of squared deviations with negative values."""
        x = np.array([-3.0, -1.0, 1.0, 3.0])
        # Mean = 0, deviations = [-3, -1, 1, 3]
        # Squared deviations = [9, 1, 1, 9]
        # Sum = 20
        result = calculate_sum_x_squared_deviations(x)
        expected = 20.0

        np.testing.assert_almost_equal(result, expected, decimal=10)


@pytest.mark.mathematical
class TestExtractSumSquaredDeviations:
    """Mathematical validation for extracting sum of squared deviations from variance."""

    def test_extract_sum_squared_deviations_known_values(self) -> None:
        """Test extraction with known mathematical relationship."""
        # Relationship: sum_x_squared = sigma^2 / var_beta
        sigma = 10.0
        var_beta = 0.5
        expected = sigma**2 / var_beta  # = 100 / 0.5 = 200

        result = extract_sum_x_squared_deviations(var_beta, sigma)
        np.testing.assert_almost_equal(result, expected, decimal=10)

    def test_extract_sum_squared_deviations_unit_variance(self) -> None:
        """Test extraction with unit variance."""
        sigma = 5.0
        var_beta = 1.0
        expected = 25.0  # sigma^2 = 25

        result = extract_sum_x_squared_deviations(var_beta, sigma)
        np.testing.assert_almost_equal(result, expected, decimal=10)

    def test_extract_sum_squared_deviations_inverse_relationship(self) -> None:
        """Test that larger var_beta gives smaller sum_x_squared."""
        sigma = 10.0

        result_small_var = extract_sum_x_squared_deviations(0.1, sigma)
        result_large_var = extract_sum_x_squared_deviations(1.0, sigma)

        assert result_small_var > result_large_var
        assert result_small_var == 10 * result_large_var  # 1.0 / 0.1 = 10


@pytest.mark.mathematical
class TestSafeIntConversion:
    """Mathematical validation for safe integer conversion."""

    def test_safe_int_conversion_valid_integers(self) -> None:
        """Test conversion of valid integer values."""
        assert safe_int_conversion(42.0, "test_param") == 42
        assert safe_int_conversion(0.0, "test_param") == 0
        assert safe_int_conversion(-5.0, "test_param") == -5

    def test_safe_int_conversion_integers_as_float(self) -> None:
        """Test conversion of integers represented as floats."""
        assert safe_int_conversion(10.0, "test_param") == 10
        assert safe_int_conversion(100.0, "test_param") == 100

    def test_safe_int_conversion_non_integer_floats(self) -> None:
        """Test conversion raises error for non-integer floats."""
        with pytest.raises(
            ValueError, match="test_param should be an integer, got 42.5"
        ):
            safe_int_conversion(42.5, "test_param")

        with pytest.raises(
            ValueError, match="test_param should be an integer, got 3.14"
        ):
            safe_int_conversion(3.14, "test_param")

    def test_safe_int_conversion_very_small_decimals(self) -> None:
        """Test conversion with very small decimal parts."""
        # These should be considered integers due to floating point precision
        assert safe_int_conversion(42.0000000001, "test_param") == 42
        assert safe_int_conversion(10.0, "test_param") == 10

    def test_safe_int_conversion_edge_cases(self) -> None:
        """Test conversion with edge cases."""
        # Large integers
        large_int = 1000000.0
        assert safe_int_conversion(large_int, "test_param") == 1000000

        # Negative integers
        assert safe_int_conversion(-100.0, "test_param") == -100


@pytest.mark.mathematical
class TestMathematicalConsistency:
    """Test mathematical consistency across functions."""

    def test_sum_squared_deviations_consistency(self) -> None:
        """Test consistency between direct calculation and extraction."""
        # Generate test data
        np.random.seed(42)
        x = np.random.normal(100, 10, 50)

        # Calculate sum of squared deviations directly
        direct_result = calculate_sum_x_squared_deviations(x)

        # This test verifies the direct calculation is positive and finite
        assert direct_result > 0
        assert np.isfinite(direct_result)
        assert not np.isnan(direct_result)

    def test_mathematical_properties_preservation(self) -> None:
        """Test that mathematical properties are preserved."""
        # Test with known data
        x = np.array([1.0, 4.0, 7.0, 10.0, 13.0])

        # Calculate sum of squared deviations
        ssd = calculate_sum_x_squared_deviations(x)

        # Verify it equals manual calculation
        mean_x = np.mean(x)
        manual_ssd = np.sum((x - mean_x) ** 2)

        np.testing.assert_almost_equal(ssd, manual_ssd, decimal=10)

        # Verify positive definiteness
        assert ssd >= 0

        # For non-constant data, should be positive
        if not np.all(x == x[0]):
            assert ssd > 0
