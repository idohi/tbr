"""Basic performance tests for TBR package functions."""

import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pytest

from tbr.functional.tbr_functions import (
    calculate_sum_x_squared_deviations,
    validate_required_columns,
    validate_time_column_type,
)


@pytest.mark.performance
class TestBasicPerformance:
    """Basic performance tests for core functions."""

    def test_sum_squared_deviations_performance(self) -> None:
        """Test performance of sum of squared deviations calculation."""
        # Generate large dataset
        np.random.seed(42)
        large_array = np.random.normal(0, 1, 10000)

        start_time = time.time()
        result = calculate_sum_x_squared_deviations(large_array)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert (
            execution_time < 0.1
        ), f"Function took {execution_time:.4f} seconds, expected < 0.1s"

        # Result should be valid
        assert result > 0
        assert np.isfinite(result)

    def test_time_column_validation_performance(self) -> None:
        """Test performance of time column validation."""
        # Generate large DataFrame with datetime column
        n_rows = 10000
        dates = pd.date_range("2020-01-01", periods=n_rows, freq="D")
        df = pd.DataFrame({"date": dates, "value": np.random.randn(n_rows)})

        start_time = time.time()
        # Function returns None but validates successfully or raises exception
        validate_time_column_type(df, "date", "test_df")
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete quickly
        assert (
            execution_time < 0.05
        ), f"Validation took {execution_time:.4f} seconds, expected < 0.05s"

        # If we get here without exception, validation passed

    def test_column_validation_performance(self) -> None:
        """Test performance of column validation."""
        # Generate wide DataFrame
        n_cols = 100
        n_rows = 1000

        data = {f"col_{i}": np.random.randn(n_rows) for i in range(n_cols)}
        df = pd.DataFrame(data)

        required_cols = [f"col_{i}" for i in range(0, 50, 5)]  # Every 5th column

        start_time = time.time()
        validate_required_columns(df, required_cols, "test_df")
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete quickly even with many columns
        assert (
            execution_time < 0.01
        ), f"Validation took {execution_time:.4f} seconds, expected < 0.01s"


@pytest.mark.performance
@pytest.mark.slow
class TestScalabilityPerformance:
    """Performance tests for scalability with larger datasets."""

    def test_sum_squared_deviations_scalability(self) -> None:
        """Test scalability of sum squared deviations with increasing data size."""
        np.random.seed(42)

        sizes = [1000, 5000, 10000, 50000]
        times = []

        for size in sizes:
            data = np.random.normal(0, 1, size)

            start_time = time.time()
            result = calculate_sum_x_squared_deviations(data)
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

            # Verify result is valid
            assert result > 0
            assert np.isfinite(result)

        # Performance should scale reasonably (not exponentially)
        # This is a basic scalability check - timing can vary significantly for very small times
        for i in range(1, len(times)):
            size_ratio = sizes[i] / sizes[i - 1]

            # For very small times (< 0.001s), timing ratios are unreliable
            if times[i - 1] < 0.001 or times[i] < 0.001:
                # Just verify the function completes successfully for small datasets
                continue

            time_ratio = times[i] / times[i - 1]

            # Allow reasonable scaling - should not be exponential
            max_acceptable_ratio = (
                size_ratio * 5
            )  # Allow 5x time increase for 5x data increase
            assert (
                time_ratio < max_acceptable_ratio
            ), f"Performance degradation too severe: {time_ratio:.2f}x time for {size_ratio:.2f}x data"

    def test_dataframe_operations_scalability(self) -> None:
        """Test scalability of DataFrame operations."""
        sizes = [1000, 5000, 10000]

        for size in sizes:
            # Generate large DataFrame
            df = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=size, freq="D"),
                    "control": np.random.normal(100, 10, size),
                    "test": np.random.normal(105, 10, size),
                    "extra_col": np.random.randn(size),
                }
            )

            start_time = time.time()

            # Perform typical validation operations
            validate_time_column_type(df, "date", "test_df")
            validate_required_columns(df, ["date", "control", "test"], "test_df")

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete within reasonable time even for large DataFrames
            max_time = 0.1  # 100ms should be sufficient
            assert (
                execution_time < max_time
            ), f"DataFrame operations took {execution_time:.4f}s for {size} rows, expected < {max_time}s"


@pytest.mark.performance
class TestMemoryEfficiency:
    """Tests for memory efficiency of core functions."""

    def test_sum_squared_deviations_memory_efficiency(self) -> None:
        """Test that sum squared deviations doesn't create unnecessary copies."""
        # This is a basic test - in practice you'd use memory profiling tools
        np.random.seed(42)
        large_array = np.random.normal(0, 1, 100000)

        # The function should work with large arrays without memory issues
        result = calculate_sum_x_squared_deviations(large_array)

        assert result > 0
        assert np.isfinite(result)
        # If we get here without MemoryError, the function is reasonably efficient

    def test_dataframe_validation_memory_efficiency(self) -> None:
        """Test that DataFrame validation doesn't create unnecessary copies."""
        # Create large DataFrame
        n_rows = 50000
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
                "control": np.random.normal(100, 10, n_rows),
                "test": np.random.normal(105, 10, n_rows),
            }
        )

        # Validation should not cause memory issues
        validate_time_column_type(df, "date", "large_df")
        validate_required_columns(df, ["date", "control", "test"], "large_df")

        # If we get here without MemoryError, validation is memory-efficient


class TestPerformanceUtilities:
    """Utility functions for performance testing."""

    @staticmethod
    def time_function(func: Any, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
        """Time a function execution."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    @staticmethod
    def assert_performance_threshold(
        execution_time: float, threshold: float, operation_name: str
    ) -> None:
        """Assert that execution time is within threshold."""
        assert execution_time < threshold, (
            f"{operation_name} took {execution_time:.4f} seconds, "
            f"expected < {threshold} seconds"
        )
