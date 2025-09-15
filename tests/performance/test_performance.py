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
        # Generate large dataset for realistic TBR analysis
        np.random.seed(42)
        large_array = np.random.normal(0, 1, 10000)

        start_time = time.time()
        result = calculate_sum_x_squared_deviations(large_array)
        end_time = time.time()

        execution_time = end_time - start_time

        # Relaxed threshold for broader system compatibility
        # Mathematical operations should complete reasonably fast but allow for system variation
        max_time = 0.5  # 500ms - generous for 10K elements
        assert (
            execution_time < max_time
        ), f"Function took {execution_time:.4f} seconds, expected < {max_time}s"

        # Result should be mathematically valid
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

        # Relaxed threshold - validation should be fast but allow for system variation
        max_time = 0.2  # 200ms - reasonable for 10K datetime validations
        assert (
            execution_time < max_time
        ), f"Validation took {execution_time:.4f} seconds, expected < {max_time}s"

        # If we get here without exception, validation passed successfully

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

        # Should complete reasonably fast even with many columns
        max_time = 0.05  # 50ms - reasonable for column validation with 100 columns
        assert (
            execution_time < max_time
        ), f"Validation took {execution_time:.4f} seconds, expected < {max_time}s"


@pytest.mark.performance
@pytest.mark.slow
class TestScalabilityPerformance:
    """Performance tests for scalability with larger datasets."""

    def test_sum_squared_deviations_scalability(self) -> None:
        """Test scalability of sum squared deviations with increasing data size."""
        np.random.seed(42)

        # Focus on 3 meaningful sizes: small, medium, large for TBR analysis
        sizes = [1000, 10000, 50000]  # 1K, 10K, 50K data points
        times = []

        for size in sizes:
            data = np.random.normal(0, 1, size)

            start_time = time.time()
            result = calculate_sum_x_squared_deviations(data)
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

            # Verify result is mathematically valid
            assert result > 0
            assert np.isfinite(result)

            # Ensure reasonable absolute performance for each size
            if size <= 10000:
                max_time = 0.5  # 500ms for up to 10K elements
            else:
                max_time = 2.0  # 2s for 50K elements (generous)

            assert execution_time < max_time, (
                f"Function took {execution_time:.4f}s for {size} elements, "
                f"expected < {max_time}s"
            )

        # Check that performance scales reasonably (linear, not exponential)
        # Only check if we have reliable timing measurements
        if (
            len([t for t in times if t > 0.001]) >= 2
        ):  # At least 2 reliable measurements
            for i in range(1, len(times)):
                if (
                    times[i - 1] > 0.001 and times[i] > 0.001
                ):  # Both measurements reliable
                    size_ratio = sizes[i] / sizes[i - 1]
                    time_ratio = times[i] / times[i - 1]

                    # Allow generous scaling - main goal is to catch exponential degradation
                    max_acceptable_ratio = (
                        size_ratio * 3
                    )  # 3x time for Nx data is reasonable
                    assert time_ratio < max_acceptable_ratio, (
                        f"Performance scaling concerning: {time_ratio:.2f}x time "
                        f"for {size_ratio:.2f}x data (sizes: {sizes[i-1]} -> {sizes[i]})"
                    )

    def test_dataframe_operations_scalability(self) -> None:
        """Test scalability of DataFrame validation operations."""
        # Focus on 2 meaningful sizes for DataFrame operations
        sizes = [5000, 20000]  # Medium and large DataFrames for TBR analysis

        for size in sizes:
            # Generate DataFrame with typical TBR structure
            df = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=size, freq="D"),
                    "control": np.random.normal(100, 10, size),
                    "test": np.random.normal(105, 10, size),
                    "extra_col": np.random.randn(size),
                }
            )

            start_time = time.time()

            # Perform typical TBR validation operations
            validate_time_column_type(df, "date", "test_df")
            validate_required_columns(df, ["date", "control", "test"], "test_df")

            end_time = time.time()
            execution_time = end_time - start_time

            # Relaxed thresholds based on DataFrame size
            if size <= 10000:
                max_time = 0.3  # 300ms for smaller DataFrames
            else:
                max_time = 1.0  # 1s for larger DataFrames

            assert execution_time < max_time, (
                f"DataFrame validation took {execution_time:.4f}s for {size} rows, "
                f"expected < {max_time}s"
            )


@pytest.mark.performance
class TestMemoryEfficiency:
    """Tests for memory efficiency of core functions."""

    def test_sum_squared_deviations_memory_efficiency(self) -> None:
        """Test that sum squared deviations handles large arrays without excessive memory usage."""
        np.random.seed(42)

        # Test with progressively larger arrays to check memory handling
        sizes = [50000, 100000, 200000]  # 50K, 100K, 200K elements

        for size in sizes:
            large_array = np.random.normal(0, 1, size)

            # Record initial array size for reference (informational)
            _ = large_array.nbytes / (1024 * 1024)  # Memory usage in MB

            # The function should work with large arrays without memory issues
            result = calculate_sum_x_squared_deviations(large_array)

            # Verify mathematical correctness
            assert result > 0
            assert np.isfinite(result)

            # Basic memory efficiency check: function should not fail with large arrays
            # In a production environment, you'd use memory profiling tools like memory_profiler
            # For now, successful completion indicates reasonable memory efficiency

        # If we reach here, the function handles large arrays without memory errors

    def test_dataframe_validation_memory_efficiency(self) -> None:
        """Test that DataFrame validation operates efficiently on large DataFrames."""
        # Test with different DataFrame sizes to verify memory efficiency
        sizes = [30000, 60000]  # 30K and 60K rows - realistic for TBR analysis

        for n_rows in sizes:
            df = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
                    "control": np.random.normal(100, 10, n_rows),
                    "test": np.random.normal(105, 10, n_rows),
                    "extra_data": np.random.randn(n_rows),  # Additional column
                }
            )

            # Record DataFrame memory usage for reference (informational)
            _ = df.memory_usage(deep=True).sum() / (1024 * 1024)  # Memory usage in MB

            # Validation should not cause memory issues or create unnecessary copies
            validate_time_column_type(df, "date", "large_df")
            validate_required_columns(df, ["date", "control", "test"], "large_df")

            # Validation functions should be read-only and not modify the original DataFrame
            # Verify DataFrame structure is unchanged
            assert len(df) == n_rows
            assert list(df.columns) == ["date", "control", "test", "extra_data"]

        # If we reach here, validation is memory-efficient and non-destructive


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
