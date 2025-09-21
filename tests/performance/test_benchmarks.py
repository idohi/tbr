"""
Performance benchmarks for regression implementations.

This module provides comprehensive performance benchmarks comparing core
regression module performance against the functional implementation to
ensure optimal performance while maintaining mathematical accuracy.
"""

import gc
import time
from typing import Callable, Dict

import numpy as np
import pandas as pd
import pytest

# Core module imports
from tbr.core.regression import (
    calculate_model_variance,
    calculate_prediction_variance,
    calculate_sum_squared_deviations,
    calculate_variances,
    convert_to_integer,
    fit_regression_model,
)

# Functional module imports
from tbr.functional.tbr_functions import (
    calculate_model_variance as func_calculate_model_variance,
)
from tbr.functional.tbr_functions import (
    calculate_prediction_variance as func_calculate_prediction_variance,
)
from tbr.functional.tbr_functions import (
    calculate_sum_x_squared_deviations as func_calculate_sum_x_squared_deviations,
)
from tbr.functional.tbr_functions import (
    fit_tbr_regression_model as func_fit_tbr_regression_model,
)
from tbr.functional.tbr_functions import safe_int_conversion as func_safe_int_conversion


class PerformanceBenchmarker:
    """Utility class for performance benchmarking with statistical analysis."""

    @staticmethod
    def benchmark_function(
        func: Callable, *args, n_runs: int = 5, warmup_runs: int = 2, **kwargs
    ) -> Dict[str, float]:
        """
        Benchmark a function with statistical analysis.

        Parameters
        ----------
        func : Callable
            Function to benchmark
        *args : tuple
            Positional arguments for the function
        n_runs : int
            Number of benchmark runs
        warmup_runs : int
            Number of warmup runs (not counted in statistics)
        **kwargs : dict
            Keyword arguments for the function

        Returns
        -------
        Dict[str, float]
            Dictionary with timing statistics: mean, std, min, max
        """
        # Warmup runs
        for _ in range(warmup_runs):
            func(*args, **kwargs)

        # Benchmark runs
        times = []
        for _ in range(n_runs):
            gc.collect()  # Ensure clean memory state
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        times = np.array(times)
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "result": result,
        }

    @staticmethod
    def compare_performance(
        core_stats: Dict[str, float],
        func_stats: Dict[str, float],
        tolerance: float = 2.0,
    ) -> Dict[str, float]:
        """
        Compare performance statistics between core and functional implementations.

        Parameters
        ----------
        core_stats : Dict[str, float]
            Core implementation timing statistics
        func_stats : Dict[str, float]
            Functional implementation timing statistics
        tolerance : float
            Maximum acceptable performance ratio (core/func)

        Returns
        -------
        Dict[str, float]
            Performance comparison metrics
        """
        ratio_mean = core_stats["mean"] / func_stats["mean"]
        ratio_min = core_stats["min"] / func_stats["min"]

        return {
            "ratio_mean": ratio_mean,
            "ratio_min": ratio_min,
            "core_faster": ratio_mean < 1.0,
            "within_tolerance": ratio_mean <= tolerance,
            "performance_difference_pct": (ratio_mean - 1.0) * 100,
        }


@pytest.mark.performance
class TestRegressionFittingPerformance:
    """Performance benchmarks for regression fitting functions."""

    def test_regression_fitting_scalability(self):
        """Test regression fitting performance across different data sizes."""
        benchmarker = PerformanceBenchmarker()

        # Test different data sizes
        data_sizes = [50, 100, 500, 1000]
        results = []

        for size in data_sizes:
            np.random.seed(42)
            learning_data = pd.DataFrame(
                {
                    "control": np.random.normal(1000, 100, size),
                    "test": np.random.normal(1050, 110, size),
                }
            )

            # Benchmark core implementation
            core_stats = benchmarker.benchmark_function(
                fit_regression_model, learning_data, "control", "test"
            )

            # Benchmark functional implementation
            func_stats = benchmarker.benchmark_function(
                func_fit_tbr_regression_model, learning_data, "control", "test"
            )

            # Compare performance
            comparison = benchmarker.compare_performance(core_stats, func_stats)

            results.append(
                {
                    "size": size,
                    "core_mean": core_stats["mean"],
                    "func_mean": func_stats["mean"],
                    "ratio": comparison["ratio_mean"],
                    "within_tolerance": comparison["within_tolerance"],
                }
            )

            # Validate results are identical
            core_result = core_stats["result"]
            func_result = func_stats["result"]
            for key in core_result:
                assert abs(core_result[key] - func_result[key]) < 1e-12

            # Performance should be within tolerance
            assert comparison["within_tolerance"], (
                f"Performance regression for size {size}: "
                f"ratio={comparison['ratio_mean']:.2f}, "
                f"difference={comparison['performance_difference_pct']:.1f}%"
            )

        # Log performance results for analysis
        print("\nRegression Fitting Performance Results:")
        print("Size\tCore(ms)\tFunc(ms)\tRatio\tStatus")
        for result in results:
            status = "✓" if result["within_tolerance"] else "✗"
            print(
                f"{result['size']}\t{result['core_mean']*1000:.2f}\t\t"
                f"{result['func_mean']*1000:.2f}\t\t{result['ratio']:.2f}\t{status}"
            )

    def test_regression_fitting_memory_efficiency(self):
        """Test memory efficiency of regression fitting."""
        import os

        import psutil

        # Large dataset for memory testing
        np.random.seed(123)
        large_data = pd.DataFrame(
            {
                "control": np.random.normal(1000, 100, 5000),
                "test": np.random.normal(1050, 110, 5000),
            }
        )

        process = psutil.Process(os.getpid())

        # Measure memory usage for core implementation
        gc.collect()
        mem_before_core = process.memory_info().rss
        core_result = fit_regression_model(large_data, "control", "test")
        mem_after_core = process.memory_info().rss
        core_memory_delta = mem_after_core - mem_before_core

        # Measure memory usage for functional implementation
        gc.collect()
        mem_before_func = process.memory_info().rss
        func_result = func_fit_tbr_regression_model(large_data, "control", "test")
        mem_after_func = process.memory_info().rss
        func_memory_delta = mem_after_func - mem_before_func

        # Memory usage should be comparable (allow core to be more efficient)
        if func_memory_delta > 0:  # Avoid division by zero
            memory_ratio = core_memory_delta / func_memory_delta
            # Allow core to use less memory (ratio < 1.0) but not significantly more
            assert 0.01 <= memory_ratio <= 3.0, (
                f"Memory usage comparison: core={core_memory_delta/1024/1024:.2f}MB, "
                f"func={func_memory_delta/1024/1024:.2f}MB, ratio={memory_ratio:.2f} "
                f"(core is {'more' if memory_ratio < 1.0 else 'less'} memory efficient)"
            )

        # Results should be identical
        for key in core_result:
            assert abs(core_result[key] - func_result[key]) < 1e-12


@pytest.mark.performance
class TestSumSquaredDeviationsPerformance:
    """Performance benchmarks for sum of squared deviations calculations."""

    def test_sum_squared_deviations_scalability(self):
        """Test sum squared deviations performance across different array sizes."""
        benchmarker = PerformanceBenchmarker()

        # Test different array sizes
        array_sizes = [100, 1000, 5000, 10000]
        results = []

        for size in array_sizes:
            np.random.seed(42)
            test_array = np.random.normal(1000, 100, size)

            # Benchmark core implementation
            core_stats = benchmarker.benchmark_function(
                calculate_sum_squared_deviations, test_array
            )

            # Benchmark functional implementation
            func_stats = benchmarker.benchmark_function(
                func_calculate_sum_x_squared_deviations, test_array
            )

            # Compare performance (allow higher tolerance for scalability tests)
            comparison = benchmarker.compare_performance(
                core_stats, func_stats, tolerance=3.0
            )

            results.append(
                {
                    "size": size,
                    "core_mean": core_stats["mean"],
                    "func_mean": func_stats["mean"],
                    "ratio": comparison["ratio_mean"],
                    "within_tolerance": comparison["within_tolerance"],
                }
            )

            # Validate results are identical
            assert abs(core_stats["result"] - func_stats["result"]) < 1e-15

            # Performance should be within tolerance
            assert comparison["within_tolerance"], (
                f"Performance regression for array size {size}: "
                f"ratio={comparison['ratio_mean']:.2f}"
            )

        # Log performance results
        print("\nSum Squared Deviations Performance Results:")
        print("Size\tCore(μs)\tFunc(μs)\tRatio\tStatus")
        for result in results:
            status = "✓" if result["within_tolerance"] else "✗"
            print(
                f"{result['size']}\t{result['core_mean']*1e6:.1f}\t\t"
                f"{result['func_mean']*1e6:.1f}\t\t{result['ratio']:.2f}\t{status}"
            )

    def test_sum_squared_deviations_numerical_stability(self):
        """Test numerical stability performance with challenging data."""
        benchmarker = PerformanceBenchmarker()

        # Test scenarios with different numerical challenges
        test_scenarios = [
            ("small_values", np.random.uniform(1e-6, 1e-5, 1000)),
            ("large_values", np.random.uniform(1e6, 1e7, 1000)),
            (
                "mixed_range",
                np.concatenate(
                    [
                        np.random.uniform(1e-6, 1e-5, 500),
                        np.random.uniform(1e6, 1e7, 500),
                    ]
                ),
            ),
            ("near_zero", np.random.uniform(-1e-10, 1e-10, 1000) + 1000),
        ]

        for scenario_name, test_data in test_scenarios:
            # Benchmark both implementations
            core_stats = benchmarker.benchmark_function(
                calculate_sum_squared_deviations, test_data
            )
            func_stats = benchmarker.benchmark_function(
                func_calculate_sum_x_squared_deviations, test_data
            )

            # Compare performance
            comparison = benchmarker.compare_performance(core_stats, func_stats)

            # Validate numerical accuracy
            relative_error = abs(core_stats["result"] - func_stats["result"]) / max(
                abs(func_stats["result"]), 1e-15
            )
            assert relative_error < 1e-12, (
                f"Numerical accuracy issue in {scenario_name}: "
                f"relative_error={relative_error:.2e}"
            )

            # Performance should be reasonable
            assert comparison["within_tolerance"], (
                f"Performance issue in {scenario_name}: "
                f"ratio={comparison['ratio_mean']:.2f}"
            )


@pytest.mark.performance
class TestVarianceCalculationsPerformance:
    """Performance benchmarks for variance calculations."""

    def test_model_variance_performance(self):
        """Test model variance calculation performance."""
        benchmarker = PerformanceBenchmarker()

        # Test with different array sizes
        sizes = [100, 500, 1000, 2000]

        for size in sizes:
            np.random.seed(42)
            x_values = np.random.normal(1000, 100, size)
            x_mean = 1000.0
            sigma = 25.0
            n_pretest = 500
            sum_x_squared_dev = np.sum((x_values - x_mean) ** 2)

            # Benchmark core implementation
            core_stats = benchmarker.benchmark_function(
                calculate_model_variance,
                x_values,
                x_mean,
                sigma,
                n_pretest,
                sum_x_squared_dev,
            )

            # Benchmark functional implementation
            func_stats = benchmarker.benchmark_function(
                func_calculate_model_variance,
                x_values,
                x_mean,
                sigma,
                n_pretest,
                sum_x_squared_dev,
            )

            # Compare performance
            comparison = benchmarker.compare_performance(core_stats, func_stats)

            # Validate results
            np.testing.assert_allclose(
                core_stats["result"], func_stats["result"], rtol=1e-15
            )

            # Performance should be within tolerance
            assert comparison["within_tolerance"], (
                f"Model variance performance regression for size {size}: "
                f"ratio={comparison['ratio_mean']:.2f}"
            )

    def test_prediction_variance_performance(self):
        """Test prediction variance calculation performance."""
        benchmarker = PerformanceBenchmarker()

        # Test with different array sizes
        sizes = [100, 500, 1000, 2000]

        for size in sizes:
            np.random.seed(123)
            model_variances = np.random.uniform(0.1, 10.0, size)
            sigma = 25.0

            # Benchmark core implementation
            core_stats = benchmarker.benchmark_function(
                calculate_prediction_variance, model_variances, sigma
            )

            # Benchmark functional implementation
            func_stats = benchmarker.benchmark_function(
                func_calculate_prediction_variance, model_variances, sigma
            )

            # Compare performance
            comparison = benchmarker.compare_performance(core_stats, func_stats)

            # Validate results
            np.testing.assert_allclose(
                core_stats["result"], func_stats["result"], rtol=1e-15
            )

            # Performance should be within tolerance
            assert comparison["within_tolerance"], (
                f"Prediction variance performance regression for size {size}: "
                f"ratio={comparison['ratio_mean']:.2f}"
            )

    def test_combined_variance_calculations_performance(self):
        """Test combined variance calculations performance."""
        benchmarker = PerformanceBenchmarker()

        # Setup test data
        np.random.seed(456)
        x_values = np.random.normal(1000, 100, 1000)
        x_mean = 1000.0
        sigma = 25.0
        n_pretest = 500
        sum_x_squared_dev = np.sum((x_values - x_mean) ** 2)

        # Benchmark core combined function
        core_stats = benchmarker.benchmark_function(
            calculate_variances, x_values, x_mean, sigma, n_pretest, sum_x_squared_dev
        )

        # Benchmark functional separate functions
        def func_combined_variances(x_vals, x_mean, sigma, n_pretest, sum_sq_dev):
            model_vars = func_calculate_model_variance(
                x_vals, x_mean, sigma, n_pretest, sum_sq_dev
            )
            pred_vars = func_calculate_prediction_variance(model_vars, sigma)
            return model_vars, pred_vars

        func_stats = benchmarker.benchmark_function(
            func_combined_variances,
            x_values,
            x_mean,
            sigma,
            n_pretest,
            sum_x_squared_dev,
        )

        # Compare performance
        comparison = benchmarker.compare_performance(core_stats, func_stats)

        # Validate results
        core_model_vars, core_pred_vars = core_stats["result"]
        func_model_vars, func_pred_vars = func_stats["result"]

        np.testing.assert_allclose(core_model_vars, func_model_vars, rtol=1e-15)
        np.testing.assert_allclose(core_pred_vars, func_pred_vars, rtol=1e-15)

        # Performance should be within tolerance (combined function might be faster)
        assert comparison["within_tolerance"], (
            f"Combined variance calculations performance regression: "
            f"ratio={comparison['ratio_mean']:.2f}"
        )


@pytest.mark.performance
class TestIntegerConversionPerformance:
    """Performance benchmarks for integer conversion functions."""

    def test_integer_conversion_performance(self):
        """Test integer conversion performance with various inputs."""
        benchmarker = PerformanceBenchmarker()

        # Test different types of values
        test_values = [
            42.0,
            100.0000001,
            25.9999999,
            1.0,
            999.0000000001,
            50.0,
            200.0,
        ]

        for value in test_values:
            # Benchmark core implementation
            core_stats = benchmarker.benchmark_function(
                convert_to_integer, value, "test_param"
            )

            # Benchmark functional implementation
            func_stats = benchmarker.benchmark_function(
                func_safe_int_conversion, value, "test_param"
            )

            # Compare performance
            comparison = benchmarker.compare_performance(core_stats, func_stats)

            # Validate results
            assert core_stats["result"] == func_stats["result"]

            # Performance should be within tolerance
            assert comparison["within_tolerance"], (
                f"Integer conversion performance regression for value {value}: "
                f"ratio={comparison['ratio_mean']:.2f}"
            )

    def test_integer_conversion_batch_performance(self):
        """Test integer conversion performance with batch operations."""
        benchmarker = PerformanceBenchmarker()

        # Generate batch of values to convert
        np.random.seed(789)
        batch_values = np.random.uniform(1.0, 1000.0, 1000).astype(int).astype(float)

        # Benchmark core implementation (batch)
        def core_batch_conversion(values):
            return [convert_to_integer(val, "batch_param") for val in values]

        core_stats = benchmarker.benchmark_function(core_batch_conversion, batch_values)

        # Benchmark functional implementation (batch)
        def func_batch_conversion(values):
            return [func_safe_int_conversion(val, "batch_param") for val in values]

        func_stats = benchmarker.benchmark_function(func_batch_conversion, batch_values)

        # Compare performance
        comparison = benchmarker.compare_performance(core_stats, func_stats)

        # Validate results
        assert core_stats["result"] == func_stats["result"]

        # Performance should be within tolerance
        assert comparison["within_tolerance"], (
            f"Batch integer conversion performance regression: "
            f"ratio={comparison['ratio_mean']:.2f}"
        )


@pytest.mark.performance
class TestEndToEndPerformanceBenchmarks:
    """End-to-end performance benchmarks for complete regression workflows."""

    def test_complete_regression_workflow_performance(self):
        """Test performance of complete regression analysis workflow."""
        benchmarker = PerformanceBenchmarker()

        # Generate realistic TBR dataset
        np.random.seed(42)
        n_samples = 1000
        control_values = np.random.normal(1000, 100, n_samples)
        test_values = 50 + 0.95 * control_values + np.random.normal(0, 30, n_samples)

        learning_data = pd.DataFrame({"control": control_values, "test": test_values})

        # Define complete workflow using core functions
        def core_workflow(data):
            # Step 1: Fit regression
            params = fit_regression_model(data, "control", "test")

            # Step 2: Calculate sum squared deviations
            sum_sq_dev = calculate_sum_squared_deviations(data["control"].values)

            # Step 3: Calculate variances for prediction
            test_x = np.array([950, 1000, 1050, 1100])
            model_vars, pred_vars = calculate_variances(
                test_x,
                params["x_mean"],
                params["sigma"],
                params["n_pretest"],
                sum_sq_dev,
            )

            # Step 4: Convert degrees of freedom
            dof = convert_to_integer(params["degrees_freedom"], "degrees_freedom")

            return params, sum_sq_dev, model_vars, pred_vars, dof

        # Define complete workflow using functional functions
        def func_workflow(data):
            # Step 1: Fit regression
            params = func_fit_tbr_regression_model(data, "control", "test")

            # Step 2: Calculate sum squared deviations
            sum_sq_dev = func_calculate_sum_x_squared_deviations(data["control"].values)

            # Step 3: Calculate variances for prediction
            test_x = np.array([950, 1000, 1050, 1100])
            model_vars = func_calculate_model_variance(
                test_x,
                params["x_mean"],
                params["sigma"],
                params["n_pretest"],
                sum_sq_dev,
            )
            pred_vars = func_calculate_prediction_variance(model_vars, params["sigma"])

            # Step 4: Convert degrees of freedom
            dof = func_safe_int_conversion(params["degrees_freedom"], "degrees_freedom")

            return params, sum_sq_dev, model_vars, pred_vars, dof

        # Benchmark both workflows
        core_stats = benchmarker.benchmark_function(core_workflow, learning_data)
        func_stats = benchmarker.benchmark_function(func_workflow, learning_data)

        # Compare performance
        comparison = benchmarker.compare_performance(core_stats, func_stats)

        # Validate results are identical
        core_results = core_stats["result"]
        func_results = func_stats["result"]

        # Compare all components
        for key in core_results[0]:  # params
            assert abs(core_results[0][key] - func_results[0][key]) < 1e-12

        assert abs(core_results[1] - func_results[1]) < 1e-15  # sum_sq_dev
        np.testing.assert_allclose(
            core_results[2], func_results[2], rtol=1e-15
        )  # model_vars
        np.testing.assert_allclose(
            core_results[3], func_results[3], rtol=1e-15
        )  # pred_vars
        assert core_results[4] == func_results[4]  # dof

        # Performance should be within tolerance
        assert comparison["within_tolerance"], (
            f"End-to-end workflow performance regression: "
            f"ratio={comparison['ratio_mean']:.2f}, "
            f"difference={comparison['performance_difference_pct']:.1f}%"
        )

        # Log detailed performance results
        print("\nEnd-to-End Workflow Performance:")
        print(
            f"Core implementation: {core_stats['mean']*1000:.2f} ± {core_stats['std']*1000:.2f} ms"
        )
        print(
            f"Functional implementation: {func_stats['mean']*1000:.2f} ± {func_stats['std']*1000:.2f} ms"
        )
        print(f"Performance ratio: {comparison['ratio_mean']:.2f}")
        print(
            f"Performance difference: {comparison['performance_difference_pct']:.1f}%"
        )
        print(f"Core is faster: {comparison['core_faster']}")
        print(f"Within tolerance: {comparison['within_tolerance']}")
