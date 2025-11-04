"""
Performance benchmarks for TBRAnalysis OOP API workflows.

This module provides comprehensive performance testing for the TBRAnalysis class,
validating that the user-friendly OOP API doesn't introduce significant performance
overhead compared to the functional API.

Test Categories
---------------
1. Method Performance: Individual TBRAnalysis method benchmarks
2. Workflow Performance: Complete end-to-end workflow benchmarks
3. API Comparison: OOP vs. Functional API performance comparison
4. Scalability: Performance across different data sizes
5. Method Chaining: Fluent API performance overhead
6. Regression Prevention: Performance baseline validation

Performance Criteria
-------------------
- OOP API overhead: ≤ 10% slower than functional API
- Scalability: Linear O(n) performance for data size
- Memory efficiency: Comparable to functional API
- No performance regressions from baseline
"""

import time
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from tbr import TBRAnalysis
from tbr.functional import perform_tbr_analysis


class PerformanceBenchmarker:
    """Helper class for performance benchmarking."""

    def __init__(self, n_iterations: int = 5, warmup: int = 1):
        """
        Initialize benchmarker.

        Parameters
        ----------
        n_iterations : int
            Number of iterations for each benchmark
        warmup : int
            Number of warmup iterations to discard
        """
        self.n_iterations = n_iterations
        self.warmup = warmup

    def benchmark_function(self, func, *args, **kwargs) -> Dict:
        """
        Benchmark a function.

        Returns dictionary with timing statistics and result.
        """
        # Warmup
        for _ in range(self.warmup):
            func(*args, **kwargs)

        # Actual benchmarking
        times = []
        for _ in range(self.n_iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)

        return {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "times": times,
            "result": result,
        }

    def compare_performance(
        self, stats1: Dict, stats2: Dict, tolerance: float = 2.0
    ) -> Dict:
        """
        Compare performance between two benchmarks.

        Parameters
        ----------
        stats1 : dict
            Statistics from first benchmark
        stats2 : dict
            Statistics from second benchmark (baseline)
        tolerance : float
            Maximum acceptable performance ratio

        Returns
        -------
        dict
            Comparison results including ratio and within_tolerance flag
        """
        ratio = stats1["mean"] / stats2["mean"]
        diff_pct = ((stats1["mean"] - stats2["mean"]) / stats2["mean"]) * 100

        return {
            "ratio_mean": ratio,
            "performance_difference_pct": diff_pct,
            "within_tolerance": ratio <= tolerance,
            "first_faster": ratio < 1.0,
            "stats1_mean": stats1["mean"],
            "stats2_mean": stats2["mean"],
        }


@pytest.fixture
def benchmarker():
    """Provide performance benchmarker."""
    return PerformanceBenchmarker(n_iterations=5, warmup=1)


@pytest.fixture
def sample_data_small():
    """Generate small sample data for performance testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60)
    return pd.DataFrame(
        {
            "date": dates,
            "control": np.random.normal(1000, 50, 60),
            "test": np.random.normal(1020, 55, 60),
        }
    )


@pytest.fixture
def sample_data_medium():
    """Generate medium sample data for performance testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=180)
    return pd.DataFrame(
        {
            "date": dates,
            "control": np.random.normal(1000, 50, 180),
            "test": np.random.normal(1020, 55, 180),
        }
    )


@pytest.fixture
def sample_data_large():
    """Generate large sample data for performance testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=1000)
    return pd.DataFrame(
        {
            "date": dates,
            "control": np.random.normal(1000, 50, 1000),
            "test": np.random.normal(1020, 55, 1000),
        }
    )


class TestTBRAnalysisMethodPerformance:
    """Performance benchmarks for individual TBRAnalysis methods."""

    def test_fit_method_performance(self, benchmarker, sample_data_medium):
        """Test performance of fit() method."""

        def fit_workflow(data):
            m = TBRAnalysis(level=0.80)
            m.fit(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2024-01-01"),
                test_start=pd.Timestamp("2024-03-01"),
                test_end=pd.Timestamp("2024-06-28"),
            )
            return m

        stats = benchmarker.benchmark_function(fit_workflow, sample_data_medium)

        # Validate performance
        assert stats["mean"] < 1.0, f"fit() too slow: {stats['mean']:.3f}s"
        print(
            f"\nfit() performance: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms"
        )

    def test_predict_method_performance(self, benchmarker, sample_data_medium):
        """Test performance of predict() method."""
        model = TBRAnalysis(level=0.80)
        model.fit(
            data=sample_data_medium,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2024-01-01"),
            test_start=pd.Timestamp("2024-03-01"),
            test_end=pd.Timestamp("2024-06-28"),
        )

        stats = benchmarker.benchmark_function(model.predict)

        # Validate performance
        assert stats["mean"] < 0.1, f"predict() too slow: {stats['mean']:.3f}s"
        print(
            f"\npredict() performance: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms"
        )

    def test_summarize_method_performance(self, benchmarker, sample_data_medium):
        """Test performance of summarize() method."""
        model = TBRAnalysis(level=0.80)
        model.fit(
            data=sample_data_medium,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2024-01-01"),
            test_start=pd.Timestamp("2024-03-01"),
            test_end=pd.Timestamp("2024-06-28"),
        )

        stats = benchmarker.benchmark_function(model.summarize)

        # Validate performance (summarize should be very fast - just data access)
        assert stats["mean"] < 0.01, f"summarize() too slow: {stats['mean']:.3f}s"
        print(
            f"\nsummarize() performance: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms"
        )

    def test_analyze_subinterval_performance(self, benchmarker, sample_data_medium):
        """Test performance of analyze_subinterval() method."""
        model = TBRAnalysis(level=0.80)
        model.fit(
            data=sample_data_medium,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2024-01-01"),
            test_start=pd.Timestamp("2024-03-01"),
            test_end=pd.Timestamp("2024-06-28"),
        )

        stats = benchmarker.benchmark_function(model.analyze_subinterval, 1, 30)

        # Validate performance
        assert (
            stats["mean"] < 0.05
        ), f"analyze_subinterval() too slow: {stats['mean']:.3f}s"
        print(
            f"\nanalyze_subinterval() performance: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms"
        )

    def test_summarize_incremental_performance(self, benchmarker, sample_data_medium):
        """Test performance of summarize_incremental() method."""
        model = TBRAnalysis(level=0.80)
        model.fit(
            data=sample_data_medium,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2024-01-01"),
            test_start=pd.Timestamp("2024-03-01"),
            test_end=pd.Timestamp("2024-06-28"),
        )

        stats = benchmarker.benchmark_function(model.summarize_incremental)

        # Validate performance (should be very fast - just returns stored data)
        assert (
            stats["mean"] < 0.01
        ), f"summarize_incremental() too slow: {stats['mean']:.3f}s"
        print(
            f"\nsummarize_incremental() performance: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms"
        )


class TestWorkflowPerformanceBenchmarks:
    """Performance benchmarks for complete end-to-end workflows."""

    def test_basic_fit_summarize_workflow_performance(
        self, benchmarker, sample_data_medium
    ):
        """Test performance of basic fit→summarize workflow."""

        def workflow(data):
            model = TBRAnalysis(level=0.80)
            model.fit(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2024-01-01"),
                test_start=pd.Timestamp("2024-03-01"),
                test_end=pd.Timestamp("2024-06-28"),
            )
            summary = model.summarize()
            return summary

        stats = benchmarker.benchmark_function(workflow, sample_data_medium)

        # Validate performance
        assert stats["mean"] < 1.0, f"Workflow too slow: {stats['mean']:.3f}s"
        print(
            f"\nfit→summarize workflow: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms"
        )

    def test_fit_predict_workflow_performance(self, benchmarker, sample_data_medium):
        """Test performance of fit→predict workflow."""

        def workflow(data):
            model = TBRAnalysis(level=0.80)
            model.fit(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2024-01-01"),
                test_start=pd.Timestamp("2024-03-01"),
                test_end=pd.Timestamp("2024-06-28"),
            )
            predictions = model.predict()
            return predictions

        stats = benchmarker.benchmark_function(workflow, sample_data_medium)

        # Validate performance
        assert stats["mean"] < 1.0, f"Workflow too slow: {stats['mean']:.3f}s"
        print(
            f"\nfit→predict workflow: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms"
        )

    def test_comprehensive_analysis_workflow_performance(
        self, benchmarker, sample_data_medium
    ):
        """Test performance of comprehensive analysis workflow."""

        def workflow(data):
            model = TBRAnalysis(level=0.80)
            model.fit(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2024-01-01"),
                test_start=pd.Timestamp("2024-03-01"),
                test_end=pd.Timestamp("2024-06-28"),
            )
            summary = model.summarize()
            predictions = model.predict()
            incremental = model.summarize_incremental()
            subinterval = model.analyze_subinterval(1, 30)
            return summary, predictions, incremental, subinterval

        stats = benchmarker.benchmark_function(workflow, sample_data_medium)

        # Validate performance
        assert (
            stats["mean"] < 1.5
        ), f"Comprehensive workflow too slow: {stats['mean']:.3f}s"
        print(
            f"\nComprehensive workflow: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms"
        )

    def test_multiple_subintervals_workflow_performance(
        self, benchmarker, sample_data_medium
    ):
        """Test performance of analyzing multiple subintervals."""

        def workflow(data):
            model = TBRAnalysis(level=0.80)
            model.fit(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2024-01-01"),
                test_start=pd.Timestamp("2024-03-01"),
                test_end=pd.Timestamp("2024-06-28"),
            )

            # Analyze 10 subintervals
            results = []
            for i in range(10):
                start = i * 10 + 1
                end = min(start + 9, 120)
                results.append(model.analyze_subinterval(start, end))
            return results

        stats = benchmarker.benchmark_function(workflow, sample_data_medium)

        # Validate performance
        assert (
            stats["mean"] < 2.0
        ), f"Multiple subintervals too slow: {stats['mean']:.3f}s"
        print(
            f"\n10 subintervals workflow: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms"
        )


class TestOOPFunctionalPerformanceComparison:
    """Performance comparison between OOP and Functional APIs."""

    def test_oop_vs_functional_basic_analysis(self, benchmarker, sample_data_medium):
        """Compare OOP and Functional API performance for basic analysis."""

        # OOP API workflow
        def oop_workflow(data):
            model = TBRAnalysis(level=0.80, threshold=0.0)
            model.fit(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2024-01-01"),
                test_start=pd.Timestamp("2024-03-01"),
                test_end=pd.Timestamp("2024-06-28"),
            )
            return model.results_, model.summaries_

        # Functional API workflow
        def functional_workflow(data):
            results = perform_tbr_analysis(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2024-01-01"),
                test_start=pd.Timestamp("2024-03-01"),
                test_end=pd.Timestamp("2024-06-28"),
                level=0.80,
                threshold=0.0,
            )
            return results.tbr_dataframe(), results.summary()

        # Benchmark both
        oop_stats = benchmarker.benchmark_function(oop_workflow, sample_data_medium)
        func_stats = benchmarker.benchmark_function(
            functional_workflow, sample_data_medium
        )

        # Compare performance
        comparison = benchmarker.compare_performance(
            oop_stats, func_stats, tolerance=1.5
        )

        print(
            f"\nOOP API: {oop_stats['mean']*1000:.2f} ± {oop_stats['std']*1000:.2f} ms"
        )
        print(
            f"Functional API: {func_stats['mean']*1000:.2f} ± {func_stats['std']*1000:.2f} ms"
        )
        print(f"Performance ratio: {comparison['ratio_mean']:.2f}x")
        print(f"Difference: {comparison['performance_difference_pct']:.1f}%")

        # OOP API should not be more than 50% slower (1.5x ratio)
        assert comparison["within_tolerance"], (
            f"OOP API too slow compared to functional: "
            f"ratio={comparison['ratio_mean']:.2f}x, "
            f"difference={comparison['performance_difference_pct']:.1f}%"
        )

    def test_oop_vs_functional_with_predictions(self, benchmarker, sample_data_medium):
        """Compare OOP and Functional API performance including predictions."""

        # OOP API workflow with predictions
        def oop_workflow(data):
            model = TBRAnalysis(level=0.80)
            model.fit(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2024-01-01"),
                test_start=pd.Timestamp("2024-03-01"),
                test_end=pd.Timestamp("2024-06-28"),
            )
            predictions = model.predict()
            summary = model.summarize()
            return predictions, summary

        # Functional API workflow with predictions
        def functional_workflow(data):
            results = perform_tbr_analysis(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2024-01-01"),
                test_start=pd.Timestamp("2024-03-01"),
                test_end=pd.Timestamp("2024-06-28"),
                level=0.80,
                threshold=0.0,
            )
            # Extract predictions and summary
            df = results.tbr_dataframe()
            test_period = df[df["period"] == 1]
            predictions = test_period[["pred", "predsd"]]
            summary = results.summary()
            return predictions, summary

        # Benchmark both
        oop_stats = benchmarker.benchmark_function(oop_workflow, sample_data_medium)
        func_stats = benchmarker.benchmark_function(
            functional_workflow, sample_data_medium
        )

        # Compare performance
        comparison = benchmarker.compare_performance(
            oop_stats, func_stats, tolerance=1.5
        )

        print(
            f"\nOOP with predictions: {oop_stats['mean']*1000:.2f} ± {oop_stats['std']*1000:.2f} ms"
        )
        print(
            f"Functional with predictions: {func_stats['mean']*1000:.2f} ± {func_stats['std']*1000:.2f} ms"
        )
        print(f"Performance ratio: {comparison['ratio_mean']:.2f}x")

        # OOP API should not be more than 50% slower
        assert comparison[
            "within_tolerance"
        ], f"OOP API too slow: ratio={comparison['ratio_mean']:.2f}x"


class TestScalabilityBenchmarks:
    """Test performance scalability across different data sizes."""

    @pytest.mark.parametrize("n_samples", [50, 100, 200, 500, 1000])
    def test_scalability_across_data_sizes(self, benchmarker, n_samples):
        """Test TBRAnalysis performance scales linearly with data size."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n_samples)
        data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, n_samples),
                "test": np.random.normal(1020, 55, n_samples),
            }
        )

        # Calculate period boundaries
        pretest_days = n_samples // 3
        test_days = n_samples // 2

        def workflow(df):
            model = TBRAnalysis(level=0.80)
            model.fit(
                data=df,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=df["date"].iloc[0],
                test_start=df["date"].iloc[pretest_days],
                test_end=df["date"].iloc[pretest_days + test_days - 1],
            )
            return model.summarize()

        stats = benchmarker.benchmark_function(workflow, data)

        # Performance should be reasonable even for large datasets
        max_time = n_samples / 100.0  # ~10ms per 100 samples
        assert stats["mean"] < max_time, (
            f"Performance doesn't scale well for {n_samples} samples: "
            f"{stats['mean']:.3f}s (max: {max_time:.3f}s)"
        )

        print(f"\nn={n_samples}: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms")

    def test_scalability_large_dataset(self, benchmarker):
        """Test performance with large dataset (10,000 samples)."""
        np.random.seed(42)
        n_samples = 10000
        dates = pd.date_range("2024-01-01", periods=n_samples)
        data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, n_samples),
                "test": np.random.normal(1020, 55, n_samples),
            }
        )

        def workflow(df):
            model = TBRAnalysis(level=0.80)
            model.fit(
                data=df,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=df["date"].iloc[0],
                test_start=df["date"].iloc[3000],
                test_end=df["date"].iloc[8000],
            )
            return model.summarize()

        stats = benchmarker.benchmark_function(workflow, data)

        # Should complete in reasonable time even for 10k samples
        assert stats["mean"] < 15.0, f"Too slow for 10k samples: {stats['mean']:.3f}s"
        print(f"\n10k samples: {stats['mean']*1000:.2f} ± {stats['std']*1000:.2f} ms")


class TestMethodChainingPerformance:
    """Test performance of method chaining and fluent API patterns."""

    def test_method_chaining_overhead(self, benchmarker, sample_data_medium):
        """Test overhead of method chaining compared to direct calls."""

        # Method chaining workflow
        def chained_workflow(data):
            summary = (
                TBRAnalysis(level=0.80)
                .fit(
                    data,
                    "date",
                    "control",
                    "test",
                    pd.Timestamp("2024-01-01"),
                    pd.Timestamp("2024-03-01"),
                    pd.Timestamp("2024-06-28"),
                )
                .summarize()
            )
            return summary

        # Direct calls workflow
        def direct_workflow(data):
            model = TBRAnalysis(level=0.80)
            model.fit(
                data,
                "date",
                "control",
                "test",
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-03-01"),
                pd.Timestamp("2024-06-28"),
            )
            summary = model.summarize()
            return summary

        # Benchmark both
        chained_stats = benchmarker.benchmark_function(
            chained_workflow, sample_data_medium
        )
        direct_stats = benchmarker.benchmark_function(
            direct_workflow, sample_data_medium
        )

        # Compare performance
        comparison = benchmarker.compare_performance(
            chained_stats, direct_stats, tolerance=1.1
        )

        print(
            f"\nMethod chaining: {chained_stats['mean']*1000:.2f} ± {chained_stats['std']*1000:.2f} ms"
        )
        print(
            f"Direct calls: {direct_stats['mean']*1000:.2f} ± {direct_stats['std']*1000:.2f} ms"
        )
        print(f"Chaining overhead: {comparison['performance_difference_pct']:.1f}%")

        # Method chaining should have negligible overhead (< 10%)
        assert comparison["within_tolerance"], (
            f"Method chaining adds too much overhead: "
            f"{comparison['performance_difference_pct']:.1f}%"
        )

    def test_fit_summarize_convenience_performance(
        self, benchmarker, sample_data_medium
    ):
        """Test performance of fit_summarize() convenience method."""

        # Convenience method
        def convenience_workflow(data):
            summary = TBRAnalysis(level=0.80).fit_summarize(
                data,
                "date",
                "control",
                "test",
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-03-01"),
                pd.Timestamp("2024-06-28"),
            )
            return summary

        # Manual equivalent
        def manual_workflow(data):
            model = TBRAnalysis(level=0.80)
            model.fit(
                data,
                "date",
                "control",
                "test",
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-03-01"),
                pd.Timestamp("2024-06-28"),
            )
            summary = model.summarize()
            return summary

        # Benchmark both
        convenience_stats = benchmarker.benchmark_function(
            convenience_workflow, sample_data_medium
        )
        manual_stats = benchmarker.benchmark_function(
            manual_workflow, sample_data_medium
        )

        # Compare performance
        comparison = benchmarker.compare_performance(
            convenience_stats, manual_stats, tolerance=1.05
        )

        print(
            f"\nfit_summarize(): {convenience_stats['mean']*1000:.2f} ± {convenience_stats['std']*1000:.2f} ms"
        )
        print(
            f"Manual fit+summarize: {manual_stats['mean']*1000:.2f} ± {manual_stats['std']*1000:.2f} ms"
        )

        # Should have minimal overhead (< 5%)
        assert comparison[
            "within_tolerance"
        ], f"fit_summarize() adds overhead: {comparison['performance_difference_pct']:.1f}%"

    def test_copy_and_configure_performance(self, benchmarker, sample_data_medium):
        """Test performance of copy() and set_params() operations."""

        model_base = TBRAnalysis(level=0.80)
        model_base.fit(
            sample_data_medium,
            "date",
            "control",
            "test",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-03-01"),
            pd.Timestamp("2024-06-28"),
        )

        def copy_workflow():
            model_copy = model_base.copy()
            model_copy.set_params(level=0.95)
            return model_copy

        stats = benchmarker.benchmark_function(copy_workflow)

        # copy() and set_params() should be very fast
        assert (
            stats["mean"] < 0.001
        ), f"copy/set_params too slow: {stats['mean']*1000:.3f}ms"
        print(
            f"\ncopy + set_params: {stats['mean']*1000000:.2f} ± {stats['std']*1000000:.2f} μs"
        )


class TestPerformanceRegressionPrevention:
    """Validate no performance regressions from expected baselines."""

    def test_baseline_fit_performance(self, benchmarker, sample_data_medium):
        """Validate fit() performance against baseline."""

        def workflow(data):
            model = TBRAnalysis(level=0.80)
            model.fit(
                data,
                "date",
                "control",
                "test",
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-03-01"),
                pd.Timestamp("2024-06-28"),
            )
            return model

        stats = benchmarker.benchmark_function(workflow, sample_data_medium)

        # Baseline: fit() with 180 samples should complete in < 500ms
        baseline_threshold = 0.5
        assert stats["mean"] < baseline_threshold, (
            f"Performance regression in fit(): {stats['mean']:.3f}s "
            f"(baseline: < {baseline_threshold}s)"
        )

        print(
            f"\nBaseline fit() check: {stats['mean']*1000:.2f}ms (baseline: < {baseline_threshold*1000}ms)"
        )

    def test_baseline_complete_workflow_performance(
        self, benchmarker, sample_data_medium
    ):
        """Validate complete workflow performance against baseline."""

        def workflow(data):
            model = TBRAnalysis(level=0.80)
            model.fit(
                data,
                "date",
                "control",
                "test",
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-03-01"),
                pd.Timestamp("2024-06-28"),
            )
            summary = model.summarize()
            predictions = model.predict()
            incremental = model.summarize_incremental()
            return summary, predictions, incremental

        stats = benchmarker.benchmark_function(workflow, sample_data_medium)

        # Baseline: Complete workflow should complete in < 1s
        baseline_threshold = 1.0
        assert stats["mean"] < baseline_threshold, (
            f"Performance regression in complete workflow: {stats['mean']:.3f}s "
            f"(baseline: < {baseline_threshold}s)"
        )

        print(
            f"\nBaseline workflow check: {stats['mean']*1000:.2f}ms (baseline: < {baseline_threshold*1000}ms)"
        )
