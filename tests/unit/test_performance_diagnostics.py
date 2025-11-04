"""
Unit tests for performance diagnostics and computational efficiency metrics.

This module provides comprehensive tests for the performance diagnostics framework,
including profiling, efficiency analysis, monitoring, and TBR-specific performance
analysis capabilities.
"""

import time
import warnings
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from tbr.analysis.performance import TBRPerformanceAnalyzer, quick_performance_check
from tbr.utils.performance import (
    EfficiencyMetrics,
    EfficiencyReport,
    PerformanceMetrics,
    PerformanceMonitor,
    PerformanceProfiler,
    benchmark_tbr_functions,
    profile_tbr_workflow,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creation of PerformanceMetrics objects."""
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            duration=1.5,
            memory_peak=100.0,
            cpu_percent=75.0,
        )

        assert metrics.operation_name == "test_operation"
        assert metrics.duration == 1.5
        assert metrics.memory_peak == 100.0
        assert metrics.cpu_percent == 75.0
        assert metrics.function_calls == 0
        assert isinstance(metrics.metadata, dict)

    def test_performance_metrics_with_metadata(self):
        """Test PerformanceMetrics with custom metadata."""
        metadata = {"data_size": 1000, "algorithm": "OLS"}
        metrics = PerformanceMetrics(
            operation_name="regression", duration=0.5, metadata=metadata
        )

        assert metrics.metadata == metadata
        assert metrics.metadata["data_size"] == 1000


class TestEfficiencyReport:
    """Test EfficiencyReport dataclass."""

    def test_efficiency_report_creation(self):
        """Test creation of EfficiencyReport objects."""
        report = EfficiencyReport(
            operation_name="tbr_analysis",
            data_size=5000,
            computational_complexity="O(n)",
            efficiency_score=7.5,
            bottlenecks=["regression_fitting"],
            recommendations=["Use vectorized operations"],
        )

        assert report.operation_name == "tbr_analysis"
        assert report.data_size == 5000
        assert report.computational_complexity == "O(n)"
        assert report.efficiency_score == 7.5
        assert "regression_fitting" in report.bottlenecks
        assert "Use vectorized operations" in report.recommendations

    def test_efficiency_report_summary(self):
        """Test EfficiencyReport summary generation."""
        report = EfficiencyReport(
            operation_name="test_op",
            data_size=1000,
            computational_complexity="O(n log n)",
            efficiency_score=6.0,
            bottlenecks=["slow_function"],
            recommendations=["Optimize algorithm", "Use caching"],
        )

        summary = report.summary()

        assert "Performance Analysis: test_op" in summary
        assert "Data Size: 1,000 elements" in summary
        assert "Efficiency Score: 6.00/10.0" in summary
        assert "Computational Complexity: O(n log n)" in summary
        assert "Bottlenecks: slow_function" in summary
        assert "Optimize algorithm" in summary
        assert "Use caching" in summary


class TestPerformanceProfiler:
    """Test PerformanceProfiler class."""

    def test_profiler_initialization(self):
        """Test PerformanceProfiler initialization."""
        profiler = PerformanceProfiler()

        assert profiler.enable_memory_tracking is True
        assert profiler.enable_cpu_tracking is True
        assert isinstance(profiler.metrics, dict)
        assert len(profiler.metrics) == 0

    def test_profiler_initialization_with_options(self):
        """Test PerformanceProfiler initialization with custom options."""
        profiler = PerformanceProfiler(
            enable_memory_tracking=False, enable_cpu_tracking=False
        )

        assert profiler.enable_memory_tracking is False
        assert profiler.enable_cpu_tracking is False

    def test_profile_context_basic(self):
        """Test basic context manager profiling."""
        profiler = PerformanceProfiler()

        with profiler.profile_context("test_operation") as metrics:
            time.sleep(0.01)  # Small delay for measurable duration
            assert metrics.operation_name == "test_operation"

        # Check stored metrics
        stored_metrics = profiler.get_metrics("test_operation")
        assert stored_metrics is not None
        assert stored_metrics.duration > 0
        assert stored_metrics.operation_name == "test_operation"

    def test_profile_context_with_metadata(self):
        """Test context manager profiling with metadata."""
        profiler = PerformanceProfiler()
        metadata = {"test_param": "value", "data_size": 100}

        with profiler.profile_context("test_with_metadata", metadata=metadata):
            pass

        stored_metrics = profiler.get_metrics("test_with_metadata")
        assert stored_metrics.metadata == metadata

    def test_profile_function_decorator(self):
        """Test function decorator profiling."""
        profiler = PerformanceProfiler()

        @profiler.profile_function
        def test_function(x, y):
            return x + y

        result = test_function(2, 3)
        assert result == 5

        # Check that profiling occurred
        metrics = profiler.get_metrics()
        assert len(metrics) == 1

        # Get the metrics (function name will be module.function_name)
        metric_name = list(metrics.keys())[0]
        assert "test_function" in metric_name
        assert metrics[metric_name].function_calls == 1

    def test_benchmark_function(self):
        """Test function benchmarking."""
        profiler = PerformanceProfiler()

        def simple_function(n):
            return sum(range(n))

        stats = profiler.benchmark_function(
            simple_function, 100, n_runs=3, warmup_runs=1
        )

        assert "mean_time" in stats
        assert "std_time" in stats
        assert "min_time" in stats
        assert "max_time" in stats
        assert "median_time" in stats
        assert "n_runs" in stats
        assert "result" in stats
        assert stats["n_runs"] == 3
        assert stats["result"] == sum(range(100))
        assert stats["mean_time"] > 0

    def test_get_metrics_all(self):
        """Test getting all metrics."""
        profiler = PerformanceProfiler()

        with profiler.profile_context("op1"):
            pass
        with profiler.profile_context("op2"):
            pass

        all_metrics = profiler.get_metrics()
        assert len(all_metrics) == 2
        assert "op1" in all_metrics
        assert "op2" in all_metrics

    def test_get_metrics_specific(self):
        """Test getting specific operation metrics."""
        profiler = PerformanceProfiler()

        with profiler.profile_context("target_op"):
            pass
        with profiler.profile_context("other_op"):
            pass

        target_metrics = profiler.get_metrics("target_op")
        assert target_metrics.operation_name == "target_op"

        nonexistent_metrics = profiler.get_metrics("nonexistent")
        assert nonexistent_metrics is None

    def test_clear_metrics(self):
        """Test clearing stored metrics."""
        profiler = PerformanceProfiler()

        with profiler.profile_context("test_op"):
            pass

        assert len(profiler.get_metrics()) == 1

        profiler.clear_metrics()
        assert len(profiler.get_metrics()) == 0

    def test_print_summary(self, capsys):
        """Test printing performance summary."""
        profiler = PerformanceProfiler()

        with profiler.profile_context("test_operation", metadata={"key": "value"}):
            time.sleep(0.001)  # Small measurable delay

        profiler.print_summary()

        captured = capsys.readouterr()
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out
        assert "test_operation" in captured.out
        assert "Duration:" in captured.out


class TestEfficiencyMetrics:
    """Test EfficiencyMetrics class."""

    def test_efficiency_metrics_initialization(self):
        """Test EfficiencyMetrics initialization."""
        efficiency = EfficiencyMetrics()

        assert isinstance(efficiency.baseline_metrics, dict)
        assert isinstance(efficiency.profiler, PerformanceProfiler)

    def test_analyze_workflow_efficiency(self):
        """Test workflow efficiency analysis."""
        efficiency = EfficiencyMetrics()

        # Create mock operation metrics
        operation_metrics = {
            "op1": PerformanceMetrics("op1", duration=0.1, memory_peak=50.0),
            "op2": PerformanceMetrics("op2", duration=0.2, memory_peak=75.0),
        }

        report = efficiency.analyze_workflow_efficiency(
            data_size=1000,
            operation_metrics=operation_metrics,
            operation_name="test_workflow",
        )

        assert isinstance(report, EfficiencyReport)
        assert report.operation_name == "test_workflow"
        assert report.data_size == 1000
        assert report.efficiency_score >= 0
        assert report.efficiency_score <= 10
        assert isinstance(report.bottlenecks, list)
        assert isinstance(report.recommendations, list)

    def test_analyze_scaling_behavior(self):
        """Test scaling behavior analysis."""
        efficiency = EfficiencyMetrics()

        def test_function(data):
            return np.sum(data)

        data_sizes = [100, 500, 1000]
        scaling_analysis = efficiency.analyze_scaling_behavior(
            function=test_function, data_sizes=data_sizes
        )

        assert "scaling_results" in scaling_analysis
        assert "complexity_analysis" in scaling_analysis
        assert "best_fit_complexity" in scaling_analysis
        assert "scaling_efficiency" in scaling_analysis

        assert len(scaling_analysis["scaling_results"]) == len(data_sizes)

        for i, result in enumerate(scaling_analysis["scaling_results"]):
            assert result["data_size"] == data_sizes[i]
            assert "mean_time" in result

    def test_set_baseline(self):
        """Test setting performance baseline."""
        efficiency = EfficiencyMetrics()

        metrics = {
            "op1": PerformanceMetrics("op1", duration=0.1, memory_peak=50.0),
            "op2": PerformanceMetrics("op2", duration=0.2, memory_peak=75.0),
        }

        efficiency.set_baseline("test_baseline", metrics)

        assert "test_baseline" in efficiency.baseline_metrics
        baseline = efficiency.baseline_metrics["test_baseline"]
        assert abs(baseline["total_duration"] - 0.3) < 1e-10  # 0.1 + 0.2
        assert baseline["peak_memory"] == 75.0  # max of 50.0, 75.0
        assert baseline["operation_count"] == 2

    def test_time_complexity_estimation_linear(self):
        """Test time complexity estimation for O(n) linear operations."""
        efficiency = EfficiencyMetrics()

        # Create metrics with linear time complexity
        # duration = 0.05 seconds for 1000 elements → 50 microseconds per element
        # This should be classified as O(n) - Linear
        operation_metrics = {
            "linear_op": PerformanceMetrics(
                operation_name="linear_op",
                duration=0.05,
                memory_peak=100.0,
            )
        }

        report = efficiency.analyze_workflow_efficiency(
            data_size=1000, operation_metrics=operation_metrics
        )

        assert isinstance(report, EfficiencyReport)
        assert "Linear" in report.computational_complexity

    def test_memory_efficiency_scoring_low_usage(self):
        """Test memory efficiency scoring for low memory per element."""
        efficiency = EfficiencyMetrics()

        # Create metrics with very low memory per element (< 1KB per element)
        # 0.5 MB for 1000 elements = 0.0005 MB per element < 0.001 MB (1KB)
        # This should contribute memory_score of 3.0 to the efficiency_score
        operation_metrics = {
            "memory_efficient_op": PerformanceMetrics(
                operation_name="memory_efficient_op",
                duration=0.1,
                memory_peak=0.5,
            )
        }

        report = efficiency.analyze_workflow_efficiency(
            data_size=1000, operation_metrics=operation_metrics
        )

        # Verify the report is created and memory_per_element < 1KB path is exercised
        assert isinstance(report, EfficiencyReport)
        assert report.efficiency_score >= 0
        # With low memory usage, we should get high memory score contribution
        assert report.efficiency_score >= 3.0  # At least memory_score of 3.0

    def test_memory_efficiency_scoring_medium_usage(self):
        """Test memory efficiency scoring for medium memory per element."""
        efficiency = EfficiencyMetrics()

        # Create metrics with medium memory per element (1-10 KB per element)
        # 5 MB for 1000 elements = 0.005 MB per element (~5KB per element)
        # This should contribute memory_score of 2.0 to the efficiency_score
        operation_metrics = {
            "medium_memory_op": PerformanceMetrics(
                operation_name="medium_memory_op",
                duration=0.1,
                memory_peak=5.0,
            )
        }

        report = efficiency.analyze_workflow_efficiency(
            data_size=1000, operation_metrics=operation_metrics
        )

        # Verify the report is created and memory_per_element 1-10KB path is exercised
        assert isinstance(report, EfficiencyReport)
        assert report.efficiency_score >= 0
        # With medium memory usage (1-10KB per element), efficiency should be lower than low usage
        # but still get at least memory_score of 2.0
        assert report.efficiency_score >= 2.0


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_monitor_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor(sampling_interval=0.05)

        assert monitor.sampling_interval == 0.05
        assert monitor.monitoring is False
        assert isinstance(monitor.samples, list)
        assert len(monitor.samples) == 0

    @patch("psutil.Process")
    def test_start_stop_monitoring(self, mock_process_class):
        """Test starting and stopping monitoring."""
        # Mock psutil.Process
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 50.0
        mock_process.memory_info.return_value = Mock(rss=1024 * 1024 * 100)  # 100MB
        mock_process.memory_percent.return_value = 25.0
        mock_process_class.return_value = mock_process

        # Mock system-wide functions
        with patch("psutil.cpu_percent", return_value=30.0), patch(
            "psutil.virtual_memory", return_value=Mock(percent=60.0)
        ):
            monitor = PerformanceMonitor()

            assert monitor.monitoring is False

            monitor.start_monitoring()
            assert monitor.monitoring is True
            assert len(monitor.samples) >= 1

            monitor.stop_monitoring()
            assert monitor.monitoring is False

    @patch("psutil.Process")
    def test_get_monitoring_report(self, mock_process_class):
        """Test generating monitoring report."""
        # Mock psutil.Process
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 50.0
        mock_process.memory_info.return_value = Mock(rss=1024 * 1024 * 100)  # 100MB
        mock_process.memory_percent.return_value = 25.0
        mock_process_class.return_value = mock_process

        with patch("psutil.cpu_percent", return_value=30.0), patch(
            "psutil.virtual_memory", return_value=Mock(percent=60.0)
        ):
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            time.sleep(0.01)  # Small delay
            monitor.stop_monitoring()

            report = monitor.get_monitoring_report()

            assert "duration" in report
            assert "sample_count" in report
            assert "cpu_stats" in report
            assert "memory_stats" in report
            assert "system_stats" in report
            assert "alerts" in report

            assert report["sample_count"] >= 2  # Start and stop samples
            assert isinstance(report["alerts"], list)

    def test_get_monitoring_report_no_data(self):
        """Test monitoring report with no data."""
        monitor = PerformanceMonitor()

        report = monitor.get_monitoring_report()

        assert "error" in report
        assert report["error"] == "No monitoring data available"


class TestTBRPerformanceAnalyzer:
    """Test TBRPerformanceAnalyzer class."""

    def setup_method(self):
        """Set up test data for each test."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        self.test_data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, 100),
                "test": np.random.normal(1020, 55, 100),
            }
        )

    def test_analyzer_initialization(self):
        """Test TBRPerformanceAnalyzer initialization."""
        analyzer = TBRPerformanceAnalyzer()

        assert isinstance(analyzer.profiler, PerformanceProfiler)
        assert isinstance(analyzer.efficiency_metrics, EfficiencyMetrics)
        assert isinstance(analyzer.monitor, PerformanceMonitor)
        assert isinstance(analyzer.baseline_metrics, dict)

    def test_analyze_tbr_performance(self):
        """Test TBR performance analysis."""
        analyzer = TBRPerformanceAnalyzer()

        # Use smaller dataset and disable monitoring for faster testing
        small_data = self.test_data.iloc[:30].copy()

        performance_report = analyzer.analyze_tbr_performance(
            data=small_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-15"),
            test_end=pd.Timestamp("2023-01-25"),
            enable_monitoring=False,  # Disable for testing
        )

        # Check report structure
        assert "workflow_metrics" in performance_report
        assert "operation_metrics" in performance_report
        assert "efficiency_report" in performance_report
        assert "data_characteristics" in performance_report
        assert "tbr_results" in performance_report
        assert "tbr_summaries" in performance_report

        # Check workflow metrics
        workflow_metrics = performance_report["workflow_metrics"]
        assert isinstance(workflow_metrics, PerformanceMetrics)
        assert workflow_metrics.operation_name == "tbr_complete_workflow"
        assert workflow_metrics.duration > 0

        # Check efficiency report
        efficiency_report = performance_report["efficiency_report"]
        assert isinstance(efficiency_report, EfficiencyReport)
        assert efficiency_report.data_size == len(small_data)

        # Check data characteristics
        data_chars = performance_report["data_characteristics"]
        assert data_chars["data_size"] == len(small_data)
        assert data_chars["data_memory_mb"] > 0

    def test_analyze_data_size_scaling(self):
        """Test data size scaling analysis."""
        analyzer = TBRPerformanceAnalyzer()

        # Use very small dataset for fast testing
        small_data = self.test_data.iloc[:20].copy()

        scaling_analysis = analyzer.analyze_data_size_scaling(
            base_data=small_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-10"),
            test_end=pd.Timestamp("2023-01-15"),
            size_multipliers=[0.5, 1.0],  # Only test small multipliers
            level=0.80,
            threshold=0.0,
        )

        assert "scaling_results" in scaling_analysis
        assert "scaling_analysis" in scaling_analysis
        assert "recommendations" in scaling_analysis

        scaling_results = scaling_analysis["scaling_results"]
        assert len(scaling_results) == 2  # Two multipliers

        for result in scaling_results:
            assert "size_multiplier" in result
            assert "data_size" in result
            if result["success"]:
                assert "total_duration" in result
                assert "efficiency_score" in result

    def test_analyze_data_size_scaling_default_multipliers(self):
        """Test data size scaling with default size_multipliers parameter."""
        analyzer = TBRPerformanceAnalyzer()

        # Use very small dataset for fast testing
        small_data = self.test_data.iloc[:20].copy()

        # Call without size_multipliers to test default [0.5, 1.0, 2.0, 5.0]
        scaling_analysis = analyzer.analyze_data_size_scaling(
            base_data=small_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-10"),
            test_end=pd.Timestamp("2023-01-15"),
            # size_multipliers not provided - should use default
            level=0.80,
            threshold=0.0,
        )

        assert "scaling_results" in scaling_analysis
        assert "scaling_analysis" in scaling_analysis
        assert "recommendations" in scaling_analysis

        scaling_results = scaling_analysis["scaling_results"]
        # Default size_multipliers are [0.5, 1.0, 2.0, 5.0] - 4 values
        assert len(scaling_results) == 4

        # Verify all multipliers are present
        multipliers_found = [r["size_multiplier"] for r in scaling_results]
        assert 0.5 in multipliers_found
        assert 1.0 in multipliers_found
        assert 2.0 in multipliers_found
        assert 5.0 in multipliers_found

    def test_get_optimization_recommendations(self):
        """Test optimization recommendations generation."""
        analyzer = TBRPerformanceAnalyzer()

        # Create mock performance report
        mock_workflow_metrics = PerformanceMetrics(
            operation_name="test_workflow",
            duration=5.0,  # 5 seconds
            memory_peak=500.0,  # 500 MB
        )

        mock_efficiency_report = EfficiencyReport(
            operation_name="test_workflow",
            data_size=10000,
            computational_complexity="O(n)",
            efficiency_score=4.0,  # Low efficiency
            bottlenecks=["regression_fitting (60% of total time)"],
            recommendations=["Use vectorized operations"],
        )

        mock_performance_report = {
            "workflow_metrics": mock_workflow_metrics,
            "efficiency_report": mock_efficiency_report,
            "data_characteristics": {"data_size": 10000, "data_memory_mb": 200.0},
        }

        recommendations = analyzer.get_optimization_recommendations(
            mock_performance_report
        )

        assert "priority_actions" in recommendations
        assert "data_optimization" in recommendations
        assert "computational_optimization" in recommendations
        assert "memory_optimization" in recommendations
        assert "general_recommendations" in recommendations

        # Check that low efficiency triggers priority action
        assert len(recommendations["priority_actions"]) > 0

        # Check that bottleneck triggers computational optimization
        assert len(recommendations["computational_optimization"]) > 0

    def test_set_performance_baseline(self):
        """Test setting performance baseline."""
        analyzer = TBRPerformanceAnalyzer()

        mock_performance_report = {
            "workflow_metrics": PerformanceMetrics(
                "test", duration=2.0, memory_peak=100.0
            ),
            "efficiency_report": EfficiencyReport(
                "test",
                data_size=1000,
                computational_complexity="O(n)",
                efficiency_score=7.0,
                bottlenecks=[],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 1000},
            "operation_metrics": {"op1": PerformanceMetrics("op1", duration=1.0)},
        }

        analyzer.set_performance_baseline("test_baseline", mock_performance_report)

        assert "test_baseline" in analyzer.baseline_metrics
        baseline = analyzer.baseline_metrics["test_baseline"]
        assert baseline["workflow_duration"] == 2.0
        assert baseline["efficiency_score"] == 7.0
        assert baseline["data_size"] == 1000
        assert baseline["memory_peak"] == 100.0

    def test_compare_to_baseline(self):
        """Test comparison to baseline."""
        analyzer = TBRPerformanceAnalyzer()

        # Set baseline
        baseline_report = {
            "workflow_metrics": PerformanceMetrics(
                "baseline", duration=2.0, memory_peak=100.0
            ),
            "efficiency_report": EfficiencyReport(
                "baseline",
                data_size=1000,
                computational_complexity="O(n)",
                efficiency_score=7.0,
                bottlenecks=[],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 1000},
            "operation_metrics": {},
        }

        analyzer.set_performance_baseline("test_baseline", baseline_report)

        # Create current report
        current_report = {
            "workflow_metrics": PerformanceMetrics(
                "current", duration=3.0, memory_peak=150.0
            ),
            "efficiency_report": EfficiencyReport(
                "current",
                data_size=2000,
                computational_complexity="O(n)",
                efficiency_score=6.0,
                bottlenecks=[],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 2000},
        }

        comparison = analyzer.compare_to_baseline(current_report, "test_baseline")

        assert "baseline_name" in comparison
        assert "size_ratio" in comparison
        assert "duration_ratio" in comparison
        assert "normalized_duration_ratio" in comparison
        assert "efficiency_ratio" in comparison
        assert "performance_regression" in comparison
        assert "performance_improvement" in comparison

        assert comparison["baseline_name"] == "test_baseline"
        assert comparison["size_ratio"] == 2.0  # 2000/1000
        assert comparison["duration_ratio"] == 1.5  # 3.0/2.0
        assert comparison["efficiency_ratio"] == 6.0 / 7.0  # 6.0/7.0

    def test_analyze_data_size_scaling_with_exceptions(self):
        """Test scaling analysis when TBR analysis raises exceptions."""
        analyzer = TBRPerformanceAnalyzer()

        small_data = self.test_data.iloc[:30].copy()

        # Mock perform_tbr_analysis to raise exceptions
        with patch(
            "tbr.analysis.performance.perform_tbr_analysis",
            side_effect=Exception("Analysis failed"),
        ):
            scaling_analysis = analyzer.analyze_data_size_scaling(
                base_data=small_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-01-15"),
                test_end=pd.Timestamp("2023-01-25"),
                size_multipliers=[0.5, 1.0],
            )

            # All results should capture the error
            for result in scaling_analysis["scaling_results"]:
                assert "error" in result
                assert result["success"] is False

            # Should have insufficient successful runs error
            assert "error" in scaling_analysis["scaling_analysis"]

    def test_compare_tbr_configurations_with_failures(self):
        """Test configuration comparison when some configurations fail."""
        analyzer = TBRPerformanceAnalyzer()

        small_data = self.test_data.iloc[:30].copy()

        base_config = {
            "time_col": "date",
            "control_col": "control",
            "test_col": "test",
            "pretest_start": pd.Timestamp("2023-01-01"),
            "test_start": pd.Timestamp("2023-01-15"),
            "test_end": pd.Timestamp("2023-01-25"),
        }

        # Mock to fail on non-baseline configs
        original = analyzer.analyze_tbr_performance
        call_count = [0]

        def mock_analyze(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return original(*args, **kwargs)
            raise Exception("Config failed")

        with patch.object(
            analyzer, "analyze_tbr_performance", side_effect=mock_analyze
        ):
            results = analyzer.compare_tbr_configurations(
                data=small_data, configurations=[base_config], base_config=base_config
            )

            # Baseline should succeed, other should fail
            assert results["comparison_results"][0]["config_name"] == "baseline"
            assert results["comparison_results"][1]["success"] is False

    def test_get_optimization_recommendations_low_efficiency(self):
        """Test recommendations with very low efficiency score."""
        analyzer = TBRPerformanceAnalyzer()

        performance_report = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=5.0, memory_peak=200
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=1000,
                efficiency_score=2.5,  # Very low
                computational_complexity="O(n)",
                bottlenecks=[],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 1000, "data_memory_mb": 50},
        }

        recs = analyzer.get_optimization_recommendations(performance_report)

        # Should have critical priority action
        assert len(recs["priority_actions"]) > 0
        assert any("Critical" in action for action in recs["priority_actions"])

    def test_get_optimization_recommendations_large_dataset(self):
        """Test recommendations with large dataset."""
        analyzer = TBRPerformanceAnalyzer()

        performance_report = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=5.0, memory_peak=200
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=150000,  # Large
                efficiency_score=7.0,
                computational_complexity="O(n)",
                bottlenecks=[],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 150000, "data_memory_mb": 50},
        }

        recs = analyzer.get_optimization_recommendations(performance_report)

        # Should recommend sampling or chunking
        assert any(
            "sampling" in rec.lower() or "chunked" in rec.lower()
            for rec in recs["data_optimization"]
        )

    def test_get_optimization_recommendations_high_memory(self):
        """Test recommendations with high memory usage."""
        analyzer = TBRPerformanceAnalyzer()

        performance_report = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=5.0, memory_peak=2500  # High
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=1000,
                efficiency_score=7.0,
                computational_complexity="O(n)",
                bottlenecks=[],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 1000, "data_memory_mb": 1500},  # High
        }

        recs = analyzer.get_optimization_recommendations(performance_report)

        # Should recommend memory optimization
        assert len(recs["memory_optimization"]) > 0

    def test_get_optimization_recommendations_long_duration(self):
        """Test recommendations with long execution time."""
        analyzer = TBRPerformanceAnalyzer()

        performance_report = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=75.0, memory_peak=200  # Long
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=1000,
                efficiency_score=7.0,
                computational_complexity="O(n)",
                bottlenecks=[],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 1000, "data_memory_mb": 50},
        }

        recs = analyzer.get_optimization_recommendations(performance_report)

        # Should recommend computational optimization
        assert any(
            "significant time" in rec.lower()
            for rec in recs["computational_optimization"]
        )

    def test_get_optimization_recommendations_bottlenecks(self):
        """Test recommendations with specific bottlenecks."""
        analyzer = TBRPerformanceAnalyzer()

        # Test regression bottleneck
        report1 = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=5.0, memory_peak=200
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=1000,
                efficiency_score=7.0,
                computational_complexity="O(n)",
                bottlenecks=["regression_fitting"],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 1000, "data_memory_mb": 50},
        }

        recs1 = analyzer.get_optimization_recommendations(report1)
        assert any(
            "regression" in rec.lower() or "BLAS" in rec
            for rec in recs1["computational_optimization"]
        )

        # Test validation bottleneck
        report2 = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=5.0, memory_peak=200
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=1000,
                efficiency_score=7.0,
                computational_complexity="O(n)",
                bottlenecks=["data_validation"],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 1000, "data_memory_mb": 50},
        }

        recs2 = analyzer.get_optimization_recommendations(report2)
        assert any("validation" in rec.lower() for rec in recs2["data_optimization"])

    def test_compare_to_baseline_not_found(self):
        """Test comparing to non-existent baseline."""
        analyzer = TBRPerformanceAnalyzer()

        performance_report = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=5.0, memory_peak=200
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=1000,
                efficiency_score=7.0,
                computational_complexity="O(n)",
                bottlenecks=[],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 1000, "data_memory_mb": 50},
        }

        comparison = analyzer.compare_to_baseline(performance_report, "nonexistent")

        assert "error" in comparison
        assert "not found" in comparison["error"].lower()

    def test_compare_to_baseline_zero_memory(self):
        """Test baseline comparison with zero memory peak."""
        analyzer = TBRPerformanceAnalyzer()

        # Baseline with zero memory
        baseline_report = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=5.0, memory_peak=0
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=1000,
                efficiency_score=7.0,
                computational_complexity="O(n)",
                bottlenecks=[],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 1000, "data_memory_mb": 50},
            "operation_metrics": {},
        }

        analyzer.set_performance_baseline("zero_mem", baseline_report)

        # Current with non-zero memory
        current_report = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=6.0, memory_peak=200
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=1000,
                efficiency_score=6.5,
                computational_complexity="O(n)",
                bottlenecks=[],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 1000, "data_memory_mb": 50},
        }

        comparison = analyzer.compare_to_baseline(current_report, "zero_mem")

        # Should use 1.0 as memory ratio when baseline is zero
        assert comparison["memory_ratio"] == 1.0

    def test_print_performance_summary_with_monitoring(self):
        """Test printing summary with monitoring report."""
        import io
        import sys

        analyzer = TBRPerformanceAnalyzer()

        performance_report = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=5.0, memory_peak=200
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=1000,
                efficiency_score=7.0,
                computational_complexity="O(n)",
                bottlenecks=["test"],
                recommendations=["test"],
            ),
            "data_characteristics": {"data_size": 1000, "data_memory_mb": 50},
            "monitoring_report": {
                "duration": 5.0,
                "sample_count": 10,
                "cpu_stats": {"mean": 45.0, "max": 80.0, "min": 20.0, "std": 15.0},
                "memory_stats": {
                    "mean_mb": 150.0,
                    "max_mb": 200.0,
                    "min_mb": 100.0,
                    "peak_percent": 75.0,
                },
                "system_stats": {"cpu_mean": 40.0, "memory_mean": 60.0},
                "alerts": ["High CPU"],
            },
        }

        captured = io.StringIO()
        sys.stdout = captured

        try:
            analyzer.print_performance_summary(performance_report)
            output = captured.getvalue()

            assert "Resource Utilization:" in output
            assert "Average CPU:" in output
            assert "Alerts:" in output
        finally:
            sys.stdout = sys.__stdout__

    def test_helper_calculate_period_length_inclusive(self):
        """Test period length calculation with inclusive parameter."""
        analyzer = TBRPerformanceAnalyzer()

        length = analyzer._calculate_period_length(
            data=self.test_data,
            time_col="date",
            start=pd.Timestamp("2023-01-01"),
            end=pd.Timestamp("2023-01-10"),
            inclusive=True,
        )

        assert length == 10  # Should include both start and end

    def test_helper_upsample_data_no_change(self):
        """Test upsampling when target size is smaller than data."""
        analyzer = TBRPerformanceAnalyzer()

        result = analyzer._upsample_data(
            data=self.test_data,
            target_size=50,  # Less than 100
            control_col="control",
            test_col="test",
        )

        assert len(result) == len(self.test_data)

    def test_helper_upsample_data_with_noise(self):
        """Test upsampling with noise addition."""
        analyzer = TBRPerformanceAnalyzer()

        result = analyzer._upsample_data(
            data=self.test_data,
            target_size=200,
            control_col="control",
            test_col="test",
            noise_factor=0.05,
        )

        assert len(result) == 200

    def test_helper_analyze_scaling_patterns_errors(self):
        """Test scaling pattern analysis error handling."""
        analyzer = TBRPerformanceAnalyzer()

        # Test with polyfit failure
        with patch("numpy.polyfit", side_effect=Exception("Fit error")):
            result = analyzer._analyze_scaling_patterns(
                [
                    {"data_size": 100, "total_duration": 5.0, "success": True},
                    {"data_size": 200, "total_duration": 10.0, "success": True},
                ]
            )
            assert "error" in result

        # Test with insufficient data
        result = analyzer._analyze_scaling_patterns(
            [
                {"data_size": 100, "total_duration": 5.0, "success": True},
            ]
        )
        assert "error" in result

    def test_helper_generate_scaling_recommendations(self):
        """Test scaling recommendations generation."""
        analyzer = TBRPerformanceAnalyzer()

        # Insufficient data
        recs1 = analyzer._generate_scaling_recommendations(
            [
                {"data_size": 100, "total_duration": 5.0, "success": True},
                {"data_size": 200, "error": "Failed", "success": False},
            ]
        )
        assert any("Insufficient" in rec for rec in recs1)

        # Performance degradation
        recs2 = analyzer._generate_scaling_recommendations(
            [
                {
                    "data_size": 100,
                    "total_duration": 1.0,
                    "efficiency_score": 8.0,
                    "success": True,
                },
                {
                    "data_size": 200,
                    "total_duration": 5.0,
                    "efficiency_score": 7.0,
                    "success": True,
                },
            ]
        )
        assert any("chunked" in rec.lower() for rec in recs2)

        # High memory
        recs3 = analyzer._generate_scaling_recommendations(
            [
                {
                    "data_size": 100,
                    "total_duration": 1.0,
                    "memory_peak_mb": 500,
                    "efficiency_score": 8.0,
                    "success": True,
                },
                {
                    "data_size": 200,
                    "total_duration": 2.0,
                    "memory_peak_mb": 1500,
                    "efficiency_score": 7.5,
                    "success": True,
                },
            ]
        )
        assert any("memory" in rec.lower() for rec in recs3)

    def test_helper_generate_configuration_recommendations(self):
        """Test configuration recommendations generation."""
        analyzer = TBRPerformanceAnalyzer()

        # Insufficient data
        recs1 = analyzer._generate_configuration_recommendations(
            [
                {"config_name": "baseline", "duration_ratio": 1.0},
                {"config_name": "config_1", "error": "Failed"},
            ]
        )
        assert any("Insufficient" in rec for rec in recs1)

        # Best config
        recs2 = analyzer._generate_configuration_recommendations(
            [
                {"config_name": "baseline", "duration_ratio": 1.0},
                {"config_name": "fast", "duration_ratio": 0.7},
            ]
        )
        assert any("fast" in rec and "improvement" in rec.lower() for rec in recs2)

        # Worst config
        recs3 = analyzer._generate_configuration_recommendations(
            [
                {"config_name": "baseline", "duration_ratio": 1.0},
                {"config_name": "slow", "duration_ratio": 2.0},
            ]
        )
        assert any("slow" in rec and "Avoid" in rec for rec in recs3)

    def test_print_summary_with_bottlenecks_and_recommendations(self):
        """Test printing summary with bottlenecks and full recommendations."""
        import io
        import sys

        analyzer = TBRPerformanceAnalyzer()

        performance_report = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=75.0, memory_peak=2500
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=150000,
                efficiency_score=2.5,
                computational_complexity="O(n²)",
                bottlenecks=["regression_fitting", "data_validation"],
                recommendations=["General rec"],
            ),
            "data_characteristics": {"data_size": 150000, "data_memory_mb": 1500},
        }

        captured = io.StringIO()
        sys.stdout = captured

        try:
            analyzer.print_performance_summary(
                performance_report, include_recommendations=True
            )
            output = captured.getvalue()

            # Check bottlenecks are printed
            assert "Bottlenecks:" in output
            # Check recommendations sections are printed
            assert (
                "Priority Actions:" in output
                or "Computational Optimizations:" in output
                or "Memory Optimizations:" in output
            )
        finally:
            sys.stdout = sys.__stdout__

    def test_helper_upsample_with_zero_noise(self):
        """Test upsampling with noise_factor=0."""
        analyzer = TBRPerformanceAnalyzer()

        # Test with noise_factor=0 (should skip noise addition)
        result = analyzer._upsample_data(
            data=self.test_data,
            target_size=150,
            control_col="control",
            test_col="test",
            noise_factor=0,  # No noise
        )

        assert len(result) == 150

    def test_helper_analyze_scaling_patterns_slope_ranges(self):
        """Test scaling pattern analysis with different slope ranges."""
        analyzer = TBRPerformanceAnalyzer()

        # Test with very good scaling (slope < 0.5)
        result1 = analyzer._analyze_scaling_patterns(
            [
                {"data_size": 100, "total_duration": 1.0, "success": True},
                {"data_size": 200, "total_duration": 1.3, "success": True},
                {"data_size": 400, "total_duration": 1.6, "success": True},
            ]
        )
        if "error" not in result1:
            assert "complexity_estimate" in result1

        # Test with moderate scaling (slope 1.2-2.0)
        result2 = analyzer._analyze_scaling_patterns(
            [
                {"data_size": 100, "total_duration": 1.0, "success": True},
                {"data_size": 200, "total_duration": 2.5, "success": True},
                {"data_size": 400, "total_duration": 6.0, "success": True},
            ]
        )
        if "error" not in result2:
            assert "complexity_estimate" in result2

        # Test with poor scaling (slope > 2.0)
        result3 = analyzer._analyze_scaling_patterns(
            [
                {"data_size": 100, "total_duration": 1.0, "success": True},
                {"data_size": 200, "total_duration": 4.5, "success": True},
                {"data_size": 400, "total_duration": 18.0, "success": True},
            ]
        )
        if "error" not in result3:
            assert "complexity_estimate" in result3

    def test_helper_generate_scaling_recommendations_efficiency_decrease(self):
        """Test scaling recommendations with efficiency decrease."""
        analyzer = TBRPerformanceAnalyzer()

        # Test with decreasing efficiency (80% drop triggers line 789)
        recs = analyzer._generate_scaling_recommendations(
            [
                {
                    "data_size": 100,
                    "total_duration": 1.0,
                    "efficiency_score": 10.0,
                    "success": True,
                },
                {
                    "data_size": 200,
                    "total_duration": 2.0,
                    "efficiency_score": 7.5,
                    "success": True,
                },
                {
                    "data_size": 400,
                    "total_duration": 4.0,
                    "efficiency_score": 3.0,
                    "success": True,
                },  # 30% of original
            ]
        )

        # Should recommend reviewing algorithm complexity
        assert any(
            "algorithm" in rec.lower() or "complexity" in rec.lower() for rec in recs
        )

    def test_optimize_tbr_data_size_integration(self):
        """Test optimize_tbr_data_size with actual execution."""
        from tbr.analysis.performance import optimize_tbr_data_size

        small_data = self.test_data.iloc[:30].copy()

        # This should execute successfully
        result = optimize_tbr_data_size(
            data=small_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-15"),
            test_end=pd.Timestamp("2023-01-25"),
            target_duration=5.0,
            size_multipliers=[0.5, 1.0],  # Small multipliers for fast execution
        )

        # Should have results
        if "error" not in result:
            assert "recommended_size" in result

    def test_print_summary_without_recommendations(self):
        """Test printing summary with include_recommendations=False."""
        import io
        import sys

        analyzer = TBRPerformanceAnalyzer()

        performance_report = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=5.0, memory_peak=200
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=1000,
                efficiency_score=7.0,
                computational_complexity="O(n)",
                bottlenecks=[],  # Empty bottlenecks
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 1000, "data_memory_mb": 50},
        }

        captured = io.StringIO()
        sys.stdout = captured

        try:
            # Test with include_recommendations=False
            analyzer.print_performance_summary(
                performance_report, include_recommendations=False
            )
            output = captured.getvalue()

            # Recommendations should NOT be printed
            assert "Priority Actions:" not in output
            assert "Computational Optimizations:" not in output
        finally:
            sys.stdout = sys.__stdout__

    def test_print_summary_with_monitoring_no_alerts(self):
        """Test printing summary with monitoring but no alerts."""
        import io
        import sys

        analyzer = TBRPerformanceAnalyzer()

        performance_report = {
            "workflow_metrics": PerformanceMetrics(
                operation_name="test", duration=5.0, memory_peak=200
            ),
            "efficiency_report": EfficiencyReport(
                operation_name="test",
                data_size=1000,
                efficiency_score=7.0,
                computational_complexity="O(n)",
                bottlenecks=[],
                recommendations=[],
            ),
            "data_characteristics": {"data_size": 1000, "data_memory_mb": 50},
            "monitoring_report": {
                "duration": 5.0,
                "sample_count": 10,
                "cpu_stats": {"mean": 45.0, "max": 80.0, "min": 20.0, "std": 15.0},
                "memory_stats": {
                    "mean_mb": 150.0,
                    "max_mb": 200.0,
                    "min_mb": 100.0,
                    "peak_percent": 75.0,
                },
                "system_stats": {"cpu_mean": 40.0, "memory_mean": 60.0},
                "alerts": [],  # Empty alerts list
            },
        }

        captured = io.StringIO()
        sys.stdout = captured

        try:
            analyzer.print_performance_summary(performance_report)
            output = captured.getvalue()

            # Should print monitoring info but no alerts
            assert "Resource Utilization:" in output
            assert "Alerts:" not in output
        finally:
            sys.stdout = sys.__stdout__

    def test_optimize_tbr_data_size_all_failures(self):
        """Test optimize_tbr_data_size when all scaling tests fail (line 949)."""
        from tbr.analysis.performance import optimize_tbr_data_size

        small_data = self.test_data.iloc[:30].copy()

        # Mock analyze_data_size_scaling to return all failures
        with patch(
            "tbr.analysis.performance.TBRPerformanceAnalyzer.analyze_data_size_scaling"
        ) as mock_scaling:
            mock_scaling.return_value = {
                "scaling_results": [
                    {"success": False, "error": "Failed"},
                    {"success": False, "error": "Failed"},
                ],
                "scaling_analysis": {"error": "All failed"},
                "recommendations": [],
            }

            result = optimize_tbr_data_size(
                data=small_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-01-15"),
                test_end=pd.Timestamp("2023-01-25"),
                target_duration=5.0,
            )

            # Should return error (line 949)
            assert "error" in result
            assert "No successful scaling tests" in result["error"]

    def test_data_size_scaling_with_upsampling(self):
        """Test data size scaling analysis with upsampling for larger datasets."""
        analyzer = TBRPerformanceAnalyzer()

        # Use small dataset and test upsampling (multipliers > 1.0)
        small_data = self.test_data.iloc[:20].copy()

        scaling_analysis = analyzer.analyze_data_size_scaling(
            base_data=small_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-10"),
            test_end=pd.Timestamp("2023-01-15"),
            size_multipliers=[1.5, 2.0],  # Test upsampling path
            level=0.80,
            threshold=0.0,
        )

        assert "scaling_results" in scaling_analysis
        scaling_results = scaling_analysis["scaling_results"]
        assert len(scaling_results) == 2

        # Verify upsampled data sizes are larger than base
        for result in scaling_results:
            if result["success"]:
                assert result["data_size"] > len(small_data)

    def test_compare_tbr_configurations_multiple(self):
        """Test performance comparison across multiple TBR configurations."""
        analyzer = TBRPerformanceAnalyzer()

        # Use small dataset for fast testing
        small_data = self.test_data.iloc[:30].copy()

        # Define base configuration
        base_config = {
            "time_col": "date",
            "control_col": "control",
            "test_col": "test",
            "pretest_start": pd.Timestamp("2023-01-01"),
            "test_start": pd.Timestamp("2023-01-20"),
            "test_end": pd.Timestamp("2023-01-25"),
            "level": 0.90,
            "threshold": 0.0,
        }

        # Define alternative configurations with different confidence levels
        configurations = [
            {
                "time_col": "date",
                "control_col": "control",
                "test_col": "test",
                "pretest_start": pd.Timestamp("2023-01-01"),
                "test_start": pd.Timestamp("2023-01-20"),
                "test_end": pd.Timestamp("2023-01-25"),
                "level": 0.80,
                "threshold": 0.0,
            },
            {
                "time_col": "date",
                "control_col": "control",
                "test_col": "test",
                "pretest_start": pd.Timestamp("2023-01-01"),
                "test_start": pd.Timestamp("2023-01-20"),
                "test_end": pd.Timestamp("2023-01-25"),
                "level": 0.95,
                "threshold": 0.0,
            },
        ]

        comparison_results = analyzer.compare_tbr_configurations(
            data=small_data, configurations=configurations, base_config=base_config
        )

        assert "comparison_results" in comparison_results
        assert "comparison_summary" in comparison_results
        assert len(comparison_results["comparison_results"]) >= 2

        # Verify comparison summary includes best and worst configs
        summary = comparison_results["comparison_summary"]
        assert "best_config" in summary
        assert "best_speedup" in summary
        assert "worst_config" in summary
        assert "worst_slowdown" in summary

    def test_scaling_patterns_linear_complexity_estimation(self):
        """Test scaling pattern analysis correctly identifies linear complexity."""
        analyzer = TBRPerformanceAnalyzer()

        # Create scaling results with approximately linear complexity (slope ~1.0)
        scaling_results = [
            {
                "size_multiplier": 1.0,
                "data_size": 100,
                "total_duration": 1.0,
                "success": True,
            },
            {
                "size_multiplier": 2.0,
                "data_size": 200,
                "total_duration": 2.1,  # ~2x slower for 2x data
                "success": True,
            },
            {
                "size_multiplier": 3.0,
                "data_size": 300,
                "total_duration": 3.2,  # ~3x slower for 3x data
                "success": True,
            },
        ]

        patterns = analyzer._analyze_scaling_patterns(scaling_results)

        assert "complexity_estimate" in patterns
        # Should estimate approximately linear complexity
        assert "linear" in patterns["complexity_estimate"].lower()


class TestConvenienceFunctions:
    """Test convenience functions for performance analysis."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        self.test_data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, 50),
                "test": np.random.normal(1020, 55, 50),
            }
        )

    def test_profile_tbr_workflow(self):
        """Test profile_tbr_workflow convenience function."""

        def mock_analysis_function(data, **_kwargs):
            # Simple mock that returns expected structure
            tbr_df = pd.DataFrame(
                {
                    "date": data["date"],
                    "period": [0] * len(data),
                    "y": data["test"],
                    "x": data["control"],
                    "pred": data["control"] * 1.02,
                    "predsd": [1.0] * len(data),
                    "dif": data["test"] - data["control"] * 1.02,
                    "cumdif": [0.0] * len(data),
                    "cumsd": [1.0] * len(data),
                    "estsd": [1.0] * len(data),
                }
            )

            summary_df = pd.DataFrame(
                {
                    "estimate": [10.0],
                    "precision": [5.0],
                    "lower": [5.0],
                    "upper": [15.0],
                }
            )

            return tbr_df, summary_df

        result, metrics, monitoring_report = profile_tbr_workflow(
            mock_analysis_function,
            data=self.test_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-15"),
            test_end=pd.Timestamp("2023-01-25"),
            level=0.80,
            threshold=0.0,
            enable_monitoring=False,
        )

        # Check result structure
        assert len(result) == 2  # tbr_df, summary_df
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation_name == "tbr_workflow"
        assert metrics.duration > 0

        # Monitoring should be None when disabled
        assert monitoring_report is None

    def test_profile_tbr_workflow_with_monitoring(self):
        """Test profile_tbr_workflow with monitoring enabled."""

        def mock_analysis_function(data, **_kwargs):
            # Simple mock that returns expected structure
            time.sleep(0.01)  # Small delay to ensure monitoring captures data
            tbr_df = pd.DataFrame(
                {
                    "date": data["date"],
                    "period": [0] * len(data),
                    "y": data["test"],
                    "x": data["control"],
                    "pred": data["control"] * 1.02,
                    "predsd": [1.0] * len(data),
                    "dif": data["test"] - data["control"] * 1.02,
                    "cumdif": [0.0] * len(data),
                    "cumsd": [1.0] * len(data),
                    "estsd": [1.0] * len(data),
                }
            )

            summary_df = pd.DataFrame(
                {
                    "estimate": [10.0],
                    "precision": [5.0],
                    "lower": [5.0],
                    "upper": [15.0],
                }
            )

            return tbr_df, summary_df

        result, metrics, monitoring_report = profile_tbr_workflow(
            mock_analysis_function,
            data=self.test_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-15"),
            test_end=pd.Timestamp("2023-01-25"),
            level=0.80,
            threshold=0.0,
            enable_monitoring=True,  # Enable monitoring
        )

        # Check result structure
        assert len(result) == 2  # tbr_df, summary_df
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation_name == "tbr_workflow"
        assert metrics.duration > 0

        # Monitoring should contain data when enabled
        assert monitoring_report is not None
        assert isinstance(monitoring_report, dict)
        # Check that monitoring report has expected keys
        if "error" not in monitoring_report:
            assert "duration" in monitoring_report
            assert "sample_count" in monitoring_report

    def test_benchmark_tbr_functions(self):
        """Test benchmark_tbr_functions convenience function."""

        def fast_function(data):
            return np.sum(data)

        def slow_function(data):
            time.sleep(0.001)  # Small delay
            return np.sum(data)

        functions = {"fast_func": fast_function, "slow_func": slow_function}

        test_array = np.random.randn(100)

        results = benchmark_tbr_functions(
            functions=functions, test_data=test_array, n_runs=3
        )

        assert "fast_func" in results
        assert "slow_func" in results

        # Check that both functions have benchmark stats
        for func_name in ["fast_func", "slow_func"]:
            stats = results[func_name]
            if "error" not in stats:
                assert "mean_time" in stats
                assert "n_runs" in stats
                assert stats["n_runs"] == 3

    def test_quick_performance_check(self, capsys):
        """Test quick_performance_check convenience function."""
        # Use a very small dataset for fast testing
        small_data = self.test_data.iloc[:10].copy()

        # This should print performance summary
        quick_performance_check(
            data=small_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-05"),
            test_end=pd.Timestamp("2023-01-08"),
            level=0.80,
            threshold=0.0,
        )

        captured = capsys.readouterr()
        assert "Running TBR performance analysis..." in captured.out
        assert "TBR PERFORMANCE ANALYSIS SUMMARY" in captured.out


class TestPerformanceIntegration:
    """Integration tests for performance diagnostics."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        self.test_data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, 30),
                "test": np.random.normal(1020, 55, 30),
            }
        )

    def test_end_to_end_performance_analysis(self):
        """Test complete end-to-end performance analysis workflow."""
        analyzer = TBRPerformanceAnalyzer()

        # Analyze performance
        performance_report = analyzer.analyze_tbr_performance(
            data=self.test_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-15"),
            test_end=pd.Timestamp("2023-01-25"),
            enable_monitoring=False,
        )

        # Get recommendations
        recommendations = analyzer.get_optimization_recommendations(performance_report)

        # Set as baseline
        analyzer.set_performance_baseline("test_baseline", performance_report)

        # Compare to baseline (should be identical)
        comparison = analyzer.compare_to_baseline(performance_report, "test_baseline")

        # Verify complete workflow
        assert isinstance(performance_report, dict)
        assert isinstance(recommendations, dict)
        assert isinstance(comparison, dict)

        # Check that comparison shows no regression (identical data)
        assert abs(comparison["duration_ratio"] - 1.0) < 0.1
        assert abs(comparison["efficiency_ratio"] - 1.0) < 0.1
        assert comparison["performance_regression"] is False

    def test_performance_with_different_data_sizes(self):
        """Test performance analysis with different data sizes."""
        analyzer = TBRPerformanceAnalyzer()

        # Test with different sized datasets
        sizes = [10, 20]  # Small sizes for fast testing
        results = []

        for size in sizes:
            data_subset = self.test_data.iloc[:size].copy()

            performance_report = analyzer.analyze_tbr_performance(
                data=data_subset,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-01-08"),
                test_end=pd.Timestamp("2023-01-12"),
                enable_monitoring=False,
            )

            results.append(
                {
                    "size": size,
                    "duration": performance_report["workflow_metrics"].duration,
                    "efficiency_score": performance_report[
                        "efficiency_report"
                    ].efficiency_score,
                }
            )

        # Verify we got results for both sizes
        assert len(results) == 2

        # Both should have positive duration and reasonable efficiency scores
        for result in results:
            assert result["duration"] > 0
            assert 0 <= result["efficiency_score"] <= 10


class TestPerformanceCoverageGaps:
    """Test cases specifically designed to cover missing lines in performance.py."""

    def test_efficiency_report_with_bottlenecks_and_recommendations(self):
        """Test EfficiencyReport summary method with bottlenecks and recommendations."""
        # Test with bottlenecks and recommendations (lines 91-99)
        report = EfficiencyReport(
            operation_name="test_operation",
            data_size=1000,
            efficiency_score=7.5,
            computational_complexity="O(n)",
            bottlenecks=["memory_usage", "cpu_intensive"],
            recommendations=["Use vectorization", "Reduce memory allocation"],
        )

        summary_output = report.summary()
        assert "Bottlenecks: memory_usage, cpu_intensive" in summary_output
        assert "Recommendations:" in summary_output
        assert "  - Use vectorization" in summary_output
        assert "  - Reduce memory allocation" in summary_output

    def test_efficiency_report_without_bottlenecks_and_recommendations(self):
        """Test EfficiencyReport summary method without bottlenecks and recommendations."""
        # Test without bottlenecks and recommendations (lines 91-99 branches)
        report = EfficiencyReport(
            operation_name="test_operation",
            data_size=1000,
            efficiency_score=9.0,
            computational_complexity="O(1)",
            bottlenecks=[],
            recommendations=[],
        )

        summary_output = report.summary()
        assert "Bottlenecks:" not in summary_output
        assert "Recommendations:" not in summary_output

    @patch("psutil.Process")
    def test_profiler_cpu_tracking_initialization_failure(self, mock_process):
        """Test profiler initialization when CPU tracking fails."""
        # Test psutil.NoSuchProcess exception (lines 148-150)
        import psutil

        mock_process.side_effect = psutil.NoSuchProcess(pid=123)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            profiler = PerformanceProfiler(enable_cpu_tracking=True)

            assert not profiler.enable_cpu_tracking
            assert len(w) == 1
            assert "CPU tracking disabled due to system limitations" in str(
                w[0].message
            )

    @patch("psutil.Process")
    def test_profiler_cpu_tracking_access_denied(self, mock_process):
        """Test profiler initialization when CPU tracking access is denied."""
        # Test psutil.AccessDenied exception (lines 148-150)
        import psutil

        mock_process.side_effect = psutil.AccessDenied(pid=123)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            profiler = PerformanceProfiler(enable_cpu_tracking=True)

            assert not profiler.enable_cpu_tracking
            assert len(w) == 1
            assert "CPU tracking disabled due to system limitations" in str(
                w[0].message
            )

    @patch("tracemalloc.start")
    @patch("gc.collect")
    def test_profiler_memory_tracking_enabled(self, mock_gc, mock_tracemalloc):
        """Test profiler context with memory tracking enabled."""
        # Test memory tracking branch (lines 177-182)
        profiler = PerformanceProfiler(enable_memory_tracking=True)

        with profiler.profile_context("test_op") as metrics:
            time.sleep(0.01)

        mock_tracemalloc.assert_called_once()
        mock_gc.assert_called_once()
        assert metrics.duration > 0

    @patch("psutil.Process")
    def test_profiler_cpu_tracking_during_profiling_failure(self, mock_process_class):
        """Test CPU tracking failure during profiling."""
        # Test CPU tracking exceptions during profiling (lines 188-191, 217-218)
        import psutil

        mock_process = Mock()
        mock_process.cpu_percent.side_effect = [10.0, psutil.NoSuchProcess(pid=123)]
        mock_process_class.return_value = mock_process

        profiler = PerformanceProfiler(enable_cpu_tracking=True)
        profiler.process = mock_process

        with profiler.profile_context("test_op") as metrics:
            time.sleep(0.01)

        # Should handle the exception gracefully
        assert metrics.duration > 0
        assert metrics.cpu_percent is None  # Should be None due to exception

    @patch("psutil.Process")
    def test_profiler_cpu_tracking_start_failure_nosuchprocess(
        self, mock_process_class
    ):
        """Test CPU tracking failure at start of profiling - NoSuchProcess."""
        # Test CPU tracking exception at start (lines 188-189)
        import psutil

        mock_process = Mock()
        mock_process.cpu_percent.side_effect = psutil.NoSuchProcess(pid=123)
        mock_process_class.return_value = mock_process

        profiler = PerformanceProfiler(enable_cpu_tracking=True)
        profiler.process = mock_process

        with profiler.profile_context("test_op") as metrics:
            time.sleep(0.01)

        # Should handle the exception gracefully and set cpu_start to None
        assert metrics.duration > 0
        assert metrics.cpu_percent is None  # Should be None due to exception at start

    @patch("psutil.Process")
    def test_profiler_cpu_tracking_start_failure_accessdenied(self, mock_process_class):
        """Test CPU tracking failure at start of profiling - AccessDenied."""
        # Test CPU tracking exception at start (lines 188-189)
        import psutil

        mock_process = Mock()
        mock_process.cpu_percent.side_effect = psutil.AccessDenied(pid=123)
        mock_process_class.return_value = mock_process

        profiler = PerformanceProfiler(enable_cpu_tracking=True)
        profiler.process = mock_process

        with profiler.profile_context("test_op") as metrics:
            time.sleep(0.01)

        # Should handle the exception gracefully and set cpu_start to None
        assert metrics.duration > 0
        assert metrics.cpu_percent is None  # Should be None due to exception at start

    @patch("tracemalloc.get_traced_memory")
    @patch("tracemalloc.stop")
    def test_profiler_memory_tracking_exception(self, _mock_stop, mock_get_memory):
        """Test memory tracking exception handling."""
        # Test memory tracking exception (lines 208-210)
        mock_get_memory.side_effect = Exception("Memory tracking failed")

        profiler = PerformanceProfiler(enable_memory_tracking=True)

        with profiler.profile_context("test_op") as metrics:
            time.sleep(0.01)

        # Should handle the exception gracefully
        assert metrics.duration > 0
        assert metrics.memory_current is None
        assert metrics.memory_peak is None

    def test_profiler_cpu_tracking_without_process(self):
        """Test CPU tracking when process is not available."""
        # Test CPU tracking branch when process is not set (lines 185-191)
        profiler = PerformanceProfiler(enable_cpu_tracking=False)

        with profiler.profile_context("test_op") as metrics:
            time.sleep(0.01)

        assert metrics.duration > 0
        assert metrics.cpu_percent is None

    def test_benchmark_function_operation_name_default(self):
        """Test benchmark function with default operation name."""
        # Test default operation name generation (lines 280-284)
        profiler = PerformanceProfiler()

        def test_func(x):
            return x * 2

        stats = profiler.benchmark_function(test_func, 5, n_runs=2, warmup_runs=1)

        assert stats["result"] == 10  # 5 * 2
        assert stats["n_runs"] == 2
        assert "mean_time" in stats

    def test_benchmark_function_operation_name_none_with_module(self):
        """Test benchmark function with None operation_name generates module.name."""
        # Test lines 280-281: operation_name generation from func.__module__ and func.__name__
        profiler = PerformanceProfiler()

        def sample_function():
            return 42

        # Explicitly pass None for operation_name to trigger the default generation
        profiler.benchmark_function(
            sample_function,
            operation_name=None,  # This should trigger lines 280-281
            n_runs=1,
            warmup_runs=0,
        )

        # Should generate operation_name as "module.function_name" and use it in profile contexts
        expected_name = f"{sample_function.__module__}.{sample_function.__name__}"

        # Check that the operation was recorded with the expected name pattern
        all_metrics = profiler.get_metrics()
        assert len(all_metrics) == 1  # Should have 1 run

        # The profile context should be named "{operation_name}_run_0"
        expected_context_name = f"{expected_name}_run_0"
        assert expected_context_name in all_metrics
        assert "sample_function" in expected_context_name

    def test_benchmark_function_warmup_runs_execution(self):
        """Test that warmup runs are executed before benchmark runs."""
        # Test lines 283-284: warmup runs loop execution
        profiler = PerformanceProfiler()

        call_count = 0

        def counting_function():
            nonlocal call_count
            call_count += 1
            return call_count

        # Run with 3 warmup runs and 2 benchmark runs
        stats = profiler.benchmark_function(
            counting_function,
            operation_name="test_warmup",
            n_runs=2,
            warmup_runs=3,  # This should trigger the warmup loop (lines 283-284)
        )

        # Total calls should be warmup_runs + n_runs = 3 + 2 = 5
        assert call_count == 5
        assert stats["n_runs"] == 2

        # Check that the operation was recorded with the expected name pattern
        all_metrics = profiler.get_metrics()
        assert len(all_metrics) == 2  # Should have 2 benchmark runs

        # The profile contexts should be named "test_warmup_run_0" and "test_warmup_run_1"
        assert "test_warmup_run_0" in all_metrics
        assert "test_warmup_run_1" in all_metrics

    def test_print_summary_specific_operation(self, capsys):
        """Test print_summary with specific operation name."""
        # Test line 353: metrics_dict = {operation_name: self.metrics.get(operation_name)}
        profiler = PerformanceProfiler()

        # Create some metrics by profiling operations
        with profiler.profile_context("operation_1"):
            time.sleep(0.01)

        with profiler.profile_context("operation_2"):
            time.sleep(0.01)

        # Print summary for specific operation (should trigger line 353)
        profiler.print_summary(operation_name="operation_1")

        captured = capsys.readouterr()

        # Should only show operation_1, not operation_2
        assert "Operation: operation_1" in captured.out
        assert "operation_2" not in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_nonexistent_operation(self, capsys):
        """Test print_summary with nonexistent operation name."""
        # Test line 353 with nonexistent operation (metrics.get returns None)
        profiler = PerformanceProfiler()

        # Create one operation
        with profiler.profile_context("existing_operation"):
            time.sleep(0.01)

        # Print summary for nonexistent operation (should trigger line 353 and then line 363)
        profiler.print_summary(operation_name="nonexistent_operation")

        captured = capsys.readouterr()

        # Should show "No metrics found" message
        assert "No metrics found for operation: nonexistent_operation" in captured.out
        assert "existing_operation" not in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_mixed_existing_and_none_metrics(self, capsys):
        """Test print_summary with mix of existing and None metrics."""
        # Test lines 363-364: handling None metrics in loop with continue statement
        profiler = PerformanceProfiler()

        # Create some real metrics
        with profiler.profile_context("valid_operation_1"):
            time.sleep(0.01)

        with profiler.profile_context("valid_operation_2"):
            time.sleep(0.01)

        # Manually inject None metrics to test the None handling path
        # This simulates a scenario where metrics could be None
        profiler.metrics["none_operation_1"] = None
        profiler.metrics["none_operation_2"] = None

        # Print summary for all operations (should trigger lines 363-364 for None metrics)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show "No metrics found" for None operations (lines 363-364)
        assert "No metrics found for operation: none_operation_1" in captured.out
        assert "No metrics found for operation: none_operation_2" in captured.out

        # Should also show valid operations
        assert "Operation: valid_operation_1" in captured.out
        assert "Operation: valid_operation_2" in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_all_none_metrics(self, capsys):
        """Test print_summary when all metrics are None."""
        # Test lines 363-364: continue statement when all metrics are None
        profiler = PerformanceProfiler()

        # Manually inject only None metrics
        profiler.metrics["none_operation_1"] = None
        profiler.metrics["none_operation_2"] = None
        profiler.metrics["none_operation_3"] = None

        # Print summary (should trigger lines 363-364 for all operations)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show "No metrics found" for all operations
        assert "No metrics found for operation: none_operation_1" in captured.out
        assert "No metrics found for operation: none_operation_2" in captured.out
        assert "No metrics found for operation: none_operation_3" in captured.out

        # Should not show any "Operation:" lines (since all continue)
        assert "Operation: none_operation_1" not in captured.out
        assert "Operation: none_operation_2" not in captured.out
        assert "Operation: none_operation_3" not in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_with_memory_peak(self, capsys):
        """Test print_summary with memory peak information."""
        # Test lines 369-370: memory peak conditional printing
        from src.tbr.utils.performance import PerformanceMetrics

        profiler = PerformanceProfiler()

        # Create metrics with memory peak directly
        metrics_with_memory = PerformanceMetrics(
            operation_name="memory_operation",
            duration=0.0125,
            memory_peak=15.75,  # MB
            memory_current=12.50,
            cpu_percent=5.2,
        )

        # Add to profiler metrics
        profiler.metrics["memory_operation"] = metrics_with_memory

        # Print summary (should trigger lines 369-370)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show memory peak information (lines 369-370)
        assert "Operation: memory_operation" in captured.out
        assert "Peak Memory: 15.75 MB" in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_without_memory_peak(self, capsys):
        """Test print_summary without memory peak information."""
        # Test line 369: memory peak is None, should skip line 370
        from src.tbr.utils.performance import PerformanceMetrics

        profiler = PerformanceProfiler()

        # Create metrics without memory peak (None)
        metrics_without_memory = PerformanceMetrics(
            operation_name="no_memory_operation",
            duration=0.0125,
            memory_peak=None,  # This should skip lines 369-370
            memory_current=None,
            cpu_percent=3.1,
        )

        # Add to profiler metrics
        profiler.metrics["no_memory_operation"] = metrics_without_memory

        # Print summary (memory_peak is None, should skip lines 369-370)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show operation but NOT show memory peak information
        assert "Operation: no_memory_operation" in captured.out
        assert "Peak Memory:" not in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_mixed_memory_tracking(self, capsys):
        """Test print_summary with mixed memory tracking scenarios."""
        # Test lines 369-370: some operations have memory_peak, others don't
        from src.tbr.utils.performance import PerformanceMetrics

        profiler = PerformanceProfiler()

        # Create operation with memory peak
        metrics_with_memory = PerformanceMetrics(
            operation_name="with_memory",
            duration=0.0125,
            memory_peak=25.50,  # MB
            memory_current=20.25,
            cpu_percent=4.5,
        )

        # Create operation without memory peak
        metrics_without_memory = PerformanceMetrics(
            operation_name="without_memory",
            duration=0.0130,
            memory_peak=None,  # No memory tracking
            memory_current=None,
            cpu_percent=2.8,
        )

        # Add both to profiler metrics
        profiler.metrics["with_memory"] = metrics_with_memory
        profiler.metrics["without_memory"] = metrics_without_memory

        # Print summary (should handle both cases)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show memory for the operation that has it
        assert "Operation: with_memory" in captured.out
        assert "Peak Memory: 25.50 MB" in captured.out

        # Should show operation without memory info
        assert "Operation: without_memory" in captured.out
        # Count occurrences to ensure only one "Peak Memory" line
        peak_memory_count = captured.out.count("Peak Memory:")
        assert peak_memory_count == 1

    def test_print_summary_with_current_memory(self, capsys):
        """Test print_summary with current memory information."""
        # Test lines 371-372: current memory conditional printing
        from src.tbr.utils.performance import PerformanceMetrics

        profiler = PerformanceProfiler()

        # Create metrics with current memory
        metrics_with_current_memory = PerformanceMetrics(
            operation_name="current_memory_operation",
            duration=0.0135,
            memory_peak=18.25,
            memory_current=14.80,  # MB - should trigger lines 371-372
            cpu_percent=6.3,
        )

        # Add to profiler metrics
        profiler.metrics["current_memory_operation"] = metrics_with_current_memory

        # Print summary (should trigger lines 371-372)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show current memory information (lines 371-372)
        assert "Operation: current_memory_operation" in captured.out
        assert "Current Memory: 14.80 MB" in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_without_current_memory(self, capsys):
        """Test print_summary without current memory information."""
        # Test line 371: memory_current is None, should skip line 372
        from src.tbr.utils.performance import PerformanceMetrics

        profiler = PerformanceProfiler()

        # Create metrics without current memory
        metrics_without_current_memory = PerformanceMetrics(
            operation_name="no_current_memory_operation",
            duration=0.0140,
            memory_peak=20.15,
            memory_current=None,  # This should skip lines 371-372
            cpu_percent=4.7,
        )

        # Add to profiler metrics
        profiler.metrics["no_current_memory_operation"] = metrics_without_current_memory

        # Print summary (memory_current is None, should skip lines 371-372)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show operation but NOT show current memory information
        assert "Operation: no_current_memory_operation" in captured.out
        assert "Current Memory:" not in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_mixed_current_memory_tracking(self, capsys):
        """Test print_summary with mixed current memory tracking scenarios."""
        # Test lines 371-372: some operations have memory_current, others don't
        from src.tbr.utils.performance import PerformanceMetrics

        profiler = PerformanceProfiler()

        # Create operation with current memory
        metrics_with_current = PerformanceMetrics(
            operation_name="with_current_memory",
            duration=0.0125,
            memory_peak=22.75,
            memory_current=18.90,  # MB
            cpu_percent=5.2,
        )

        # Create operation without current memory
        metrics_without_current = PerformanceMetrics(
            operation_name="without_current_memory",
            duration=0.0130,
            memory_peak=15.60,
            memory_current=None,  # No current memory tracking
            cpu_percent=3.8,
        )

        # Add both to profiler metrics
        profiler.metrics["with_current_memory"] = metrics_with_current
        profiler.metrics["without_current_memory"] = metrics_without_current

        # Print summary (should handle both cases)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show current memory for the operation that has it
        assert "Operation: with_current_memory" in captured.out
        assert "Current Memory: 18.90 MB" in captured.out

        # Should show operation without current memory info
        assert "Operation: without_current_memory" in captured.out
        # Count occurrences to ensure only one "Current Memory" line
        current_memory_count = captured.out.count("Current Memory:")
        assert current_memory_count == 1

    def test_print_summary_with_cpu_usage(self, capsys):
        """Test print_summary with CPU usage information."""
        # Test lines 373-374: CPU usage conditional printing
        from src.tbr.utils.performance import PerformanceMetrics

        profiler = PerformanceProfiler()

        # Create metrics with CPU usage
        metrics_with_cpu = PerformanceMetrics(
            operation_name="cpu_operation",
            duration=0.0145,
            memory_peak=16.50,
            memory_current=13.25,
            cpu_percent=8.7,  # Should trigger lines 373-374
            function_calls=0,
        )

        # Add to profiler metrics
        profiler.metrics["cpu_operation"] = metrics_with_cpu

        # Print summary (should trigger lines 373-374)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show CPU usage information (lines 373-374)
        assert "Operation: cpu_operation" in captured.out
        assert "CPU Usage: 8.7%" in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_without_cpu_usage(self, capsys):
        """Test print_summary without CPU usage information."""
        # Test line 373: cpu_percent is None, should skip line 374
        from src.tbr.utils.performance import PerformanceMetrics

        profiler = PerformanceProfiler()

        # Create metrics without CPU usage
        metrics_without_cpu = PerformanceMetrics(
            operation_name="no_cpu_operation",
            duration=0.0150,
            memory_peak=19.75,
            memory_current=15.40,
            cpu_percent=None,  # This should skip lines 373-374
            function_calls=0,
        )

        # Add to profiler metrics
        profiler.metrics["no_cpu_operation"] = metrics_without_cpu

        # Print summary (cpu_percent is None, should skip lines 373-374)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show operation but NOT show CPU usage information
        assert "Operation: no_cpu_operation" in captured.out
        assert "CPU Usage:" not in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_with_function_calls(self, capsys):
        """Test print_summary with function calls information."""
        # Test lines 375-376: function calls conditional printing
        from src.tbr.utils.performance import PerformanceMetrics

        profiler = PerformanceProfiler()

        # Create metrics with function calls
        metrics_with_calls = PerformanceMetrics(
            operation_name="function_calls_operation",
            duration=0.0155,
            memory_peak=21.30,
            memory_current=17.80,
            cpu_percent=6.9,
            function_calls=42,  # Should trigger lines 375-376 (> 0)
        )

        # Add to profiler metrics
        profiler.metrics["function_calls_operation"] = metrics_with_calls

        # Print summary (should trigger lines 375-376)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show function calls information (lines 375-376)
        assert "Operation: function_calls_operation" in captured.out
        assert "Function Calls: 42" in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_without_function_calls(self, capsys):
        """Test print_summary without function calls information."""
        # Test line 375: function_calls is 0, should skip line 376
        from src.tbr.utils.performance import PerformanceMetrics

        profiler = PerformanceProfiler()

        # Create metrics without function calls
        metrics_without_calls = PerformanceMetrics(
            operation_name="no_calls_operation",
            duration=0.0160,
            memory_peak=18.90,
            memory_current=14.60,
            cpu_percent=5.3,
            function_calls=0,  # This should skip lines 375-376 (not > 0)
        )

        # Add to profiler metrics
        profiler.metrics["no_calls_operation"] = metrics_without_calls

        # Print summary (function_calls is 0, should skip lines 375-376)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show operation but NOT show function calls information
        assert "Operation: no_calls_operation" in captured.out
        assert "Function Calls:" not in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_with_metadata(self, capsys):
        """Test print_summary with metadata information."""
        # Test lines 378-381: metadata conditional printing
        from src.tbr.utils.performance import PerformanceMetrics

        profiler = PerformanceProfiler()

        # Create metrics with metadata
        metrics_with_metadata = PerformanceMetrics(
            operation_name="metadata_operation",
            duration=0.0165,
            memory_peak=23.45,
            memory_current=19.20,
            cpu_percent=7.8,
            function_calls=15,
            metadata={
                "algorithm": "quicksort",
                "iterations": 100,
                "mode": "optimized",
            },  # Should trigger lines 378-381
        )

        # Add to profiler metrics
        profiler.metrics["metadata_operation"] = metrics_with_metadata

        # Print summary (should trigger lines 378-381)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show metadata information (lines 378-381)
        assert "Operation: metadata_operation" in captured.out
        assert "Metadata:" in captured.out
        assert "  algorithm: quicksort" in captured.out
        assert "  iterations: 100" in captured.out
        assert "  mode: optimized" in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_print_summary_without_metadata(self, capsys):
        """Test print_summary without metadata information."""
        # Test line 378: metadata is empty/None, should skip lines 378-381
        from src.tbr.utils.performance import PerformanceMetrics

        profiler = PerformanceProfiler()

        # Create metrics without metadata
        metrics_without_metadata = PerformanceMetrics(
            operation_name="no_metadata_operation",
            duration=0.0170,
            memory_peak=20.85,
            memory_current=16.40,
            cpu_percent=4.2,
            function_calls=8,
            metadata={},  # Empty dict should skip lines 378-381
        )

        # Add to profiler metrics
        profiler.metrics["no_metadata_operation"] = metrics_without_metadata

        # Print summary (metadata is empty, should skip lines 378-381)
        profiler.print_summary()

        captured = capsys.readouterr()

        # Should show operation but NOT show metadata information
        assert "Operation: no_metadata_operation" in captured.out
        assert "Metadata:" not in captured.out
        assert "PERFORMANCE PROFILING SUMMARY" in captured.out

    def test_efficiency_metrics_baseline_comparison(self):
        """Test baseline comparison in analyze_workflow_efficiency."""
        # Test line 463: baseline comparison when baseline exists
        from src.tbr.utils.performance import PerformanceMetrics

        efficiency = EfficiencyMetrics()

        # Set a baseline first with proper metrics dict
        baseline_metrics = {
            "test_op": PerformanceMetrics(
                operation_name="test_op",
                duration=1.0,
                memory_peak=10.0,
                cpu_percent=5.0,
            )
        }
        efficiency.set_baseline("test_operation", baseline_metrics)

        # Create operation metrics for current analysis
        current_metrics = {
            "test_op": PerformanceMetrics(
                operation_name="test_op",
                duration=1.2,  # Slightly slower than baseline
                memory_peak=12.0,
                cpu_percent=6.0,
            )
        }

        # Analyze workflow efficiency (should trigger line 463)
        report = efficiency.analyze_workflow_efficiency(
            data_size=1000,
            operation_metrics=current_metrics,
            operation_name="test_operation",
        )

        # Should have baseline comparison
        assert report.baseline_comparison is not None
        assert "duration_ratio" in report.baseline_comparison
        assert "memory_ratio" in report.baseline_comparison
        assert "operation_ratio" in report.baseline_comparison
        # Verify that line 463 was executed (baseline comparison exists)
        assert (
            report.baseline_comparison["duration_ratio"] > 1.0
        )  # Slower than baseline

    def test_efficiency_metrics_scaling_behavior_regression_function(self):
        """Test scaling behavior with regression function name."""
        # Test lines 509-513: DataFrame generation for regression functions
        efficiency = EfficiencyMetrics()

        # Create a mock function with 'regression' in the name
        def mock_regression_function(data):
            return (
                data.mean().mean() if hasattr(data, "mean") else sum(data) / len(data)
            )

        # Analyze scaling behavior (should trigger lines 509-513)
        results = efficiency.analyze_scaling_behavior(
            mock_regression_function, data_sizes=[10, 20]  # Small sizes for testing
        )

        # Should have results for both data sizes
        assert "scaling_results" in results
        assert len(results["scaling_results"]) == 2
        assert all("data_size" in result for result in results["scaling_results"])
        assert all("mean_time" in result for result in results["scaling_results"])
        assert all("memory_usage" in result for result in results["scaling_results"])

    def test_efficiency_metrics_complexity_constant_time(self):
        """Test computational complexity estimation for O(1) operations."""
        # Test line 570: return "O(1) - Constant"
        efficiency = EfficiencyMetrics()

        # Very fast operation (< 1e-6 seconds per element)
        complexity = efficiency._estimate_computational_complexity(
            data_size=1000,  # 1000 elements
            duration=0.0000005,  # 0.5 microseconds total = 0.5e-6 seconds per element
        )

        # Should return O(1) complexity (line 570)
        assert complexity == "O(1) - Constant"

    def test_efficiency_metrics_complexity_logarithmic_time(self):
        """Test computational complexity estimation for O(log n) operations."""
        # Test line 572: return "O(log n) - Logarithmic"
        efficiency = EfficiencyMetrics()

        # Moderately fast operation (< 1e-5 but >= 1e-6 seconds per element)
        complexity = efficiency._estimate_computational_complexity(
            data_size=1000,  # 1000 elements
            duration=0.005,  # 5 milliseconds total = 5e-6 seconds per element
        )

        # Should return O(log n) complexity (line 572)
        assert complexity == "O(log n) - Logarithmic"

    def test_efficiency_metrics_scoring_very_fast_operations(self):
        """Test efficiency scoring for very fast operations."""
        # Test line 592: time_score = 5.0
        from src.tbr.utils.performance import PerformanceMetrics

        efficiency = EfficiencyMetrics()

        # Create mock operation metrics
        operation_metrics = {
            "fast_op": PerformanceMetrics(
                operation_name="fast_op",
                duration=0.0000005,  # 0.5 microseconds total
                memory_peak=10.0,
                cpu_percent=5.0,
            )
        }

        # Very fast operation (< 1e-6 seconds per element)
        score = efficiency._calculate_efficiency_score(
            data_size=1000,  # 0.5e-6 seconds per element
            total_duration=0.0000005,
            operation_metrics=operation_metrics,
        )

        # Should get maximum time score of 5.0 (line 592)
        # Total score should be high (time_score + memory_score)
        assert score >= 5.0  # At least the time component
        assert score <= 10.0  # Maximum possible score

    def test_efficiency_metrics_scoring_fast_operations(self):
        """Test efficiency scoring for fast operations."""
        # Test line 594: time_score = 4.0
        from src.tbr.utils.performance import PerformanceMetrics

        efficiency = EfficiencyMetrics()

        # Create mock operation metrics
        operation_metrics = {
            "fast_op": PerformanceMetrics(
                operation_name="fast_op",
                duration=0.005,  # 5 milliseconds total
                memory_peak=15.0,
                cpu_percent=8.0,
            )
        }

        # Fast operation (< 1e-5 but >= 1e-6 seconds per element)
        score = efficiency._calculate_efficiency_score(
            data_size=1000,  # 5e-6 seconds per element
            total_duration=0.005,
            operation_metrics=operation_metrics,
        )

        # Should get time score of 4.0 (line 594)
        # Total score should be good but not maximum
        assert score >= 4.0  # At least the time component
        assert score < 10.0  # Less than maximum
        assert score <= 9.0  # Should be reasonable given memory usage

    def test_efficiency_metrics_operation_count_scoring_medium(self):
        """Test efficiency scoring for medium operation count."""
        # Test lines 624-625: operation_score = 1.0 for medium count (6-10 operations)
        from src.tbr.utils.performance import PerformanceMetrics

        efficiency = EfficiencyMetrics()

        # Create 8 operation metrics (medium count: 6-10)
        operation_metrics = {}
        for i in range(8):
            operation_metrics[f"op_{i}"] = PerformanceMetrics(
                operation_name=f"op_{i}",
                duration=0.01,  # Moderate duration
                memory_peak=20.0,
                cpu_percent=5.0,
            )

        # Should get operation score of 1.0 (lines 624-625)
        score = efficiency._calculate_efficiency_score(
            data_size=1000,
            total_duration=0.08,  # 8 * 0.01
            operation_metrics=operation_metrics,
        )

        # Should include operation score of 1.0
        assert score >= 1.0  # At least the operation component
        assert score <= 10.0  # Maximum possible score

    def test_efficiency_metrics_operation_count_scoring_high(self):
        """Test efficiency scoring for high operation count."""
        # Test lines 626-627: operation_score = 0.0 for high count (> 10 operations)
        from src.tbr.utils.performance import PerformanceMetrics

        efficiency = EfficiencyMetrics()

        # Create 12 operation metrics (high count: > 10)
        operation_metrics = {}
        for i in range(12):
            operation_metrics[f"op_{i}"] = PerformanceMetrics(
                operation_name=f"op_{i}",
                duration=0.01,  # Moderate duration
                memory_peak=15.0,
                cpu_percent=4.0,
            )

        # Should get operation score of 0.0 (lines 626-627)
        score = efficiency._calculate_efficiency_score(
            data_size=1000,
            total_duration=0.12,  # 12 * 0.01
            operation_metrics=operation_metrics,
        )

        # Should not include operation score (0.0)
        assert score >= 0.0  # Minimum possible score
        assert score <= 8.0  # No operation score bonus

    def test_efficiency_metrics_bottlenecks_empty_metrics(self):
        """Test bottleneck identification with empty operation metrics."""
        # Test line 636: return bottlenecks when no operation metrics
        efficiency = EfficiencyMetrics()

        # Empty operation metrics should trigger line 636
        bottlenecks = efficiency._identify_bottlenecks({})

        # Should return empty list (line 636)
        assert bottlenecks == []

    def test_efficiency_metrics_bottlenecks_duration_analysis(self):
        """Test bottleneck identification with duration analysis."""
        # Test lines 640->650: duration analysis and bottleneck identification
        from src.tbr.utils.performance import PerformanceMetrics

        efficiency = EfficiencyMetrics()

        # Create metrics with one slow operation (no memory peaks to avoid memory bottleneck)
        # Mean will be (0.1 + 2.0 + 0.1) / 3 = 0.733, so 2 * mean = 1.466
        # slow_op duration of 2.0 > 1.466, so it will be identified as bottleneck
        operation_metrics = {
            "fast_op": PerformanceMetrics(
                operation_name="fast_op",
                duration=0.1,  # Fast
                memory_peak=None,  # No memory data
            ),
            "slow_op": PerformanceMetrics(
                operation_name="slow_op",
                duration=2.0,  # Slow (> 2 * mean of 0.733)
                memory_peak=None,  # No memory data
            ),
            "medium_op": PerformanceMetrics(
                operation_name="medium_op",
                duration=0.1,  # Fast
                memory_peak=None,  # No memory data
            ),
        }

        # Should identify slow operation as bottleneck (lines 640->650)
        bottlenecks = efficiency._identify_bottlenecks(operation_metrics)

        # Should identify the slow operation (could be both duration and memory bottleneck)
        assert len(bottlenecks) >= 1
        assert any("slow_op" in bottleneck for bottleneck in bottlenecks)

        # Check that at least one bottleneck is duration-based (lines 640->650)
        duration_bottlenecks = [b for b in bottlenecks if "% of total time)" in b]
        assert len(duration_bottlenecks) >= 1
        assert any("slow_op" in b for b in duration_bottlenecks)

    def test_efficiency_metrics_bottlenecks_with_valid_durations(self):
        """Test bottleneck identification specifically for lines 640-650 coverage."""
        # This test specifically targets the branch 640->650 that was missing coverage
        from src.tbr.utils.performance import PerformanceMetrics

        efficiency = EfficiencyMetrics()

        # Create operation metrics with valid durations to ensure line 640 condition is True
        # Mean will be (1.0 + 6.0 + 1.5) / 3 = 2.833, so 2 * mean = 5.666
        # operation_2 duration of 6.0 > 5.666, so it will be identified as bottleneck
        operation_metrics = {
            "operation_1": PerformanceMetrics(
                operation_name="operation_1",
                duration=1.0,  # Valid duration
                memory_peak=None,
            ),
            "operation_2": PerformanceMetrics(
                operation_name="operation_2",
                duration=6.0,  # Significantly slower - should be identified as bottleneck
                memory_peak=None,
            ),
            "operation_3": PerformanceMetrics(
                operation_name="operation_3",
                duration=1.5,  # Valid duration
                memory_peak=None,
            ),
        }

        # Call _identify_bottlenecks to trigger lines 640-650
        bottlenecks = efficiency._identify_bottlenecks(operation_metrics)

        # Verify that the duration analysis was performed (lines 640-650)
        duration_bottlenecks = [b for b in bottlenecks if "% of total time)" in b]
        assert len(duration_bottlenecks) >= 1

        # Verify that operation_2 is identified as a bottleneck
        operation_2_bottlenecks = [
            b for b in bottlenecks if "operation_2" in b and "% of total time)" in b
        ]
        assert len(operation_2_bottlenecks) >= 1

        # Verify the percentage calculation (line 646-647)
        # operation_2 takes 6.0 out of total 8.5 seconds = 70.6%
        assert any("70.6%" in b or "70.5%" in b for b in operation_2_bottlenecks)

    def test_efficiency_metrics_bottlenecks_with_no_valid_durations(self):
        """Test bottleneck identification when no valid durations exist."""
        # This test covers the case where durations list is empty (line 640 condition is False)
        from src.tbr.utils.performance import PerformanceMetrics

        efficiency = EfficiencyMetrics()

        # Create operation metrics with None durations to ensure line 640 condition is False
        operation_metrics = {
            "operation_1": PerformanceMetrics(
                operation_name="operation_1",
                duration=None,  # No duration data
                memory_peak=50.0,  # Has memory data
            ),
            "operation_2": PerformanceMetrics(
                operation_name="operation_2",
                duration=None,  # No duration data
                memory_peak=100.0,  # Has memory data
            ),
            "operation_3": PerformanceMetrics(
                operation_name="operation_3",
                duration=None,  # No duration data
                memory_peak=25.0,  # Has memory data
            ),
        }

        # Call _identify_bottlenecks - should skip duration analysis (line 640 is False)
        bottlenecks = efficiency._identify_bottlenecks(operation_metrics)

        # Should not have any duration-based bottlenecks since durations list is empty
        duration_bottlenecks = [b for b in bottlenecks if "% of total time)" in b]
        assert len(duration_bottlenecks) == 0

        # Should still identify memory-based bottlenecks
        memory_bottlenecks = [b for b in bottlenecks if "high memory usage" in b]
        assert (
            len(memory_bottlenecks) >= 1
        )  # operation_2 should be identified for high memory

    def test_efficiency_metrics_bottlenecks_memory_analysis(self):
        """Test bottleneck identification with memory analysis."""
        # Test lines 654->660: memory peak analysis and high memory usage identification
        from src.tbr.utils.performance import PerformanceMetrics

        efficiency = EfficiencyMetrics()

        # Create metrics with one high-memory operation
        operation_metrics = {
            "low_memory_op": PerformanceMetrics(
                operation_name="low_memory_op",
                duration=0.1,
                memory_peak=10.0,  # Low memory
            ),
            "high_memory_op": PerformanceMetrics(
                operation_name="high_memory_op",
                duration=0.1,
                memory_peak=100.0,  # High memory (> 70% of max)
            ),
            "medium_memory_op": PerformanceMetrics(
                operation_name="medium_memory_op",
                duration=0.1,
                memory_peak=50.0,  # Medium memory
            ),
        }

        # Should identify high memory operation as bottleneck (lines 654->660)
        bottlenecks = efficiency._identify_bottlenecks(operation_metrics)

        # Should identify the high memory operation
        assert len(bottlenecks) >= 1
        assert any("high_memory_op" in bottleneck for bottleneck in bottlenecks)
        assert any("high memory usage" in bottleneck for bottleneck in bottlenecks)

    def test_efficiency_metrics_recommendations_high_memory(self):
        """Test recommendations for high memory usage."""
        # Test lines 684-685: high memory usage recommendations
        from src.tbr.utils.performance import PerformanceMetrics

        efficiency = EfficiencyMetrics()

        # Create metrics with very high memory usage (> 1GB = 1000 MB)
        operation_metrics = {
            "memory_intensive_op": PerformanceMetrics(
                operation_name="memory_intensive_op",
                duration=0.5,
                memory_peak=1200.0,  # > 1000 MB (1GB)
            )
        }

        # Should generate high memory recommendations (lines 684-685)
        recommendations = efficiency._generate_recommendations(
            data_size=10000,
            total_duration=0.5,
            operation_metrics=operation_metrics,
            bottlenecks=[],
        )

        # Should include memory-specific recommendations (lines 684-685)
        assert any("chunks to reduce memory usage" in rec for rec in recommendations)
        assert any("memory-efficient data types" in rec for rec in recommendations)

    def test_efficiency_metrics_recommendations_bottleneck_specific(self):
        """Test bottleneck-specific recommendations."""
        # Test lines 688->694: bottleneck-specific recommendations
        efficiency = EfficiencyMetrics()

        # Create bottlenecks including regression-related ones
        bottlenecks = [
            "slow_regression_op (45.2% of total time)",
            "memory_intensive_op (high memory usage: 150.0 MB)",
        ]

        operation_metrics = {
            "test_op": PerformanceMetrics(
                operation_name="test_op", duration=0.1, memory_peak=50.0
            )
        }

        # Should generate bottleneck-specific recommendations (lines 688->694)
        recommendations = efficiency._generate_recommendations(
            data_size=1000,
            total_duration=0.1,
            operation_metrics=operation_metrics,
            bottlenecks=bottlenecks,
        )

        # Should include bottleneck-specific recommendations (lines 688->694)
        assert any(
            "Focus optimization efforts on identified bottlenecks" in rec
            for rec in recommendations
        )
        assert any(
            "optimized linear algebra libraries" in rec for rec in recommendations
        )

    def test_efficiency_metrics_recommendations_many_operations(self):
        """Test recommendations for many operations."""
        # Test line 695: recommendation for combining operations
        from src.tbr.utils.performance import PerformanceMetrics

        efficiency = EfficiencyMetrics()

        # Create many operation metrics (> 10)
        operation_metrics = {}
        for i in range(12):
            operation_metrics[f"op_{i}"] = PerformanceMetrics(
                operation_name=f"op_{i}", duration=0.01, memory_peak=10.0
            )

        # Should generate recommendation for combining operations (line 695)
        recommendations = efficiency._generate_recommendations(
            data_size=1000,
            total_duration=0.12,  # 12 * 0.01
            operation_metrics=operation_metrics,
            bottlenecks=[],
        )

        # Should include recommendation to combine operations (line 695)
        assert any(
            "combining related operations to reduce overhead" in rec
            for rec in recommendations
        )

    def test_efficiency_metrics_recommendations_optimal_performance(self):
        """Test default recommendation for optimal performance."""
        # Test line 698: default recommendation when no specific recommendations
        from src.tbr.utils.performance import PerformanceMetrics

        efficiency = EfficiencyMetrics()

        # Create optimal operation metrics (low memory, few operations, no bottlenecks)
        operation_metrics = {
            "optimal_op": PerformanceMetrics(
                operation_name="optimal_op",
                duration=0.001,  # Very fast
                memory_peak=50.0,  # Low memory (< 1000)
            )
        }

        # Should generate default optimal recommendation (line 698)
        recommendations = efficiency._generate_recommendations(
            data_size=1000,
            total_duration=0.001,
            operation_metrics=operation_metrics,  # Few operations, low memory
            bottlenecks=[],  # No bottlenecks
        )

        # Should include default optimal recommendation (line 698)
        assert any(
            "Performance appears optimal for current data size" in rec
            for rec in recommendations
        )

    def test_benchmark_function_with_memory_peaks(self):
        """Test benchmark function with memory peak tracking."""
        # Test memory peaks branch (lines 296-297, 311-319)
        profiler = PerformanceProfiler(enable_memory_tracking=True)

        def memory_intensive_func():
            # Create some memory usage
            data = list(range(1000))
            return len(data)

        with patch(
            "tracemalloc.get_traced_memory", return_value=(1024 * 1024, 2 * 1024 * 1024)
        ):
            stats = profiler.benchmark_function(
                memory_intensive_func, n_runs=2, warmup_runs=1
            )

        assert "mean_memory" in stats
        assert "max_memory" in stats
        assert "min_memory" in stats
        assert stats["mean_memory"] > 0

    def test_benchmark_function_without_memory_peaks(self):
        """Test benchmark function without memory peak data."""
        # Test branch when no memory peaks are available (lines 311-319)
        profiler = PerformanceProfiler(enable_memory_tracking=False)

        def simple_func():
            return 42

        stats = profiler.benchmark_function(simple_func, n_runs=2, warmup_runs=1)

        assert "mean_memory" not in stats
        assert "max_memory" not in stats
        assert "min_memory" not in stats

    def test_efficiency_metrics_system_monitoring_failure(self):
        """Test EfficiencyMetrics when system monitoring fails."""
        # Test system monitoring exception handling (lines 463, 509-513)
        efficiency = EfficiencyMetrics()

        # Create mock operation metrics
        mock_metrics = {
            "test_op": PerformanceMetrics(
                operation_name="test_op",
                duration=1.0,
                memory_peak=100.0,
                cpu_percent=50.0,
            )
        }

        with patch("psutil.virtual_memory", side_effect=Exception("System error")):
            with patch("psutil.cpu_percent", side_effect=Exception("CPU error")):
                report = efficiency.analyze_workflow_efficiency(
                    data_size=1000,
                    operation_metrics=mock_metrics,
                    operation_name="test_op",
                )

        # Should handle exceptions gracefully
        assert report.operation_name == "test_op"
        assert report.efficiency_score >= 0

    def test_performance_monitor_system_info_failure(self):
        """Test PerformanceMonitor when system info collection fails."""
        # Test system info collection failure (lines 570, 572, 592, 594)
        monitor = PerformanceMonitor()
        monitor.start_monitoring()

        # Mock the _take_sample method to avoid exceptions during stop_monitoring
        with patch.object(
            monitor, "_take_sample", side_effect=Exception("Sample failed")
        ):
            time.sleep(0.1)
            try:
                monitor.stop_monitoring()
            except Exception:
                pass  # Expected to fail gracefully

            # Should still be able to get a report (even if empty)
            report = monitor.get_monitoring_report()

        # Should handle exceptions gracefully
        assert isinstance(report, dict)

    def test_performance_monitor_no_data_collected(self):
        """Test PerformanceMonitor when no data is collected."""
        # Test empty monitoring data (lines 881-882)
        monitor = PerformanceMonitor()

        # Don't start monitoring, just try to get report
        report = monitor.get_monitoring_report()

        assert "error" in report
        assert report["error"] == "No monitoring data available"

    def test_efficiency_metrics_complexity_model_fitting_exceptions(self):
        """Test exception handling in complexity model fitting."""
        # Test lines 732-733, 741-742, 751-752: exception handling in _fit_complexity_models
        efficiency = EfficiencyMetrics()

        # Create data that will cause polyfit to fail (e.g., all same values)
        sizes = np.array([100, 100, 100])  # Same sizes will cause polyfit issues
        times = np.array(
            [1.0, 1.0, 1.0]
        )  # Same times will cause division by zero in R²

        # This should trigger exception handling in all three model types
        complexity_analysis = efficiency._fit_complexity_models(sizes, times)

        # Verify that exception handling worked (lines 732-733, 741-742, 751-752)
        assert "models" in complexity_analysis
        models = complexity_analysis["models"]

        # Linear model exception handling (lines 732-733)
        assert "linear" in models
        assert models["linear"]["r2"] == 0
        assert models["linear"]["coefficients"] == [0, 0]

        # Quadratic model exception handling (lines 741-742)
        assert "quadratic" in models
        assert models["quadratic"]["r2"] == 0
        assert models["quadratic"]["coefficients"] == [0, 0, 0]

        # Logarithmic model exception handling (lines 751-752)
        assert "logarithmic" in models
        assert models["logarithmic"]["r2"] == 0
        assert models["logarithmic"]["coefficients"] == [0, 0]

        # Should still return a best fit (even if all are 0)
        assert "best_fit" in complexity_analysis
        assert "best_r2" in complexity_analysis

    def test_efficiency_metrics_scaling_efficiency_insufficient_data(self):
        """Test scaling efficiency calculation with insufficient data."""
        # Test line 766: return 5.0 for insufficient data
        efficiency = EfficiencyMetrics()

        # Test with single data point (insufficient for scaling analysis)
        sizes = np.array([100])
        times = np.array([1.0])

        # Should return neutral score of 5.0 (line 766)
        efficiency_score = efficiency._calculate_scaling_efficiency(sizes, times)
        assert efficiency_score == 5.0

        # Test with empty arrays
        empty_sizes = np.array([])
        empty_times = np.array([])

        efficiency_score_empty = efficiency._calculate_scaling_efficiency(
            empty_sizes, empty_times
        )
        assert efficiency_score_empty == 5.0

    def test_efficiency_metrics_scaling_efficiency_score_ranges(self):
        """Test different efficiency score ranges in scaling efficiency calculation."""
        # Test lines 779-786: different efficiency scoring ranges
        # We'll use mock data to directly test the different branches
        efficiency = EfficiencyMetrics()

        # Test case 1: mean_efficiency >= 0.8 but < 1.0 (lines 779-780)
        # Use simple 2x scaling for sizes and slightly worse for times
        sizes1 = np.array([100, 200])
        times1 = np.array(
            [1.0, 2.2]
        )  # Slightly worse than 2x (efficiency = 2/2.2 = 0.91)

        efficiency_score1 = efficiency._calculate_scaling_efficiency(sizes1, times1)
        # Should be between 8.0 and 10.0 (lines 779-780)
        assert 8.0 <= efficiency_score1 < 10.0

        # Test case 2: mean_efficiency >= 0.5 but < 0.8 (lines 781-782)
        sizes2 = np.array([100, 200])
        times2 = np.array([1.0, 3.0])  # Much worse than 2x (efficiency = 2/3 = 0.67)

        efficiency_score2 = efficiency._calculate_scaling_efficiency(sizes2, times2)
        # Should be between 5.0 and 8.0 (lines 781-782)
        assert 5.0 <= efficiency_score2 < 8.0

        # Test case 3: mean_efficiency >= 0.2 but < 0.5 (lines 783-784)
        sizes3 = np.array([100, 200])
        times3 = np.array([1.0, 6.0])  # Poor scaling (efficiency = 2/6 = 0.33)

        efficiency_score3 = efficiency._calculate_scaling_efficiency(sizes3, times3)
        # Should be between 2.0 and 5.0 (lines 783-784)
        assert 2.0 <= efficiency_score3 < 5.0

        # Test case 4: mean_efficiency < 0.2 (lines 785-786)
        sizes4 = np.array([100, 200])
        times4 = np.array([1.0, 12.0])  # Very poor scaling (efficiency = 2/12 = 0.17)

        efficiency_score4 = efficiency._calculate_scaling_efficiency(sizes4, times4)
        # Should be between 0.0 and 2.0 (lines 785-786)
        assert 0.0 <= efficiency_score4 < 2.0

    def test_performance_monitor_initialization_exceptions(self):
        """Test PerformanceMonitor initialization with psutil exceptions."""
        # Test lines 826-828: Exception handling during psutil.Process initialization
        import psutil

        # Test NoSuchProcess exception (lines 826-828)
        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(pid=123)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                monitor = PerformanceMonitor()

                assert monitor.process is None
                assert len(w) == 1
                assert "System monitoring disabled due to access limitations" in str(
                    w[0].message
                )

        # Test AccessDenied exception (lines 826-828)
        with patch("psutil.Process", side_effect=psutil.AccessDenied(pid=123)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                monitor = PerformanceMonitor()

                assert monitor.process is None
                assert len(w) == 1
                assert "System monitoring disabled due to access limitations" in str(
                    w[0].message
                )

    def test_performance_monitor_start_monitoring_no_process(self):
        """Test start_monitoring when process is None."""
        # Test lines 833-834: Warning and return when process is None
        monitor = PerformanceMonitor()
        monitor.process = None  # Simulate no process available

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor.start_monitoring()

            # Should issue warning and return early (lines 833-834)
            assert len(w) == 1
            assert "Cannot start monitoring: system access unavailable" in str(
                w[0].message
            )
            assert monitor.monitoring is False  # Should not have started monitoring

    def test_performance_monitor_stop_monitoring_conditional_sample(self):
        """Test stop_monitoring conditional final sample taking."""
        # Test lines 845->847: Conditional final sample taking in stop_monitoring
        monitor = PerformanceMonitor()

        # Test case 1: monitoring is True, should take final sample (line 845->846)
        monitor.monitoring = True
        with patch.object(monitor, "_take_sample") as mock_take_sample:
            monitor.stop_monitoring()

            # Should have called _take_sample for final sample (lines 845->846)
            mock_take_sample.assert_called_once()
            assert monitor.monitoring is False

        # Test case 2: monitoring is False, should not take final sample
        monitor.monitoring = False
        with patch.object(monitor, "_take_sample") as mock_take_sample:
            monitor.stop_monitoring()

            # Should not have called _take_sample (line 845 condition is False)
            mock_take_sample.assert_not_called()
            assert monitor.monitoring is False

    def test_performance_monitor_take_sample_no_process(self):
        """Test _take_sample when process is None."""
        # Test line 852: Return when process is None in _take_sample
        monitor = PerformanceMonitor()
        monitor.process = None  # Simulate no process available

        # Should return early without doing anything (line 852)
        monitor._take_sample()

        # Should not have added any samples
        assert len(monitor.samples) == 0

    def test_performance_monitor_take_sample_psutil_exceptions(self):
        """Test _take_sample with psutil exceptions."""
        # Test lines 869-870: Exception handling for psutil operations in _take_sample
        import psutil

        monitor = PerformanceMonitor()

        # Mock a process that will raise exceptions
        mock_process = Mock()
        mock_process.cpu_percent.side_effect = psutil.NoSuchProcess(pid=123)
        monitor.process = mock_process
        monitor.start_time = time.time()

        # Should handle NoSuchProcess exception gracefully (lines 869-870)
        monitor._take_sample()

        # Should not have added any samples due to exception
        assert len(monitor.samples) == 0

        # Test with AccessDenied exception
        mock_process2 = Mock()
        mock_process2.cpu_percent.side_effect = psutil.AccessDenied(pid=123)
        monitor.process = mock_process2

        # Should handle AccessDenied exception gracefully (lines 869-870)
        monitor._take_sample()

        # Should still not have added any samples due to exception
        assert len(monitor.samples) == 0

    def test_performance_monitor_memory_alerts(self):
        """Test memory-related alerts in monitoring report."""
        # Test lines 915, 917: Memory usage alerts in get_monitoring_report
        monitor = PerformanceMonitor()

        # Create mock samples with high memory usage to trigger alerts
        mock_samples = [
            {
                "timestamp": 0.0,
                "cpu_percent": 50.0,
                "memory_mb": 1000.0,
                "memory_percent": 90.0,  # High memory usage (>85%) - should trigger line 915
                "system_cpu_percent": 60.0,
                "system_memory_percent": 95.0,  # High system memory (>90%) - should trigger line 917
            },
            {
                "timestamp": 1.0,
                "cpu_percent": 55.0,
                "memory_mb": 1100.0,
                "memory_percent": 88.0,  # Still high memory usage
                "system_cpu_percent": 65.0,
                "system_memory_percent": 92.0,  # Still high system memory
            },
        ]

        monitor.samples = mock_samples

        # Get monitoring report - should trigger both memory alerts (lines 915, 917)
        report = monitor.get_monitoring_report()

        # Verify both memory alerts are present
        alerts = report["alerts"]
        assert len(alerts) >= 2

        # Check for high memory usage alert (line 915)
        memory_alert_found = any(
            "High memory usage detected (>85%)" in alert for alert in alerts
        )
        assert memory_alert_found, f"Expected memory usage alert not found in: {alerts}"

        # Check for system memory pressure alert (line 917)
        system_memory_alert_found = any(
            "System memory pressure detected (>90%)" in alert for alert in alerts
        )
        assert (
            system_memory_alert_found
        ), f"Expected system memory pressure alert not found in: {alerts}"

    def test_benchmark_tbr_functions_with_list_tuple_data(self):
        """Test benchmark_tbr_functions with list/tuple test_data."""
        # Test line 997: Benchmark function call with unpacked test_data when it's a list/tuple
        from src.tbr.utils.performance import benchmark_tbr_functions

        # Create simple test functions
        def test_func1(x, y):
            return x + y

        def test_func2(x, y):
            return x * y

        functions = {"add_func": test_func1, "multiply_func": test_func2}

        # Test with list test_data (should trigger line 997)
        list_test_data = [10, 20]  # Will be unpacked as *test_data

        results = benchmark_tbr_functions(functions, list_test_data, n_runs=2)

        # Verify results for both functions
        assert "add_func" in results
        assert "multiply_func" in results

        # Check that functions were called correctly with unpacked arguments
        add_result = results["add_func"]
        multiply_result = results["multiply_func"]

        # Both should have successful results (not errors)
        assert "error" not in add_result
        assert "error" not in multiply_result

        # Verify the actual function results are correct
        assert add_result["result"] == 30  # 10 + 20
        assert multiply_result["result"] == 200  # 10 * 20

        # Test with tuple test_data (should also trigger line 997)
        tuple_test_data = (5, 15)  # Will be unpacked as *test_data

        results_tuple = benchmark_tbr_functions(functions, tuple_test_data, n_runs=2)

        # Verify results for tuple case
        assert "add_func" in results_tuple
        assert "multiply_func" in results_tuple

        add_result_tuple = results_tuple["add_func"]
        multiply_result_tuple = results_tuple["multiply_func"]

        assert "error" not in add_result_tuple
        assert "error" not in multiply_result_tuple
        assert add_result_tuple["result"] == 20  # 5 + 15
        assert multiply_result_tuple["result"] == 75  # 5 * 15

    def test_benchmark_tbr_functions_exception_handling(self):
        """Test benchmark_tbr_functions exception handling."""
        # Test lines 1005-1006: Exception handling when functions throw exceptions
        from src.tbr.utils.performance import benchmark_tbr_functions

        # Create functions that will throw exceptions
        def working_func(x):
            return x * 2

        def failing_func(_x):
            raise ValueError("This function always fails")

        def zero_division_func(_x):
            return 1 / 0  # Will raise ZeroDivisionError

        functions = {
            "working_function": working_func,
            "failing_function": failing_func,
            "zero_division_function": zero_division_func,
        }

        # Test with simple test data
        test_data = 10

        results = benchmark_tbr_functions(functions, test_data, n_runs=2)

        # Verify results for all functions
        assert "working_function" in results
        assert "failing_function" in results
        assert "zero_division_function" in results

        # Working function should succeed
        working_result = results["working_function"]
        assert "error" not in working_result
        assert working_result["result"] == 20  # 10 * 2

        # Failing functions should have error entries (lines 1005-1006)
        failing_result = results["failing_function"]
        assert "error" in failing_result
        assert "This function always fails" in failing_result["error"]

        zero_div_result = results["zero_division_function"]
        assert "error" in zero_div_result
        assert "division by zero" in zero_div_result["error"].lower()

        # Test with list test_data and exception
        def failing_func_with_args(x, y):
            raise RuntimeError(f"Failed with args {x}, {y}")

        functions_with_args = {"failing_with_args": failing_func_with_args}

        list_test_data = [5, 10]

        results_with_args = benchmark_tbr_functions(
            functions_with_args, list_test_data, n_runs=2
        )

        # Should handle exception even with unpacked arguments (lines 1005-1006)
        failing_with_args_result = results_with_args["failing_with_args"]
        assert "error" in failing_with_args_result
        assert "Failed with args 5, 10" in failing_with_args_result["error"]


if __name__ == "__main__":
    pytest.main([__file__])
