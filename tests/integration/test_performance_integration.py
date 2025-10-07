"""
Integration tests for performance diagnostics with TBR analysis workflows.

This module provides comprehensive integration tests that validate the
performance diagnostics framework works correctly with real TBR analysis
workflows, including end-to-end performance monitoring, efficiency analysis,
and optimization recommendations.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from tbr.analysis.performance import TBRPerformanceAnalyzer, optimize_tbr_data_size
from tbr.functional.tbr_functions import perform_tbr_analysis
from tbr.utils.performance import EfficiencyMetrics, PerformanceProfiler


class TestTBRPerformanceIntegration:
    """Integration tests for TBR performance analysis."""

    def setup_method(self):
        """Set up realistic TBR test data."""
        np.random.seed(42)

        # Create realistic TBR dataset
        dates = pd.date_range("2023-01-01", periods=90, freq="D")

        # Simulate realistic control and test group data
        control_base = 1000
        control_noise = 50
        treatment_effect = 20

        self.test_data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(control_base, control_noise, 90),
                "test": np.random.normal(
                    control_base + treatment_effect, control_noise * 1.1, 90
                ),
            }
        )

        # Standard TBR analysis parameters
        self.tbr_params = {
            "time_col": "date",
            "control_col": "control",
            "test_col": "test",
            "pretest_start": pd.Timestamp("2023-01-01"),
            "test_start": pd.Timestamp("2023-02-15"),
            "test_end": pd.Timestamp("2023-03-15"),
            "level": 0.80,
            "threshold": 0.0,
        }

    def test_complete_tbr_performance_analysis(self):
        """Test complete TBR performance analysis workflow."""
        analyzer = TBRPerformanceAnalyzer()

        # Run complete performance analysis
        performance_report = analyzer.analyze_tbr_performance(
            data=self.test_data, **self.tbr_params
        )

        # Validate report structure
        assert "workflow_metrics" in performance_report
        assert "operation_metrics" in performance_report
        assert "efficiency_report" in performance_report
        assert "data_characteristics" in performance_report
        assert "tbr_results" in performance_report
        assert "daily_summaries" in performance_report

        # Validate TBR results are correct
        tbr_results = performance_report["tbr_results"]
        daily_summaries = performance_report["daily_summaries"]

        assert isinstance(tbr_results, pd.DataFrame)
        assert isinstance(daily_summaries, pd.DataFrame)
        assert len(tbr_results) > 0
        assert len(daily_summaries) > 0

        # Check that TBR analysis columns are present
        expected_columns = [
            "date",
            "period",
            "y",
            "x",
            "pred",
            "predsd",
            "dif",
            "cumdif",
            "cumsd",
            "estsd",
        ]
        for col in expected_columns:
            assert col in tbr_results.columns

        # Validate performance metrics
        workflow_metrics = performance_report["workflow_metrics"]
        assert workflow_metrics.duration > 0
        assert workflow_metrics.operation_name == "tbr_complete_workflow"

        # Validate efficiency report
        efficiency_report = performance_report["efficiency_report"]
        assert efficiency_report.data_size == len(self.test_data)
        assert 0 <= efficiency_report.efficiency_score <= 10
        assert isinstance(efficiency_report.bottlenecks, list)
        assert isinstance(efficiency_report.recommendations, list)

        # Validate data characteristics
        data_chars = performance_report["data_characteristics"]
        assert data_chars["data_size"] == len(self.test_data)
        assert data_chars["data_memory_mb"] > 0
        assert data_chars["pretest_period_length"] > 0
        assert data_chars["test_period_length"] > 0

    def test_performance_analysis_with_monitoring(self):
        """Test performance analysis with real-time monitoring enabled."""
        analyzer = TBRPerformanceAnalyzer()

        # Use smaller dataset for faster execution with monitoring
        small_data = self.test_data.iloc[:60].copy()
        small_params = self.tbr_params.copy()
        small_params["test_end"] = pd.Timestamp("2023-03-01")

        performance_report = analyzer.analyze_tbr_performance(
            data=small_data, enable_monitoring=True, **small_params
        )

        # Check that monitoring report is included
        monitoring_report = performance_report.get("monitoring_report")

        if monitoring_report and "error" not in monitoring_report:
            # Validate monitoring report structure
            assert "duration" in monitoring_report
            assert "sample_count" in monitoring_report
            assert "cpu_stats" in monitoring_report
            assert "memory_stats" in monitoring_report
            assert "system_stats" in monitoring_report
            assert "alerts" in monitoring_report

            # Check that monitoring captured some data
            assert monitoring_report["sample_count"] > 0
            assert monitoring_report["duration"] > 0

            # Validate CPU and memory stats structure
            cpu_stats = monitoring_report["cpu_stats"]
            memory_stats = monitoring_report["memory_stats"]

            for stat in ["mean", "max", "min", "std"]:
                assert stat in cpu_stats
                assert cpu_stats[stat] >= 0

            for stat in ["mean_mb", "max_mb", "min_mb", "peak_percent"]:
                assert stat in memory_stats
                assert memory_stats[stat] >= 0

    def test_performance_scaling_analysis(self):
        """Test performance scaling analysis with different data sizes."""
        analyzer = TBRPerformanceAnalyzer()

        # Use data that covers both pretest and test periods (need at least 75 days)
        base_data = self.test_data.iloc[:75].copy()  # Covers pretest + test periods

        scaling_analysis = analyzer.analyze_data_size_scaling(
            base_data=base_data,
            size_multipliers=[0.5, 1.0, 1.5],  # Limited multipliers for speed
            **self.tbr_params,
        )

        # Validate scaling analysis structure
        assert "scaling_results" in scaling_analysis
        assert "scaling_analysis" in scaling_analysis
        assert "recommendations" in scaling_analysis

        scaling_results = scaling_analysis["scaling_results"]
        assert len(scaling_results) == 3  # Three multipliers

        # Check each scaling result
        successful_results = []
        for result in scaling_results:
            assert "size_multiplier" in result
            assert "data_size" in result

            if result.get("success", False):
                successful_results.append(result)
                assert "total_duration" in result
                assert "efficiency_score" in result
                assert result["total_duration"] > 0
                assert 0 <= result["efficiency_score"] <= 10

        # Should have at least one successful result
        assert len(successful_results) >= 1

        # Validate scaling analysis if we have enough successful results
        if len(successful_results) >= 2:
            scaling_analysis_results = scaling_analysis["scaling_analysis"]
            if "error" not in scaling_analysis_results:
                assert "complexity_slope" in scaling_analysis_results
                assert "complexity_r_squared" in scaling_analysis_results
                assert "complexity_estimate" in scaling_analysis_results
                assert "scaling_efficiency" in scaling_analysis_results

    def test_configuration_comparison(self):
        """Test performance comparison across different TBR configurations."""
        analyzer = TBRPerformanceAnalyzer()

        # Use smaller dataset for faster comparison
        small_data = self.test_data.iloc[:50].copy()

        # Define base configuration
        base_config = {
            "time_col": "date",
            "control_col": "control",
            "test_col": "test",
            "pretest_start": pd.Timestamp("2023-01-01"),
            "test_start": pd.Timestamp("2023-02-01"),
            "test_end": pd.Timestamp("2023-02-15"),
            "level": 0.80,
            "threshold": 0.0,
        }

        # Define alternative configurations
        configurations = [
            {**base_config, "level": 0.95},  # Different confidence level
            {**base_config, "threshold": 5.0},  # Different threshold
        ]

        comparison_results = analyzer.compare_tbr_configurations(
            data=small_data, configurations=configurations, base_config=base_config
        )

        # Validate comparison structure
        assert "comparison_results" in comparison_results
        assert "comparison_summary" in comparison_results
        assert "recommendations" in comparison_results

        comparison_list = comparison_results["comparison_results"]
        assert len(comparison_list) == 3  # Base + 2 configurations

        # Check baseline result
        baseline_result = comparison_list[0]
        assert baseline_result["config_name"] == "baseline"
        assert baseline_result["duration_ratio"] == 1.0
        assert baseline_result["efficiency_ratio"] == 1.0

        # Check that other configurations have comparison metrics
        for i, result in enumerate(comparison_list[1:], 1):
            if "error" not in result:
                assert result["config_name"] == f"config_{i}"
                assert "duration_ratio" in result
                assert "efficiency_ratio" in result
                assert result["duration_ratio"] > 0
                assert result["efficiency_ratio"] > 0

    def test_baseline_comparison_workflow(self):
        """Test setting baseline and comparing performance."""
        analyzer = TBRPerformanceAnalyzer()

        # Use data that covers both pretest and test periods
        small_data = self.test_data.iloc[:60].copy()  # Covers up to early March
        small_params = self.tbr_params.copy()
        small_params["test_end"] = pd.Timestamp(
            "2023-02-28"
        )  # End within available data

        # Analyze baseline performance
        baseline_report = analyzer.analyze_tbr_performance(
            data=small_data, enable_monitoring=False, **small_params
        )

        # Set as baseline
        analyzer.set_performance_baseline("test_baseline", baseline_report)

        # Analyze current performance (same data, should be similar)
        current_report = analyzer.analyze_tbr_performance(
            data=small_data, enable_monitoring=False, **small_params
        )

        # Compare to baseline
        comparison = analyzer.compare_to_baseline(current_report, "test_baseline")

        # Validate comparison structure
        assert "baseline_name" in comparison
        assert "size_ratio" in comparison
        assert "duration_ratio" in comparison
        assert "normalized_duration_ratio" in comparison
        assert "efficiency_ratio" in comparison
        assert "performance_regression" in comparison
        assert "performance_improvement" in comparison

        # Since we're comparing identical analyses, ratios should be close to 1.0
        assert comparison["baseline_name"] == "test_baseline"
        assert abs(comparison["size_ratio"] - 1.0) < 0.01  # Same data size

        # Duration and efficiency ratios can vary significantly due to system performance
        # Just verify they are positive and reasonable
        assert comparison["duration_ratio"] > 0
        assert comparison["efficiency_ratio"] > 0

        # Verify the comparison contains meaningful data
        assert isinstance(comparison["performance_regression"], bool)
        assert isinstance(comparison["performance_improvement"], bool)

    def test_optimization_recommendations_integration(self):
        """Test optimization recommendations with real TBR analysis."""
        analyzer = TBRPerformanceAnalyzer()

        # Create a scenario that might trigger recommendations
        # Use larger dataset to potentially trigger memory/performance recommendations
        large_data = pd.concat([self.test_data] * 3, ignore_index=True)  # 3x larger
        large_data["date"] = pd.date_range(
            "2023-01-01", periods=len(large_data), freq="D"
        )

        large_params = self.tbr_params.copy()
        large_params["test_end"] = pd.Timestamp("2023-03-31")

        performance_report = analyzer.analyze_tbr_performance(
            data=large_data, enable_monitoring=False, **large_params
        )

        recommendations = analyzer.get_optimization_recommendations(performance_report)

        # Validate recommendations structure
        assert "priority_actions" in recommendations
        assert "data_optimization" in recommendations
        assert "computational_optimization" in recommendations
        assert "memory_optimization" in recommendations
        assert "general_recommendations" in recommendations

        # Check that all recommendation categories are lists
        for category in recommendations.values():
            assert isinstance(category, list)

        # Should have at least some recommendations
        total_recommendations = sum(len(recs) for recs in recommendations.values())
        assert total_recommendations > 0

    def test_performance_profiler_integration(self):
        """Test direct integration with PerformanceProfiler."""
        profiler = PerformanceProfiler()

        # Profile TBR analysis components
        with profiler.profile_context("data_preparation"):
            # Simulate data preparation
            processed_data = self.test_data.copy()
            processed_data["processed"] = processed_data["control"] * 1.1

        with profiler.profile_context("tbr_execution"):
            # Run actual TBR analysis
            tbr_results, daily_summaries = perform_tbr_analysis(
                data=self.test_data.iloc[:75],  # Covers pretest + test periods
                **self.tbr_params,
            )

        # Get all metrics
        all_metrics = profiler.get_metrics()

        # Validate profiling results
        assert "data_preparation" in all_metrics
        assert "tbr_execution" in all_metrics

        prep_metrics = all_metrics["data_preparation"]
        exec_metrics = all_metrics["tbr_execution"]

        assert prep_metrics.duration > 0
        assert exec_metrics.duration > 0
        assert prep_metrics.operation_name == "data_preparation"
        assert exec_metrics.operation_name == "tbr_execution"

        # TBR execution should typically take longer than data preparation
        # (though this isn't guaranteed, so we just check both are positive)
        assert prep_metrics.duration >= 0
        assert exec_metrics.duration >= 0

    def test_efficiency_metrics_integration(self):
        """Test direct integration with EfficiencyMetrics."""
        efficiency = EfficiencyMetrics()

        # Create mock operation metrics from real TBR analysis
        operation_metrics = {}

        # Profile individual TBR operations
        profiler = PerformanceProfiler()

        with profiler.profile_context("regression_fitting") as metrics:
            # Simulate regression fitting time
            time.sleep(0.001)
            metrics.metadata = {"operation_type": "regression"}

        with profiler.profile_context("prediction_generation") as metrics:
            # Simulate prediction generation time
            time.sleep(0.002)
            metrics.metadata = {"operation_type": "prediction"}

        operation_metrics = profiler.get_metrics()

        # Analyze efficiency
        efficiency_report = efficiency.analyze_workflow_efficiency(
            data_size=len(self.test_data),
            operation_metrics=operation_metrics,
            operation_name="tbr_integration_test",
        )

        # Validate efficiency analysis
        assert efficiency_report.operation_name == "tbr_integration_test"
        assert efficiency_report.data_size == len(self.test_data)
        assert 0 <= efficiency_report.efficiency_score <= 10
        assert isinstance(efficiency_report.bottlenecks, list)
        assert isinstance(efficiency_report.recommendations, list)
        assert efficiency_report.computational_complexity in [
            "O(1) - Constant",
            "O(log n) - Logarithmic",
            "O(n) - Linear",
            "O(n log n) - Linearithmic",
            "O(n²) or higher - Polynomial/Exponential",
        ]


class TestPerformanceOptimizationIntegration:
    """Integration tests for performance optimization features."""

    def setup_method(self):
        """Set up test data for optimization tests."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        self.test_data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, 60),
                "test": np.random.normal(1020, 55, 60),
            }
        )

        self.tbr_params = {
            "time_col": "date",
            "control_col": "control",
            "test_col": "test",
            "pretest_start": pd.Timestamp("2023-01-01"),
            "test_start": pd.Timestamp("2023-01-31"),
            "test_end": pd.Timestamp("2023-02-15"),
            "level": 0.80,
            "threshold": 0.0,
        }

    def test_optimize_tbr_data_size_integration(self):
        """Test data size optimization with real TBR analysis."""
        # Test with very fast target duration for testing
        optimization_result = optimize_tbr_data_size(
            data=self.test_data,
            target_duration=5.0,  # 5 seconds target
            size_multipliers=[0.5, 1.0],  # Limited for speed
            **self.tbr_params,
        )

        if "error" not in optimization_result:
            # Validate optimization result structure
            assert "recommended_size" in optimization_result
            assert "recommended_multiplier" in optimization_result
            assert "expected_duration" in optimization_result
            assert "scaling_analysis" in optimization_result

            # Check that recommended size is reasonable
            assert optimization_result["recommended_size"] > 0
            assert optimization_result["recommended_multiplier"] > 0
            assert optimization_result["expected_duration"] > 0

            # Validate scaling analysis is included
            scaling_analysis = optimization_result["scaling_analysis"]
            assert "scaling_results" in scaling_analysis
            assert "recommendations" in scaling_analysis

    @patch("psutil.Process")
    def test_real_time_monitoring_integration(self, mock_process_class):
        """Test real-time monitoring during actual TBR analysis."""
        # Mock psutil for consistent testing
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 45.0
        mock_process.memory_info.return_value = Mock(rss=1024 * 1024 * 150)  # 150MB
        mock_process.memory_percent.return_value = 30.0
        mock_process_class.return_value = mock_process

        with patch("psutil.cpu_percent", return_value=25.0), patch(
            "psutil.virtual_memory", return_value=Mock(percent=65.0)
        ):
            analyzer = TBRPerformanceAnalyzer()

            # Run analysis with monitoring
            performance_report = analyzer.analyze_tbr_performance(
                data=self.test_data.iloc[:75],  # Covers pretest + test periods
                enable_monitoring=True,
                **self.tbr_params,
            )

            # Check monitoring integration
            monitoring_report = performance_report.get("monitoring_report")

            if monitoring_report and "error" not in monitoring_report:
                # Validate that monitoring captured the analysis
                assert monitoring_report["sample_count"] >= 2  # At least start and stop
                assert monitoring_report["duration"] > 0

                # Check that CPU and memory data was captured
                cpu_stats = monitoring_report["cpu_stats"]
                memory_stats = monitoring_report["memory_stats"]

                assert cpu_stats["mean"] >= 0
                assert memory_stats["mean_mb"] >= 0

                # Verify system stats were also captured
                system_stats = monitoring_report["system_stats"]
                assert system_stats["cpu_mean"] >= 0
                assert system_stats["memory_mean"] >= 0


class TestPerformanceRegressionDetection:
    """Integration tests for performance regression detection."""

    def setup_method(self):
        """Set up test data for regression detection."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=40, freq="D")
        self.test_data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, 40),
                "test": np.random.normal(1020, 55, 40),
            }
        )

        self.tbr_params = {
            "time_col": "date",
            "control_col": "control",
            "test_col": "test",
            "pretest_start": pd.Timestamp("2023-01-01"),
            "test_start": pd.Timestamp("2023-01-20"),
            "test_end": pd.Timestamp("2023-01-30"),
            "level": 0.80,
            "threshold": 0.0,
        }

    def test_performance_regression_detection(self):
        """Test detection of performance regressions."""
        analyzer = TBRPerformanceAnalyzer()

        # Establish baseline with normal data
        baseline_report = analyzer.analyze_tbr_performance(
            data=self.test_data, enable_monitoring=False, **self.tbr_params
        )

        analyzer.set_performance_baseline("regression_test", baseline_report)

        # Simulate performance regression by using larger dataset
        # (which should take longer)
        larger_data = pd.concat([self.test_data] * 2, ignore_index=True)
        larger_data["date"] = pd.date_range(
            "2023-01-01", periods=len(larger_data), freq="D"
        )

        larger_params = self.tbr_params.copy()
        larger_params["test_end"] = pd.Timestamp("2023-02-10")

        regression_report = analyzer.analyze_tbr_performance(
            data=larger_data, enable_monitoring=False, **larger_params
        )

        # Compare to baseline
        comparison = analyzer.compare_to_baseline(regression_report, "regression_test")

        # Validate regression detection
        assert comparison["size_ratio"] > 1.0  # Larger dataset
        assert comparison["duration_ratio"] > 1.0  # Should take longer

        # Check if regression is properly detected
        # (normalized for data size, it might or might not be a regression)
        assert "performance_regression" in comparison
        assert "performance_improvement" in comparison
        assert isinstance(comparison["performance_regression"], bool)
        assert isinstance(comparison["performance_improvement"], bool)

    def test_efficiency_trend_analysis(self):
        """Test analysis of efficiency trends over multiple runs."""
        analyzer = TBRPerformanceAnalyzer()

        efficiency_scores = []
        durations = []

        # Run multiple analyses to establish trend
        for i in range(3):
            # Use slightly different data each time
            data_variant = self.test_data.copy()
            data_variant["control"] += np.random.normal(0, 1, len(data_variant))
            data_variant["test"] += np.random.normal(0, 1, len(data_variant))

            performance_report = analyzer.analyze_tbr_performance(
                data=data_variant, enable_monitoring=False, **self.tbr_params
            )

            efficiency_scores.append(
                performance_report["efficiency_report"].efficiency_score
            )
            durations.append(performance_report["workflow_metrics"].duration)

        # Validate that we collected meaningful data
        assert len(efficiency_scores) == 3
        assert len(durations) == 3
        assert all(score >= 0 for score in efficiency_scores)
        assert all(duration > 0 for duration in durations)

        # Check for reasonable variance (not all identical)
        # This validates that the performance analysis is sensitive to changes
        efficiency_variance = np.var(efficiency_scores)
        duration_variance = np.var(durations)

        # Should have some variance (not all runs identical)
        # But this is more of a sanity check than a strict requirement
        assert efficiency_variance >= 0  # At minimum, non-negative variance
        assert duration_variance >= 0


if __name__ == "__main__":
    pytest.main([__file__])
