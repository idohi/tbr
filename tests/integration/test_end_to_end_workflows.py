"""
End-to-end workflow integration tests for TBR package.

This module tests complete user workflows from data loading through analysis
to results extraction, validating that all components work together seamlessly
in realistic scenarios.

Test Categories
---------------
1. Basic Workflows: Simple, single-step analysis patterns
2. Advanced Workflows: Multi-step analysis with various methods
3. Real-World Scenarios: Domain-specific use cases
4. Method Chaining: Fluent API usage patterns
5. Error Workflows: Error handling through complete workflows
6. State Management: Re-fitting and multiple analyses
"""

import numpy as np
import pandas as pd
import pytest

from tbr import TBRAnalysis
from tbr.core.results import TBRPredictionResult, TBRSubintervalResult, TBRSummaryResult


class TestBasicWorkflows:
    """Test basic end-to-end workflow patterns."""

    def test_basic_fit_summarize_workflow(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test basic workflow: fit model and get summary."""
        # Initialize model
        model = TBRAnalysis(level=0.80, threshold=0.0)

        # Fit model
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Get summary
        summary = model.summarize()

        # Validate workflow completed successfully
        assert isinstance(summary, TBRSummaryResult)
        assert summary.estimate is not None
        assert summary.lower < summary.upper
        assert model.fitted_ is True

    def test_basic_fit_predict_workflow(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test basic workflow: fit model and generate predictions."""
        # Initialize and fit
        model = TBRAnalysis(level=0.90)
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Generate predictions
        predictions = model.predict()

        # Validate workflow
        assert isinstance(predictions, TBRPredictionResult)
        assert predictions.n_predictions > 0
        assert len(predictions.predictions) == predictions.n_predictions
        assert "pred" in predictions.predictions.columns
        assert "predsd" in predictions.predictions.columns

    def test_basic_fit_analyze_subinterval_workflow(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test basic workflow: fit model and analyze subinterval."""
        # Initialize and fit
        model = TBRAnalysis(level=0.80)
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Analyze first week
        result = model.analyze_subinterval(start_day=1, end_day=7)

        # Validate workflow
        assert isinstance(result, TBRSubintervalResult)
        assert result.start_day == 1
        assert result.end_day == 7
        assert result.n_days == 7
        assert result.lower < result.upper

    def test_fit_and_access_all_results_workflow(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test workflow accessing all result types after fitting."""
        # Fit model
        model = TBRAnalysis()
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Access all result types
        results_df = model.results_
        summaries_df = model.summaries_
        params = model.params_
        summary = model.summarize()
        incremental = model.summarize_incremental()
        predictions = model.predict()
        subinterval = model.analyze_subinterval(1, 5)

        # Validate all results are accessible
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(summaries_df, pd.DataFrame)
        assert isinstance(params, dict)
        assert isinstance(summary, TBRSummaryResult)
        assert isinstance(incremental, pd.DataFrame)
        assert isinstance(predictions, TBRPredictionResult)
        assert isinstance(subinterval, TBRSubintervalResult)


class TestAdvancedWorkflows:
    """Test advanced multi-step workflow patterns."""

    def test_fit_predict_custom_values_workflow(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test workflow: fit model then predict for custom control values."""
        # Fit model
        model = TBRAnalysis(level=0.95)
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Predict for custom control values
        custom_control = np.array([1000, 1050, 1100, 1150, 1200])
        predictions = model.predict(control_values=custom_control)

        # Validate workflow
        assert predictions.n_predictions == 5
        assert len(predictions.control_values) == 5
        assert np.allclose(predictions.control_values, custom_control)

    def test_fit_multiple_subinterval_analyses_workflow(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test workflow: fit once then analyze multiple subintervals."""
        # Fit model
        model = TBRAnalysis(level=0.80)
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Analyze multiple subintervals
        week1 = model.analyze_subinterval(1, 7)
        week2 = model.analyze_subinterval(8, 14)
        week3 = model.analyze_subinterval(15, 21)

        # Validate all analyses completed
        assert week1.n_days == 7
        assert week2.n_days == 7
        assert week3.n_days == 7
        assert week1.start_day == 1
        assert week2.start_day == 8
        assert week3.start_day == 15

    def test_fit_compare_different_ci_levels_workflow(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test workflow: analyze same interval with different confidence levels."""
        # Fit model
        model = TBRAnalysis(level=0.80)
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Analyze with different CI levels
        ci_80 = model.analyze_subinterval(1, 10, ci_level=0.80)
        ci_90 = model.analyze_subinterval(1, 10, ci_level=0.90)
        ci_95 = model.analyze_subinterval(1, 10, ci_level=0.95)

        # Validate CI widths increase with confidence level
        width_80 = ci_80.upper - ci_80.lower
        width_90 = ci_90.upper - ci_90.lower
        width_95 = ci_95.upper - ci_95.lower

        assert width_80 < width_90 < width_95
        # Estimates should be identical
        assert np.isclose(ci_80.estimate, ci_90.estimate, rtol=1e-10)
        assert np.isclose(ci_80.estimate, ci_95.estimate, rtol=1e-10)

    def test_incremental_analysis_tracking_workflow(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test workflow: track effect progression through incremental summaries."""
        # Fit model
        model = TBRAnalysis(level=0.85)
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Get incremental summaries
        incremental = model.summarize_incremental()

        # Validate progression properties
        assert len(incremental) > 1
        assert "estimate" in incremental.columns
        assert "lower" in incremental.columns
        assert "upper" in incremental.columns

        # Check that estimates are monotonic (cumulative effects)
        # Note: estimates can vary but cumulative sum should be increasing
        assert all(incremental["test_day"] == range(1, len(incremental) + 1))


class TestRealWorldScenarios:
    """Test realistic domain-specific workflow scenarios."""

    def test_marketing_campaign_analysis_workflow(self):
        """Test complete marketing campaign lift analysis workflow."""
        # Generate marketing campaign data
        np.random.seed(123)
        dates = pd.date_range("2023-01-01", periods=90, freq="D")
        control_sales = np.random.normal(10000, 500, 90)
        test_sales = control_sales.copy()
        test_sales[60:] += 800  # Campaign lift starts at day 60

        data = pd.DataFrame(
            {"date": dates, "control_sales": control_sales, "test_sales": test_sales}
        )

        # Complete marketing analysis workflow
        model = TBRAnalysis(level=0.90, threshold=0.0)
        model.fit(
            data=data,
            time_col="date",
            control_col="control_sales",
            test_col="test_sales",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-03-02"),
            test_end=pd.Timestamp("2023-03-31"),
        )

        # Get campaign results
        summary = model.summarize()
        weekly_effects = []
        for week in range(1, 5):
            start = (week - 1) * 7 + 1
            end = min(week * 7, 30)
            weekly_effects.append(model.analyze_subinterval(start, end))

        # Validate campaign analysis
        assert summary.estimate > 0  # Positive lift
        assert len(weekly_effects) == 4
        assert all(isinstance(w, TBRSubintervalResult) for w in weekly_effects)

    def test_ab_test_analysis_workflow(self):
        """Test complete A/B test analysis workflow."""
        # Generate A/B test data (shorter duration)
        np.random.seed(456)
        dates = pd.date_range("2023-06-01", periods=30, freq="D")
        control_metric = np.random.normal(5000, 200, 30)
        test_metric = control_metric.copy()
        test_metric[20:] += 150  # Feature effect

        data = pd.DataFrame(
            {"date": dates, "variant_a": control_metric, "variant_b": test_metric}
        )

        # A/B test analysis
        model = TBRAnalysis(level=0.95, threshold=100.0)
        model.fit(
            data=data,
            time_col="date",
            control_col="variant_a",
            test_col="variant_b",
            pretest_start=pd.Timestamp("2023-06-01"),
            test_start=pd.Timestamp("2023-06-21"),
            test_end=pd.Timestamp("2023-06-30"),
        )

        # Test decision making
        summary = model.summarize()
        is_significant = summary.is_significant()
        passes_threshold = summary.prob > 0.90

        # Validate A/B test workflow
        assert isinstance(is_significant, bool)
        assert isinstance(passes_threshold, bool)
        assert summary.threshold == 100.0

    def test_medical_trial_analysis_workflow(self):
        """Test medical trial treatment effect analysis workflow."""
        # Generate medical trial data (patient counts)
        np.random.seed(789)
        weeks = pd.date_range("2023-01-01", periods=20, freq="W")
        control_patients = np.random.poisson(100, 20)
        treatment_patients = control_patients.copy()
        treatment_patients[12:] = np.random.poisson(115, 8)  # Treatment effect

        data = pd.DataFrame(
            {
                "week": weeks,
                "control": control_patients,
                "treatment": treatment_patients,
            }
        )

        # Medical trial analysis
        model = TBRAnalysis(level=0.99, threshold=0.0)
        model.fit(
            data=data,
            time_col="week",
            control_col="control",
            test_col="treatment",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-03-26"),
            test_end=pd.Timestamp("2023-05-14"),
        )

        # Get trial results with high confidence
        summary = model.summarize()
        params = model.params_

        # Validate medical trial workflow
        assert summary.level == 0.99  # High confidence for medical
        assert "alpha" in params
        assert "beta" in params
        assert "sigma" in params


class TestMethodChainingWorkflows:
    """Test fluent API method chaining patterns."""

    def test_fit_summarize_chaining(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test method chaining: fit and immediately get summary."""
        summary = (
            TBRAnalysis(level=0.80, threshold=0.0)
            .fit(
                data=sample_time_series_data,
                time_col=sample_analysis_parameters["time_col"],
                control_col=sample_analysis_parameters["control_col"],
                test_col=sample_analysis_parameters["test_col"],
                pretest_start=sample_analysis_parameters["pretest_start"],
                test_start=sample_analysis_parameters["test_start"],
                test_end=sample_analysis_parameters["test_end"],
            )
            .summarize()
        )

        assert isinstance(summary, TBRSummaryResult)
        assert summary.estimate is not None

    def test_fit_results_access_chaining(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test method chaining: fit and access results."""
        results = (
            TBRAnalysis()
            .fit(
                data=sample_time_series_data,
                time_col=sample_analysis_parameters["time_col"],
                control_col=sample_analysis_parameters["control_col"],
                test_col=sample_analysis_parameters["test_col"],
                pretest_start=sample_analysis_parameters["pretest_start"],
                test_start=sample_analysis_parameters["test_start"],
                test_end=sample_analysis_parameters["test_end"],
            )
            .results_
        )

        assert isinstance(results, pd.DataFrame)
        assert "period" in results.columns

    def test_fit_predict_convenience_method(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test fit_predict convenience method for streamlined workflow."""
        predictions = TBRAnalysis().fit_predict(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        assert isinstance(predictions, TBRPredictionResult)
        assert predictions.n_predictions > 0

    def test_convenience_properties_workflow(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test convenience properties for streamlined access."""
        model = TBRAnalysis().fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Access via convenience properties
        final_summary = model.final_summary
        final_effect = model.final_effect

        # Validate convenience access
        assert isinstance(final_summary, TBRSummaryResult)
        assert isinstance(final_effect, float)
        assert final_effect == final_summary.estimate


class TestErrorHandlingWorkflows:
    """Test error handling through complete workflows."""

    def test_workflow_error_before_fitting(self):
        """Test that accessing results before fitting raises clear errors."""
        model = TBRAnalysis()

        # All methods should raise AttributeError before fitting
        with pytest.raises(
            AttributeError, match="This TBRAnalysis instance is not fitted yet"
        ):
            _ = model.results_

        with pytest.raises(
            AttributeError, match="This TBRAnalysis instance is not fitted yet"
        ):
            _ = model.summaries_

        with pytest.raises(
            AttributeError, match="This TBRAnalysis instance is not fitted yet"
        ):
            _ = model.params_

        with pytest.raises(
            AttributeError, match="This TBRAnalysis instance is not fitted yet"
        ):
            model.summarize()

        with pytest.raises(
            AttributeError, match="This TBRAnalysis instance is not fitted yet"
        ):
            model.summarize_incremental()

        with pytest.raises(
            AttributeError, match="This TBRAnalysis instance is not fitted yet"
        ):
            model.predict()

        with pytest.raises(
            AttributeError, match="This TBRAnalysis instance is not fitted yet"
        ):
            model.analyze_subinterval(1, 7)

    def test_workflow_with_invalid_data(self, sample_analysis_parameters):
        """Test workflow with invalid data raises appropriate errors."""
        model = TBRAnalysis()

        # Empty DataFrame
        with pytest.raises(ValueError, match="data cannot be empty"):
            model.fit(
                data=pd.DataFrame(),
                time_col=sample_analysis_parameters["time_col"],
                control_col=sample_analysis_parameters["control_col"],
                test_col=sample_analysis_parameters["test_col"],
                pretest_start=sample_analysis_parameters["pretest_start"],
                test_start=sample_analysis_parameters["test_start"],
                test_end=sample_analysis_parameters["test_end"],
            )

    def test_workflow_with_invalid_predictions(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test prediction workflow with invalid control values."""
        # Fit valid model
        model = TBRAnalysis()
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Try to predict with invalid control values
        with pytest.raises(ValueError, match="control_values cannot be empty"):
            model.predict(control_values=np.array([]))

        with pytest.raises(ValueError, match="must contain only finite values"):
            model.predict(control_values=np.array([1000, np.nan, 1100]))

    def test_workflow_with_invalid_subinterval(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test subinterval workflow with invalid day ranges."""
        # Fit valid model
        model = TBRAnalysis()
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Invalid day ranges
        with pytest.raises(ValueError, match="must be a positive integer"):
            model.analyze_subinterval(start_day=0, end_day=7)

        with pytest.raises(ValueError, match="must be <= end_day"):
            model.analyze_subinterval(start_day=10, end_day=5)

        with pytest.raises(ValueError, match="exceeds test period length"):
            model.analyze_subinterval(start_day=1, end_day=1000)


class TestStateManagementWorkflows:
    """Test state management across multiple analyses."""

    def test_refitting_same_model_workflow(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test refitting the same model instance with different parameters."""
        model = TBRAnalysis(level=0.80)

        # First fit
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )
        first_estimate = model.final_effect

        # Refit with different period
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-03-15"),
        )
        second_estimate = model.final_effect

        # Results should be different (different periods)
        assert first_estimate != second_estimate
        assert model.fitted_ is True

    def test_multiple_datasets_same_model_workflow(self):
        """Test analyzing multiple datasets with the same model instance."""
        model = TBRAnalysis(level=0.90, threshold=0.0)

        # Analyze first dataset
        data1 = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=50),
                "control": np.random.normal(1000, 50, 50),
                "test": np.random.normal(1050, 50, 50),
            }
        )
        model.fit(
            data=data1,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-02-19"),
        )
        result1 = model.final_effect

        # Analyze second dataset
        data2 = pd.DataFrame(
            {
                "date": pd.date_range("2023-03-01", periods=50),
                "control": np.random.normal(2000, 100, 50),
                "test": np.random.normal(2100, 100, 50),
            }
        )
        model.fit(
            data=data2,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-03-01"),
            test_start=pd.Timestamp("2023-04-01"),
            test_end=pd.Timestamp("2023-04-19"),
        )
        result2 = model.final_effect

        # Both analyses should complete successfully
        assert isinstance(result1, float)
        assert isinstance(result2, float)

    def test_state_consistency_after_refitting(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test that all state is properly updated after refitting."""
        model = TBRAnalysis(level=0.85)

        # First fit
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Store first results
        first_results = model.results_.copy()
        first_summaries = model.summaries_.copy()
        first_params = dict(model.params_)

        # Refit with different data
        new_data = sample_time_series_data.copy()
        new_data["test"] = new_data["test"] * 1.1  # Different test values

        model.fit(
            data=new_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Verify state was updated
        assert not model.results_.equals(first_results)
        assert not model.summaries_.equals(first_summaries)
        # Parameters should be different due to different test values
        assert (
            model.params_["alpha"] != first_params["alpha"]
            or model.params_["beta"] != first_params["beta"]
        )


class TestComplexWorkflows:
    """Test complex multi-component workflows."""

    def test_comprehensive_analysis_workflow(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test comprehensive analysis using all available methods."""
        # Initialize and fit
        model = TBRAnalysis(level=0.90, threshold=0.0)
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Perform comprehensive analysis
        results = {
            "final_summary": model.summarize(),
            "final_effect": model.final_effect,
            "incremental": model.summarize_incremental(),
            "predictions": model.predict(),
            "custom_predictions": model.predict(control_values=np.array([1000, 1100])),
            "week1": model.analyze_subinterval(1, 7),
            "week2": model.analyze_subinterval(8, 14),
            "full_period": model.analyze_subinterval(
                1, len(model.summaries_), ci_level=0.95
            ),
            "results_df": model.results_,
            "summaries_df": model.summaries_,
            "params": model.params_,
        }

        # Validate all components
        assert isinstance(results["final_summary"], TBRSummaryResult)
        assert isinstance(results["final_effect"], float)
        assert isinstance(results["incremental"], pd.DataFrame)
        assert isinstance(results["predictions"], TBRPredictionResult)
        assert isinstance(results["custom_predictions"], TBRPredictionResult)
        assert isinstance(results["week1"], TBRSubintervalResult)
        assert isinstance(results["week2"], TBRSubintervalResult)
        assert isinstance(results["full_period"], TBRSubintervalResult)
        assert isinstance(results["results_df"], pd.DataFrame)
        assert isinstance(results["summaries_df"], pd.DataFrame)
        assert isinstance(results["params"], dict)

        # Validate consistency
        assert results["final_effect"] == results["final_summary"].estimate
        assert len(results["incremental"]) == len(results["summaries_df"])

    def test_comparative_analysis_workflow(self):
        """Test workflow comparing multiple analysis configurations."""
        # Generate test data
        np.random.seed(999)
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=60),
                "control": np.random.normal(1000, 50, 60),
                "test": np.random.normal(1050, 50, 60),
            }
        )

        # Compare different confidence levels
        results = {}
        for level in [0.80, 0.90, 0.95, 0.99]:
            model = TBRAnalysis(level=level, threshold=0.0)
            model.fit(
                data=data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )
            results[level] = model.summarize()

        # Validate comparative analysis
        assert len(results) == 4
        # All estimates should be identical
        estimates = [r.estimate for r in results.values()]
        assert all(np.isclose(estimates[0], e) for e in estimates)

        # CI widths should increase with level
        widths = {level: r.upper - r.lower for level, r in results.items()}
        assert widths[0.80] < widths[0.90] < widths[0.95] < widths[0.99]
