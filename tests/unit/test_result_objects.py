"""
Comprehensive tests for TBR result object structures.

Tests cover all result classes (TBRPredictionResult, TBRSummaryResult,
TBRSubintervalResult) with validation of:
- Object creation and immutability
- Attribute access and types
- Conversion methods (to_dict, to_dataframe)
- Convenience methods (is_significant, contains_zero, etc.)
- String representations
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest

from tbr.core.results import TBRPredictionResult, TBRSubintervalResult, TBRSummaryResult


class TestTBRPredictionResult:
    """Tests for TBRPredictionResult class."""

    @pytest.fixture
    def sample_predictions_df(self):
        """Create sample predictions DataFrame."""
        return pd.DataFrame(
            {
                "pred": [100.0, 105.0, 110.0],
                "predsd": [5.0, 5.2, 5.4],
            }
        )

    @pytest.fixture
    def sample_model_params(self):
        """Create sample model parameters."""
        return {
            "alpha": 10.0,
            "beta": 0.95,
            "sigma": 8.5,
            "degrees_freedom": 45,
        }

    @pytest.fixture
    def sample_control_values(self):
        """Create sample control values."""
        return np.array([95.0, 100.0, 105.0])

    @pytest.fixture
    def prediction_result(
        self, sample_predictions_df, sample_model_params, sample_control_values
    ):
        """Create sample TBRPredictionResult."""
        return TBRPredictionResult(
            predictions=sample_predictions_df,
            n_predictions=3,
            model_params=sample_model_params,
            control_values=sample_control_values,
        )

    def test_prediction_result_creation(self, prediction_result):
        """Test TBRPredictionResult object creation."""
        assert isinstance(prediction_result, TBRPredictionResult)
        assert prediction_result.n_predictions == 3

    def test_prediction_result_attributes(self, prediction_result, sample_model_params):
        """Test TBRPredictionResult attribute access."""
        assert isinstance(prediction_result.predictions, pd.DataFrame)
        assert len(prediction_result.predictions) == 3
        assert list(prediction_result.predictions.columns) == ["pred", "predsd"]
        assert prediction_result.model_params == sample_model_params
        assert isinstance(prediction_result.control_values, np.ndarray)
        assert len(prediction_result.control_values) == 3

    def test_prediction_result_immutability(self, prediction_result):
        """Test that TBRPredictionResult is immutable (frozen)."""
        with pytest.raises(AttributeError):
            prediction_result.n_predictions = 5

    def test_prediction_result_to_dict(self, prediction_result):
        """Test TBRPredictionResult.to_dict() method."""
        result_dict = prediction_result.to_dict()
        assert isinstance(result_dict, dict)
        assert "predictions" in result_dict
        assert "n_predictions" in result_dict
        assert "model_params" in result_dict
        assert "control_values" in result_dict
        assert result_dict["n_predictions"] == 3

    def test_prediction_result_repr(self, prediction_result):
        """Test TBRPredictionResult.__repr__() method."""
        repr_str = repr(prediction_result)
        assert "TBRPredictionResult" in repr_str
        assert "n_predictions=3" in repr_str
        assert "mean_pred" in repr_str
        assert "mean_uncertainty" in repr_str

    def test_prediction_result_with_single_prediction(
        self, sample_model_params, sample_control_values
    ):
        """Test TBRPredictionResult with single prediction."""
        single_pred_df = pd.DataFrame({"pred": [100.0], "predsd": [5.0]})
        result = TBRPredictionResult(
            predictions=single_pred_df,
            n_predictions=1,
            model_params=sample_model_params,
            control_values=sample_control_values[:1],
        )
        assert result.n_predictions == 1
        assert len(result.predictions) == 1


class TestTBRSummaryResult:
    """Tests for TBRSummaryResult class."""

    @pytest.fixture
    def summary_result(self):
        """Create sample TBRSummaryResult."""
        return TBRSummaryResult(
            estimate=125.5,
            lower=95.2,
            upper=155.8,
            se=23.5,
            prob=0.978,
            precision=30.3,
            level=0.80,
            threshold=0.0,
            alpha=10.5,
            beta=0.95,
            sigma=8.5,
            var_alpha=4.2,
            var_beta=0.001,
            cov_alpha_beta=-0.05,
            degrees_freedom=45,
        )

    def test_summary_result_creation(self, summary_result):
        """Test TBRSummaryResult object creation."""
        assert isinstance(summary_result, TBRSummaryResult)
        assert summary_result.estimate == 125.5

    def test_summary_result_attributes(self, summary_result):
        """Test TBRSummaryResult attribute access."""
        assert summary_result.estimate == 125.5
        assert summary_result.lower == 95.2
        assert summary_result.upper == 155.8
        assert summary_result.se == 23.5
        assert summary_result.prob == 0.978
        assert summary_result.precision == 30.3
        assert summary_result.level == 0.80
        assert summary_result.threshold == 0.0
        assert summary_result.alpha == 10.5
        assert summary_result.beta == 0.95
        assert summary_result.sigma == 8.5
        assert summary_result.var_alpha == 4.2
        assert summary_result.var_beta == 0.001
        assert summary_result.cov_alpha_beta == -0.05
        assert summary_result.degrees_freedom == 45

    def test_summary_result_immutability(self, summary_result):
        """Test that TBRSummaryResult is immutable (frozen)."""
        with pytest.raises(AttributeError):
            summary_result.estimate = 200.0

    def test_summary_result_to_dict(self, summary_result):
        """Test TBRSummaryResult.to_dict() method."""
        result_dict = summary_result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["estimate"] == 125.5
        assert result_dict["lower"] == 95.2
        assert result_dict["upper"] == 155.8
        assert result_dict["prob"] == 0.978
        assert len(result_dict) == 15  # All 15 fields

    def test_summary_result_to_dataframe(self, summary_result):
        """Test TBRSummaryResult.to_dataframe() method."""
        result_df = summary_result.to_dataframe()
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1  # Single row
        assert result_df["estimate"].iloc[0] == 125.5
        assert result_df["prob"].iloc[0] == 0.978

    def test_summary_result_is_significant_true(self, summary_result):
        """Test TBRSummaryResult.is_significant() returns True."""
        assert summary_result.is_significant(probability_threshold=0.95) is True

    def test_summary_result_is_significant_false(self, summary_result):
        """Test TBRSummaryResult.is_significant() returns False."""
        assert summary_result.is_significant(probability_threshold=0.99) is False

    def test_summary_result_is_significant_custom_threshold(self, summary_result):
        """Test TBRSummaryResult.is_significant() with custom threshold."""
        assert summary_result.is_significant(probability_threshold=0.975) is True
        assert summary_result.is_significant(probability_threshold=0.98) is False

    def test_summary_result_repr(self, summary_result):
        """Test TBRSummaryResult.__repr__() method."""
        repr_str = repr(summary_result)
        assert "TBRSummaryResult" in repr_str
        assert "estimate=" in repr_str
        assert "CI=" in repr_str
        assert "prob=" in repr_str
        assert "0.978" in repr_str

    def test_summary_result_negative_effect(self):
        """Test TBRSummaryResult with negative effect."""
        result = TBRSummaryResult(
            estimate=-50.0,
            lower=-75.0,
            upper=-25.0,
            se=12.5,
            prob=0.025,
            precision=25.0,
            level=0.80,
            threshold=0.0,
            alpha=10.0,
            beta=0.95,
            sigma=8.0,
            var_alpha=4.0,
            var_beta=0.001,
            cov_alpha_beta=-0.05,
            degrees_freedom=40,
        )
        assert result.estimate < 0
        assert result.prob < 0.5  # Low probability of exceeding threshold


class TestTBRSubintervalResult:
    """Tests for TBRSubintervalResult class."""

    @pytest.fixture
    def subinterval_result(self):
        """Create sample TBRSubintervalResult."""
        return TBRSubintervalResult(
            estimate=45.2,
            lower=30.1,
            upper=60.3,
            se=11.8,
            ci_level=0.80,
            start_day=1,
            end_day=7,
            n_days=7,
        )

    def test_subinterval_result_creation(self, subinterval_result):
        """Test TBRSubintervalResult object creation."""
        assert isinstance(subinterval_result, TBRSubintervalResult)
        assert subinterval_result.estimate == 45.2

    def test_subinterval_result_attributes(self, subinterval_result):
        """Test TBRSubintervalResult attribute access."""
        assert subinterval_result.estimate == 45.2
        assert subinterval_result.lower == 30.1
        assert subinterval_result.upper == 60.3
        assert subinterval_result.se == 11.8
        assert subinterval_result.ci_level == 0.80
        assert subinterval_result.start_day == 1
        assert subinterval_result.end_day == 7
        assert subinterval_result.n_days == 7

    def test_subinterval_result_immutability(self, subinterval_result):
        """Test that TBRSubintervalResult is immutable (frozen)."""
        with pytest.raises(AttributeError):
            subinterval_result.estimate = 50.0

    def test_subinterval_result_to_dict(self, subinterval_result):
        """Test TBRSubintervalResult.to_dict() method."""
        result_dict = subinterval_result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["estimate"] == 45.2
        assert result_dict["start_day"] == 1
        assert result_dict["end_day"] == 7
        assert result_dict["n_days"] == 7
        assert len(result_dict) == 8  # All 8 fields

    def test_subinterval_result_contains_zero_false(self, subinterval_result):
        """Test TBRSubintervalResult.contains_zero() returns False."""
        assert subinterval_result.contains_zero() is False

    def test_subinterval_result_contains_zero_true(self):
        """Test TBRSubintervalResult.contains_zero() returns True."""
        result = TBRSubintervalResult(
            estimate=5.0,
            lower=-10.0,
            upper=20.0,
            se=9.0,
            ci_level=0.80,
            start_day=1,
            end_day=7,
            n_days=7,
        )
        assert result.contains_zero() is True

    def test_subinterval_result_is_positive_true(self, subinterval_result):
        """Test TBRSubintervalResult.is_positive() returns True."""
        assert subinterval_result.is_positive() is True

    def test_subinterval_result_is_positive_false(self):
        """Test TBRSubintervalResult.is_positive() returns False."""
        result = TBRSubintervalResult(
            estimate=-5.0,
            lower=-15.0,
            upper=5.0,
            se=6.0,
            ci_level=0.80,
            start_day=1,
            end_day=7,
            n_days=7,
        )
        assert result.is_positive() is False

    def test_subinterval_result_is_negative_false(self, subinterval_result):
        """Test TBRSubintervalResult.is_negative() returns False."""
        assert subinterval_result.is_negative() is False

    def test_subinterval_result_is_negative_true(self):
        """Test TBRSubintervalResult.is_negative() returns True."""
        result = TBRSubintervalResult(
            estimate=-25.0,
            lower=-40.0,
            upper=-10.0,
            se=9.0,
            ci_level=0.80,
            start_day=1,
            end_day=7,
            n_days=7,
        )
        assert result.is_negative() is True

    def test_subinterval_result_repr(self, subinterval_result):
        """Test TBRSubintervalResult.__repr__() method."""
        repr_str = repr(subinterval_result)
        assert "TBRSubintervalResult" in repr_str
        assert "days=1-7" in repr_str
        assert "n=7" in repr_str
        assert "estimate=" in repr_str
        assert "CI=" in repr_str

    def test_subinterval_result_single_day(self):
        """Test TBRSubintervalResult for single day interval."""
        result = TBRSubintervalResult(
            estimate=10.0,
            lower=5.0,
            upper=15.0,
            se=3.0,
            ci_level=0.80,
            start_day=5,
            end_day=5,
            n_days=1,
        )
        assert result.start_day == result.end_day
        assert result.n_days == 1

    def test_subinterval_result_long_interval(self):
        """Test TBRSubintervalResult for long interval."""
        result = TBRSubintervalResult(
            estimate=200.0,
            lower=150.0,
            upper=250.0,
            se=30.0,
            ci_level=0.95,
            start_day=1,
            end_day=30,
            n_days=30,
        )
        assert result.n_days == 30
        assert result.ci_level == 0.95


class TestResultObjectsIntegration:
    """Integration tests for result objects."""

    def test_all_result_types_frozen(self):
        """Test that all result classes are frozen dataclasses."""
        # Create instances
        pred_result = TBRPredictionResult(
            predictions=pd.DataFrame({"pred": [100], "predsd": [5]}),
            n_predictions=1,
            model_params={"alpha": 10, "beta": 0.95},
            control_values=np.array([95]),
        )

        summary_result = TBRSummaryResult(
            estimate=100,
            lower=80,
            upper=120,
            se=15,
            prob=0.95,
            precision=20,
            level=0.80,
            threshold=0.0,
            alpha=10,
            beta=0.95,
            sigma=8,
            var_alpha=4,
            var_beta=0.001,
            cov_alpha_beta=-0.05,
            degrees_freedom=40,
        )

        subinterval_result = TBRSubintervalResult(
            estimate=50,
            lower=40,
            upper=60,
            se=8,
            ci_level=0.80,
            start_day=1,
            end_day=7,
            n_days=7,
        )

        # Try to modify (should fail)
        with pytest.raises(AttributeError):
            pred_result.n_predictions = 2

        with pytest.raises(AttributeError):
            summary_result.estimate = 200

        with pytest.raises(AttributeError):
            subinterval_result.n_days = 10

    def test_result_objects_with_numpy_types(self):
        """Test result objects work with numpy scalar types."""
        result = TBRSubintervalResult(
            estimate=np.float64(45.2),
            lower=np.float64(30.1),
            upper=np.float64(60.3),
            se=np.float64(11.8),
            ci_level=0.80,
            start_day=np.int64(1),
            end_day=np.int64(7),
            n_days=np.int64(7),
        )
        assert result.estimate == 45.2
        assert result.start_day == 1
        assert result.n_days == 7
