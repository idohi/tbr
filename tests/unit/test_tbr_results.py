"""
Unit tests for TBRResults class.

This module tests all properties, methods, and functionality of the TBRResults
result object returned by perform_tbr_analysis.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from tbr.core.results import TBRResults
from tbr.functional import perform_tbr_analysis


class SlottedTBRResults(TBRResults):
    """Test-only subclass with state outside ``__dict__``."""

    __slots__ = ("custom_metadata",)


@pytest.fixture
def sample_analysis_data():
    """Create sample data for TBR analysis."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    control = np.random.normal(1000, 50, 60)
    test = np.random.normal(1020, 55, 60)

    data = pd.DataFrame(
        {
            "date": dates,
            "control": control,
            "test": test,
        }
    )

    return data


@pytest.fixture
def tbr_results(sample_analysis_data):
    """Create TBRResults object for testing."""
    results = perform_tbr_analysis(
        data=sample_analysis_data,
        time_col="date",
        control_col="control",
        test_col="test",
        pretest_start=pd.Timestamp("2023-01-01"),
        test_start=pd.Timestamp("2023-02-01"),
        test_end=pd.Timestamp("2023-02-28"),
        level=0.80,
        threshold=0.0,
    )
    return results


class TestTBRResultsPickleCompatibility:
    """Test explicit same-version and cross-version pickle behavior."""

    def test_same_version_pickle_round_trip(self, tbr_results):
        """Round-trip results through the current pickle format."""
        assert tbr_results.__getstate__()["_tbr_pickle_version"] == 1

        restored = pickle.loads(pickle.dumps(tbr_results))

        pd.testing.assert_frame_equal(restored.summary(), tbr_results.summary())
        assert restored.estimate == tbr_results.estimate

    def test_rejects_custom_attributes_before_serialization(self, tbr_results):
        """Fail before emitting pickle state that restoration would reject."""
        tbr_results.custom_metadata = {"owner": "example"}

        with pytest.raises(
            ValueError,
            match="Cannot serialize TBRResults.*unexpected fields",
        ):
            pickle.dumps(tbr_results)

    def test_rejects_slotted_subclass_before_serialization(self, tbr_results):
        """Reject subclass state that canonical serialization cannot preserve."""
        subclassed = SlottedTBRResults.__new__(SlottedTBRResults)
        subclassed.__dict__.update(tbr_results.__dict__)
        subclassed.custom_metadata = {"owner": "example"}

        with pytest.raises(
            ValueError,
            match="Cannot serialize TBRResults subclass",
        ):
            pickle.dumps(subclassed)

    def test_rejects_unversioned_pickle_state(self, tbr_results):
        """Reject pre-0.2 result state before restoring legacy schemas."""
        unversioned_state = dict(tbr_results.__dict__)
        restored = type(tbr_results).__new__(type(tbr_results))

        with pytest.raises(
            ValueError,
            match=r"pre-0\.2 pickle.*recreate or refit",
        ):
            restored.__setstate__(unversioned_state)

    @pytest.mark.parametrize("invalid_version", [True, 1.0, "1"])
    def test_rejects_non_integer_pickle_format_version(
        self, tbr_results, invalid_version
    ):
        """Reject format versions that are not exact integers."""
        state = tbr_results.__getstate__()
        state["_tbr_pickle_version"] = invalid_version
        restored = type(tbr_results).__new__(type(tbr_results))

        with pytest.raises(
            ValueError,
            match="invalid TBRResults pickle format version",
        ):
            restored.__setstate__(state)

    def test_rejects_unsupported_pickle_format_version(self, tbr_results):
        """Distinguish unsupported future formats from pre-0.2 state."""
        state = tbr_results.__getstate__()
        state["_tbr_pickle_version"] = 2
        restored = type(tbr_results).__new__(type(tbr_results))

        with pytest.raises(
            ValueError,
            match=r"unsupported TBRResults pickle format version 2",
        ):
            restored.__setstate__(state)

    @pytest.mark.parametrize("invalid_state", [None, []])
    def test_rejects_non_dictionary_pickle_state(self, tbr_results, invalid_state):
        """Reject malformed state before attempting version validation."""
        restored = type(tbr_results).__new__(type(tbr_results))

        with pytest.raises(
            ValueError,
            match="malformed TBRResults pickle state.*dictionary",
        ):
            restored.__setstate__(invalid_state)

    @pytest.mark.parametrize("invalid_state_kind", ["missing", "unknown"])
    def test_rejects_malformed_versioned_pickle_state(
        self, tbr_results, invalid_state_kind
    ):
        """Reject current-version state with missing or unknown fields."""
        state = tbr_results.__getstate__()
        if invalid_state_kind == "missing":
            state.pop("_summary_df")
        else:
            state["unknown"] = None
        restored = type(tbr_results).__new__(type(tbr_results))

        with pytest.raises(
            ValueError,
            match="malformed TBRResults pickle state",
        ):
            restored.__setstate__(state)


class TestTBRResultsTimeSeriesProperties:
    """Test time series property accessors."""

    def test_control_property(self, tbr_results):
        """Test control property returns correct Series."""
        control = tbr_results.control
        assert isinstance(control, pd.Series)
        assert control.name == "control"
        assert len(control) > 0
        assert not control.isnull().any()

    def test_test_property(self, tbr_results):
        """Test test property returns correct Series."""
        test = tbr_results.test
        assert isinstance(test, pd.Series)
        assert test.name == "test"
        assert len(test) > 0
        assert not test.isnull().any()

    def test_fittedvalues_property(self, tbr_results):
        """Test fittedvalues property returns correct Series."""
        fitted = tbr_results.fittedvalues
        assert isinstance(fitted, pd.Series)
        assert fitted.name == "fittedvalues"
        assert len(fitted) > 0
        assert not fitted.isnull().any()

    def test_predictions_property(self, tbr_results):
        """Test predictions property returns correct Series."""
        predictions = tbr_results.predictions
        assert isinstance(predictions, pd.Series)
        assert predictions.name == "predictions"
        assert len(predictions) > 0
        assert not predictions.isnull().any()

    def test_resid_property(self, tbr_results):
        """Test resid property returns correct Series."""
        resid = tbr_results.resid
        assert isinstance(resid, pd.Series)
        assert resid.name == "resid"
        assert len(resid) > 0
        # Residuals can have any values including NaN in some cases

    def test_effects_property(self, tbr_results):
        """Test effects property returns correct Series."""
        effects = tbr_results.effects
        assert isinstance(effects, pd.Series)
        assert effects.name == "effects"
        assert len(effects) > 0
        assert not effects.isnull().any()

    def test_cumulative_effect_property(self, tbr_results):
        """Test cumulative_effect property returns correct Series."""
        cumulative = tbr_results.cumulative_effect
        assert isinstance(cumulative, pd.Series)
        assert cumulative.name == "cumulative_effect"
        assert len(cumulative) > 0
        assert not cumulative.isnull().any()
        # Cumulative should be monotonic (approximately)
        assert cumulative.iloc[-1] == cumulative.values[-1]

    def test_prediction_se_property(self, tbr_results):
        """Test prediction_se property returns correct Series."""
        pred_se = tbr_results.prediction_se
        assert isinstance(pred_se, pd.Series)
        assert pred_se.name == "prediction_se"
        assert len(pred_se) > 0
        assert not pred_se.isnull().any()
        assert (pred_se > 0).all()

    def test_cumulative_se_property(self, tbr_results):
        """Test cumulative_se property returns correct Series."""
        cum_se = tbr_results.cumulative_se
        assert isinstance(cum_se, pd.Series)
        assert cum_se.name == "cumulative_se"
        assert len(cum_se) > 0
        assert not cum_se.isnull().any()
        assert (cum_se > 0).all()


class TestTBRResultsScalarProperties:
    """Test scalar property accessors."""

    def test_estimate_property(self, tbr_results):
        """Test estimate property returns final cumulative effect."""
        estimate = tbr_results.estimate
        assert isinstance(estimate, float)
        # Should match last value of cumulative_effect
        assert estimate == tbr_results.cumulative_effect.iloc[-1]

    def test_conf_int_lower_property(self, tbr_results):
        """Test conf_int_lower property returns lower bound."""
        lower = tbr_results.conf_int_lower
        assert isinstance(lower, float)
        # Lower bound should be less than estimate
        assert lower < tbr_results.estimate

    def test_conf_int_upper_property(self, tbr_results):
        """Test conf_int_upper property returns upper bound."""
        upper = tbr_results.conf_int_upper
        assert isinstance(upper, float)
        # Upper bound should be greater than estimate
        assert upper > tbr_results.estimate

    def test_pvalue_property(self, tbr_results):
        """Test pvalue property returns probability."""
        pvalue = tbr_results.pvalue
        assert isinstance(pvalue, float)
        assert 0.0 <= pvalue <= 1.0

    def test_n_pretest_property(self, tbr_results):
        """Test n_pretest property returns count."""
        n_pretest = tbr_results.n_pretest
        assert isinstance(n_pretest, int)
        assert n_pretest > 0

    def test_n_test_property(self, tbr_results):
        """Test n_test property returns count."""
        n_test = tbr_results.n_test
        assert isinstance(n_test, int)
        assert n_test > 0

    def test_n_test_days_property_is_removed(self, tbr_results):
        """Test that the removed day-specific count property is unavailable."""
        assert not hasattr(tbr_results, "n_test_days")
        with pytest.raises(AttributeError):
            _ = tbr_results.n_test_days


class TestTBRResultsModelParameters:
    """Test model parameter property accessors."""

    def test_alpha_property(self, tbr_results):
        """Test alpha property returns intercept."""
        alpha = tbr_results.alpha
        assert isinstance(alpha, float)
        assert np.isfinite(alpha)

    def test_beta_property(self, tbr_results):
        """Test beta property returns slope."""
        beta = tbr_results.beta
        assert isinstance(beta, float)
        assert np.isfinite(beta)

    def test_sigma_property(self, tbr_results):
        """Test sigma property returns residual std."""
        sigma = tbr_results.sigma
        assert isinstance(sigma, float)
        assert sigma > 0
        assert np.isfinite(sigma)

    def test_model_params_property(self, tbr_results):
        """Test model_params property returns complete dict."""
        params = tbr_results.model_params
        assert isinstance(params, dict)

        # Check all required keys
        required_keys = [
            "alpha",
            "beta",
            "sigma",
            "var_alpha",
            "var_beta",
            "cov_alpha_beta",
            "degrees_freedom",
            "n_pretest",
        ]
        for key in required_keys:
            assert key in params

        # Should be a copy (modifying shouldn't affect original)
        original_alpha = params["alpha"]
        params["alpha"] = 999.0
        assert tbr_results.alpha == original_alpha


class TestTBRResultsMethods:
    """Test TBRResults methods."""

    def test_summary_method(self, tbr_results):
        """Test summary method returns DataFrame."""
        summary = tbr_results.summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0

        # Check for expected columns
        expected_cols = ["estimate", "lower", "upper", "prob"]
        for col in expected_cols:
            assert col in summary.columns

    def test_tbr_dataframe_method(self, tbr_results):
        """Test tbr_dataframe method returns comprehensive DataFrame."""
        tbr_df = tbr_results.tbr_dataframe()
        assert isinstance(tbr_df, pd.DataFrame)
        assert len(tbr_df) > 0

        # Check for expected columns
        expected_cols = ["period", "x", "y", "pred", "predsd", "dif", "cumdif", "cumsd"]
        for col in expected_cols:
            assert col in tbr_df.columns

    def test_conf_int_default_level(self, tbr_results):
        """Test conf_int with default level."""
        ci = tbr_results.conf_int()
        assert isinstance(ci, pd.DataFrame)
        assert "lower" in ci.columns
        assert "upper" in ci.columns
        assert len(ci) == 1

        # Should match property values
        assert ci["lower"].iloc[0] == tbr_results.conf_int_lower
        assert ci["upper"].iloc[0] == tbr_results.conf_int_upper

    def test_conf_int_custom_level(self, tbr_results):
        """Test conf_int with custom level."""
        # Test with 95% confidence level (different from default 80%)
        ci_95 = tbr_results.conf_int(level=0.95)
        assert isinstance(ci_95, pd.DataFrame)
        assert "lower" in ci_95.columns
        assert "upper" in ci_95.columns
        assert len(ci_95) == 1

        # 95% CI should be wider than 80% CI
        ci_80 = tbr_results.conf_int(level=0.80)
        assert ci_95["upper"].iloc[0] > ci_80["upper"].iloc[0]
        assert ci_95["lower"].iloc[0] < ci_80["lower"].iloc[0]

    def test_conf_int_different_levels(self, tbr_results):
        """Test conf_int with various confidence levels."""
        levels = [0.50, 0.80, 0.90, 0.95, 0.99]

        for level in levels:
            ci = tbr_results.conf_int(level=level)
            assert isinstance(ci, pd.DataFrame)
            assert ci["lower"].iloc[0] < tbr_results.estimate
            assert ci["upper"].iloc[0] > tbr_results.estimate

    def test_repr_method(self, tbr_results):
        """Test __repr__ method returns string."""
        repr_str = repr(tbr_results)
        assert isinstance(repr_str, str)
        assert "TBRResults" in repr_str
        assert "Cumulative Effect" in repr_str
        assert "CI:" in repr_str

    def test_str_method(self, tbr_results):
        """Test __str__ method returns string."""
        str_repr = str(tbr_results)
        assert isinstance(str_repr, str)
        assert "TBRResults" in str_repr
        # __str__ should call __repr__
        assert str_repr == repr(tbr_results)


class TestTBRResultsConsistency:
    """Test consistency between different accessors."""

    def test_estimate_matches_cumulative_last(self, tbr_results):
        """Test estimate matches last cumulative effect."""
        assert tbr_results.estimate == tbr_results.cumulative_effect.iloc[-1]

    def test_confidence_interval_consistency(self, tbr_results):
        """Test confidence interval bounds are consistent."""
        lower = tbr_results.conf_int_lower
        upper = tbr_results.conf_int_upper
        estimate = tbr_results.estimate

        # Lower < estimate < upper
        assert lower < estimate < upper

        # Bounds should match conf_int() method
        ci_df = tbr_results.conf_int()
        assert lower == ci_df["lower"].iloc[0]
        assert upper == ci_df["upper"].iloc[0]

    def test_model_params_match_properties(self, tbr_results):
        """Test model_params dict matches individual properties."""
        params = tbr_results.model_params

        assert params["alpha"] == tbr_results.alpha
        assert params["beta"] == tbr_results.beta
        assert params["sigma"] == tbr_results.sigma

    def test_time_series_lengths_consistent(self, tbr_results):
        """Test time series have consistent lengths."""
        # Control and test should span both pretest and test periods
        assert len(tbr_results.control) == len(tbr_results.test)
        assert len(tbr_results.control) == tbr_results.n_pretest + tbr_results.n_test

        # Fitted values should match pretest length
        assert len(tbr_results.fittedvalues) == tbr_results.n_pretest

        # Predictions, effects, etc. should match test period length
        assert len(tbr_results.predictions) == tbr_results.n_test
        assert len(tbr_results.effects) == tbr_results.n_test
        assert len(tbr_results.cumulative_effect) == tbr_results.n_test


class TestTBRResultsIntegerTimeColumn:
    """Test TBRResults with integer time column."""

    def test_results_with_integer_time(self):
        """Test TBRResults works with integer time column."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "time": range(1, 61),
                "control": np.random.normal(1000, 50, 60),
                "test": np.random.normal(1020, 55, 60),
            }
        )

        results = perform_tbr_analysis(
            data=data,
            time_col="time",
            control_col="control",
            test_col="test",
            pretest_start=1,
            test_start=31,
            test_end=60,
            level=0.80,
            threshold=0.0,
        )

        # All properties should work
        assert isinstance(results.control, pd.Series)
        assert isinstance(results.test, pd.Series)
        assert isinstance(results.estimate, float)
        assert isinstance(results.alpha, float)
        assert isinstance(results.beta, float)
        assert isinstance(results.sigma, float)


class TestTBRResultsEdgeCases:
    """Test edge cases for TBRResults."""

    def test_minimal_test_period(self):
        """Test results with minimal test period."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=32, freq="D"),
                "control": np.random.normal(1000, 50, 32),
                "test": np.random.normal(1020, 55, 32),
            }
        )

        results = perform_tbr_analysis(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-31"),
            test_end=pd.Timestamp("2023-02-01"),
            level=0.80,
            threshold=0.0,
            test_end_inclusive=True,  # Include end date for 2 days
        )

        assert results.n_test == 2
        assert len(results.cumulative_effect) == 2
        assert isinstance(results.estimate, float)

    def test_properties_return_copies(self, tbr_results):
        """Test that DataFrame methods return copies."""
        # summary() should return a copy
        summary1 = tbr_results.summary()
        summary2 = tbr_results.summary()
        assert summary1 is not summary2

        # tbr_dataframe() should return a copy
        df1 = tbr_results.tbr_dataframe()
        df2 = tbr_results.tbr_dataframe()
        assert df1 is not df2

        # model_params should return a copy
        params1 = tbr_results.model_params
        params2 = tbr_results.model_params
        assert params1 is not params2
