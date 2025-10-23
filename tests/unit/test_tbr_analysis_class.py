"""
Tests for TBRAnalysis class structure and initialization.

This test module validates the object-oriented API for TBR analysis,
ensuring proper initialization, state management, and property access.
"""

import numpy as np
import pandas as pd
import pytest

from tbr.core.model import TBRAnalysis


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=90)
    data = pd.DataFrame(
        {
            "date": dates,
            "control": np.random.normal(1000, 50, 90),
            "test": np.random.normal(1020, 55, 90),
        }
    )
    return data


class TestTBRAnalysisInitialization:
    """Test TBRAnalysis class initialization and configuration."""

    def test_default_initialization(self):
        """Test TBRAnalysis initialization with default parameters."""
        model = TBRAnalysis()

        assert model.level == 0.80
        assert model.threshold == 0.0
        assert model.test_end_inclusive is False
        assert model.fitted_ is False

    def test_custom_initialization(self):
        """Test TBRAnalysis initialization with custom parameters."""
        model = TBRAnalysis(level=0.95, threshold=5.0, test_end_inclusive=True)

        assert model.level == 0.95
        assert model.threshold == 5.0
        assert model.test_end_inclusive is True
        assert model.fitted_ is False

    def test_level_validation_too_low(self):
        """Test that level must be greater than 0."""
        with pytest.raises(ValueError, match="level must be between 0 and 1 exclusive"):
            TBRAnalysis(level=0.0)

    def test_level_validation_too_high(self):
        """Test that level must be less than 1."""
        with pytest.raises(ValueError, match="level must be between 0 and 1 exclusive"):
            TBRAnalysis(level=1.0)

    def test_level_validation_negative(self):
        """Test that level cannot be negative."""
        with pytest.raises(ValueError, match="level must be between 0 and 1 exclusive"):
            TBRAnalysis(level=-0.5)

    def test_level_validation_greater_than_one(self):
        """Test that level cannot exceed 1."""
        with pytest.raises(ValueError, match="level must be between 0 and 1 exclusive"):
            TBRAnalysis(level=1.5)

    def test_level_type_validation(self):
        """Test that level must be numeric."""
        with pytest.raises(TypeError, match="level must be numeric"):
            TBRAnalysis(level="0.80")

    def test_threshold_type_validation(self):
        """Test that threshold must be numeric."""
        with pytest.raises(TypeError, match="threshold must be numeric"):
            TBRAnalysis(threshold="5.0")

    def test_test_end_inclusive_type_validation(self):
        """Test that test_end_inclusive must be boolean."""
        with pytest.raises(TypeError, match="test_end_inclusive must be bool"):
            TBRAnalysis(test_end_inclusive="True")

    def test_threshold_accepts_negative(self):
        """Test that threshold can be negative."""
        model = TBRAnalysis(threshold=-10.0)
        assert model.threshold == -10.0

    def test_threshold_accepts_zero(self):
        """Test that threshold can be zero."""
        model = TBRAnalysis(threshold=0.0)
        assert model.threshold == 0.0

    def test_threshold_accepts_positive(self):
        """Test that threshold can be positive."""
        model = TBRAnalysis(threshold=100.0)
        assert model.threshold == 100.0

    def test_level_coerced_to_float(self):
        """Test that level is converted to float."""
        model = TBRAnalysis(level=1 / 2)  # Python int division
        assert isinstance(model.level, float)
        assert model.level == 0.5

    def test_threshold_coerced_to_float(self):
        """Test that threshold is converted to float."""
        model = TBRAnalysis(threshold=5)  # Integer
        assert isinstance(model.threshold, float)
        assert model.threshold == 5.0


class TestTBRAnalysisRepresentation:
    """Test TBRAnalysis string representations."""

    def test_repr_not_fitted(self):
        """Test __repr__ for unfitted model."""
        model = TBRAnalysis(level=0.80, threshold=0.0, test_end_inclusive=False)
        repr_str = repr(model)

        assert "TBRAnalysis" in repr_str
        assert "level=0.8" in repr_str
        assert "threshold=0.0" in repr_str
        assert "test_end_inclusive=False" in repr_str
        assert "not fitted" in repr_str

    def test_str_not_fitted(self):
        """Test __str__ for unfitted model."""
        model = TBRAnalysis(level=0.80, threshold=0.0)
        str_rep = str(model)

        assert "TBRAnalysis (not fitted)" in str_rep
        assert "level=0.8" in str_rep
        assert "threshold=0.0" in str_rep


class TestTBRAnalysisPropertyAccess:
    """Test TBRAnalysis property access before fitting."""

    def test_results_property_not_fitted(self):
        """Test that accessing results_ before fitting raises AttributeError."""
        model = TBRAnalysis()

        with pytest.raises(
            AttributeError, match="This TBRAnalysis instance is not fitted yet"
        ):
            _ = model.results_

    def test_summaries_property_not_fitted(self):
        """Test that accessing summaries_ before fitting raises AttributeError."""
        model = TBRAnalysis()

        with pytest.raises(
            AttributeError, match="This TBRAnalysis instance is not fitted yet"
        ):
            _ = model.summaries_

    def test_params_property_not_fitted(self):
        """Test that accessing params_ before fitting raises AttributeError."""
        model = TBRAnalysis()

        with pytest.raises(
            AttributeError, match="This TBRAnalysis instance is not fitted yet"
        ):
            _ = model.params_

    def test_fitted_property_accessible(self):
        """Test that fitted_ property is accessible before fitting."""
        model = TBRAnalysis()
        assert model.fitted_ is False


class TestTBRAnalysisImport:
    """Test TBRAnalysis import from various locations."""

    def test_import_from_core_model(self):
        """Test importing TBRAnalysis from core.model."""
        from tbr.core.model import TBRAnalysis as TBRAnalysisImport

        model = TBRAnalysisImport()
        assert isinstance(model, TBRAnalysis)

    def test_import_from_core(self):
        """Test importing TBRAnalysis from core module."""
        from tbr.core import TBRAnalysis as TBRAnalysisImport

        model = TBRAnalysisImport()
        assert isinstance(model, TBRAnalysis)


class TestTBRAnalysisFittedState:
    """Test TBRAnalysis behavior with manually fitted state for 100% coverage."""

    def test_results_property_when_fitted(self):
        """Test that results_ property returns data when fitted."""
        import pandas as pd

        model = TBRAnalysis()

        # Manually set fitted state (until fit() is implemented in Task 7.2)
        model._fitted = True
        model._results = pd.DataFrame({"date": [1, 2], "control": [10, 20]})

        results = model.results_
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2

    def test_summaries_property_when_fitted(self):
        """Test that summaries_ property returns data when fitted."""
        import pandas as pd

        model = TBRAnalysis()

        # Manually set fitted state
        model._fitted = True
        model._summaries = pd.DataFrame(
            {"estimate": [5.0, 10.0], "lower": [2.0, 6.0], "upper": [8.0, 14.0]}
        )

        summaries = model.summaries_
        assert isinstance(summaries, pd.DataFrame)
        assert len(summaries) == 2

    def test_params_property_when_fitted(self):
        """Test that params_ property returns dictionary when fitted."""
        model = TBRAnalysis()

        # Manually set fitted state
        model._fitted = True
        model._params = {
            "alpha": 1.5,
            "beta": 0.8,
            "sigma": 2.3,
            "var_alpha": 0.1,
            "var_beta": 0.05,
        }

        params = model.params_
        assert isinstance(params, dict)
        assert params["alpha"] == 1.5
        assert params["beta"] == 0.8

    def test_repr_when_fitted(self):
        """Test __repr__ for fitted model."""
        model = TBRAnalysis(level=0.95, threshold=5.0, test_end_inclusive=True)

        # Manually set fitted state
        model._fitted = True

        repr_str = repr(model)
        assert "TBRAnalysis" in repr_str
        assert "level=0.95" in repr_str
        assert "threshold=5.0" in repr_str
        assert "test_end_inclusive=True" in repr_str
        assert "fitted" in repr_str
        assert "not fitted" not in repr_str

    def test_str_when_fitted(self):
        """Test __str__ for fitted model with results summary."""
        import pandas as pd

        model = TBRAnalysis(level=0.80, threshold=0.0)

        # Manually set fitted state with realistic data
        model._fitted = True
        model._summaries = pd.DataFrame(
            {
                "estimate": [5.0, 10.0, 15.0],
                "lower": [2.0, 6.0, 10.0],
                "upper": [8.0, 14.0, 20.0],
            }
        )

        str_rep = str(model)
        assert "TBRAnalysis (fitted)" in str_rep
        assert "Configuration:" in str_rep
        assert "level=0.8" in str_rep
        assert "threshold=0.0" in str_rep
        assert "Results:" in str_rep
        assert "Test period days: 3" in str_rep
        assert "Final effect estimate: 15.00" in str_rep
        assert "80% CI: [10.00, 20.00]" in str_rep


class TestTBRAnalysisFitMethod:
    """Test TBRAnalysis fit() method functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=90)
        data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, 90),
                "test": np.random.normal(1020, 55, 90),
            }
        )
        return data

    def test_fit_basic_functionality(self, sample_data):
        """Test basic fit() functionality with datetime data."""
        model = TBRAnalysis(level=0.80, threshold=0.0)

        result = model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Check that fit returns self for method chaining
        assert result is model

        # Check that model is now fitted
        assert model.fitted_ is True

        # Check that results are accessible
        assert isinstance(model.results_, pd.DataFrame)
        assert isinstance(model.summaries_, pd.DataFrame)
        assert isinstance(model.params_, dict)

    def test_fit_sets_fitted_flag(self, sample_data):
        """Test that fit() sets the fitted_ flag correctly."""
        model = TBRAnalysis()
        assert model.fitted_ is False

        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        assert model.fitted_ is True

    def test_fit_stores_results(self, sample_data):
        """Test that fit() stores TBR results DataFrame."""
        model = TBRAnalysis(level=0.80, threshold=0.0)

        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        results = model.results_

        # Check DataFrame structure
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

        # Check for expected columns (note: control/test are renamed to x/y internally)
        expected_cols = [
            "date",
            "x",
            "y",
            "period",
            "pred",
            "predsd",
            "dif",
            "cumdif",
            "cumsd",
            "estsd",
        ]
        for col in expected_cols:
            assert col in results.columns

    def test_fit_stores_summaries(self, sample_data):
        """Test that fit() stores incremental summaries."""
        model = TBRAnalysis(level=0.80, threshold=0.0)

        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        summaries = model.summaries_

        # Check DataFrame structure
        assert isinstance(summaries, pd.DataFrame)
        assert len(summaries) > 0

        # Check for expected columns (note: threshold is abbreviated as thres)
        expected_cols = [
            "estimate",
            "precision",
            "lower",
            "upper",
            "se",
            "level",
            "thres",
            "prob",
            "alpha",
            "beta",
            "sigma",
        ]
        for col in expected_cols:
            assert col in summaries.columns

    def test_fit_stores_params(self, sample_data):
        """Test that fit() stores model parameters correctly."""
        model = TBRAnalysis(level=0.80, threshold=0.0)

        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        params = model.params_

        # Check dictionary structure
        assert isinstance(params, dict)

        # Check for required parameter keys
        required_keys = [
            "alpha",
            "beta",
            "sigma",
            "var_alpha",
            "var_beta",
            "cov_alpha_beta",
            "degrees_freedom",
            "pretest_x_mean",
            "pretest_sum_x_squared_deviations",
        ]
        for key in required_keys:
            assert key in params, f"Missing parameter: {key}"

        # Check parameter types
        assert isinstance(params["alpha"], float)
        assert isinstance(params["beta"], float)
        assert isinstance(params["sigma"], float)
        assert isinstance(params["degrees_freedom"], int)

    def test_fit_uses_stored_configuration(self, sample_data):
        """Test that fit() uses level and threshold from initialization."""
        model = TBRAnalysis(level=0.95, threshold=5.0)

        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        summaries = model.summaries_

        # Check that summaries use the correct level and threshold (note: threshold is abbreviated as thres)
        assert summaries.iloc[0]["level"] == 0.95
        assert summaries.iloc[0]["thres"] == 5.0

    def test_fit_method_chaining(self, sample_data):
        """Test that fit() supports method chaining."""
        results = (
            TBRAnalysis(level=0.80, threshold=0.0)
            .fit(
                data=sample_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )
            .results_
        )

        assert isinstance(results, pd.DataFrame)

    def test_fit_with_integer_time(self):
        """Test fit() with integer time column."""
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        data = pd.DataFrame(
            {
                "time": list(range(1, 91)),  # Integer time
                "control": np.random.normal(1000, 50, 90),
                "test": np.random.normal(1020, 55, 90),
            }
        )

        model = TBRAnalysis(level=0.80, threshold=0.0)
        model.fit(
            data=data,
            time_col="time",
            control_col="control",
            test_col="test",
            pretest_start=1,
            test_start=45,
            test_end=60,
        )

        assert model.fitted_ is True
        assert isinstance(model.results_, pd.DataFrame)

    def test_fit_with_test_end_inclusive(self, sample_data):
        """Test fit() with test_end_inclusive=True."""
        model = TBRAnalysis(level=0.80, threshold=0.0, test_end_inclusive=True)

        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-02-15"),  # Same-day analysis
        )

        assert model.fitted_ is True
        assert len(model.summaries_) == 1  # Single-day test period

    def test_fit_invalid_data(self):
        """Test that fit() raises error with invalid data."""
        import pandas as pd

        model = TBRAnalysis()

        # Empty DataFrame
        with pytest.raises(ValueError):
            model.fit(
                data=pd.DataFrame(),
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_missing_columns(self, sample_data):
        """Test that fit() raises error with missing columns."""
        model = TBRAnalysis()

        with pytest.raises(ValueError):
            model.fit(
                data=sample_data,
                time_col="nonexistent",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_stores_fit_info(self, sample_data):
        """Test that fit() stores fit metadata information."""
        model = TBRAnalysis()

        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Check internal fit_info storage
        assert model._fit_info is not None
        assert model._fit_info["time_col"] == "date"
        assert model._fit_info["control_col"] == "control"
        assert model._fit_info["test_col"] == "test"
        assert "n_pretest" in model._fit_info
        assert "n_test" in model._fit_info

    def test_fit_parameter_extraction_accuracy(self, sample_data):
        """Test that fit() correctly extracts parameters from summaries."""
        model = TBRAnalysis(level=0.80, threshold=0.0)

        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Extract parameters from both sources
        params = model.params_
        summary_row = model.summaries_.iloc[0]

        # Verify parameter consistency
        assert params["alpha"] == summary_row["alpha"]
        assert params["beta"] == summary_row["beta"]
        assert params["sigma"] == summary_row["sigma"]
        assert params["var_alpha"] == summary_row["var_alpha"]
        assert params["var_beta"] == summary_row["var_beta"]
        assert params["cov_alpha_beta"] == summary_row["alpha_beta_cov"]
        assert params["degrees_freedom"] == summary_row["t_dist_df"]

    def test_fit_can_refit_model(self, sample_data):
        """Test that fit() can be called multiple times to refit."""
        model = TBRAnalysis(level=0.80, threshold=0.0)

        # First fit
        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Second fit with different period
        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-02-15"),
        )

        # Verify model is still fitted after refitting
        assert model.fitted_ is True
        assert model.summaries_ is not None
        assert len(model.summaries_) > 0


class TestTBRAnalysisPredictMethod:
    """Tests for the predict() method of TBRAnalysis class."""

    def test_predict_basic_functionality(self, sample_data):
        """Test basic predict functionality with default arguments."""
        model = TBRAnalysis(level=0.80, threshold=0.0)
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Predict using test period data
        predictions = model.predict()

        # Check structure
        assert isinstance(predictions, pd.DataFrame)
        assert "pred" in predictions.columns
        assert "predsd" in predictions.columns
        assert (
            len(predictions) == 28
        )  # 28 days in test period (Feb 1 - Mar 1 exclusive)
        assert all(predictions["predsd"] > 0)

    def test_predict_with_custom_control_values(self, sample_data):
        """Test predict with custom control values."""
        model = TBRAnalysis(level=0.80)
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Predict for custom values
        custom_control = np.array([1000.0, 1100.0, 1200.0])
        predictions = model.predict(control_values=custom_control)

        assert len(predictions) == 3
        assert all(predictions["pred"] > 0)
        assert all(predictions["predsd"] > 0)

    def test_predict_with_series_input(self, sample_data):
        """Test predict accepts pandas Series as input."""
        model = TBRAnalysis(level=0.80)
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Use pandas Series
        custom_control = pd.Series([1000.0, 1100.0, 1200.0])
        predictions = model.predict(control_values=custom_control)

        assert len(predictions) == 3
        assert isinstance(predictions, pd.DataFrame)

    def test_predict_with_list_input(self, sample_data):
        """Test predict accepts Python list as input."""
        model = TBRAnalysis(level=0.80)
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Use Python list
        custom_control = [1000.0, 1100.0, 1200.0]
        predictions = model.predict(control_values=custom_control)

        assert len(predictions) == 3
        assert isinstance(predictions, pd.DataFrame)

    def test_predict_not_fitted_error(self):
        """Test predict raises error when model not fitted."""
        model = TBRAnalysis()

        with pytest.raises(AttributeError, match="not fitted yet"):
            model.predict()

    def test_predict_invalid_shape_error(self, sample_data):
        """Test predict raises error for invalid shape."""
        model = TBRAnalysis()
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # 2D array should fail
        invalid_control = np.array([[1000, 1100], [1200, 1300]])
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            model.predict(control_values=invalid_control)

    def test_predict_non_finite_values_error(self, sample_data):
        """Test predict raises error for non-finite values."""
        model = TBRAnalysis()
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Non-finite values should fail
        invalid_control = np.array([1000.0, np.nan, 1200.0])
        with pytest.raises(ValueError, match="must contain only finite values"):
            model.predict(control_values=invalid_control)


class TestTBRAnalysisSummarizeMethod:
    """Tests for the summarize() method of TBRAnalysis class."""

    def test_summarize_final_default(self, sample_data):
        """Test summarize returns final summary by default."""
        model = TBRAnalysis(level=0.80, threshold=0.0)
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        summary = model.summarize()

        # Should be single row
        assert len(summary) == 1
        assert isinstance(summary, pd.DataFrame)

        # Check expected columns
        expected_cols = ["estimate", "lower", "upper", "se", "prob", "level", "thres"]
        for col in expected_cols:
            assert col in summary.columns

    def test_summarize_incremental(self, sample_data):
        """Test summarize with incremental=True returns all summaries."""
        model = TBRAnalysis(level=0.80)
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        summaries = model.summarize(incremental=True)

        # Should have multiple rows (one per test day)
        assert len(summaries) == 28  # 28 days in test period
        assert isinstance(summaries, pd.DataFrame)

    def test_summarize_not_fitted_error(self):
        """Test summarize raises error when model not fitted."""
        model = TBRAnalysis()

        with pytest.raises(AttributeError, match="not fitted yet"):
            model.summarize()

    def test_summarize_returns_copy(self, sample_data):
        """Test summarize returns a copy, not original data."""
        model = TBRAnalysis(level=0.80)
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        summary = model.summarize()
        original_value = summary.iloc[0]["estimate"]

        # Modify summary
        summary.iloc[0, summary.columns.get_loc("estimate")] = 999999.0

        # Original should not be affected
        new_summary = model.summarize()
        assert new_summary.iloc[0]["estimate"] == original_value

    def test_summarize_final_matches_last_incremental(self, sample_data):
        """Test final summary matches last row of incremental summaries."""
        model = TBRAnalysis(level=0.80)
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        final = model.summarize(incremental=False)
        incremental = model.summarize(incremental=True)

        # Final should equal last row of incremental
        assert final.iloc[0]["estimate"] == incremental.iloc[-1]["estimate"]
        assert final.iloc[0]["lower"] == incremental.iloc[-1]["lower"]
        assert final.iloc[0]["upper"] == incremental.iloc[-1]["upper"]


class TestTBRAnalysisAnalyzeSubintervalMethod:
    """Tests for the analyze_subinterval() method of TBRAnalysis class."""

    def test_analyze_subinterval_basic_functionality(self, sample_data):
        """Test basic subinterval analysis."""
        model = TBRAnalysis(level=0.80)
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Analyze first week
        result = model.analyze_subinterval(start_day=1, end_day=7)

        # Check structure
        assert isinstance(result, dict)
        assert "estimate" in result
        assert "lower" in result
        assert "upper" in result
        assert "se" in result
        assert "ci_level" in result
        assert "start_day" in result
        assert "end_day" in result
        assert "n_days" in result

        # Check values
        assert result["start_day"] == 1
        assert result["end_day"] == 7
        assert result["n_days"] == 7
        assert result["ci_level"] == 0.80
        assert isinstance(result["estimate"], float)

    def test_analyze_subinterval_custom_ci_level(self, sample_data):
        """Test subinterval analysis with custom CI level."""
        model = TBRAnalysis(level=0.80)
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        result = model.analyze_subinterval(start_day=1, end_day=7, ci_level=0.95)

        assert result["ci_level"] == 0.95
        # 95% CI should be wider than 80% CI
        result_80 = model.analyze_subinterval(start_day=1, end_day=7, ci_level=0.80)
        assert (result["upper"] - result["lower"]) > (
            result_80["upper"] - result_80["lower"]
        )

    def test_analyze_subinterval_single_day(self, sample_data):
        """Test subinterval analysis for a single day."""
        model = TBRAnalysis(level=0.80)
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        result = model.analyze_subinterval(start_day=5, end_day=5)

        assert result["n_days"] == 1
        assert result["start_day"] == 5
        assert result["end_day"] == 5

    def test_analyze_subinterval_not_fitted_error(self):
        """Test analyze_subinterval raises error when model not fitted."""
        model = TBRAnalysis()

        with pytest.raises(AttributeError, match="not fitted yet"):
            model.analyze_subinterval(start_day=1, end_day=7)

    def test_analyze_subinterval_invalid_start_day(self, sample_data):
        """Test error for invalid start_day."""
        model = TBRAnalysis()
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Zero start_day
        with pytest.raises(ValueError, match="start_day must be a positive integer"):
            model.analyze_subinterval(start_day=0, end_day=7)

        # Negative start_day
        with pytest.raises(ValueError, match="start_day must be a positive integer"):
            model.analyze_subinterval(start_day=-1, end_day=7)

    def test_analyze_subinterval_invalid_end_day(self, sample_data):
        """Test error for invalid end_day."""
        model = TBRAnalysis()
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        with pytest.raises(ValueError, match="end_day must be a positive integer"):
            model.analyze_subinterval(start_day=1, end_day=0)

    def test_analyze_subinterval_start_greater_than_end(self, sample_data):
        """Test error when start_day > end_day."""
        model = TBRAnalysis()
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        with pytest.raises(ValueError, match="start_day.*must be <= end_day"):
            model.analyze_subinterval(start_day=10, end_day=5)

    def test_analyze_subinterval_exceeds_test_period(self, sample_data):
        """Test error when end_day exceeds test period length."""
        model = TBRAnalysis()
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        with pytest.raises(ValueError, match="exceeds test period length"):
            model.analyze_subinterval(start_day=1, end_day=999)

    def test_analyze_subinterval_full_period(self, sample_data):
        """Test subinterval for entire test period."""
        model = TBRAnalysis(level=0.80)
        data = sample_data
        model.fit(
            data=data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )

        # Full test period is 28 days
        result = model.analyze_subinterval(start_day=1, end_day=28)
        final_summary = model.summarize(incremental=False)

        # Should be very close to final summary estimate
        assert abs(result["estimate"] - final_summary.iloc[0]["estimate"]) < 0.01


class TestTBRAnalysisFitValidation:
    """Comprehensive validation tests for fit() method input validation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=90)
        data = pd.DataFrame(
            {
                "date": dates,
                "control": np.random.normal(1000, 50, 90),
                "test": np.random.normal(1020, 55, 90),
            }
        )
        return data

    def test_fit_non_dataframe_type_error(self):
        """Test fit() raises TypeError for non-DataFrame input."""
        model = TBRAnalysis()

        with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
            model.fit(
                data="not a dataframe",
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_empty_dataframe_error(self):
        """Test fit() raises ValueError for empty DataFrame."""
        model = TBRAnalysis()

        with pytest.raises(ValueError, match="data cannot be empty"):
            model.fit(
                data=pd.DataFrame(),
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_time_col_not_string_error(self, sample_data):
        """Test fit() raises TypeError when time_col is not a string."""
        model = TBRAnalysis()

        with pytest.raises(TypeError, match="time_col must be a string"):
            model.fit(
                data=sample_data,
                time_col=123,
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_control_col_not_string_error(self, sample_data):
        """Test fit() raises TypeError when control_col is not a string."""
        model = TBRAnalysis()

        with pytest.raises(TypeError, match="control_col must be a string"):
            model.fit(
                data=sample_data,
                time_col="date",
                control_col=["control"],
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_test_col_not_string_error(self, sample_data):
        """Test fit() raises TypeError when test_col is not a string."""
        model = TBRAnalysis()

        with pytest.raises(TypeError, match="test_col must be a string"):
            model.fit(
                data=sample_data,
                time_col="date",
                control_col="control",
                test_col={"test": "value"},
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_missing_time_column_error(self, sample_data):
        """Test fit() raises ValueError when time column is missing."""
        model = TBRAnalysis()

        with pytest.raises(ValueError, match="Missing required columns"):
            model.fit(
                data=sample_data,
                time_col="nonexistent_col",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_missing_control_column_error(self, sample_data):
        """Test fit() raises ValueError when control column is missing."""
        model = TBRAnalysis()

        with pytest.raises(ValueError, match="Missing required columns"):
            model.fit(
                data=sample_data,
                time_col="date",
                control_col="missing_control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_missing_test_column_error(self, sample_data):
        """Test fit() raises ValueError when test column is missing."""
        model = TBRAnalysis()

        with pytest.raises(ValueError, match="Missing required columns"):
            model.fit(
                data=sample_data,
                time_col="date",
                control_col="control",
                test_col="missing_test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_invalid_time_column_dtype_error(self, sample_data):
        """Test fit() raises ValueError for unsupported time column dtype."""
        model = TBRAnalysis()

        # Create data with object dtype time column
        bad_data = sample_data.copy()
        bad_data["date"] = bad_data["date"].astype(str)

        with pytest.raises(ValueError, match="Unsupported dtype"):
            model.fit(
                data=bad_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start="2023-01-01",
                test_start="2023-02-15",
                test_end="2023-03-01",
            )

    def test_fit_non_numeric_control_column_error(self, sample_data):
        """Test fit() raises ValueError when control column is not numeric."""
        model = TBRAnalysis()

        # Create data with non-numeric control
        bad_data = sample_data.copy()
        bad_data["control"] = ["low", "medium", "high"] * 30

        with pytest.raises(ValueError, match="Control column.*must be numeric"):
            model.fit(
                data=bad_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_non_numeric_test_column_error(self, sample_data):
        """Test fit() raises ValueError when test column is not numeric."""
        model = TBRAnalysis()

        # Create data with non-numeric test
        bad_data = sample_data.copy()
        bad_data["test"] = ["small", "medium", "large"] * 30

        with pytest.raises(ValueError, match="Test column.*must be numeric"):
            model.fit(
                data=bad_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_null_values_in_time_column_error(self, sample_data):
        """Test fit() raises ValueError when time column contains nulls."""
        model = TBRAnalysis()

        # Create data with null time values
        bad_data = sample_data.copy()
        bad_data.loc[5, "date"] = pd.NaT

        with pytest.raises(ValueError, match="Null values found"):
            model.fit(
                data=bad_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_null_values_in_control_column_error(self, sample_data):
        """Test fit() raises ValueError when control column contains nulls."""
        model = TBRAnalysis()

        # Create data with null control values
        bad_data = sample_data.copy()
        bad_data.loc[10, "control"] = np.nan

        with pytest.raises(ValueError, match="Null values found"):
            model.fit(
                data=bad_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_null_values_in_test_column_error(self, sample_data):
        """Test fit() raises ValueError when test column contains nulls."""
        model = TBRAnalysis()

        # Create data with null test values
        bad_data = sample_data.copy()
        bad_data.loc[15, "test"] = np.nan

        with pytest.raises(ValueError, match="Null values found"):
            model.fit(
                data=bad_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_mixed_boundary_types_error(self, sample_data):
        """Test fit() raises ValueError when boundary types are mixed."""
        model = TBRAnalysis()

        # Mix pd.Timestamp and int
        with pytest.raises(
            ValueError, match="All time boundaries must have the same type"
        ):
            model.fit(
                data=sample_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=1,  # Wrong type
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_boundary_type_mismatch_error(self):
        """Test fit() raises ValueError when boundary types don't match column dtype."""
        model = TBRAnalysis()

        # Integer time column with Timestamp boundaries
        int_data = pd.DataFrame(
            {
                "time": list(range(1, 91)),
                "control": np.random.normal(1000, 50, 90),
                "test": np.random.normal(1020, 55, 90),
            }
        )

        with pytest.raises(ValueError, match="Use int for integer time columns"):
            model.fit(
                data=int_data,
                time_col="time",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),  # Wrong type
                test_start=pd.Timestamp("2023-02-15"),
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_pretest_after_test_start_error(self, sample_data):
        """Test fit() raises ValueError when pretest_start >= test_start."""
        model = TBRAnalysis()

        with pytest.raises(ValueError, match="pretest_start must be before test_start"):
            model.fit(
                data=sample_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-02-15"),
                test_start=pd.Timestamp("2023-02-15"),  # Same as pretest_start
                test_end=pd.Timestamp("2023-03-01"),
            )

    def test_fit_test_start_after_test_end_exclusive_error(self, sample_data):
        """Test fit() raises ValueError when test_start >= test_end (exclusive)."""
        model = TBRAnalysis(test_end_inclusive=False)

        with pytest.raises(ValueError, match="test_start must be < test_end"):
            model.fit(
                data=sample_data,
                time_col="date",
                control_col="control",
                test_col="test",
                pretest_start=pd.Timestamp("2023-01-01"),
                test_start=pd.Timestamp("2023-03-01"),
                test_end=pd.Timestamp("2023-03-01"),  # Same as test_start
            )

    def test_fit_test_start_after_test_end_inclusive_valid(self, sample_data):
        """Test fit() allows test_start == test_end when test_end_inclusive=True."""
        model = TBRAnalysis(test_end_inclusive=True)

        # Should not raise error
        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-15"),
            test_end=pd.Timestamp("2023-02-15"),  # Same-day analysis
        )

        assert model.fitted_ is True


class TestTBRAnalysisPredictValidation:
    """Comprehensive validation tests for predict() method input validation."""

    @pytest.fixture
    def fitted_model(self, sample_data):
        """Create a fitted TBRAnalysis model for testing."""
        model = TBRAnalysis(level=0.80, threshold=0.0)
        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )
        return model

    def test_predict_empty_control_values_error(self, fitted_model):
        """Test predict() raises ValueError for empty control values array."""
        with pytest.raises(ValueError, match="control_values cannot be empty"):
            fitted_model.predict(control_values=np.array([]))

    def test_predict_empty_list_error(self, fitted_model):
        """Test predict() raises ValueError for empty list."""
        with pytest.raises(ValueError, match="control_values cannot be empty"):
            fitted_model.predict(control_values=[])

    def test_predict_invalid_type_error(self, fitted_model):
        """Test predict() raises TypeError for non-numeric control values."""
        with pytest.raises(
            TypeError, match="control_values must contain numeric values"
        ):
            fitted_model.predict(control_values="not an array")

    def test_predict_with_inf_values_error(self, fitted_model):
        """Test predict() raises ValueError for infinite values."""
        with pytest.raises(ValueError, match="must contain only finite values"):
            fitted_model.predict(control_values=np.array([1000.0, np.inf, 1200.0]))

    def test_predict_with_nan_values_error(self, fitted_model):
        """Test predict() raises ValueError for NaN values."""
        with pytest.raises(ValueError, match="must contain only finite values"):
            fitted_model.predict(control_values=np.array([1000.0, np.nan, 1200.0]))

    def test_predict_2d_array_error(self, fitted_model):
        """Test predict() raises ValueError for 2D array."""
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            fitted_model.predict(control_values=np.array([[1000], [1100]]))

    def test_predict_3d_array_error(self, fitted_model):
        """Test predict() raises ValueError for 3D array."""
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            fitted_model.predict(control_values=np.array([[[1000]]]))

    def test_predict_unconvertible_object_error(self, fitted_model):
        """Test predict() raises TypeError for completely unconvertible objects."""

        # Create an object that raises TypeError when np.array() tries to convert it
        class UnconvertibleObject:
            def __array__(self):
                raise TypeError("Cannot convert to array")

        with pytest.raises(TypeError, match="control_values must be array-like"):
            fitted_model.predict(control_values=UnconvertibleObject())


class TestTBRAnalysisAnalyzeSubintervalValidation:
    """Comprehensive validation tests for analyze_subinterval() method input validation."""

    @pytest.fixture
    def fitted_model(self, sample_data):
        """Create a fitted TBRAnalysis model for testing."""
        model = TBRAnalysis(level=0.80, threshold=0.0)
        model.fit(
            data=sample_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-01"),
        )
        return model

    def test_analyze_subinterval_start_day_not_int_error(self, fitted_model):
        """Test analyze_subinterval() raises TypeError when start_day is not int."""
        with pytest.raises(TypeError, match="start_day must be an integer"):
            fitted_model.analyze_subinterval(start_day=1.5, end_day=7)

    def test_analyze_subinterval_end_day_not_int_error(self, fitted_model):
        """Test analyze_subinterval() raises TypeError when end_day is not int."""
        with pytest.raises(TypeError, match="end_day must be an integer"):
            fitted_model.analyze_subinterval(start_day=1, end_day="7")

    def test_analyze_subinterval_start_day_zero_error(self, fitted_model):
        """Test analyze_subinterval() raises ValueError for start_day=0."""
        with pytest.raises(ValueError, match="start_day must be a positive integer"):
            fitted_model.analyze_subinterval(start_day=0, end_day=7)

    def test_analyze_subinterval_start_day_negative_error(self, fitted_model):
        """Test analyze_subinterval() raises ValueError for negative start_day."""
        with pytest.raises(ValueError, match="start_day must be a positive integer"):
            fitted_model.analyze_subinterval(start_day=-5, end_day=7)

    def test_analyze_subinterval_end_day_zero_error(self, fitted_model):
        """Test analyze_subinterval() raises ValueError for end_day=0."""
        with pytest.raises(ValueError, match="end_day must be a positive integer"):
            fitted_model.analyze_subinterval(start_day=1, end_day=0)

    def test_analyze_subinterval_end_day_negative_error(self, fitted_model):
        """Test analyze_subinterval() raises ValueError for negative end_day."""
        with pytest.raises(ValueError, match="end_day must be a positive integer"):
            fitted_model.analyze_subinterval(start_day=1, end_day=-3)

    def test_analyze_subinterval_start_exceeds_period_error(self, fitted_model):
        """Test analyze_subinterval() raises ValueError when start_day exceeds test period."""
        with pytest.raises(ValueError, match="start_day.*exceeds test period length"):
            fitted_model.analyze_subinterval(start_day=999, end_day=1000)

    def test_analyze_subinterval_end_exceeds_period_error(self, fitted_model):
        """Test analyze_subinterval() raises ValueError when end_day exceeds test period."""
        with pytest.raises(ValueError, match="end_day.*exceeds test period length"):
            fitted_model.analyze_subinterval(start_day=1, end_day=999)

    def test_analyze_subinterval_invalid_ci_level_type_error(self, fitted_model):
        """Test analyze_subinterval() raises TypeError for non-numeric ci_level."""
        with pytest.raises(TypeError, match="ci_level must be numeric"):
            fitted_model.analyze_subinterval(start_day=1, end_day=7, ci_level="0.95")

    def test_analyze_subinterval_ci_level_too_low_error(self, fitted_model):
        """Test analyze_subinterval() raises ValueError for ci_level <= 0."""
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            fitted_model.analyze_subinterval(start_day=1, end_day=7, ci_level=0.0)

    def test_analyze_subinterval_ci_level_too_high_error(self, fitted_model):
        """Test analyze_subinterval() raises ValueError for ci_level >= 1."""
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            fitted_model.analyze_subinterval(start_day=1, end_day=7, ci_level=1.0)

    def test_analyze_subinterval_ci_level_negative_error(self, fitted_model):
        """Test analyze_subinterval() raises ValueError for negative ci_level."""
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            fitted_model.analyze_subinterval(start_day=1, end_day=7, ci_level=-0.5)

    def test_analyze_subinterval_numpy_integers_accepted(self, fitted_model):
        """Test analyze_subinterval() accepts numpy integer types."""
        # Should not raise error
        result = fitted_model.analyze_subinterval(
            start_day=np.int64(1), end_day=np.int64(7)
        )

        assert result["start_day"] == 1
        assert result["end_day"] == 7

    def test_analyze_subinterval_valid_ci_level(self, fitted_model):
        """Test analyze_subinterval() accepts valid ci_level."""
        # Should not raise error
        result = fitted_model.analyze_subinterval(start_day=1, end_day=7, ci_level=0.95)

        assert result["ci_level"] == 0.95


if __name__ == "__main__":
    pytest.main([__file__])
