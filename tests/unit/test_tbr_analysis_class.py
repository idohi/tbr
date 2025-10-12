"""
Tests for TBRAnalysis class structure and initialization.

This test module validates the object-oriented API for TBR analysis,
ensuring proper initialization, state management, and property access.
"""

import pytest

from tbr.core.model import TBRAnalysis


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


if __name__ == "__main__":
    pytest.main([__file__])
