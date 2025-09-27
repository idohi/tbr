"""
Unit tests for TBR Analysis Incremental Module.

This module provides comprehensive unit tests for the incremental analysis
functionality, ensuring mathematical accuracy, proper error handling, and
professional standards compliance following patterns from top scientific
PyPI packages.

Test Categories:
- Basic functionality and mathematical accuracy
- Input validation and error handling
- Edge cases and boundary conditions
- Backward compatibility with functional implementation
- Integration with lazy loading system
- Professional module organization standards

The tests maintain 100% coverage and validate against the proven functional
implementation to ensure mathematical consistency.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Import the module under test
from tbr.analysis.incremental import create_incremental_tbr_summaries

# Import functional implementation for cross-validation
from tbr.functional.tbr_functions import (
    create_incremental_tbr_summaries as functional_create_incremental_tbr_summaries,
)


class TestCreateIncrementalTbrSummaries:
    """Test suite for create_incremental_tbr_summaries function."""

    @pytest.fixture
    def sample_tbr_dataframe(self):
        """Create sample TBR dataframe for testing."""
        return pd.DataFrame(
            {
                "period": [0, 0, 0, 1, 1, 1],
                "cumdif": [0.0, 0.0, 0.0, 10.5, 22.3, 35.8],
                "cumsd": [0.0, 0.0, 0.0, 5.2, 7.8, 9.1],
            }
        )

    @pytest.fixture
    def sample_parameters(self):
        """Create sample regression parameters for testing."""
        return {
            "alpha": 50.2,
            "beta": 0.95,
            "sigma": 25.3,
            "var_alpha": 100.5,
            "var_beta": 0.001,
            "cov_alpha_beta": -0.05,
            "degrees_freedom": 43,
            "level": 0.80,
            "threshold": 0.0,
        }

    def test_basic_functionality(self, sample_tbr_dataframe, sample_parameters):
        """Test basic incremental summaries functionality."""
        result = create_incremental_tbr_summaries(
            sample_tbr_dataframe, **sample_parameters
        )

        # Verify result is DataFrame
        assert isinstance(result, pd.DataFrame)

        # Verify has test_day column
        assert "test_day" in result.columns

        # Verify has standard summary columns
        expected_columns = [
            "estimate",
            "precision",
            "lower",
            "upper",
            "se",
            "level",
            "thres",
            "prob",
            "test_day",
        ]
        for col in expected_columns:
            assert col in result.columns

        # Verify test_day progression
        assert result["test_day"].tolist() == [1, 2, 3]

        # Verify estimates are cumulative (increasing)
        estimates = result["estimate"].tolist()
        assert estimates[0] <= estimates[1] <= estimates[2]

    def test_mathematical_accuracy_cross_validation(
        self, sample_tbr_dataframe, sample_parameters
    ):
        """Test mathematical accuracy against functional implementation."""
        # Get results from both implementations
        core_result = create_incremental_tbr_summaries(
            sample_tbr_dataframe, **sample_parameters
        )
        functional_result = functional_create_incremental_tbr_summaries(
            sample_tbr_dataframe, **sample_parameters
        )

        # Verify identical results (exact match)
        pd.testing.assert_frame_equal(core_result, functional_result)

        # Verify numerical precision for key columns
        for col in ["estimate", "precision", "lower", "upper", "se", "prob"]:
            np.testing.assert_array_equal(
                core_result[col].values,
                functional_result[col].values,
                err_msg=f"Mismatch in {col} column",
            )

    def test_input_validation_empty_dataframe(self, sample_parameters):
        """Test error handling for empty dataframe."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty|missing"):
            create_incremental_tbr_summaries(empty_df, **sample_parameters)

    def test_input_validation_missing_columns(self, sample_parameters):
        """Test error handling for missing required columns."""
        # Missing 'period' column
        invalid_df = pd.DataFrame({"cumdif": [10.5, 22.3], "cumsd": [5.2, 7.8]})

        with pytest.raises(ValueError, match="missing.*columns|period"):
            create_incremental_tbr_summaries(invalid_df, **sample_parameters)

    def test_input_validation_invalid_level(
        self, sample_tbr_dataframe, sample_parameters
    ):
        """Test error handling for invalid level parameter."""
        # Level > 1
        invalid_params = sample_parameters.copy()
        invalid_params["level"] = 1.5

        with pytest.raises(ValueError, match="Level must be between 0 and 1"):
            create_incremental_tbr_summaries(sample_tbr_dataframe, **invalid_params)

        # Level < 0
        invalid_params["level"] = -0.1

        with pytest.raises(ValueError, match="Level must be between 0 and 1"):
            create_incremental_tbr_summaries(sample_tbr_dataframe, **invalid_params)

    def test_input_validation_invalid_degrees_freedom(
        self, sample_tbr_dataframe, sample_parameters
    ):
        """Test error handling for invalid degrees of freedom."""
        invalid_params = sample_parameters.copy()
        invalid_params["degrees_freedom"] = 0

        with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
            create_incremental_tbr_summaries(sample_tbr_dataframe, **invalid_params)

    def test_input_validation_invalid_sigma(
        self, sample_tbr_dataframe, sample_parameters
    ):
        """Test error handling for invalid sigma parameter."""
        invalid_params = sample_parameters.copy()
        invalid_params["sigma"] = -1.0

        with pytest.raises(ValueError, match="Sigma must be positive"):
            create_incremental_tbr_summaries(sample_tbr_dataframe, **invalid_params)

    def test_no_test_period_data(self, sample_parameters):
        """Test error handling when no test period data exists."""
        # Only learning period data (period == 0)
        learning_only_df = pd.DataFrame(
            {"period": [0, 0, 0], "cumdif": [0.0, 0.0, 0.0], "cumsd": [0.0, 0.0, 0.0]}
        )

        with pytest.raises(ValueError, match="no test period data|period == 1"):
            create_incremental_tbr_summaries(learning_only_df, **sample_parameters)

    def test_single_test_day(self, sample_parameters):
        """Test incremental analysis with single test day."""
        single_day_df = pd.DataFrame(
            {"period": [0, 0, 1], "cumdif": [0.0, 0.0, 15.2], "cumsd": [0.0, 0.0, 6.1]}
        )

        result = create_incremental_tbr_summaries(single_day_df, **sample_parameters)

        # Should have exactly one row
        assert len(result) == 1
        assert result["test_day"].iloc[0] == 1
        assert result["estimate"].iloc[0] == 15.2

    def test_multiple_test_days_progression(self, sample_parameters):
        """Test incremental analysis with multiple test days."""
        multi_day_df = pd.DataFrame(
            {
                "period": [0, 0, 0, 1, 1, 1, 1, 1],
                "cumdif": [0.0, 0.0, 0.0, 8.5, 18.2, 29.1, 42.3, 56.8],
                "cumsd": [0.0, 0.0, 0.0, 4.1, 6.2, 8.0, 9.5, 10.8],
            }
        )

        result = create_incremental_tbr_summaries(multi_day_df, **sample_parameters)

        # Should have 5 rows (5 test days)
        assert len(result) == 5

        # Verify test_day progression
        expected_days = [1, 2, 3, 4, 5]
        assert result["test_day"].tolist() == expected_days

        # Verify cumulative estimates are increasing
        estimates = result["estimate"].tolist()
        for i in range(1, len(estimates)):
            assert estimates[i] >= estimates[i - 1], "Estimates should be cumulative"

    def test_credible_interval_properties(
        self, sample_tbr_dataframe, sample_parameters
    ):
        """Test credible interval mathematical properties."""
        result = create_incremental_tbr_summaries(
            sample_tbr_dataframe, **sample_parameters
        )

        for _, row in result.iterrows():
            # Lower bound should be less than estimate
            assert row["lower"] < row["estimate"]

            # Upper bound should be greater than estimate
            assert row["upper"] > row["estimate"]

            # Precision should be positive
            assert row["precision"] > 0

            # Interval width should equal 2 * precision
            interval_width = row["upper"] - row["lower"]
            expected_width = 2 * row["precision"]
            np.testing.assert_allclose(interval_width, expected_width, rtol=1e-10)

    def test_posterior_probability_properties(
        self, sample_tbr_dataframe, sample_parameters
    ):
        """Test posterior probability mathematical properties."""
        result = create_incremental_tbr_summaries(
            sample_tbr_dataframe, **sample_parameters
        )

        for _, row in result.iterrows():
            # Probability should be between 0 and 1
            assert 0 <= row["prob"] <= 1

            # For positive estimates with threshold=0, prob should be > 0.5
            if row["estimate"] > 0 and sample_parameters["threshold"] == 0:
                assert row["prob"] > 0.5

    def test_different_threshold_values(self, sample_tbr_dataframe, sample_parameters):
        """Test incremental analysis with different threshold values."""
        # Test with positive threshold
        params_positive = sample_parameters.copy()
        params_positive["threshold"] = 10.0

        result_positive = create_incremental_tbr_summaries(
            sample_tbr_dataframe, **params_positive
        )

        # Test with negative threshold
        params_negative = sample_parameters.copy()
        params_negative["threshold"] = -5.0

        result_negative = create_incremental_tbr_summaries(
            sample_tbr_dataframe, **params_negative
        )

        # Probabilities should be different for different thresholds
        assert not np.array_equal(
            result_positive["prob"].values, result_negative["prob"].values
        )

        # For same estimate, higher threshold should give lower probability
        for i in range(len(result_positive)):
            if result_positive.iloc[i]["estimate"] > 0:  # Only for positive estimates
                assert result_negative.iloc[i]["prob"] > result_positive.iloc[i]["prob"]

    def test_different_confidence_levels(self, sample_tbr_dataframe, sample_parameters):
        """Test incremental analysis with different confidence levels."""
        # Test with 90% confidence
        params_90 = sample_parameters.copy()
        params_90["level"] = 0.90

        result_90 = create_incremental_tbr_summaries(sample_tbr_dataframe, **params_90)

        # Test with 95% confidence
        params_95 = sample_parameters.copy()
        params_95["level"] = 0.95

        result_95 = create_incremental_tbr_summaries(sample_tbr_dataframe, **params_95)

        # Higher confidence should give wider intervals
        for i in range(len(result_90)):
            width_90 = result_90.iloc[i]["upper"] - result_90.iloc[i]["lower"]
            width_95 = result_95.iloc[i]["upper"] - result_95.iloc[i]["lower"]
            assert (
                width_95 > width_90
            ), f"95% CI should be wider than 90% CI for row {i}"

    def test_lazy_loading_integration(self):
        """Test integration with lazy loading system."""
        # Test direct import from analysis module
        # Test import from main package
        from tbr import create_incremental_tbr_summaries as main_func
        from tbr.analysis import create_incremental_tbr_summaries as analysis_func

        # Should be the same function
        assert analysis_func is main_func

        # Test specific module import
        from tbr.analysis.incremental import (
            create_incremental_tbr_summaries as incremental_func,
        )

        # Should be the same function
        assert incremental_func is main_func

    def test_module_organization_standards(self):
        """Test professional module organization standards."""
        # Test module has proper docstring
        from tbr.analysis import incremental

        assert incremental.__doc__ is not None
        assert len(incremental.__doc__) > 100  # Substantial documentation

        # Test function has comprehensive docstring
        assert create_incremental_tbr_summaries.__doc__ is not None
        assert "Parameters" in create_incremental_tbr_summaries.__doc__
        assert "Returns" in create_incremental_tbr_summaries.__doc__
        assert "Examples" in create_incremental_tbr_summaries.__doc__

    def test_backward_compatibility_imports(self):
        """Test backward compatibility with existing import patterns."""
        # These should all work without errors
        from tbr import create_incremental_tbr_summaries  # noqa: F401
        from tbr import create_incremental_tbr_summaries as main_import

        # All should reference the same function
        from tbr.analysis import create_incremental_tbr_summaries as analysis_import
        from tbr.analysis.incremental import (
            create_incremental_tbr_summaries as direct_import,
        )

        assert analysis_import is main_import is direct_import

    def test_edge_case_extreme_values(self, sample_parameters):
        """Test incremental analysis with extreme parameter values."""
        # Create dataframe with large cumulative differences
        extreme_df = pd.DataFrame(
            {
                "period": [0, 0, 1, 1],
                "cumdif": [0.0, 0.0, 1000.5, 2500.8],
                "cumsd": [0.0, 0.0, 50.2, 75.1],
            }
        )

        result = create_incremental_tbr_summaries(extreme_df, **sample_parameters)

        # Should handle extreme values without errors
        assert len(result) == 2
        assert all(np.isfinite(result["estimate"]))
        assert all(np.isfinite(result["precision"]))
        assert all(result["prob"] >= 0) and all(result["prob"] <= 1)

    def test_functional_wrapper_pattern(self, sample_tbr_dataframe, sample_parameters):
        """Test that the function properly wraps the functional implementation."""
        # Mock the functional implementation to verify it's called
        with patch(
            "tbr.analysis.incremental.functional_create_incremental_tbr_summaries"
        ) as mock_func:
            # Set up mock return value
            expected_result = pd.DataFrame({"test": [1, 2, 3]})
            mock_func.return_value = expected_result

            # Call the wrapper function
            result = create_incremental_tbr_summaries(
                sample_tbr_dataframe, **sample_parameters
            )

            # Verify the functional implementation was called with correct arguments
            mock_func.assert_called_once_with(
                tbr_dataframe=sample_tbr_dataframe, **sample_parameters
            )

            # Verify the result is returned unchanged
            pd.testing.assert_frame_equal(result, expected_result)
