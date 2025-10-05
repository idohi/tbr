"""
Tests for data structure validation utilities.

This module contains comprehensive tests for the structure_validation module,
covering validation of dictionaries, tuples, nested structures, and TBR-specific
data structures used throughout the analysis pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from tbr.utils.structure_validation import (
    validate_analysis_results_tuple,
    validate_model_parameters_dict,
    validate_nested_dict_structure,
    validate_tbr_output_structure,
)


class TestModelParametersValidation:
    """Test validate_model_parameters_dict function."""

    def test_valid_model_parameters(self):
        """Test validation of valid model parameters dictionary."""
        params = {
            "alpha": 50.0,
            "beta": 0.95,
            "sigma": 25.0,
            "var_alpha": 100.0,
            "var_beta": 0.001,
            "cov_alpha_beta": -0.05,
            "degrees_freedom": 43,
            "n_pretest": 45,
            "pretest_x_mean": 1000.0,
        }

        # Should not raise any error
        validate_model_parameters_dict(params)

    def test_valid_model_parameters_custom_keys(self):
        """Test validation with custom required keys."""
        params = {"alpha": 50.0, "beta": 0.95, "sigma": 25.0}

        required_keys = ["alpha", "beta", "sigma"]
        validate_model_parameters_dict(params, required_keys)

    def test_invalid_type_not_dict(self):
        """Test validation fails when input is not a dictionary."""
        with pytest.raises(TypeError, match="Model parameters must be a dictionary"):
            validate_model_parameters_dict([1, 2, 3])

    def test_missing_required_keys(self):
        """Test validation fails when required keys are missing."""
        incomplete_params = {
            "alpha": 50.0,
            "beta": 0.95
            # Missing other required keys
        }

        with pytest.raises(ValueError, match="Missing required model parameters"):
            validate_model_parameters_dict(incomplete_params)

    def test_invalid_value_types(self):
        """Test validation fails when values have wrong types."""
        params = {
            "alpha": "not_a_number",  # Should be numeric
            "beta": 0.95,
            "sigma": 25.0,
            "var_alpha": 100.0,
            "var_beta": 0.001,
            "cov_alpha_beta": -0.05,
            "degrees_freedom": 43,
            "n_pretest": 45,
            "pretest_x_mean": 1000.0,
        }

        with pytest.raises(TypeError, match="Parameter 'alpha' must be numeric"):
            validate_model_parameters_dict(params)

    def test_non_finite_values(self):
        """Test validation fails for non-finite values."""
        params = {
            "alpha": float("inf"),  # Non-finite
            "beta": 0.95,
            "sigma": 25.0,
            "var_alpha": 100.0,
            "var_beta": 0.001,
            "cov_alpha_beta": -0.05,
            "degrees_freedom": 43,
            "n_pretest": 45,
            "pretest_x_mean": 1000.0,
        }

        with pytest.raises(ValueError, match="Parameter 'alpha' must be finite"):
            validate_model_parameters_dict(params)

    def test_nan_values(self):
        """Test validation fails for NaN values."""
        params = {
            "alpha": float("nan"),  # NaN value
            "beta": 0.95,
            "sigma": 25.0,
            "var_alpha": 100.0,
            "var_beta": 0.001,
            "cov_alpha_beta": -0.05,
            "degrees_freedom": 43,
            "n_pretest": 45,
            "pretest_x_mean": 1000.0,
        }

        with pytest.raises(ValueError, match="Parameter 'alpha' must be finite"):
            validate_model_parameters_dict(params)

    def test_negative_sigma(self):
        """Test validation fails for negative sigma."""
        params = {
            "alpha": 50.0,
            "beta": 0.95,
            "sigma": -25.0,  # Must be positive
            "var_alpha": 100.0,
            "var_beta": 0.001,
            "cov_alpha_beta": -0.05,
            "degrees_freedom": 43,
            "n_pretest": 45,
            "pretest_x_mean": 1000.0,
        }

        with pytest.raises(ValueError, match="Parameter 'sigma' must be positive"):
            validate_model_parameters_dict(params)

    def test_negative_variances(self):
        """Test validation fails for negative variances."""
        params = {
            "alpha": 50.0,
            "beta": 0.95,
            "sigma": 25.0,
            "var_alpha": -100.0,  # Must be positive
            "var_beta": 0.001,
            "cov_alpha_beta": -0.05,
            "degrees_freedom": 43,
            "n_pretest": 45,
            "pretest_x_mean": 1000.0,
        }

        with pytest.raises(ValueError, match="Parameter 'var_alpha' must be positive"):
            validate_model_parameters_dict(params)

    def test_invalid_degrees_freedom(self):
        """Test validation fails for invalid degrees of freedom."""
        params = {
            "alpha": 50.0,
            "beta": 0.95,
            "sigma": 25.0,
            "var_alpha": 100.0,
            "var_beta": 0.001,
            "cov_alpha_beta": -0.05,
            "degrees_freedom": 0,  # Must be positive
            "n_pretest": 45,
            "pretest_x_mean": 1000.0,
        }

        with pytest.raises(
            ValueError, match="Parameter 'degrees_freedom' must be a positive integer"
        ):
            validate_model_parameters_dict(params)

    def test_invalid_n_pretest(self):
        """Test validation fails for invalid n_pretest."""
        params = {
            "alpha": 50.0,
            "beta": 0.95,
            "sigma": 25.0,
            "var_alpha": 100.0,
            "var_beta": 0.001,
            "cov_alpha_beta": -0.05,
            "degrees_freedom": 43,
            "n_pretest": 2,  # Must be >= 3
            "pretest_x_mean": 1000.0,
        }

        with pytest.raises(
            ValueError, match="Parameter 'n_pretest' must be an integer >= 3"
        ):
            validate_model_parameters_dict(params)


class TestTBROutputStructureValidation:
    """Test validate_tbr_output_structure function."""

    def test_valid_tbr_output(self):
        """Test validation of valid TBR output DataFrame."""
        tbr_df = pd.DataFrame(
            {
                "period": [0, 0, 1, 1],
                "y": [100, 110, 120, 130],
                "x": [95, 105, 115, 125],
                "pred": [98, 108, 118, 128],
                "cumdif": [np.nan, np.nan, 2.0, 4.0],
                "cumsd": [0.0, 0.0, 1.5, 2.1],
            }
        )

        # Should not raise any error
        validate_tbr_output_structure(tbr_df)

    def test_valid_tbr_output_custom_columns(self):
        """Test validation with custom required columns."""
        tbr_df = pd.DataFrame({"period": [0, 1], "y": [100, 120], "x": [95, 115]})

        required_columns = ["period", "y", "x"]
        validate_tbr_output_structure(tbr_df, required_columns)

    def test_invalid_type_not_dataframe(self):
        """Test validation fails when input is not a DataFrame."""
        with pytest.raises(TypeError, match="TBR output must be a pandas DataFrame"):
            validate_tbr_output_structure({"not": "a_dataframe"})

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="TBR output DataFrame cannot be empty"):
            validate_tbr_output_structure(empty_df)

    def test_missing_required_columns(self):
        """Test validation fails when required columns are missing."""
        incomplete_df = pd.DataFrame(
            {
                "period": [0, 1],
                "y": [100, 120]
                # Missing other required columns
            }
        )

        with pytest.raises(ValueError, match="Missing required columns in TBR output"):
            validate_tbr_output_structure(incomplete_df)

    def test_invalid_period_values(self):
        """Test validation fails for invalid period values."""
        invalid_df = pd.DataFrame(
            {
                "period": [0, 1, 2, 5],  # 2 and 5 are invalid
                "y": [100, 110, 120, 130],
                "x": [95, 105, 115, 125],
                "pred": [98, 108, 118, 128],
                "cumdif": [np.nan, np.nan, 2.0, 4.0],
                "cumsd": [0.0, 0.0, 1.5, 2.1],
            }
        )

        with pytest.raises(ValueError, match="Invalid period values found"):
            validate_tbr_output_structure(invalid_df)

    def test_non_numeric_columns(self):
        """Test validation fails for non-numeric required columns."""
        invalid_df = pd.DataFrame(
            {
                "period": [0, 0, 1, 1],
                "y": ["a", "b", "c", "d"],  # Should be numeric
                "x": [95, 105, 115, 125],
                "pred": [98, 108, 118, 128],
                "cumdif": [np.nan, np.nan, 2.0, 4.0],
                "cumsd": [0.0, 0.0, 1.5, 2.1],
            }
        )

        with pytest.raises(ValueError, match="Column 'y' must be numeric"):
            validate_tbr_output_structure(invalid_df)

    def test_valid_period_values_with_baseline_cooldown(self):
        """Test validation passes for all valid period values."""
        complete_df = pd.DataFrame(
            {
                "period": [-1, 0, 1, 3],  # All valid periods
                "y": [90, 100, 120, 140],
                "x": [85, 95, 115, 135],
                "pred": [88, 98, 118, 138],
                "cumdif": [np.nan, np.nan, 2.0, 4.0],
                "cumsd": [0.0, 0.0, 1.5, 2.1],
            }
        )

        # Should not raise any error
        validate_tbr_output_structure(complete_df)

    def test_tbr_output_without_period_column(self):
        """Test validation passes when period column is not present in custom columns."""
        df_without_period = pd.DataFrame(
            {
                "y": [100, 110, 120, 130],
                "x": [95, 105, 115, 125],
                "pred": [98, 108, 118, 128],
                "cumdif": [np.nan, np.nan, 2.0, 4.0],
                "cumsd": [0.0, 0.0, 1.5, 2.1],
            }
        )

        # Use custom required columns without 'period' - period validation is skipped
        custom_columns = ["y", "x", "pred", "cumdif", "cumsd"]
        validate_tbr_output_structure(
            df_without_period, required_columns=custom_columns
        )


class TestAnalysisResultsTupleValidation:
    """Test validate_analysis_results_tuple function."""

    def test_valid_results_tuple(self):
        """Test validation of valid results tuple."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"c": [5, 6], "d": [7, 8]})
        results = (df1, df2)

        # Should not raise any error
        validate_analysis_results_tuple(results)

    def test_valid_results_tuple_custom_length(self):
        """Test validation with custom expected length."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        df3 = pd.DataFrame({"c": [5, 6]})
        results = (df1, df2, df3)

        validate_analysis_results_tuple(results, expected_length=3)

    def test_invalid_type_not_tuple(self):
        """Test validation fails when input is not a tuple."""
        df = pd.DataFrame({"a": [1, 2]})

        with pytest.raises(TypeError, match="Analysis results must be a tuple"):
            validate_analysis_results_tuple([df, df])  # List instead of tuple

    def test_wrong_tuple_length(self):
        """Test validation fails for wrong tuple length."""
        df = pd.DataFrame({"a": [1, 2]})
        results = (df,)  # Length 1, but expecting 2

        with pytest.raises(
            ValueError, match="Expected tuple of length 2, got length 1"
        ):
            validate_analysis_results_tuple(results)

    def test_non_dataframe_elements(self):
        """Test validation fails when tuple contains non-DataFrame elements."""
        df = pd.DataFrame({"a": [1, 2]})
        results = (df, "not_a_dataframe")

        with pytest.raises(TypeError, match="Element 1 must be a pandas DataFrame"):
            validate_analysis_results_tuple(results)

    def test_empty_dataframe_in_tuple(self):
        """Test validation fails when tuple contains empty DataFrames."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame()  # Empty
        results = (df1, df2)

        with pytest.raises(ValueError, match="DataFrame at position 1 cannot be empty"):
            validate_analysis_results_tuple(results)

    def test_single_element_tuple(self):
        """Test validation with single element tuple."""
        df = pd.DataFrame({"a": [1, 2]})
        results = (df,)

        validate_analysis_results_tuple(results, expected_length=1)


class TestNestedDictStructureValidation:
    """Test validate_nested_dict_structure function."""

    def test_valid_nested_dict(self):
        """Test validation of valid nested dictionary."""
        config = {"level": 0.80, "threshold": 0.0, "test_end_inclusive": False}

        validate_nested_dict_structure(
            config,
            required_keys=["level", "threshold"],
            value_types={
                "level": float,
                "threshold": float,
                "test_end_inclusive": bool,
            },
        )

    def test_valid_nested_dict_minimal(self):
        """Test validation with minimal requirements."""
        data = {"key1": "value1", "key2": 42}

        validate_nested_dict_structure(data, required_keys=["key1"])

    def test_invalid_type_not_dict(self):
        """Test validation fails when input is not a dictionary."""
        with pytest.raises(TypeError, match="Data must be a dictionary"):
            validate_nested_dict_structure("not_a_dict", required_keys=["key"])

    def test_missing_required_keys(self):
        """Test validation fails when required keys are missing."""
        incomplete_data = {"level": 0.80}

        with pytest.raises(ValueError, match="Missing required keys"):
            validate_nested_dict_structure(
                incomplete_data, required_keys=["level", "threshold"]
            )

    def test_unexpected_keys_not_allowed(self):
        """Test validation fails for unexpected keys when not allowed."""
        data = {"level": 0.80, "threshold": 0.0, "extra_key": "unexpected"}

        with pytest.raises(ValueError, match="Unexpected keys found"):
            validate_nested_dict_structure(
                data, required_keys=["level", "threshold"], allow_extra_keys=False
            )

    def test_no_extra_keys_when_not_allowed(self):
        """Test validation passes when no extra keys and allow_extra_keys=False."""
        data = {"level": 0.80, "threshold": 0.0}

        # Should not raise error - no extra keys to complain about
        validate_nested_dict_structure(
            data, required_keys=["level", "threshold"], allow_extra_keys=False
        )

    def test_unexpected_keys_allowed(self):
        """Test validation passes for unexpected keys when allowed."""
        data = {"level": 0.80, "threshold": 0.0, "extra_key": "allowed"}

        # Should not raise error (allow_extra_keys=True by default)
        validate_nested_dict_structure(data, required_keys=["level", "threshold"])

    def test_wrong_value_types(self):
        """Test validation fails for wrong value types."""
        data = {
            "level": "0.80",  # Should be float
            "threshold": 0.0,
            "test_end_inclusive": False,
        }

        with pytest.raises(TypeError, match="Key 'level' expected type float"):
            validate_nested_dict_structure(
                data,
                required_keys=["level", "threshold"],
                value_types={
                    "level": float,
                    "threshold": float,
                    "test_end_inclusive": bool,
                },
            )

    def test_partial_type_checking(self):
        """Test validation with partial type checking."""
        data = {
            "level": 0.80,
            "threshold": 0.0,
            "test_end_inclusive": False,
            "extra": "no_type_check",
        }

        # Only check types for specified keys
        validate_nested_dict_structure(
            data,
            required_keys=["level"],
            value_types={"level": float},  # Only checking 'level'
        )

    def test_value_types_key_not_in_data(self):
        """Test validation with value_types key not present in data."""
        data = {"level": 0.80, "threshold": 0.0}

        # value_types specifies a key not in data - should not cause error
        validate_nested_dict_structure(
            data,
            required_keys=["level", "threshold"],
            value_types={
                "level": float,
                "missing_key": str,
            },  # 'missing_key' not in data
        )

    def test_complex_nested_structure(self):
        """Test validation with complex nested structure."""
        complex_data = {
            "analysis_params": {"level": 0.80, "threshold": 0.0},
            "data_params": {"time_col": "date", "control_col": "control"},
            "flags": {"test_end_inclusive": False, "validate_continuity": True},
        }

        # Validate top-level structure
        validate_nested_dict_structure(
            complex_data,
            required_keys=["analysis_params", "data_params"],
            value_types={"analysis_params": dict, "data_params": dict, "flags": dict},
        )


class TestStructureValidationIntegration:
    """Test integration of structure validation with TBR analysis."""

    def test_model_parameters_integration(self):
        """Test model parameters validation in context of TBR analysis."""
        # Simulate model parameters from fit_tbr_regression_model
        model_params = {
            "alpha": 52.3,
            "beta": 0.987,
            "sigma": 23.5,
            "var_alpha": 98.7,
            "var_beta": 0.0012,
            "cov_alpha_beta": -0.045,
            "degrees_freedom": 42,
            "n_pretest": 44,
            "pretest_x_mean": 1005.2,
        }

        # Should validate successfully
        validate_model_parameters_dict(model_params)

    def test_tbr_output_integration(self):
        """Test TBR output validation in context of analysis pipeline."""
        # Simulate TBR analysis output
        tbr_output = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=6),
                "period": [-1, 0, 0, 1, 1, 3],
                "y": [95, 100, 110, 120, 130, 125],
                "x": [90, 95, 105, 115, 125, 120],
                "pred": [np.nan, 98, 108, 118, 128, 123],
                "predsd": [np.nan, 0, 0, 2.1, 2.3, 2.5],
                "dif": [np.nan, 2, 2, 2, 2, 2],
                "cumdif": [np.nan, np.nan, np.nan, 2, 4, 6],
                "cumsd": [np.nan, 0, 0, 1.5, 2.1, 2.6],
                "estsd": [np.nan, 1.2, 1.3, np.nan, np.nan, np.nan],
            }
        )

        # Should validate successfully with extended column set
        extended_columns = [
            "period",
            "y",
            "x",
            "pred",
            "cumdif",
            "cumsd",
            "predsd",
            "dif",
            "estsd",
        ]
        validate_tbr_output_structure(tbr_output, required_columns=extended_columns)

    def test_analysis_results_integration(self):
        """Test analysis results validation in context of perform_tbr_analysis."""
        # Simulate perform_tbr_analysis return values
        tbr_dataframe = pd.DataFrame(
            {
                "period": [0, 1, 1],
                "y": [100, 120, 130],
                "x": [95, 115, 125],
                "pred": [98, 118, 128],
                "cumdif": [np.nan, 2, 4],
                "cumsd": [0, 1.5, 2.1],
            }
        )

        daily_summaries = pd.DataFrame(
            {
                "test_day": [1, 2],
                "estimate": [2.0, 4.0],
                "precision": [3.2, 4.5],
                "lower": [-1.2, -0.5],
                "upper": [5.2, 8.5],
            }
        )

        results = (tbr_dataframe, daily_summaries)

        # Should validate successfully
        validate_analysis_results_tuple(results)

    def test_configuration_validation(self):
        """Test configuration dictionary validation."""
        analysis_config = {
            "level": 0.80,
            "threshold": 0.0,
            "test_end_inclusive": False,
            "validate_continuity": True,
            "min_pretest_days": 14,
        }

        validate_nested_dict_structure(
            analysis_config,
            required_keys=["level", "threshold"],
            value_types={
                "level": float,
                "threshold": float,
                "test_end_inclusive": bool,
                "validate_continuity": bool,
                "min_pretest_days": int,
            },
        )
