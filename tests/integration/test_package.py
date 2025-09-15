"""
Integration tests for TBR package.

This module tests package-level imports, module structure, and integration
between different components of the TBR package.
"""

import numpy as np
import pytest


class TestPackageImports:
    """Test cases for package-level imports."""

    def test_main_package_import(self) -> None:
        """Test main package can be imported."""
        import tbr

        assert hasattr(tbr, "__version__")
        assert hasattr(tbr, "__author__")
        assert hasattr(tbr, "__license__")

    def test_functional_module_import(self) -> None:
        """Test functional module can be imported."""
        from tbr.functional import tbr_functions

        assert hasattr(tbr_functions, "perform_tbr_analysis")

    def test_utils_imports(self) -> None:
        """Test utils module imports work correctly."""
        from tbr.utils import CONTROL_VAL, TEST_VAL, ConvergenceError, TBRError

        assert CONTROL_VAL == "control"
        assert TEST_VAL == "test"
        assert issubclass(ConvergenceError, TBRError)

    def test_main_function_import(self) -> None:
        """Test main TBR analysis function can be imported."""
        from tbr.functional.tbr_functions import perform_tbr_analysis

        assert callable(perform_tbr_analysis)

    def test_validation_utilities_import(self) -> None:
        """Test validation utilities can be imported."""
        from tbr.utils.validation import validate_array_not_empty, validate_sample_size

        assert callable(validate_array_not_empty)
        assert callable(validate_sample_size)


class TestBasicFunctionality:
    """Test cases for basic package functionality."""

    def test_constants_are_accessible(self) -> None:
        """Test package constants are accessible and correct."""
        from tbr.utils.constants import CONTROL_VAL, TEST_VAL

        assert isinstance(CONTROL_VAL, str)
        assert isinstance(TEST_VAL, str)
        assert CONTROL_VAL != TEST_VAL

    def test_exceptions_can_be_raised(self) -> None:
        """Test custom exceptions can be raised and caught."""
        from tbr.utils.exceptions import ConvergenceError, TBRError

        with pytest.raises(TBRError):
            raise TBRError("Test error")

        with pytest.raises(ConvergenceError):  # type: ignore[unreachable]
            raise ConvergenceError("Test convergence error")

    def test_validation_functions_work(self) -> None:
        """Test validation functions work with real data."""
        from tbr.utils.validation import validate_array_not_empty, validate_sample_size

        # Test with valid data
        arr = np.array([1, 2, 3, 4, 5])
        validate_array_not_empty(arr, "test_array")
        validate_sample_size(len(arr), 3, "test_size")

        # Test with invalid data
        empty_arr = np.array([])
        with pytest.raises(ValueError):
            validate_array_not_empty(empty_arr, "empty_array")


@pytest.mark.integration
class TestTBRAnalysisImportAndStructure:
    """Integration tests for TBR analysis function structure."""

    def test_perform_tbr_analysis_signature(self) -> None:
        """Test perform_tbr_analysis function has expected signature."""
        import inspect

        from tbr.functional.tbr_functions import perform_tbr_analysis

        sig = inspect.signature(perform_tbr_analysis)
        params = list(sig.parameters.keys())

        # Check key parameters exist
        expected_params = [
            "data",
            "time_col",
            "control_col",
            "test_col",
            "pretest_start",
            "test_start",
            "test_end",
            "level",
            "threshold",
        ]

        for param in expected_params:
            assert (
                param in params
            ), f"Expected parameter '{param}' not found in function signature"

    def test_tbr_functions_module_structure(self) -> None:
        """Test tbr_functions module has expected functions."""
        from tbr.functional import tbr_functions

        expected_functions = [
            "perform_tbr_analysis",
            "fit_tbr_regression_model",
            "create_tbr_summary",
            "calculate_sum_x_squared_deviations",
        ]

        for func_name in expected_functions:
            assert hasattr(
                tbr_functions, func_name
            ), f"Function '{func_name}' not found in module"
            assert callable(
                getattr(tbr_functions, func_name)
            ), f"'{func_name}' is not callable"


@pytest.mark.integration
class TestPackageMetadata:
    """Integration tests for package metadata."""

    def test_package_version_format(self) -> None:
        """Test package version follows expected format."""
        import tbr

        version = tbr.__version__
        assert isinstance(version, str)
        assert len(version) > 0

        # Check it follows semantic versioning pattern (basic check)
        parts = version.split(".")
        assert len(parts) >= 2, "Version should have at least major.minor format"

    def test_package_author_and_license(self) -> None:
        """Test package has author and license information."""
        import tbr

        assert hasattr(tbr, "__author__")
        assert hasattr(tbr, "__license__")
        assert isinstance(tbr.__author__, str)
        assert isinstance(tbr.__license__, str)
        assert len(tbr.__author__) > 0
        assert len(tbr.__license__) > 0
