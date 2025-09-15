"""
Test suite for utility functions and constants.

This module tests utility functions, constants, and exception handling
used throughout the TBR package.
"""

import pytest

from tbr.utils.constants import CONTROL_VAL, TEST_VAL
from tbr.utils.exceptions import (
    ConvergenceError,
    InsufficientDataError,
    NumericalInstabilityError,
    TBRError,
)


class TestConstants:
    """Test package constants."""

    def test_control_val_type_and_value(self):
        """Test CONTROL_VAL constant."""
        assert isinstance(CONTROL_VAL, str)
        assert CONTROL_VAL == "control"

    def test_test_val_type_and_value(self):
        """Test TEST_VAL constant."""
        assert isinstance(TEST_VAL, str)
        assert TEST_VAL == "test"

    def test_constants_are_different(self):
        """Test that constants have different values."""
        assert CONTROL_VAL != TEST_VAL

    def test_constants_immutability(self):
        """Test that constants maintain their values."""
        # Constants should be consistent
        assert CONTROL_VAL == "control"
        assert TEST_VAL == "test"


class TestExceptions:
    """Test custom exception classes."""

    def test_tbr_error_inheritance(self):
        """Test TBRError inheritance."""
        assert issubclass(TBRError, Exception)

    def test_tbr_error_instantiation(self):
        """Test TBRError can be instantiated."""
        error = TBRError("Test error message")
        assert str(error) == "Test error message"

    def test_tbr_error_catches_subclasses(self):
        """Test that TBRError catches its subclasses."""
        with pytest.raises(TBRError):
            raise ConvergenceError("Convergence failed")

    def test_convergence_error_inheritance(self):
        """Test ConvergenceError inheritance."""
        assert issubclass(ConvergenceError, TBRError)
        assert issubclass(ConvergenceError, Exception)

    def test_convergence_error_basic_usage(self):
        """Test basic ConvergenceError usage."""
        error = ConvergenceError("Algorithm did not converge")
        assert str(error) == "Algorithm did not converge"

    def test_convergence_error_with_parameters(self):
        """Test ConvergenceError with additional parameters."""
        error = ConvergenceError("Failed to converge")
        error.max_iterations = 100
        error.tolerance = 1e-6
        assert "Failed to converge" in str(error)
        assert error.max_iterations == 100
        assert error.tolerance == 1e-6

    def test_convergence_error_optional_parameters(self):
        """Test ConvergenceError with optional parameters."""
        error = ConvergenceError("Simple convergence error")
        # Should be able to add attributes dynamically
        assert not hasattr(error, "max_iterations")

    def test_numerical_error_inheritance(self):
        """Test NumericalInstabilityError inheritance."""
        assert issubclass(NumericalInstabilityError, TBRError)
        assert issubclass(NumericalInstabilityError, Exception)

    def test_numerical_error_usage(self):
        """Test NumericalInstabilityError usage."""
        error = NumericalInstabilityError("Numerical instability detected")
        assert str(error) == "Numerical instability detected"

    def test_insufficient_data_error_inheritance(self):
        """Test InsufficientDataError inheritance."""
        assert issubclass(InsufficientDataError, TBRError)
        assert issubclass(InsufficientDataError, Exception)

    def test_insufficient_data_error_basic_usage(self):
        """Test basic InsufficientDataError usage."""
        error = InsufficientDataError("Not enough data points")
        assert str(error) == "Not enough data points"

    def test_insufficient_data_error_with_parameters(self):
        """Test InsufficientDataError with additional parameters."""
        error = InsufficientDataError("Insufficient data")
        error.required = 100
        error.actual = 50
        assert "Insufficient data" in str(error)
        assert error.required == 100
        assert error.actual == 50

    def test_insufficient_data_error_optional_parameters(self):
        """Test InsufficientDataError with optional parameters."""
        error = InsufficientDataError("Simple data error")
        # Should be able to add attributes dynamically
        assert not hasattr(error, "actual")


class TestExceptionHandling:
    """Test exception handling patterns."""

    def test_catch_specific_exception(self):
        """Test catching specific exception types."""
        with pytest.raises(ConvergenceError):
            raise ConvergenceError("Test convergence error")

        with pytest.raises(NumericalInstabilityError):
            raise NumericalInstabilityError("Test numerical error")

        with pytest.raises(InsufficientDataError):
            raise InsufficientDataError("Test data error")

    def test_catch_base_exception(self):
        """Test catching base TBRError."""
        # All custom exceptions should be catchable as TBRError
        exceptions_to_test = [
            ConvergenceError("Test"),
            NumericalInstabilityError("Test"),
            InsufficientDataError("Test"),
        ]

        for exc in exceptions_to_test:
            with pytest.raises(TBRError):
                raise exc

    def test_catch_builtin_exceptions(self):
        """Test that custom exceptions don't interfere with builtin exceptions."""
        # Standard exceptions should still work normally
        with pytest.raises(ValueError):
            raise ValueError("Standard value error")

        with pytest.raises(TypeError):
            raise TypeError("Standard type error")

        with pytest.raises(KeyError):
            raise KeyError("Standard key error")
