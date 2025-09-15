"""Unit tests for TBR package custom exceptions."""

import pytest

from tbr.utils.exceptions import (
    ConvergenceError,
    InsufficientDataError,
    NumericalInstabilityError,
    TBRError,
)


class TestTBRError:
    """Test cases for base TBRError exception."""

    def test_tbr_error_inheritance(self) -> None:
        """Test TBRError inherits from Exception."""
        assert issubclass(TBRError, Exception)

    def test_tbr_error_instantiation(self) -> None:
        """Test TBRError can be instantiated and raised."""
        error = TBRError("Test error message")
        assert str(error) == "Test error message"

        with pytest.raises(TBRError):
            raise TBRError("Test error")

    def test_tbr_error_catches_subclasses(self) -> None:
        """Test TBRError catches all custom TBR exceptions."""
        with pytest.raises(TBRError):
            raise ConvergenceError("Convergence failed")

        with pytest.raises(TBRError):  # type: ignore[unreachable]
            raise NumericalInstabilityError("Numerical issue")

        with pytest.raises(TBRError):
            raise InsufficientDataError("Not enough data")


class TestConvergenceError:
    """Test cases for ConvergenceError exception."""

    def test_convergence_error_inheritance(self) -> None:
        """Test ConvergenceError inherits correctly."""
        assert issubclass(ConvergenceError, TBRError)
        assert issubclass(ConvergenceError, RuntimeError)

    def test_convergence_error_basic_usage(self) -> None:
        """Test basic ConvergenceError functionality."""
        error = ConvergenceError("Algorithm failed to converge")
        assert str(error) == "Algorithm failed to converge"

        with pytest.raises(ConvergenceError):
            raise ConvergenceError("Test convergence failure")

    def test_convergence_error_with_parameters(self) -> None:
        """Test ConvergenceError with optional parameters."""
        error = ConvergenceError("Failed to converge", iterations=1000, tolerance=1e-6)

        assert str(error) == "Failed to converge"
        assert error.iterations == 1000
        assert error.tolerance == 1e-6

    def test_convergence_error_optional_parameters(self) -> None:
        """Test ConvergenceError with None parameters."""
        error = ConvergenceError("Simple message")
        assert error.iterations is None
        assert error.tolerance is None


class TestNumericalInstabilityError:
    """Test cases for NumericalInstabilityError exception."""

    def test_numerical_error_inheritance(self) -> None:
        """Test NumericalInstabilityError inherits correctly."""
        assert issubclass(NumericalInstabilityError, TBRError)
        assert issubclass(NumericalInstabilityError, RuntimeError)

    def test_numerical_error_usage(self) -> None:
        """Test NumericalInstabilityError functionality."""
        error = NumericalInstabilityError("Matrix is singular")
        assert str(error) == "Matrix is singular"

        with pytest.raises(NumericalInstabilityError):
            raise NumericalInstabilityError("Numerical instability")


class TestInsufficientDataError:
    """Test cases for InsufficientDataError exception."""

    def test_insufficient_data_error_inheritance(self) -> None:
        """Test InsufficientDataError inherits correctly."""
        assert issubclass(InsufficientDataError, TBRError)
        assert issubclass(InsufficientDataError, ValueError)

    def test_insufficient_data_error_basic_usage(self) -> None:
        """Test basic InsufficientDataError functionality."""
        error = InsufficientDataError("Need more data")
        assert str(error) == "Need more data"

        with pytest.raises(InsufficientDataError):
            raise InsufficientDataError("Insufficient sample size")

    def test_insufficient_data_error_with_parameters(self) -> None:
        """Test InsufficientDataError with optional parameters."""
        error = InsufficientDataError(
            "Need at least 30 observations", required=30, available=15
        )

        assert str(error) == "Need at least 30 observations"
        assert error.required == 30
        assert error.available == 15

    def test_insufficient_data_error_optional_parameters(self) -> None:
        """Test InsufficientDataError with None parameters."""
        error = InsufficientDataError("Simple message")
        assert error.required is None
        assert error.available is None


class TestExceptionHandling:
    """Test cases for exception handling patterns."""

    def test_catch_specific_exception(self) -> None:
        """Test catching specific custom exceptions."""
        with pytest.raises(ConvergenceError):
            raise ConvergenceError("Specific error")

    def test_catch_base_exception(self) -> None:
        """Test catching base TBRError catches all custom exceptions."""
        exceptions_to_test = [
            ConvergenceError("Conv error"),
            NumericalInstabilityError("Num error"),
            InsufficientDataError("Data error"),
        ]

        for exc in exceptions_to_test:
            with pytest.raises(TBRError):
                raise exc

    def test_catch_builtin_exceptions(self) -> None:
        """Test custom exceptions can be caught by their builtin parents."""
        with pytest.raises(RuntimeError):
            raise ConvergenceError("Runtime error")

        with pytest.raises(RuntimeError):  # type: ignore[unreachable]
            raise NumericalInstabilityError("Runtime error")

        with pytest.raises(ValueError):
            raise InsufficientDataError("Value error")
