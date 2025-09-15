"""Input validation utilities for TBR analysis."""

import numpy as np


def validate_array_not_empty(arr: np.ndarray, param_name: str) -> None:
    """
    Validate that array is not empty.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate
    param_name : str
        Parameter name for error messages

    Raises
    ------
    ValueError
        If array is empty
    """
    if len(arr) == 0:
        raise ValueError(f"{param_name} cannot be empty")


def validate_sample_size(
    n: int, min_size: int, param_name: str = "sample size"
) -> None:
    """
    Validate sample size for statistical operations.

    Parameters
    ----------
    n : int
        Sample size to validate
    min_size : int
        Minimum required sample size
    param_name : str, default "sample size"
        Parameter name for error messages

    Raises
    ------
    ValueError
        If sample size is insufficient
    """
    if n < 0:
        raise ValueError(f"{param_name} cannot be negative, got {n}")
    if n < min_size:
        raise ValueError(
            f"Insufficient {param_name}: {n} observations. Need at least {min_size}."
        )
