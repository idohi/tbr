# Python Docstring Conventions for Professional PyPI Packages

This document provides comprehensive guidelines for writing professional docstrings that comply with Python standards and are used by major PyPI packages.

## Table of Contents
1. [Basic Docstring Structure](#basic-docstring-structure)
2. [PEP 257 Standards](#pep-257-standards)
3. [Docstring Styles](#docstring-styles)
4. [Module-Level Docstrings](#module-level-docstrings)
5. [Function/Method Docstrings](#functionmethod-docstrings)
6. [Class Docstrings](#class-docstrings)
7. [Standard Sections](#standard-sections)
8. [Code Examples and Doctests](#code-examples-and-doctests)
9. [Professional PyPI Examples](#professional-pypi-examples)
10. [Common Mistakes to Avoid](#common-mistakes-to-avoid)
11. [Tools and Validation](#tools-and-validation)

---

## Basic Docstring Structure

### Triple Quotes
Always use triple double quotes (`"""`) for docstrings:

```python
def function():
    """This is the correct format."""
    pass

# WRONG - don't use single quotes or triple single quotes
def function():
    '''This is incorrect.'''
    pass
```

### Basic Format
```python
"""One-line summary ending with a period.

Optional longer description that provides more detail about what
the function/class/module does. This can span multiple paragraphs.

Parameters
----------
param1 : type
    Description of parameter 1
param2 : type, optional
    Description of parameter 2 (default is None)

Returns
-------
type
    Description of return value

Examples
--------
>>> example_usage()
'expected_output'
"""
```

---

## PEP 257 Standards

[PEP 257](https://peps.python.org/pep-0257/) defines the official Python docstring conventions:

### Key Rules:
1. **Triple quotes**: Always use `"""`
2. **One-line summary**: First line should be a brief summary
3. **Blank line**: Separate summary from detailed description
4. **Imperative mood**: Use "Return the result" not "Returns the result"
5. **Period ending**: End the summary line with a period
6. **No blank line**: Don't put a blank line after the opening quotes

### Correct PEP 257 Format:
```python
def calculate_mean(values):
    """Calculate the arithmetic mean of a list of numbers.

    This function takes a list of numeric values and returns their
    arithmetic mean. It handles empty lists by returning 0.

    Parameters
    ----------
    values : list of float
        List of numeric values

    Returns
    -------
    float
        Arithmetic mean of the input values
    """
```

---

## Docstring Styles

### 1. NumPy Style (Most Popular for Scientific Packages)
```python
def function(param1, param2=None):
    """Brief summary of function.

    Longer description of function behavior and purpose.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, optional
        Description of param2. Default is None.

    Returns
    -------
    bool
        Description of return value

    Raises
    ------
    ValueError
        When param1 is invalid
    TypeError
        When param2 is not an integer

    Examples
    --------
    >>> function("hello", 42)
    True
    """
```

### 2. Google Style
```python
def function(param1, param2=None):
    """Brief summary of function.

    Longer description of function behavior and purpose.

    Args:
        param1 (str): Description of param1
        param2 (int, optional): Description of param2. Defaults to None.

    Returns:
        bool: Description of return value

    Raises:
        ValueError: When param1 is invalid
        TypeError: When param2 is not an integer

    Example:
        >>> function("hello", 42)
        True
    """
```

### 3. Sphinx Style (reStructuredText)
```python
def function(param1, param2=None):
    """Brief summary of function.

    Longer description of function behavior and purpose.

    :param param1: Description of param1
    :type param1: str
    :param param2: Description of param2
    :type param2: int, optional
    :returns: Description of return value
    :rtype: bool
    :raises ValueError: When param1 is invalid
    :raises TypeError: When param2 is not an integer

    Example::

        >>> function("hello", 42)
        True
    """
```

**Recommendation**: Use **NumPy style** for scientific/data packages, **Google style** for general applications.

---

## Module-Level Docstrings

Module docstrings should be placed at the top of the file, before any imports:

```python
"""Module for statistical analysis functions.

This module provides functions for descriptive and inferential statistics,
including measures of central tendency, dispersion, and hypothesis testing.

The module is designed to work with NumPy arrays and pandas DataFrames,
providing a consistent interface for statistical computations.

Examples
--------
>>> import statistics_module
>>> data = [1, 2, 3, 4, 5]
>>> statistics_module.mean(data)
3.0

Notes
-----
All functions assume input data is numeric and handle missing values
by excluding them from calculations.
"""

import numpy as np
import pandas as pd
```

### Standard Module Sections:
- **Brief description**: What the module does
- **Longer description**: How it fits into the larger system
- **Examples**: Basic usage patterns
- **Notes**: Important implementation details or assumptions

---

## Function/Method Docstrings

### Complete Function Example:
```python
def process_data(data, method='mean', axis=0, skipna=True):
    """Process data using specified statistical method.

    Apply a statistical aggregation method to the input data along
    the specified axis. This function provides a unified interface
    for common data processing operations.

    Parameters
    ----------
    data : array_like
        Input data to process. Can be list, tuple, numpy array,
        or pandas DataFrame.
    method : {'mean', 'median', 'sum', 'std'}, default 'mean'
        Statistical method to apply to the data.
    axis : int, default 0
        Axis along which to apply the method.
        - 0: Apply along rows (column-wise aggregation)
        - 1: Apply along columns (row-wise aggregation)
    skipna : bool, default True
        Whether to skip NaN values in the calculation.

    Returns
    -------
    numpy.ndarray or scalar
        Result of applying the statistical method. Returns scalar
        if input is 1-dimensional, otherwise returns array.

    Raises
    ------
    ValueError
        If method is not one of the supported options.
    TypeError
        If data cannot be converted to numeric format.

    Examples
    --------
    >>> import numpy as np
    >>> data = [[1, 2, 3], [4, 5, 6]]
    >>> process_data(data, method='mean', axis=0)
    array([2.5, 3.5, 4.5])

    >>> process_data([1, 2, 3, 4, 5], method='median')
    3.0

    Notes
    -----
    This function internally converts input to numpy arrays for
    consistent behavior across different input types.
    """
```

---

## Class Docstrings

### Class Documentation:
```python
class DataProcessor:
    """Process and analyze numerical data with statistical methods.

    This class provides a unified interface for common data processing
    operations including cleaning, transformation, and statistical analysis.
    It supports both pandas DataFrames and numpy arrays as input.

    Parameters
    ----------
    data : array_like
        Input data to process
    validate : bool, default True
        Whether to validate input data on initialization

    Attributes
    ----------
    data : numpy.ndarray
        Processed input data
    is_valid : bool
        Whether the data passed validation
    shape : tuple
        Shape of the processed data

    Examples
    --------
    >>> processor = DataProcessor([[1, 2], [3, 4]])
    >>> processor.mean()
    2.5

    Notes
    -----
    The class automatically handles missing values by default,
    but this behavior can be customized using the skipna parameter
    in individual methods.
    """

    def __init__(self, data, validate=True):
        """Initialize the DataProcessor."""
        # Implementation here
        pass

    def mean(self, axis=None, skipna=True):
        """Calculate mean along specified axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to calculate mean. If None, calculate
            mean of flattened array.
        skipna : bool, default True
            Whether to skip NaN values.

        Returns
        -------
        float or numpy.ndarray
            Mean value(s)
        """
        # Implementation here
        pass
```

---

## Standard Sections

### Common Section Headers (NumPy Style):

#### Required Sections:
- **Parameters**: Function/method arguments
- **Returns**: What the function returns

#### Optional Sections:
- **Raises**: Exceptions that may be raised
- **Examples**: Usage examples with doctests
- **Notes**: Additional implementation details
- **References**: Academic papers or external documentation
- **See Also**: Related functions

### Section Format:
```python
"""
Parameters
----------
param_name : type
    Description of the parameter

param_name : type, optional
    Description of optional parameter. Default is value.

param_name : {'option1', 'option2'}, default 'option1'
    Description of parameter with limited options.

Returns
-------
return_type
    Description of return value

Raises
------
ExceptionType
    When this exception occurs

Examples
--------
>>> example_code()
expected_output

Notes
-----
Additional technical details or implementation notes.

References
----------
.. [1] Author, "Title", Journal, Year.

See Also
--------
related_function : Brief description
another_function : Brief description
"""
```

---

## Code Examples and Doctests

### Doctest Format:
Use the interactive Python prompt format:

```python
"""
Examples
--------
>>> import numpy as np
>>> data = [1, 2, 3, 4, 5]
>>> result = calculate_mean(data)
>>> print(f"{result:.1f}")
3.0

>>> # Multi-line example
>>> complex_data = np.array([[1, 2, 3],
...                         [4, 5, 6]])
>>> result = calculate_mean(complex_data, axis=0)
>>> result
array([2.5, 3.5, 4.5])
"""
```

### Doctest Rules:
- Use `>>>` for the primary prompt
- Use `...` for continuation lines
- Include expected output when helpful
- Keep examples simple and focused
- Test edge cases when relevant

### Example Formatting:
```python
# Good examples:
>>> simple_function(42)
84

>>> multi_line_call(
...     parameter1="value",
...     parameter2=42
... )
'expected_result'

# Show output when it's informative:
>>> data_frame.head()
   A  B  C
0  1  2  3
1  4  5  6

# Don't show output for obvious cases:
>>> processor.validate_data()  # No output needed for void functions
```

---

## Professional PyPI Examples

### NumPy Example:
```python
def array_split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays.

    Please refer to the ``split`` documentation.  The only difference
    between these functions is that ``array_split`` allows
    `indices_or_sections` to be an integer that does not equally
    divide the axis. For an array of length l that should be split
    into n sections, it returns l % n sub-arrays of size l//n + 1
    and the rest of size l//n.

    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`.  If such a split is not possible,
        an array of length l that should be split into n sections, it returns
        l % n sub-arrays of size l//n + 1 and the rest of size l//n.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of sub-arrays.

    See Also
    --------
    split : Split array into multiple sub-arrays of equal size.
    """
```

### Pandas Example:
```python
def read_csv(filepath_or_buffer, sep=',', delimiter=None, **kwargs):
    """Read a comma-separated values (csv) file into DataFrame.

    Also supports optionally iterating or breaking of the file
    into chunks.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL.
    sep : str, default ','
        Delimiter to use. If sep is None, the C engine cannot automatically
        detect the separator, but the Python parsing engine can.
    delimiter : str, optional
        Alias for sep.

    Returns
    -------
    DataFrame or TextFileReader
        A comma-separated values (csv) file is returned as two-dimensional
        data structure with labeled axes.

    Examples
    --------
    >>> pd.read_csv('data.csv')  # doctest: +SKIP
       col1  col2
    0     1     2
    1     3     4
    """
```

### Scikit-learn Example:
```python
def fit(self, X, y=None, sample_weight=None):
    """Fit the model according to the given training data.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples,)
        Target vector relative to X.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.

    Returns
    -------
    self : object
        Returns the instance itself.
    """
```

---

## Common Mistakes to Avoid

### 1. Inconsistent Style
```python
# BAD - mixing styles
def function(param):
    """Brief description.

    Args:  # Google style
        param (str): Description

    Parameters  # NumPy style
    ----------
    param : str
        Description
    """

# GOOD - consistent style
def function(param):
    """Brief description.

    Parameters
    ----------
    param : str
        Description
    """
```

### 2. Missing Type Information
```python
# BAD - no type hints
def process(data, method):
    """Process data."""

# GOOD - clear types
def process(data, method):
    """Process data using specified method.

    Parameters
    ----------
    data : array_like
        Input data to process
    method : {'mean', 'sum', 'max'}
        Processing method to apply
    """
```

### 3. Overly Long Lines
```python
# BAD - too long
def function():
    """This is a very long description that goes on and on and exceeds the recommended line length which makes it hard to read."""

# GOOD - wrapped properly
def function():
    """This is a long description that is properly wrapped to maintain
    readability and follows Python style guidelines for line length.
    """
```

### 4. No Examples
```python
# BAD - no examples
def complex_function(data, params):
    """Perform complex data transformation."""

# GOOD - with examples
def complex_function(data, params):
    """Perform complex data transformation.

    Examples
    --------
    >>> data = [1, 2, 3, 4, 5]
    >>> params = {'method': 'normalize'}
    >>> complex_function(data, params)
    [0.0, 0.25, 0.5, 0.75, 1.0]
    """
```

### 5. Using Markdown in Docstrings
```python
# BAD - Markdown formatting
def function():
    """Process data.

    **Parameters:**
    - data: Input data
    - method: Processing method
    """

# GOOD - Standard Python formatting
def function():
    """Process data.

    Parameters
    ----------
    data : array_like
        Input data
    method : str
        Processing method
    """
```

---

## Tools and Validation

### Documentation Tools:
1. **Sphinx**: Generates HTML documentation from docstrings
2. **pdoc**: Simpler alternative to Sphinx
3. **pydoc**: Built-in Python documentation generator

### Linting Tools:
1. **pydocstyle**: Checks docstring style compliance
2. **flake8-docstrings**: Flake8 plugin for docstring checking
3. **interrogate**: Measures docstring coverage

### Example pydocstyle Configuration:
```ini
# setup.cfg
[pydocstyle]
convention = numpy
add-ignore = D100,D104
```

### Testing Doctests:
```bash
# Run doctests
python -m doctest module.py

# Run with pytest
pytest --doctest-modules
```

---

## Summary Checklist

For professional PyPI package docstrings:

- [ ] Use triple double quotes (`"""`)
- [ ] Start with one-line summary ending in period
- [ ] Use consistent style (NumPy recommended for data/science packages)
- [ ] Include Parameters and Returns sections
- [ ] Add Examples with doctest format (`>>>`)
- [ ] Use proper section headers with underlines
- [ ] Keep lines under 79-88 characters
- [ ] Use imperative mood ("Calculate the mean")
- [ ] Include type information for all parameters
- [ ] Add Raises section for exceptions
- [ ] Test examples with doctest
- [ ] Avoid Markdown formatting
- [ ] Be concise but complete
- [ ] Follow PEP 257 standards

This guide ensures your docstrings meet professional standards used by major PyPI packages like NumPy, Pandas, Scikit-learn, and others.
