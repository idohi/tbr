# TBR Testing Guide

> **Professional Testing Documentation for Scientific PyPI Package**

This guide provides comprehensive instructions for running, writing, and contributing tests to the TBR package. Our testing framework follows scientific PyPI package standards used by NumPy, SciPy, pandas, and other leading packages.

## 🎯 **Quick Start**

```bash
# Run all tests (recommended)
make test

# Run all tests with coverage report
make test-cov

# Run tests without coverage (faster)
pytest tests/ -v --no-cov
```

## 📊 **Test Structure Overview**

TBR uses a **4-tier testing architecture** designed for scientific packages:

```
tests/
├── unit/           # Unit tests (35 tests)
├── integration/    # Integration tests (12 tests)
├── mathematical/   # Mathematical validation (15 tests)
├── performance/    # Performance tests (7 tests)
└── conftest.py     # Shared fixtures and utilities
```

### **Test Categories Explained**

| **Category** | **Purpose** | **When to Use** | **Examples** |
|--------------|-------------|-----------------|--------------|
| **Unit** | Test individual functions/classes | Testing single function behavior | `test_constants.py`, `test_exceptions.py` |
| **Integration** | Test module interactions | Package imports, API structure | `test_package_imports.py` |
| **Mathematical** | Validate mathematical accuracy | Known-value testing, algorithm validation | `test_basic_calculations.py` |
| **Performance** | Test speed and scalability | Performance regression, memory usage | `test_basic_performance.py` |

## 🏷️ **Test Markers and Categories**

### **Available Markers**

Our pytest configuration includes professional test markers:

```python
# Defined in pyproject.toml
markers = [
    "unit: Unit tests for individual functions and classes",
    "integration: Integration tests for complete workflows",
    "mathematical: Mathematical validation tests against reference implementations",
    "performance: Performance and benchmarking tests",
    "slow: Tests that take significant time to run",
    "requires_data: Tests that require external data files",
]
```

### **Running Tests by Marker**

```bash
# Run only mathematical validation tests
pytest -m "mathematical" -v

# Run only performance tests
pytest -m "performance" -v

# Skip slow tests for quick development
pytest -m "not slow" -v

# Run multiple categories
pytest -m "unit or integration" -v
```

## 🚀 **Test Execution Options**

### **Basic Test Execution**

```bash
# All tests with verbose output
pytest tests/ -v

# All tests with minimal output
pytest tests/ -q

# Stop on first failure (fast feedback)
pytest tests/ -x

# Show local variables in tracebacks
pytest tests/ -v -l
```

### **Coverage Testing**

```bash
# Coverage with terminal report
pytest tests/ --cov=src/tbr --cov-report=term-missing

# Coverage with HTML report (opens in browser)
pytest tests/ --cov=src/tbr --cov-report=html
open htmlcov/index.html

# Coverage with XML report (for CI)
pytest tests/ --cov=src/tbr --cov-report=xml

# Fail if coverage below threshold
pytest tests/ --cov=src/tbr --cov-fail-under=85
```

### **Test Discovery and Filtering**

```bash
# Run specific test file
pytest tests/unit/test_constants.py -v

# Run specific test class
pytest tests/unit/test_constants.py::TestConstants -v

# Run specific test method
pytest tests/unit/test_constants.py::TestConstants::test_control_val_type_and_value -v

# Run tests matching pattern
pytest -k "constants" -v

# Run tests in specific directory
pytest tests/mathematical/ -v
```

### **Performance and Timing**

```bash
# Show 10 slowest tests
pytest tests/ --durations=10

# Show all test durations
pytest tests/ --durations=0

# Benchmark mode (for performance tests)
pytest tests/performance/ --benchmark-only
```

## 🔧 **Development Testing**

### **Pre-commit Testing**

```bash
# Run all quality checks (includes tests)
make all

# Run only linting and formatting
make lint

# Run only tests
make test

# Run complete pre-commit pipeline
pre-commit run --all-files
```

### **Continuous Testing During Development**

```bash
# Watch mode (re-run tests on file changes)
pytest-watch tests/

# Quick feedback loop (stop on first failure)
pytest tests/ -x --tb=short

# Test specific functionality while developing
pytest tests/unit/test_validation.py -v --no-cov
```

## 📝 **Writing Tests**

### **Test File Organization**

```python
# tests/unit/test_example.py
"""Unit tests for example module."""

import pytest
import numpy as np
from tbr.utils.example import example_function


class TestExampleFunction:
    """Test cases for example_function."""

    def test_valid_input(self):
        """Test function with valid input."""
        result = example_function([1, 2, 3])
        assert result == 6

    def test_edge_case_empty_input(self):
        """Test function with empty input."""
        with pytest.raises(ValueError, match="Input cannot be empty"):
            example_function([])

    def test_mathematical_property(self):
        """Test mathematical property holds."""
        data = np.random.randn(100)
        result = example_function(data)
        assert np.isfinite(result)
        assert result >= 0  # Non-negative property
```

### **Using Fixtures**

```python
# Use shared fixtures from conftest.py
def test_with_sample_data(sample_time_series_data):
    """Test using shared fixture."""
    df = sample_time_series_data
    assert len(df) == 100
    assert 'date' in df.columns

# Custom fixtures for specific tests
@pytest.fixture
def small_dataset():
    """Small dataset for quick tests."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'value': range(10)
    })
```

### **Mathematical Test Standards**

```python
# Mathematical validation with known values
def test_sum_squared_deviations_known_values():
    """Test with mathematically verified values."""
    # Known case: [1, 2, 3] -> mean=2, deviations=[-1,0,1], sum_sq=2
    x = np.array([1.0, 2.0, 3.0])
    result = calculate_sum_x_squared_deviations(x)
    expected = 2.0
    assert np.isclose(result, expected, rtol=1e-10)

# Property-based testing
def test_mathematical_properties():
    """Test mathematical properties hold."""
    x = np.random.randn(50)
    result = calculate_sum_x_squared_deviations(x)

    # Properties that must always hold
    assert result >= 0  # Non-negative
    assert np.isfinite(result)  # Finite result
    assert result == 0 if len(set(x)) == 1 else result > 0  # Zero iff constant
```

## 🎯 **Test Quality Standards**

### **Coverage Requirements**

- **Minimum Coverage**: 85% (enforced by CI)
- **Target Coverage**: 90%+ for production modules
- **Mathematical Functions**: 100% coverage required

### **Test Categories Requirements**

| **Module Type** | **Required Tests** | **Coverage Target** |
|-----------------|-------------------|-------------------|
| **Core Functions** | Unit + Mathematical + Performance | 95%+ |
| **Utilities** | Unit + Integration | 90%+ |
| **Validation** | Unit + Edge Cases | 100% |
| **Constants** | Unit (basic verification) | 100% |

### **Code Quality for Tests**

```bash
# Tests must pass all quality checks
make lint          # Formatting and style
make type-check    # Type checking (mypy)
make docstring     # Docstring validation
```

## 🚨 **Troubleshooting Tests**

### **Common Issues**

#### **Import Errors**
```bash
# ModuleNotFoundError: No module named 'tbr'
# Solution: Ensure pythonpath is set correctly
pytest tests/ --pythonpath=src

# Or use our configured setup
pytest tests/  # Uses pyproject.toml pythonpath setting
```

#### **Coverage Issues**
```bash
# Coverage failure: total coverage below threshold
# Solution: Check missing coverage
pytest tests/ --cov=src/tbr --cov-report=term-missing

# See detailed HTML report
pytest tests/ --cov=src/tbr --cov-report=html
open htmlcov/index.html
```

#### **Performance Test Failures**
```bash
# Performance tests can be sensitive to system load
# Solution: Run with relaxed timing (already configured)
pytest tests/performance/ -v --no-cov
```

#### **Mathematical Precision Issues**
```python
# Use appropriate tolerances for floating point comparisons
assert np.isclose(result, expected, rtol=1e-10, atol=1e-12)

# For arrays
np.testing.assert_array_almost_equal(result, expected, decimal=10)
```

## 🔄 **Continuous Integration**

### **GitHub Actions Integration**

Our tests are designed for CI/CD pipelines:

```yaml
# .github/workflows/test.yml (future)
- name: Run tests
  run: |
    pytest tests/ \
      --cov=src/tbr \
      --cov-report=xml \
      --cov-report=term \
      --cov-fail-under=85
```

### **Local CI Simulation**

```bash
# Simulate CI environment locally
make all  # Run complete pipeline

# Individual CI steps
make test-cov      # Test with coverage
make lint          # Code quality
make type-check    # Type checking
make docstring     # Documentation quality
```

## 📊 **Test Metrics and Monitoring**

### **Current Test Statistics**

- **Total Tests**: 69 tests
- **Test Categories**: 4 (unit, integration, mathematical, performance)
- **Coverage Target**: 85%
- **Execution Time**: ~0.7 seconds
- **Success Rate**: 100% (all tests pass)

### **Performance Benchmarks**

```bash
# Track test performance over time
pytest tests/performance/ --durations=0

# Memory usage monitoring
pytest tests/ --memray

# Benchmark specific functions
pytest tests/mathematical/ --benchmark-compare
```

## 🤝 **Contributing Tests**

### **Test Contribution Guidelines**

1. **Every new function/class needs tests**
2. **Mathematical functions require known-value validation**
3. **Performance-critical code needs performance tests**
4. **Edge cases and error conditions must be tested**
5. **Tests must be documented with clear docstrings**

### **Test Review Checklist**

- [ ] Tests cover main functionality
- [ ] Edge cases and error conditions tested
- [ ] Mathematical validation with known values (if applicable)
- [ ] Performance implications considered
- [ ] Clear, descriptive test names and docstrings
- [ ] Appropriate use of fixtures and utilities
- [ ] Follows existing test patterns and style

## 📚 **Additional Resources**

### **pytest Documentation**
- [pytest Official Docs](https://docs.pytest.org/)
- [pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)

### **Scientific Testing Standards**
- [NumPy Testing Guidelines](https://numpy.org/devdocs/dev/development_workflow.html#development-workflow)
- [SciPy Testing Guide](https://docs.scipy.org/doc/scipy/dev/contributor/adding_new.html#unit-tests)

### **Coverage and Quality**
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [pytest-cov Plugin](https://pytest-cov.readthedocs.io/)

---

**This testing framework meets and exceeds the standards used by top scientific PyPI packages like NumPy, SciPy, and pandas.**
