==============
Testing Guide
==============

This document provides a comprehensive guide to testing in the TBR (Time-Based Regression) package. It covers the testing framework, methodologies, and procedures used to ensure mathematical accuracy, performance efficiency, and code reliability throughout the package.

Overview
========

The TBR package employs a comprehensive testing framework that ensures:

- **Mathematical Accuracy**: Precise numerical results for all statistical computations
- **Performance Efficiency**: Optimal performance across different data sizes and scenarios
- **Code Reliability**: Robust behavior under various conditions and edge cases
- **API Stability**: Consistent interfaces and backward compatibility
- **Documentation Quality**: Comprehensive test coverage and clear testing procedures

Test Suite Organization
=======================

The test suite is organized into four main categories:

**Unit Tests** (``tests/unit/``)
  Individual function and class testing with focus on mathematical correctness

**Integration Tests** (``tests/integration/``)
  Package-level functionality and module interaction testing

**Mathematical Tests** (``tests/mathematical/``)
  Validation of statistical properties and numerical accuracy

**Performance Tests** (``tests/performance/``)
  Benchmarking and scalability validation

Testing Architecture
====================

1. Cross-Implementation Validation (``test_comprehensive_regression_validation.py``)
---------------------------------------------------------------------------------------

TestComprehensiveCrossImplementationValidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose**: Validate mathematical equivalence between core and functional implementations
- **Scope**: All regression functions with parametrized testing across multiple scenarios
- **Validation Criteria**: Relative error < 1e-12 for all parameters

**Key Test Categories:**

- **Regression Fitting**: Multiple data sizes, distributions, and statistical scenarios
- **Sum Squared Deviations**: Various array sizes and numerical distributions
- **Variance Calculations**: Comprehensive parameter combinations and edge cases
- **Integer Conversion**: Edge cases and boundary conditions
- **Parameter Extraction**: Mathematical relationship validation

TestExtendedRobustnessValidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose**: Ensure robustness across extreme and edge case scenarios
- **Scope**: Statistical edge cases, extreme values, and numerical precision

**Robustness Categories:**

- **Extreme Values**: Very small (1e-6) and very large (1e6) value ranges
- **Statistical Edge Cases**: Perfect correlation, zero variance scenarios
- **Numerical Precision**: Float32 vs Float64 precision validation
- **Data Type Compatibility**: Various pandas dtypes and numpy arrays

TestComprehensiveIntegrationValidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Purpose**: Validate complete end-to-end regression pipeline
- **Scope**: Full workflow integration with realistic TBR analysis scenarios

2. Performance Regression Benchmarks (``test_regression_performance_benchmarks.py``)
-------------------------------------------------------------------------------------

Performance Benchmarking Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Statistical Analysis**: Multiple runs with warmup periods
- **Memory Profiling**: Memory usage comparison and efficiency validation
- **Scalability Testing**: Performance across different data sizes

Performance Test Categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**TestRegressionFittingPerformance:**

- Scalability testing across data sizes (50 to 1000 samples)
- Memory efficiency validation
- Performance tolerance: ≤ 2.0x functional implementation time

**TestSumSquaredDeviationsPerformance:**

- Array size scalability (100 to 10,000 elements)
- Numerical stability with challenging data distributions
- Microsecond-level precision benchmarking

**TestVarianceCalculationsPerformance:**

- Model and prediction variance calculation benchmarks
- Combined variance function optimization validation
- Performance across different array sizes

**TestEndToEndPerformanceBenchmarks:**

- Complete regression workflow performance
- Real-world TBR analysis scenario benchmarking
- Comprehensive pipeline optimization validation

3. Existing Test Integration
----------------------------

Core Regression Tests (``test_core_regression.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Basic functionality validation
- Backward compatibility testing
- Integration workflow testing
- Mathematical property validation

Mathematical Validation (``test_reference_validation.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Statistical property validation
- Numerical stability testing
- Mathematical consistency verification

Performance Testing (``test_performance.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Basic performance benchmarks
- Scalability validation
- Memory efficiency testing

Validation Criteria
===================

Mathematical Accuracy Standards
-------------------------------

- **High Precision**: Relative error < 1e-12 for all regression parameters
- **Numerical Stability**: Consistent results across different data types and ranges
- **Edge Case Handling**: Identical behavior for boundary conditions and special cases

Performance Standards
---------------------

- **Performance Parity**: Core implementation ≤ 2.0x functional implementation time
- **Memory Efficiency**: Memory usage within 50% of functional implementation
- **Scalability**: Linear performance scaling with data size

Robustness Standards
--------------------

- **Data Range Coverage**: Testing from 1e-6 to 1e7 value ranges
- **Distribution Coverage**: Normal, uniform, and exponential distributions
- **Precision Coverage**: Float32 and Float64 numerical precision
- **Statistical Coverage**: Perfect correlation, zero variance, and extreme scenarios

Running Tests
=============

Quick Start
-----------

To run the complete test suite:

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run with coverage report
   pytest tests/ --cov=src/tbr --cov-report=term-missing

   # Run specific test categories
   pytest tests/unit/        # Unit tests only
   pytest tests/integration/ # Integration tests only
   pytest tests/mathematical/ # Mathematical validation
   pytest tests/performance/  # Performance benchmarks

Test Categories
---------------

**Unit Tests**

.. code-block:: bash

   # Run all unit tests
   pytest tests/unit/ -v

   # Run specific module tests
   pytest tests/unit/test_core_regression.py -v
   pytest tests/unit/test_core_prediction.py -v
   pytest tests/unit/test_validation.py -v

**Integration Tests**

.. code-block:: bash

   # Run package integration tests
   pytest tests/integration/ -v

**Mathematical Validation**

.. code-block:: bash

   # Run mathematical property tests
   pytest tests/mathematical/ -v

**Performance Benchmarks**

.. code-block:: bash

   # Run performance tests
   pytest tests/performance/ -v

   # Run with detailed timing
   pytest tests/performance/ -v --durations=10

Performance Monitoring
----------------------

.. code-block:: bash

   # Run with performance profiling
   pytest tests/performance/ -v -m performance --tb=short

   # Generate performance reports
   pytest tests/performance/ --benchmark-only --benchmark-sort=mean

Coverage Validation
-------------------

.. code-block:: bash

   # Ensure 100% coverage maintained
   pytest tests/ --cov=src/tbr --cov-report=term-missing --cov-fail-under=100

Quality Assurance Metrics
==========================

Test Coverage Metrics
----------------------

- **Function Coverage**: 100% of all regression functions tested
- **Branch Coverage**: 100% of all code paths validated
- **Parameter Coverage**: All parameter combinations tested
- **Edge Case Coverage**: All identified edge cases validated

Performance Metrics
-------------------

- **Execution Time**: Mean execution time with standard deviation
- **Memory Usage**: Peak memory consumption during execution
- **Scalability Factor**: Performance scaling coefficient with data size
- **Efficiency Ratio**: Core vs functional implementation performance ratio

Accuracy Metrics
----------------

- **Numerical Precision**: Maximum relative error across all test scenarios
- **Statistical Consistency**: Variance in results across multiple runs
- **Mathematical Correctness**: Validation of mathematical relationships and properties

Continuous Integration Integration
==================================

Pre-commit Validation
---------------------

- All regression tests must pass before code commits
- Performance benchmarks must meet tolerance criteria
- Coverage must remain at 100%

CI Pipeline Integration
-----------------------

- Automated execution on all pull requests
- Performance regression detection and reporting
- Mathematical accuracy validation across different environments

Maintenance and Updates
=======================

Test Suite Maintenance
-----------------------

- **Regular Updates**: Test scenarios updated with new edge cases
- **Performance Baseline Updates**: Benchmarks updated with infrastructure changes
- **Coverage Monitoring**: Continuous monitoring of test coverage metrics

Documentation Updates
---------------------

- **Methodology Updates**: Documentation updated with new testing approaches
- **Results Documentation**: Performance and accuracy results documented
- **Best Practices**: Testing best practices documented and shared

Conclusion
==========

This comprehensive regression testing methodology ensures that the TBR core modular implementation maintains the highest standards of mathematical accuracy, performance efficiency, and robustness expected of top scientific PyPI packages. The multi-layered testing approach provides confidence in the reliability and correctness of the modular architecture while preserving the proven mathematical foundations of the functional implementation.

The testing suite serves as both a validation framework and a continuous quality assurance system, ensuring that future developments maintain the rigorous standards established for the TBR package.

----

**Testing Framework Status**

The TBR package maintains a comprehensive testing framework with:

- **378 total tests** across all categories
- **100% code coverage** for all modules
- **Automated CI/CD integration** with GitHub Actions
- **Performance benchmarking** with regression detection
- **Mathematical validation** with high-precision standards
