TBR Examples
============

This directory contains practical examples demonstrating various aspects of the TBR package.

Available Examples
------------------

Basic Analysis
~~~~~~~~~~~~~~

**File:** ``plot_basic_analysis.py``

A complete walkthrough of a basic TBR analysis from data preparation to result export.

**Topics Covered:**

- Data preparation and exploration
- Model initialization and configuration
- Fitting the model
- Interpreting summary results
- Subinterval analysis
- Generating predictions
- Exporting results

**Run the example:**

.. code-block:: bash

    cd examples
    python plot_basic_analysis.py


Method Chaining
~~~~~~~~~~~~~~~

**File:** ``plot_method_chaining.py``

Demonstrates the fluent API and method chaining capabilities for concise workflows.

**Topics Covered:**

- One-liner analysis with ``fit_summarize()``
- Configuration and fit chaining
- Model copying and variants
- Fit and predict chains
- Sklearn-style parameter management
- Comparing multiple configurations
- Property access shortcuts

**Run the example:**

.. code-block:: bash

    python plot_method_chaining.py


Incremental Analysis
~~~~~~~~~~~~~~~~~~~~

**File:** ``plot_incremental_analysis.py``

Shows how to analyze treatment effects over time using incremental summaries and subintervals.

**Topics Covered:**

- Incremental summary progression
- Subinterval analysis in 15-day windows
- Period comparison (early vs. late)
- Cumulative growth tracking
- Credible interval width over time
- Exporting incremental results

**Run the example:**

.. code-block:: bash

    python plot_incremental_analysis.py


Data and Domains
----------------

All examples use synthetic data with a simulated treatment effect.

TBR is not tied to a particular domain. It applies to any before-after
experiment with control and test groups observed over time, for example
marketing campaigns, product feature rollouts, economic policy changes, or
medical interventions.

Before applying TBR to real data, see `Assumptions and Limitations
<https://tbr.readthedocs.io/en/stable/mathematical_methodology.html#assumptions-and-limitations>`_
in the mathematical methodology.


Running Examples
----------------

Prerequisites
~~~~~~~~~~~~~

.. code-block:: bash

    pip install tbr


Basic Execution
~~~~~~~~~~~~~~~

Each example is a standalone Python script:

.. code-block:: bash

    python plot_basic_analysis.py
    python plot_method_chaining.py
    python plot_incremental_analysis.py


Output
~~~~~~

Examples produce:

- Console output with analysis results
- JSON export files (e.g., ``tbr_summary.json``, ``tbr_predictions.json``)
- CSV export files (e.g., ``tbr_summary.csv``, ``incremental_summaries.csv``)


Customization
~~~~~~~~~~~~~

To adapt examples to your data:

1. **Replace data generation code** with your data loading:

.. code-block:: python

    # Instead of synthetic data generation
    # Use:
    data = pd.read_csv('your_data.csv')
    data['date'] = pd.to_datetime(data['date'])

2. **Update column names** to match your data:

.. code-block:: python

    model.fit(
        data=data,
        time_col='your_time_column',
        control_col='your_control_column',
        test_col='your_test_column',
        ...
    )

3. **Adjust time periods** to your analysis window:

.. code-block:: python

    pretest_start=pd.Timestamp('2024-01-01'),
    test_start=pd.Timestamp('2024-02-15'),
    test_end=pd.Timestamp('2024-03-31')


Learning Path
-------------

**Recommended order for beginners:**

1. **Start with** ``plot_basic_analysis.py``

   - Understand the complete workflow
   - Learn core concepts and terminology
   - See all major components in action

2. **Then explore** ``plot_method_chaining.py``

   - Learn fluent API patterns
   - Discover shortcuts and convenience methods
   - Understand model configuration

3. **Finally study** ``plot_incremental_analysis.py``

   - See how the cumulative effect evolves over the test period
   - Estimate the effect in subintervals of the test period


Additional Resources
--------------------

- `Quick Start Guide <https://tbr.readthedocs.io/en/stable/api/quickstart.html>`_ - Fast introduction to TBR
- `API Reference <https://tbr.readthedocs.io/en/stable/api/api_reference.html>`_ - Complete API documentation
- `Common Patterns <https://tbr.readthedocs.io/en/stable/api/patterns.html>`_ - Best practices and patterns
- `Result Objects <https://tbr.readthedocs.io/en/stable/api/results.html>`_ - Understanding result structures


Contributing Examples
---------------------

Have a great example? Consider contributing:

1. Create a new example file (e.g., ``plot_your_example.py``)
2. Follow the existing example structure
3. Include comprehensive comments
4. Add domain-agnostic synthetic data
5. Update this README


Questions or Issues?
--------------------

- Check the API Reference for detailed documentation
- Review Common Patterns for best practices
- See Quick Start for basic usage


License
-------

All examples are part of the TBR package and subject to the same license.
