TBR API Reference
=================

This reference is generated from the public TBR docstrings. It is intended for
signatures, parameters, return values, and symbol-level examples.

For workflow-oriented documentation, see :doc:`quickstart`, :doc:`patterns`,
:doc:`results`, and :doc:`../mathematical_methodology`.

Primary API
-----------

.. currentmodule:: tbr

.. autosummary::
   :toctree: generated
   :nosignatures:

   TBRAnalysis
   perform_tbr_analysis

Result Objects
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   core.results.TBRResults
   core.results.TBRPredictionResult
   core.results.TBRSummaryResult
   core.results.TBRSubintervalResult

Analysis Helpers
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   create_tbr_summary
   create_incremental_tbr_summaries
   compute_interval_estimate_and_ci
   analyze_multiple_subintervals
   create_subinterval_summary

Diagnostics And Performance
---------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   validate_tbr_model
   diagnose_tbr_analysis
   check_tbr_assumptions
   analyze_tbr_residuals
   assess_tbr_performance
   create_tbr_diagnostic_report

Constants
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   CONTROL_VAL
   TEST_VAL
