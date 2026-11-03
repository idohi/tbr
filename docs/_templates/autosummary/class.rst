{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

{% set annotation_only_attributes = {
   "tbr.core.results.TBRPredictionResult": (
      "predictions",
      "n_predictions",
      "model_params",
      "control_values",
   ),
   "tbr.core.results.TBRSummaryResult": (
      "estimate",
      "lower",
      "upper",
      "se",
      "prob",
      "precision",
      "level",
      "threshold",
      "alpha",
      "beta",
      "sigma",
      "var_alpha",
      "var_beta",
      "cov_alpha_beta",
      "degrees_freedom",
   ),
   "tbr.core.results.TBRSubintervalResult": (
      "estimate",
      "lower",
      "upper",
      "se",
      "precision",
      "ci_level",
      "start_day",
      "end_day",
      "n_days",
   ),
   "tbr.TBRResidualStatistics": (
      "mean",
      "std",
      "min",
      "max",
      "q25",
      "median",
      "q75",
   ),
   "tbr.TBRResidualAnalysis": (
      "residuals",
      "standardized_residuals",
      "studentized_residuals",
      "outliers",
      "outlier_threshold",
      "outlier_percentage",
      "residual_stats",
      "residual_std",
      "n_observations",
   ),
   "tbr.TBRDiagnosticReport": (
      "executive_summary",
      "overall_validity",
      "warnings_count",
      "key_findings",
      "recommendations",
      "detailed_results",
      "report_timestamp",
   ),
} %}
{% if fullname.startswith("tbr.core.results.") %}
{% if fullname in annotation_only_attributes %}
.. autoclass:: {{ objname }}()
   :members:
   :exclude-members: {{ annotation_only_attributes[fullname] | join(", ") }}
   :show-inheritance:
   :class-doc-from: class
{% else %}
.. autoclass:: {{ objname }}()
   :members:
   :show-inheritance:
   :class-doc-from: class
{% endif %}
{% else %}
{% if fullname in annotation_only_attributes %}
.. autoclass:: {{ objname }}
   :members:
   :exclude-members: {{ annotation_only_attributes[fullname] | join(", ") }}
   :show-inheritance:
{% else %}
.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
{% endif %}
{% endif %}
