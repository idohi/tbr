# Migrating from 0.1.x to 0.2.0

Version 0.2.0 is a breaking pre-1.0 release. It replaces generic
day-specific names with terminology that applies to daily, hourly, weekly, and
other regularly observed data. Statistical calculations and positional
selection behavior are unchanged.

## Subinterval analysis

Subinterval bounds remain inclusive, one-based positions among test-period
observations. Only their names change:

- `start_day` becomes `start_position`;
- `end_day` becomes `end_position`;
- `n_days` and the summary column `days` become `n_observations`; and
- `avg_daily_effect` becomes `average_effect_per_observation`.

The callable signatures change from:

```python
compute_interval_estimate_and_ci(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    start_day: int,
    end_day: int,
    ci_level: float,
) -> Dict[str, float]

TBRAnalysis.analyze_subinterval(
    start_day: int,
    end_day: int,
    ci_level: Optional[float] = None,
) -> TBRSubintervalResult
```

to:

```python
compute_interval_estimate_and_ci(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    start_position: int,
    end_position: int,
    ci_level: float,
) -> Dict[str, float]

TBRAnalysis.analyze_subinterval(
    start_position: int,
    end_position: int,
    ci_level: Optional[float] = None,
) -> TBRSubintervalResult
```

The compute signature applies to every supported route:

- `tbr.compute_interval_estimate_and_ci`;
- `tbr.analysis.compute_interval_estimate_and_ci`;
- `tbr.analysis.subinterval.compute_interval_estimate_and_ci`;
- `tbr.core.compute_interval_estimate_and_ci`;
- `tbr.core.prediction.compute_interval_estimate_and_ci`;
- `tbr.core.effects.compute_interval_estimate_and_ci`; and
- `tbr.functional.tbr_functions.compute_interval_estimate_and_ci`.

The direct validator likewise becomes
`tbr.analysis.validate_subinterval_parameters(tbr_df, tbr_summary,
start_position, end_position, ci_level) -> None`.

Before:

```python
result = model.analyze_subinterval(start_day=1, end_day=7)
print(result.start_day, result.end_day, result.n_days)
```

After:

```python
result = model.analyze_subinterval(start_position=1, end_position=7)
print(
    result.start_position,
    result.end_position,
    result.n_observations,
)
```

The same keyword change applies to every supported
`compute_interval_estimate_and_ci` route and to
`tbr.analysis.validate_subinterval_parameters`. The `intervals` argument of
`analyze_multiple_subintervals` and `create_subinterval_summary` remains a
list of integer pairs; each pair now has the documented meaning
`(start_position, end_position)`.

`TBRSubintervalResult` now serializes fields in this order:

```text
estimate, lower, upper, se, precision, ci_level,
start_position, end_position, n_observations
```

`create_subinterval_summary()` now returns columns in this order:

```text
interval, start_position, end_position, n_observations, estimate, se,
precision, lower, upper, interval_excludes_threshold,
average_effect_per_observation, ci_level
```

No temporary aliases or duplicate columns are provided. Calls using removed
keywords raise the normal Python `TypeError`, and removed result attributes
raise `AttributeError`.

Trusted `TBRSubintervalResult` pickle files created by 0.1.x remain loadable.
Their old field names are migrated to canonical names during deserialization;
newly saved objects contain only canonical fields. As with all Python pickle
files, load them only from trusted sources.

## Incremental analysis

Incremental summaries still contain one row for each cumulative prefix of
ordered test observations. The one-based position column changes from
`test_day` to `test_position`; values, ordering, dtypes, and calculations are
unchanged.

Before:

```python
incremental = model.summarize_incremental()
positions = incremental["test_day"]
```

After:

```python
incremental = model.summarize_incremental()
positions = incremental["test_position"]
```

This column change applies to every supported
`create_incremental_tbr_summaries` route and to incremental frames returned by
`TBRAnalysis.summaries_`, `TBRAnalysis.summarize_incremental()`, and
`TBRResults.summary()`.

The supported callable routes are:

- `tbr.create_incremental_tbr_summaries`;
- `tbr.analysis.create_incremental_tbr_summaries`;
- `tbr.analysis.incremental.create_incremental_tbr_summaries`;
- `tbr.core.create_incremental_tbr_summaries`;
- `tbr.core.effects.create_incremental_tbr_summaries`; and
- `tbr.functional.tbr_functions.create_incremental_tbr_summaries`.

`calculate_incremental_posterior_probabilities()` now returns the key
`"position"` instead of `"day"`:

```python
incremental = calculate_incremental_posterior_probabilities(
    estimates,
    standard_errors,
    degrees_freedom,
)
positions = incremental["position"]
```

The mapping order remains `position`, `estimates`, `standard_errors`,
`probabilities`, and `probability_change`.
This applies to `tbr.core.calculate_incremental_posterior_probabilities` and
`tbr.core.posterior.calculate_incremental_posterior_probabilities`.

## Test-observation count

Use `TBRResults.n_test` for the number of test observations.
`TBRResults.n_test_days` has been removed:

```python
results = perform_tbr_analysis(...)
print(results.n_test)
```

No temporary aliases or duplicate columns and keys are provided. For objects
created or recomputed with 0.2.0, accessing `test_day`, `"day"`, or
`n_test_days` raises the normal `KeyError` or `AttributeError`.

Whole-object pickles of `TBRAnalysis` and `TBRResults` created with 0.1.x
are not automatically migrated. Their stored incremental DataFrames can
retain the `test_day` column, so recreate or refit these objects with 0.2.0.
The migration described above applies only to `TBRSubintervalResult`.
