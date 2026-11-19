# Migrating from 0.1.x to 0.2.0

Version 0.2.0 is a breaking pre-1.0 release. It replaces generic
day-specific names with terminology that applies to daily, hourly, weekly, and
other regularly observed data. The terminology migration does not change
positional selection behavior. Version 0.2.0 separately corrects subinterval
posterior variance; see the changelog for that numerical correction.

## Migration reference

### Callable signature changes

| Surface | 0.1.x | 0.2.0 | Compatibility |
| --- | --- | --- | --- |
| `tbr.compute_interval_estimate_and_ci` | `(tbr_df, tbr_summary, start_day, end_day, ci_level)` | `(tbr_df, tbr_summary, start_position, end_position, ci_level)` | Removed keywords are not accepted. |
| `tbr.analysis.compute_interval_estimate_and_ci` | `(tbr_df, tbr_summary, start_day, end_day, ci_level)` | `(tbr_df, tbr_summary, start_position, end_position, ci_level)` | Removed keywords are not accepted. |
| `tbr.analysis.subinterval.compute_interval_estimate_and_ci` | `(tbr_df, tbr_summary, start_day, end_day, ci_level)` | `(tbr_df, tbr_summary, start_position, end_position, ci_level)` | Removed keywords are not accepted. |
| `tbr.core.compute_interval_estimate_and_ci` | `(tbr_df, tbr_summary, start_day, end_day, ci_level)` | `(tbr_df, tbr_summary, start_position, end_position, ci_level)` | Removed keywords are not accepted. |
| `tbr.core.prediction.compute_interval_estimate_and_ci` | `(tbr_df, tbr_summary, start_day, end_day, ci_level)` | `(tbr_df, tbr_summary, start_position, end_position, ci_level)` | Removed keywords are not accepted. |
| `tbr.core.effects.compute_interval_estimate_and_ci` | `(tbr_df, tbr_summary, start_day, end_day, ci_level)` | `(tbr_df, tbr_summary, start_position, end_position, ci_level)` | Removed keywords are not accepted. |
| `tbr.functional.tbr_functions.compute_interval_estimate_and_ci` | `(tbr_df, tbr_summary, start_day, end_day, ci_level)` | `(tbr_df, tbr_summary, start_position, end_position, ci_level)` | Removed keywords are not accepted. |
| `TBRAnalysis.analyze_subinterval` | `(start_day, end_day, ci_level=None)` | `(start_position, end_position, ci_level=None)` | Removed keywords are not accepted. |
| `tbr.analysis.validate_subinterval_parameters` | `(tbr_df, tbr_summary, start_day, end_day, ci_level)` | `(tbr_df, tbr_summary, start_position, end_position, ci_level)` | Removed keywords are not accepted. |
| `analyze_multiple_subintervals` | `(tbr_df, tbr_summary, intervals, ci_level=0.80)` | Unchanged | Each tuple remains an inclusive one-based `(start_position, end_position)` pair. |
| `create_subinterval_summary` | `(tbr_df, tbr_summary, intervals, ci_level=0.80, significance_threshold=0.0)` | `(tbr_df, tbr_summary, intervals, ci_level=0.80, threshold=0.0)` | `significance_threshold` is removed; each tuple remains an inclusive one-based `(start_position, end_position)` pair. |

### Result and schema changes

| Surface | 0.1.x | 0.2.0 | Compatibility |
| --- | --- | --- | --- |
| `TBRSubintervalResult` public fields and serialized data keys, in order | `estimate`, `lower`, `upper`, `se`, `ci_level`, `start_day`, `end_day`, `n_days` | `estimate`, `lower`, `upper`, `se`, `precision`, `ci_level`, `start_position`, `end_position`, `n_observations` | `precision` is added. The bounds are inclusive one-based test-observation positions, and the count includes both endpoints. Removed attributes are not available. Cross-version pickle loading is unsupported; recreate or refit results with 0.2.0. |
| `create_subinterval_summary()` columns, in order | `interval`, `start_day`, `end_day`, `days`, `estimate`, `precision`, `lower`, `upper`, `significant`, `avg_daily_effect`, `ci_level` | `interval`, `start_position`, `end_position`, `n_observations`, `estimate`, `se`, `precision`, `lower`, `upper`, `interval_excludes_threshold`, `average_effect_per_observation`, `ci_level` | `se` is added; `significant` is replaced by `interval_excludes_threshold` with the documented threshold-exclusion semantics. Positions are inclusive and one-based; `n_observations` includes both endpoints; `average_effect_per_observation` is the cumulative estimate divided by that count. No duplicate compatibility columns are returned. |
| Incremental summary columns, in order | `test_day`, `estimate`, `precision`, `lower`, `upper`, `se`, `level`, `thres`, `prob`, `alpha`, `beta`, `alpha_beta_cov`, `var_alpha`, `var_beta`, `sigma`, `t_dist_df` | `test_position`, `estimate`, `precision`, `lower`, `upper`, `se`, `level`, `thres`, `prob`, `alpha`, `beta`, `alpha_beta_cov`, `var_alpha`, `var_beta`, `sigma`, `t_dist_df` | `test_position` is the one-based endpoint of each cumulative test-observation prefix. Applies to all documented incremental-summary routes; no duplicate compatibility column is returned. |
| Incremental posterior keys, in order | `"day"`, `"estimates"`, `"standard_errors"`, `"probabilities"`, `"probability_change"` | `"position"`, `"estimates"`, `"standard_errors"`, `"probabilities"`, `"probability_change"` | `"position"` contains the one-based cumulative positions. Applies to both documented posterior routes; no duplicate compatibility key is returned. |
| Test-observation count on `TBRResults` | `TBRResults.n_test_days` | `TBRResults.n_test` | `n_test_days` is removed; `n_test` was already available in 0.1.x. |
| Whole-object 0.1.x `TBRAnalysis` and `TBRResults` pickles | May contain incremental frames with `test_day`. | Newly created objects contain `test_position`. | Cross-version loading is unsupported; recreate or refit these objects with 0.2.0. |

`_tbr_pickle_version` is an internal pickle-format marker stored alongside the
public data keys in pickle state. It is not part of the public result schema or
the mappings returned by conversion methods.

## Subinterval analysis

Subinterval bounds remain inclusive, one-based positions among test-period
observations. Their positional interpretation is unchanged:

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

0.1.x `TBRSubintervalResult` pickle files are not supported in 0.2.0. In
0.1.x, the field named `se` held the credible-interval half-width, and the
pickle does not contain enough information to reconstruct the corrected
standard error. Recreate or refit these results with 0.2.0.
Same-version 0.2.0 pickle round trips remain supported for exact
`TBRSubintervalResult` instances; subclass instances are rejected. As with all
Python pickle files, load them only from trusted sources.

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

Whole-object pickles of `TBRAnalysis` and `TBRResults` created with 0.1.x are
also unsupported. Their stored incremental DataFrames can retain the
`test_day` column, so recreate or refit these objects with 0.2.0. Same-version
0.2.0 pickle round trips remain supported for canonical package-managed state
on exact `TBRAnalysis` and `TBRResults` instances. Subclass instances and
objects with user-added instance attributes are rejected during serialization
rather than silently discarding additional state.
