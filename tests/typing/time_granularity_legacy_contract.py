"""Consumer-style typing contract for the pre-migration public API."""

from typing import Dict, List, Tuple

import pandas as pd

import tbr
import tbr.analysis as analysis
import tbr.analysis.subinterval as analysis_subinterval
import tbr.core as core
import tbr.core.effects as core_effects
import tbr.core.prediction as core_prediction
import tbr.functional.tbr_functions as functional_functions
from tbr.core.results import TBRSubintervalResult


def legacy_time_granularity_consumer(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    model: tbr.TBRAnalysis,
) -> None:
    """Type-check supported legacy keyword calls and result fields."""
    interval: Dict[str, float]
    interval = tbr.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    interval = analysis.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    interval = analysis_subinterval.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    interval = core.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    interval = core_prediction.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    interval = core_effects.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    interval = functional_functions.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )

    intervals: List[Tuple[int, int]] = [(1, 2)]
    batch: List[Dict[str, float]] = analysis.analyze_multiple_subintervals(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        intervals=intervals,
        ci_level=0.80,
    )
    summary: pd.DataFrame = analysis.create_subinterval_summary(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        intervals=intervals,
        ci_level=0.80,
        threshold=0.0,
    )
    analysis.validate_subinterval_parameters(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )

    result: TBRSubintervalResult = model.analyze_subinterval(
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    constructed = TBRSubintervalResult(
        estimate=interval["estimate"],
        lower=interval["lower"],
        upper=interval["upper"],
        se=interval["se"],
        precision=interval["precision"],
        ci_level=0.80,
        start_day=1,
        end_day=2,
        n_days=2,
    )

    legacy_values: Tuple[int, int, int] = (
        result.start_day,
        result.end_day,
        constructed.n_days,
    )
    _ = (batch, summary, legacy_values)
