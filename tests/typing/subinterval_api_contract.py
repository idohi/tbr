"""Consumer-style typing contract for the canonical subinterval API."""

from typing import Dict, List, Tuple

import pandas as pd

import tbr
import tbr.analysis as analysis
import tbr.analysis.subinterval as analysis_subinterval
import tbr.core as core
import tbr.core.effects as core_effects
import tbr.core.prediction as core_prediction
import tbr.functional.tbr_functions as functional_functions
from tbr.core.model import TBRAnalysis
from tbr.core.results import TBRSubintervalResult


def canonical_subinterval_consumer(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    model: TBRAnalysis,
) -> None:
    """Type-check canonical keyword calls and result fields."""
    interval: Dict[str, float]
    interval = tbr.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_position=1,
        end_position=2,
        ci_level=0.80,
    )
    interval = analysis.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_position=1,
        end_position=2,
        ci_level=0.80,
    )
    interval = analysis_subinterval.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_position=1,
        end_position=2,
        ci_level=0.80,
    )
    interval = core.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_position=1,
        end_position=2,
        ci_level=0.80,
    )
    interval = core_prediction.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_position=1,
        end_position=2,
        ci_level=0.80,
    )
    interval = core_effects.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_position=1,
        end_position=2,
        ci_level=0.80,
    )
    interval = functional_functions.compute_interval_estimate_and_ci(
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_position=1,
        end_position=2,
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
        start_position=1,
        end_position=2,
        ci_level=0.80,
    )

    result: TBRSubintervalResult = model.analyze_subinterval(
        start_position=1,
        end_position=2,
        ci_level=0.80,
    )
    constructed = TBRSubintervalResult(
        estimate=interval["estimate"],
        lower=interval["lower"],
        upper=interval["upper"],
        se=interval["se"],
        precision=interval["precision"],
        ci_level=0.80,
        start_position=1,
        end_position=2,
        n_observations=2,
    )

    canonical_values: Tuple[int, int, int] = (
        result.start_position,
        result.end_position,
        constructed.n_observations,
    )
    _ = (batch, summary, canonical_values)


def removed_subinterval_names_are_rejected(
    tbr_df: pd.DataFrame,
    tbr_summary: pd.DataFrame,
    model: TBRAnalysis,
) -> None:
    """Require mypy errors for removed keywords and result attributes."""
    tbr.compute_interval_estimate_and_ci(  # type: ignore[call-arg]
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    analysis.compute_interval_estimate_and_ci(  # type: ignore[call-arg]
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    analysis_subinterval.compute_interval_estimate_and_ci(  # type: ignore[call-arg]
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    core.compute_interval_estimate_and_ci(  # type: ignore[call-arg]
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    core_prediction.compute_interval_estimate_and_ci(  # type: ignore[call-arg]
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    core_effects.compute_interval_estimate_and_ci(  # type: ignore[call-arg]
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    functional_functions.compute_interval_estimate_and_ci(  # type: ignore[call-arg]
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    analysis.validate_subinterval_parameters(  # type: ignore[call-arg]
        tbr_df=tbr_df,
        tbr_summary=tbr_summary,
        start_day=1,
        end_day=2,
        ci_level=0.80,
    )
    model.analyze_subinterval(  # type: ignore[call-arg]
        start_day=1,
        end_day=2,
    )
    TBRSubintervalResult(  # type: ignore[call-arg]
        estimate=1.0,
        lower=0.0,
        upper=2.0,
        se=0.5,
        precision=1.0,
        ci_level=0.80,
        start_day=1,
        end_day=2,
        n_days=2,
    )
    canonical_result: TBRSubintervalResult = model.analyze_subinterval(
        start_position=1,
        end_position=2,
    )
    _ = (
        canonical_result.start_day,  # type: ignore[attr-defined]
        canonical_result.end_day,  # type: ignore[attr-defined]
        canonical_result.n_days,  # type: ignore[attr-defined]
    )
