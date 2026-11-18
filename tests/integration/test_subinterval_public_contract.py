"""Phase 0 contracts for the public subinterval API surface."""

import inspect
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

import tbr
import tbr.analysis as analysis
import tbr.analysis.incremental as analysis_incremental
import tbr.analysis.subinterval as analysis_subinterval
import tbr.core as core
import tbr.core.effects as core_effects
import tbr.core.prediction as core_prediction
import tbr.functional as functional
import tbr.functional.tbr_functions as functional_functions

pytestmark = pytest.mark.integration


@pytest.fixture
def subinterval_inputs():
    """Return deterministic data for public-route contract tests."""
    tbr_df = pd.DataFrame(
        {
            "period": [0, 1, 1, 1],
            "y": [0.0, 11.0, 22.0, 33.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "pred": [0.0, 10.0, 20.0, 30.0],
        }
    )
    tbr_summary = pd.DataFrame(
        {
            "sigma": [1.0],
            "t_dist_df": [10],
            "var_alpha": [0.1],
            "var_beta": [0.01],
            "alpha_beta_cov": [0.0],
        }
    )
    return tbr_df, tbr_summary


@pytest.fixture
def fitted_analysis():
    """Return a deterministic fitted model for OOP wrapper contracts."""
    time = np.arange(10, dtype=np.int64)
    control = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0])
    data = pd.DataFrame(
        {
            "time": time,
            "control": control,
            "test": 4.0 + 1.5 * control + np.linspace(-0.2, 0.2, 10),
        }
    )
    model = tbr.TBRAnalysis(
        level=0.80,
        threshold=0.0,
        test_end_inclusive=True,
    )
    model.fit(
        data,
        time_col="time",
        control_col="control",
        test_col="test",
        pretest_start=0,
        test_start=7,
        test_end=9,
    )
    return model


def _parameter_contract(callable_object):
    """Return stable parameter metadata without stringifying annotations."""
    return [
        (name, parameter.kind, parameter.default, parameter.annotation)
        for name, parameter in inspect.signature(callable_object).parameters.items()
    ]


def test_compute_interval_public_import_route_inventory():
    """Freeze every supported subinterval callable import route."""
    validated_wrapper = analysis_subinterval.compute_interval_estimate_and_ci
    assert tbr.compute_interval_estimate_and_ci is validated_wrapper
    assert analysis.compute_interval_estimate_and_ci is validated_wrapper
    assert (
        tbr.analyze_multiple_subintervals
        is analysis.analyze_multiple_subintervals
        is analysis_subinterval.analyze_multiple_subintervals
    )
    assert (
        tbr.create_subinterval_summary
        is analysis.create_subinterval_summary
        is analysis_subinterval.create_subinterval_summary
    )
    assert (
        analysis.validate_subinterval_parameters
        is analysis_subinterval.validate_subinterval_parameters
    )
    assert not hasattr(tbr, "validate_subinterval_parameters")

    assert (
        core.compute_interval_estimate_and_ci
        is core_prediction.compute_interval_estimate_and_ci
    )
    assert core_effects.compute_interval_estimate_and_ci is not validated_wrapper
    assert (
        core_effects.compute_interval_estimate_and_ci
        is not core.compute_interval_estimate_and_ci
    )
    assert (
        functional_functions.compute_interval_estimate_and_ci is not validated_wrapper
    )
    assert (
        functional_functions.compute_interval_estimate_and_ci
        is not core.compute_interval_estimate_and_ci
    )

    assert not hasattr(functional, "compute_interval_estimate_and_ci")
    assert "compute_interval_estimate_and_ci" not in functional.__all__


def test_subinterval_public_signatures_are_exact():
    """Freeze canonical parameter names, order, defaults, kinds, and annotations."""
    required_compute_parameters = [
        (
            "tbr_df",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            pd.DataFrame,
        ),
        (
            "tbr_summary",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            pd.DataFrame,
        ),
        (
            "start_position",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            int,
        ),
        (
            "end_position",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            int,
        ),
        (
            "ci_level",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            float,
        ),
    ]
    compute_routes = [
        tbr.compute_interval_estimate_and_ci,
        analysis.compute_interval_estimate_and_ci,
        analysis_subinterval.compute_interval_estimate_and_ci,
        core.compute_interval_estimate_and_ci,
        core_prediction.compute_interval_estimate_and_ci,
        core_effects.compute_interval_estimate_and_ci,
        functional_functions.compute_interval_estimate_and_ci,
    ]
    for route in compute_routes:
        assert _parameter_contract(route) == required_compute_parameters
        assert inspect.signature(route).return_annotation == Dict[str, float]

    assert _parameter_contract(analysis.analyze_multiple_subintervals) == [
        (
            "tbr_df",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            pd.DataFrame,
        ),
        (
            "tbr_summary",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            pd.DataFrame,
        ),
        (
            "intervals",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            List[Tuple[int, int]],
        ),
        ("ci_level", inspect.Parameter.POSITIONAL_OR_KEYWORD, 0.80, float),
    ]
    assert (
        inspect.signature(analysis.analyze_multiple_subintervals).return_annotation
        == List[Dict[str, float]]
    )

    assert _parameter_contract(analysis.create_subinterval_summary) == [
        (
            "tbr_df",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            pd.DataFrame,
        ),
        (
            "tbr_summary",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            pd.DataFrame,
        ),
        (
            "intervals",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            List[Tuple[int, int]],
        ),
        ("ci_level", inspect.Parameter.POSITIONAL_OR_KEYWORD, 0.80, float),
        ("threshold", inspect.Parameter.POSITIONAL_OR_KEYWORD, 0.0, float),
    ]
    assert (
        inspect.signature(analysis.create_subinterval_summary).return_annotation
        is pd.DataFrame
    )

    assert (
        _parameter_contract(analysis.validate_subinterval_parameters)
        == required_compute_parameters
    )
    assert (
        inspect.signature(analysis.validate_subinterval_parameters).return_annotation
        is None
    )

    assert _parameter_contract(tbr.TBRAnalysis.analyze_subinterval) == [
        (
            "self",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            inspect.Parameter.empty,
        ),
        (
            "start_position",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            int,
        ),
        (
            "end_position",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            int,
        ),
        ("ci_level", inspect.Parameter.POSITIONAL_OR_KEYWORD, None, Optional[float]),
    ]
    assert (
        inspect.signature(tbr.TBRAnalysis.analyze_subinterval).return_annotation
        is core.TBRSubintervalResult
    )


def test_incremental_public_routes_and_signatures_are_exact():
    """Freeze every incremental callable route and its legacy signature."""
    analysis_wrapper = analysis_incremental.create_incremental_tbr_summaries
    assert tbr.create_incremental_tbr_summaries is analysis_wrapper
    assert analysis.create_incremental_tbr_summaries is analysis_wrapper
    assert core.create_incremental_tbr_summaries is (
        core_effects.create_incremental_tbr_summaries
    )
    assert functional_functions.create_incremental_tbr_summaries is not analysis_wrapper
    assert (
        functional_functions.create_incremental_tbr_summaries
        is not core.create_incremental_tbr_summaries
    )
    assert not hasattr(functional, "create_incremental_tbr_summaries")

    required_parameters = [
        (
            "tbr_dataframe",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            pd.DataFrame,
        ),
        (
            "alpha",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            float,
        ),
        (
            "beta",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            float,
        ),
        (
            "sigma",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            float,
        ),
        (
            "var_alpha",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            float,
        ),
        (
            "var_beta",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            float,
        ),
        (
            "cov_alpha_beta",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            float,
        ),
        (
            "degrees_freedom",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            int,
        ),
        (
            "level",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            float,
        ),
        (
            "threshold",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            float,
        ),
    ]
    routes = [
        tbr.create_incremental_tbr_summaries,
        analysis.create_incremental_tbr_summaries,
        analysis_incremental.create_incremental_tbr_summaries,
        core.create_incremental_tbr_summaries,
        core_effects.create_incremental_tbr_summaries,
        functional_functions.create_incremental_tbr_summaries,
    ]
    for route in routes:
        assert _parameter_contract(route) == required_parameters
        assert inspect.signature(route).return_annotation is pd.DataFrame


@pytest.mark.parametrize(
    "route",
    [
        tbr.compute_interval_estimate_and_ci,
        core.compute_interval_estimate_and_ci,
        core_effects.compute_interval_estimate_and_ci,
        functional_functions.compute_interval_estimate_and_ci,
    ],
)
def test_canonical_keywords_are_accepted_and_legacy_keywords_are_rejected(
    route, subinterval_inputs
):
    """Accept canonical calls and reject removed names without warnings."""
    tbr_df, tbr_summary = subinterval_inputs

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = route(
            tbr_df=tbr_df,
            tbr_summary=tbr_summary,
            start_position=np.int64(1),
            end_position=np.int64(2),
            ci_level=0.80,
        )
    assert result["estimate"] == pytest.approx(3.0)
    assert caught == []

    with pytest.raises(TypeError, match="unexpected keyword argument 'start_day'"):
        route(
            tbr_df=tbr_df,
            tbr_summary=tbr_summary,
            start_day=1,
            end_day=2,
            ci_level=0.80,
        )

    with pytest.raises(TypeError, match="unexpected keyword argument 'end_day'"):
        route(
            tbr_df=tbr_df,
            tbr_summary=tbr_summary,
            start_position=1,
            end_day=2,
            ci_level=0.80,
        )

    with pytest.raises(TypeError, match="unexpected keyword argument 'start_day'"):
        route(
            tbr_df=tbr_df,
            tbr_summary=tbr_summary,
            start_day=1,
            end_position=2,
            ci_level=0.80,
        )

    with pytest.raises(TypeError, match="unexpected keyword argument 'start_day'"):
        route(
            tbr_df=tbr_df,
            tbr_summary=tbr_summary,
            start_position=1,
            start_day=1,
            end_position=2,
            ci_level=0.80,
        )

    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'end_position'"
    ):
        route(
            tbr_df=tbr_df,
            tbr_summary=tbr_summary,
            start_position=1,
            ci_level=0.80,
        )


def test_validated_wrapper_invalid_input_contract(subinterval_inputs):
    """Freeze exact validation behavior of the analysis wrapper."""
    tbr_df, tbr_summary = subinterval_inputs
    route = tbr.compute_interval_estimate_and_ci

    for start_position in (True, np.int64(1)):
        result = route(tbr_df, tbr_summary, start_position, 2, 0.80)
        assert result["estimate"] == pytest.approx(3.0)

    invalid_calls = [
        ((0, 2, 0.80), ValueError, "start_position must be >= 1, got: 0"),
        (
            (1, 4, 0.80),
            ValueError,
            "end_position (4) exceeds the number of test observations (3)",
        ),
        (
            (3, 2, 0.80),
            ValueError,
            "end_position (2) must be >= start_position (3)",
        ),
        (
            (None, 2, 0.80),
            TypeError,
            "start_position and end_position must be integers",
        ),
        (
            (1, True, 0.80),
            TypeError,
            "cannot do positional indexing on RangeIndex with these indexers "
            "[True] of type bool",
        ),
        ((1, 2, 0), ValueError, "ci_level must be between 0 and 1, got: 0"),
        ((1, 2, 1), ValueError, "ci_level must be between 0 and 1, got: 1"),
        ((1, 2, "x"), TypeError, "ci_level must be a number"),
    ]
    for arguments, exception_type, message in invalid_calls:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(exception_type, match=f"^{re.escape(message)}$"):
                route(tbr_df, tbr_summary, *arguments)
        assert caught == []


def test_direct_validator_runtime_contract(subinterval_inputs):
    """Freeze direct validator success, failure, and warning behavior."""
    tbr_df, tbr_summary = subinterval_inputs

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analysis.validate_subinterval_parameters(
            tbr_df,
            tbr_summary,
            np.int64(1),
            np.int64(2),
            np.float64(0.80),
        )
    assert result is None
    assert caught == []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(
            ValueError,
            match=(
                r"^end_position \(4\) exceeds the number of "
                r"test observations \(3\)$"
            ),
        ):
            analysis.validate_subinterval_parameters(
                tbr_df,
                tbr_summary,
                1,
                4,
                0.80,
            )
    assert caught == []


@pytest.mark.parametrize(
    "route",
    [
        core.compute_interval_estimate_and_ci,
        core_effects.compute_interval_estimate_and_ci,
        functional_functions.compute_interval_estimate_and_ci,
    ],
)
def test_low_level_wrapper_validation_differences(route, subinterval_inputs):
    """Freeze low-level behavior that differs from the validated wrapper."""
    tbr_df, tbr_summary = subinterval_inputs

    assert route(tbr_df, tbr_summary, True, 2, 0.80)["estimate"] == pytest.approx(3.0)
    with pytest.raises(
        TypeError,
        match=(
            "^cannot do positional indexing on RangeIndex with these indexers "
            r"\[True\] of type bool$"
        ),
    ):
        route(tbr_df, tbr_summary, 1, True, 0.80)
    assert route(tbr_df, tbr_summary, 1, 4, 0.80)["estimate"] == pytest.approx(6.0)
    assert route(tbr_df, tbr_summary, 1, None, 0.80)["estimate"] == pytest.approx(6.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="^x_values cannot be empty$"):
            route(tbr_df, tbr_summary, 0, 2, 0.80)
        with pytest.raises(ValueError, match="^x_values cannot be empty$"):
            route(tbr_df, tbr_summary, 3, 2, 0.80)
        with pytest.raises(
            TypeError,
            match=r"^unsupported operand type\(s\) for -: 'NoneType' and 'int'$",
        ):
            route(tbr_df, tbr_summary, None, 2, 0.80)
        with pytest.raises(
            TypeError,
            match=r"^unsupported operand type\(s\) for -: 'int' and 'str'$",
        ):
            route(tbr_df, tbr_summary, 1, 2, "x")
    assert caught == []

    zero_level = route(tbr_df, tbr_summary, 1, 2, 0)
    assert zero_level["lower"] == pytest.approx(zero_level["estimate"])
    assert zero_level["upper"] == pytest.approx(zero_level["estimate"])

    unit_level = route(tbr_df, tbr_summary, 1, 2, 1)
    assert np.isneginf(unit_level["lower"])
    assert np.isposinf(unit_level["upper"])


def test_oop_wrapper_call_forms_validation_and_warnings(fitted_analysis):
    """Freeze OOP wrapper behavior independently from functional routes."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        python_result = fitted_analysis.analyze_subinterval(
            start_position=1,
            end_position=2,
            ci_level=0.80,
        )
        numpy_result = fitted_analysis.analyze_subinterval(
            start_position=np.int64(1),
            end_position=np.int64(2),
        )
        boolean_result = fitted_analysis.analyze_subinterval(True, 2, 0.80)

    assert python_result.start_position == 1
    assert python_result.end_position == 2
    assert python_result.n_observations == 2
    assert numpy_result == python_result
    assert boolean_result == python_result
    assert caught == []

    with pytest.raises(TypeError, match="unexpected keyword argument 'start_day'"):
        fitted_analysis.analyze_subinterval(
            start_day=1,
            end_day=2,
            ci_level=0.80,
        )
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'end_position'"
    ):
        fitted_analysis.analyze_subinterval(start_position=1)

    invalid_calls = [
        (
            (None, 2, 0.80),
            TypeError,
            "start_position must be an integer, got NoneType",
        ),
        (
            (1, None, 0.80),
            TypeError,
            "end_position must be an integer, got NoneType",
        ),
        (
            (0, 2, 0.80),
            ValueError,
            "start_position must be a positive integer (>= 1), got 0",
        ),
        (
            (3, 2, 0.80),
            ValueError,
            "start_position (3) must be <= end_position (2)",
        ),
        (
            (1, 4, 0.80),
            ValueError,
            "end_position (4) exceeds the number of test observations (3)",
        ),
        ((1, 2, 0), ValueError, "ci_level must be between 0 and 1 (exclusive), got 0"),
        ((1, 2, 1), ValueError, "ci_level must be between 0 and 1 (exclusive), got 1"),
        ((1, 2, "x"), TypeError, "ci_level must be numeric, got str"),
    ]
    for arguments, exception_type, message in invalid_calls:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(exception_type, match=f"^{re.escape(message)}$"):
                fitted_analysis.analyze_subinterval(*arguments)
        assert caught == []
