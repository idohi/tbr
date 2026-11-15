"""Phase 0 reflection, serialization, and presentation contracts."""

import base64
import dataclasses
import inspect
import json
import pickle

import numpy as np
import pandas as pd
import pytest

import tbr
import tbr.core as core
from tbr.core import results as result_objects

pytestmark = pytest.mark.integration

LEGACY_FIELDS = (
    "estimate",
    "lower",
    "upper",
    "se",
    "precision",
    "ci_level",
    "start_day",
    "end_day",
    "n_days",
)

# Trusted repository fixture created from the pre-migration result class with
# portable pickle protocol 4, supported across Python 3.8-3.12.
PRE_MIGRATION_SUBINTERVAL_PICKLE = base64.b64decode(
    b"gASVxwAAAAAAAACMEHRici5jb3JlLnJlc3VsdHOUjBRUQlJTdWJpbnRlcnZhbFJlc3Vs"
    b"dJSTlCmBlH2UKIwIZXN0aW1hdGWUR0BGmZmZmZmajAVsb3dlcpRHQD4ZmZmZmZqMBXVw"
    b"cGVylEdATiZmZmZmZowCc2WUR0AnmZmZmZmajAlwcmVjaXNpb26UR0AuMzMzMzMzjAhj"
    b"aV9sZXZlbJRHP+mZmZmZmZqMCXN0YXJ0X2RheZRLAowHZW5kX2RheZRLB4wGbl9kYXlz"
    b"lEsGdWIu",
    validate=True,
)


@pytest.fixture
def subinterval_result():
    """Return a deterministic legacy subinterval result."""
    return result_objects.TBRSubintervalResult(
        estimate=45.2,
        lower=30.1,
        upper=60.3,
        se=11.8,
        precision=15.1,
        ci_level=0.80,
        start_day=2,
        end_day=7,
        n_days=6,
    )


def test_subinterval_result_routes_constructor_and_reflection(subinterval_result):
    """Freeze routes, constructor forms, fields, annotations, and matching."""
    result_class = result_objects.TBRSubintervalResult
    assert core.TBRSubintervalResult is result_class
    assert not hasattr(tbr, "TBRSubintervalResult")

    positional = result_class(45.2, 30.1, 60.3, 11.8, 15.1, 0.80, 2, 7, 6)
    assert positional == subinterval_result
    assert inspect.signature(result_class) == inspect.Signature(
        [
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
            )
            for name, annotation in result_class.__annotations__.items()
        ],
        return_annotation=None,
    )
    assert (
        tuple(field.name for field in dataclasses.fields(result_class)) == LEGACY_FIELDS
    )
    assert tuple(result_class.__annotations__) == LEGACY_FIELDS
    if hasattr(result_class, "__match_args__"):
        assert result_class.__match_args__ == LEGACY_FIELDS
    assert dataclasses.asdict(subinterval_result) == subinterval_result.to_dict()

    replaced = dataclasses.replace(subinterval_result, start_day=3, n_days=5)
    assert replaced.start_day == 3
    assert replaced.end_day == 7
    assert replaced.n_days == 5
    assert subinterval_result.start_day == 2

    with pytest.raises(TypeError, match="unexpected keyword argument 'start_position'"):
        result_class(
            estimate=45.2,
            lower=30.1,
            upper=60.3,
            se=11.8,
            precision=15.1,
            ci_level=0.80,
            start_position=2,
            end_position=7,
            n_observations=6,
        )


def test_subinterval_result_equality_hash_pickle_and_repr(subinterval_result):
    """Freeze value semantics, pickle compatibility, and exact representation."""
    same = dataclasses.replace(subinterval_result)
    different = dataclasses.replace(subinterval_result, end_day=8)

    assert same == subinterval_result
    assert hash(same) == hash(subinterval_result)
    assert different != subinterval_result

    restored = pickle.loads(pickle.dumps(subinterval_result))
    assert type(restored) is result_objects.TBRSubintervalResult
    assert restored == subinterval_result
    assert dataclasses.asdict(restored) == dataclasses.asdict(subinterval_result)

    assert repr(subinterval_result) == (
        "TBRSubintervalResult(\n"
        "  days=2-7 (n=6),\n"
        "  estimate=45.200,\n"
        "  CI=[30.100, 60.300] (level=0.8),\n"
        "  se=11.800\n"
        ")"
    )


def test_subinterval_result_loads_pre_migration_pickle(subinterval_result):
    """Freeze loading of an artifact written before the field migration."""
    restored = pickle.loads(PRE_MIGRATION_SUBINTERVAL_PICKLE)

    assert type(restored) is result_objects.TBRSubintervalResult
    assert restored == subinterval_result
    assert dataclasses.asdict(restored) == subinterval_result.to_dict()


def test_subinterval_result_json_contract(tmp_path, subinterval_result):
    """Freeze JSON envelopes, key order, and legacy serialized names."""
    with_metadata = tmp_path / "with-metadata.json"
    without_metadata = tmp_path / "without-metadata.json"

    subinterval_result.to_json(with_metadata)
    subinterval_result.to_json(without_metadata, include_metadata=False)

    metadata_payload = json.loads(with_metadata.read_text(encoding="utf-8"))
    plain_payload = json.loads(without_metadata.read_text(encoding="utf-8"))
    assert list(metadata_payload) == ["type", "data"]
    assert metadata_payload["type"] == "TBRSubintervalResult"
    assert list(metadata_payload["data"]) == list(LEGACY_FIELDS)
    assert metadata_payload["data"] == subinterval_result.to_dict()
    assert list(plain_payload) == list(LEGACY_FIELDS)
    assert plain_payload == subinterval_result.to_dict()


def test_subinterval_result_csv_default_and_explicit_index(
    tmp_path, subinterval_result
):
    """Freeze CSV column order and the current default-index behavior."""
    default_path = tmp_path / "default.csv"
    no_index_path = tmp_path / "no-index.csv"

    subinterval_result.to_csv(default_path)
    subinterval_result.to_csv(no_index_path, index=False)

    default_lines = default_path.read_text(encoding="utf-8").splitlines()
    no_index_lines = no_index_path.read_text(encoding="utf-8").splitlines()
    expected_header = ",".join(LEGACY_FIELDS)
    expected_values = "45.2,30.1,60.3,11.8,15.1,0.8,2,7,6"

    assert default_lines == [f",{expected_header}", f"0,{expected_values}"]
    assert no_index_lines == [expected_header, expected_values]


def test_fitted_analysis_string_uses_exact_legacy_runtime_text():
    """Freeze fitted and unfitted string representations before migration."""
    model = tbr.TBRAnalysis(level=0.80, threshold=0.0, test_end_inclusive=True)
    assert str(model) == "TBRAnalysis (not fitted)\n  level=0.8\n  threshold=0.0"

    time = np.arange(10, dtype=np.int64)
    control = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0])
    data = pd.DataFrame(
        {
            "time": time,
            "control": control,
            "test": 4.0 + 1.5 * control + np.linspace(-0.2, 0.2, 10),
        }
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

    fitted_text = str(model)
    assert fitted_text == (
        "TBRAnalysis (fitted)\n"
        "  Configuration:\n"
        "    level=0.8\n"
        "    threshold=0.0\n"
        "  Results:\n"
        "    Test period days: 3\n"
        f"    Final effect estimate: {model.summaries_.iloc[-1]['estimate']:.2f}\n"
        "    80% CI: "
        f"[{model.summaries_.iloc[-1]['lower']:.2f}, "
        f"{model.summaries_.iloc[-1]['upper']:.2f}]"
    )
