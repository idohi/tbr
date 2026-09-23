"""Focused regression checks for the public API documentation contract."""

from __future__ import annotations

import inspect
import pydoc
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence, get_args, get_origin, get_type_hints

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
API_REFERENCE = REPOSITORY_ROOT / "docs" / "api" / "api_reference.rst"
METHODOLOGY = REPOSITORY_ROOT / "docs" / "mathematical_methodology.md"

EXPECTED_AUTOSUMMARY_SYMBOLS = (
    "TBRAnalysis",
    "perform_tbr_analysis",
    "core.results.TBRResults",
    "core.results.TBRPredictionResult",
    "core.results.TBRSummaryResult",
    "core.results.TBRSubintervalResult",
    "create_tbr_summary",
    "create_incremental_tbr_summaries",
    "compute_interval_estimate_and_ci",
    "analyze_multiple_subintervals",
    "create_subinterval_summary",
    "validate_tbr_model",
    "diagnose_tbr_analysis",
    "check_tbr_assumptions",
    "analyze_tbr_residuals",
    "assess_tbr_performance",
    "create_tbr_diagnostic_report",
)

VAGUE_SCHEMA_ENTRIES = {
    "additional metrics",
    "detailed results",
    "model parameters",
    "result details",
    "summary data",
}

RESULT_CLASS_RETURN_REFERENCES = {
    "TBRAnalysis.analyze_subinterval": "tbr.core.results.TBRSubintervalResult",
    "TBRAnalysis.final_summary": "tbr.core.results.TBRSummaryResult",
    "TBRAnalysis.fit_predict": "tbr.core.results.TBRPredictionResult",
    "TBRAnalysis.fit_summarize": "tbr.core.results.TBRSummaryResult",
    "TBRAnalysis.predict": "tbr.core.results.TBRPredictionResult",
    "TBRAnalysis.summarize": "tbr.core.results.TBRSummaryResult",
    "perform_tbr_analysis": "tbr.core.results.TBRResults",
}

MODEL_PARAMETER_FIELDS = (
    "alpha",
    "beta",
    "sigma",
    "var_alpha",
    "var_beta",
    "cov_alpha_beta",
    "degrees_freedom",
    "pretest_x_mean",
    "pretest_sum_x_squared_deviations",
)

DAILY_RESULT_FIELDS = (
    "period",
    "y",
    "x",
    "pred",
    "predsd",
    "dif",
    "cumdif",
    "cumsd",
    "estsd",
)

SUMMARY_FIELDS = (
    "estimate",
    "precision",
    "lower",
    "upper",
    "se",
    "level",
    "thres",
    "prob",
    "alpha",
    "beta",
    "alpha_beta_cov",
    "var_alpha",
    "var_beta",
    "sigma",
    "t_dist_df",
)

INCREMENTAL_SUMMARY_FIELDS = ("test_day", *SUMMARY_FIELDS)

RESULT_SUMMARY_FIELDS = (
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
)

EXPECTED_SCHEMA_FIELDS = {
    "TBRAnalysis.get_params": ("level", "threshold", "test_end_inclusive"),
    "TBRAnalysis.params_": MODEL_PARAMETER_FIELDS,
    "TBRAnalysis.results_": DAILY_RESULT_FIELDS,
    "TBRAnalysis.summaries_": INCREMENTAL_SUMMARY_FIELDS,
    "TBRAnalysis.summarize_incremental": INCREMENTAL_SUMMARY_FIELDS,
    "core.results.TBRResults.conf_int": ("lower", "upper"),
    "core.results.TBRResults.model_params": (
        "alpha",
        "beta",
        "sigma",
        "var_alpha",
        "var_beta",
        "cov_alpha_beta",
        "degrees_freedom",
        "n_pretest",
        "pretest_x_mean",
    ),
    "core.results.TBRResults.summary": INCREMENTAL_SUMMARY_FIELDS,
    "core.results.TBRResults.tbr_dataframe": DAILY_RESULT_FIELDS,
    "core.results.TBRPredictionResult.to_dict": (
        "predictions",
        "n_predictions",
        "model_params",
        *MODEL_PARAMETER_FIELDS,
        "control_values",
    ),
    "core.results.TBRSummaryResult.to_dataframe": RESULT_SUMMARY_FIELDS,
    "core.results.TBRSummaryResult.to_dict": RESULT_SUMMARY_FIELDS,
    "core.results.TBRSubintervalResult.to_dict": (
        "estimate",
        "lower",
        "upper",
        "se",
        "ci_level",
        "start_day",
        "end_day",
        "n_days",
    ),
    "create_tbr_summary": SUMMARY_FIELDS,
    "create_incremental_tbr_summaries": INCREMENTAL_SUMMARY_FIELDS,
    "compute_interval_estimate_and_ci": (
        "estimate",
        "precision",
        "lower",
        "upper",
    ),
    "analyze_multiple_subintervals": (
        "estimate",
        "precision",
        "lower",
        "upper",
    ),
    "create_subinterval_summary": (
        "interval",
        "start_day",
        "end_day",
        "days",
        "estimate",
        "precision",
        "lower",
        "upper",
        "significant",
        "avg_daily_effect",
        "ci_level",
    ),
    "validate_tbr_model": (
        "overall_validity",
        "warnings",
        "assumption_tests",
        "linearity_valid",
        "normality_valid",
        "homoscedasticity_valid",
        "independence_valid",
        "all_assumptions_valid",
        "significance_level",
        "validation_summary",
        "error",
        "goodness_of_fit",
        "r_squared",
        "adj_r_squared",
        "f_statistic",
        "f_p_value",
        "mse",
        "rmse",
        "error",
        "residual_analysis",
        "residuals",
        "standardized_residuals",
        "studentized_residuals",
        "outliers",
        "outlier_threshold",
        "outlier_percentage",
        "error",
        "prediction_quality",
        "mae",
        "mse",
        "rmse",
        "mape",
        "prediction_interval_coverage",
        "n_predictions",
        "error",
    ),
    "diagnose_tbr_analysis": (
        "model_validation",
        "diagnostic_summary",
        "goodness_of_fit",
        "information_criteria",
        "aic",
        "bic",
        "log_likelihood",
        "normality_test",
        "statistic",
        "p_value",
        "is_normal",
        "test_name",
        "homoscedasticity_test",
        "statistic",
        "p_value",
        "is_homoscedastic",
        "test_name",
        "independence_test",
        "statistic",
        "interpretation",
        "is_independent",
        "test_name",
        "overall_validity",
        "warnings",
        "error",
        "performance_metrics",
        "recommendations",
    ),
    "check_tbr_assumptions": (
        "linearity_valid",
        "normality_valid",
        "homoscedasticity_valid",
        "independence_valid",
        "all_assumptions_valid",
        "significance_level",
        "validation_summary",
    ),
    "analyze_tbr_residuals": (
        "residuals",
        "standardized_residuals",
        "studentized_residuals",
        "outliers",
        "outlier_threshold",
        "outlier_percentage",
        "residual_stats",
        "mean",
        "std",
        "min",
        "max",
        "q25",
        "median",
        "q75",
        "residual_std",
        "n_observations",
    ),
    "assess_tbr_performance": (
        "data_metrics",
        "total_observations",
        "learning_observations",
        "test_observations",
        "learning_test_ratio",
        "prediction_metrics",
        "mae",
        "mse",
        "rmse",
        "mape",
        "mean_error",
        "std_error",
        "interval_coverage",
        "model_complexity",
        "degrees_freedom",
        "sigma",
        "r_squared_proxy",
        "efficiency_score",
        "performance_summary",
        "data_quality",
        "prediction_quality",
        "overall_performance",
    ),
    "create_tbr_diagnostic_report": (
        "executive_summary",
        "overall_validity",
        "warnings_count",
        "key_findings",
        "recommendations",
        "detailed_results",
        "model_validation",
        "diagnostic_summary",
        "performance_metrics",
        "recommendations",
        "report_timestamp",
    ),
}

SCHEMA_NOTES_OBJECTS = tuple(EXPECTED_SCHEMA_FIELDS)

QUOTED_SCHEMA_ENTRY = re.compile(
    r"^\s*(?:-\s+)?(?P<name>``[^`]+``|'[^']+'|\"[^\"]+\")" r"(?P<type>\s*:\s*\S.*)?\s*$"
)
INLINE_SCHEMA_ENTRY = re.compile(
    r"^\s*(?:-\s+)?``(?P<name>[^`:]+?)\s*:\s*(?P<type>[^`]+)``(?::.*)?$"
)

STABLE_SYMBOL_MAPPINGS = (
    (r"$\beta_0$", ("`alpha`",)),
    (r"$\beta_1$", ("`beta`",)),
    (r"$\sigma$", ("`sigma`",)),
    (
        r"$\text{Cov}(\hat{\beta}_0, \hat{\beta}_1)$",
        ("`cov_alpha_beta`",),
    ),
    (r"$\hat{\Delta}(T)$", ("`estimate`",)),
    (r"$\text{SE}$", ("`se`",)),
    (
        r"$t_{\alpha/2,\nu} \cdot \text{SE}$",
        ("`precision`",),
    ),
    (
        r"$P(\Delta(T) > \theta \mid \text{data})$",
        ("`prob`",),
    ),
    (r"$\sqrt{\mathbb{V}[\Delta(T)]}$", ("`cumsd`",)),
    (r"$\sqrt{\mathbb{V}[y_t^*]}$", ("`predsd`",)),
)


def _autosummary_symbols(source: str) -> tuple[str, ...]:
    """Extract object names from every autosummary directive."""
    symbols: list[str] = []
    in_autosummary = False

    for line in source.splitlines():
        stripped = line.strip()
        if stripped == ".. autosummary::":
            in_autosummary = True
            continue
        if not in_autosummary:
            continue
        if not stripped or stripped.startswith(":"):
            continue
        if line.startswith("   "):
            symbols.append(stripped)
            continue
        in_autosummary = False

    return tuple(symbols)


def _numpy_doc_section(docstring: str, heading: str) -> str:
    """Return one NumPy-style docstring section without snapshotting prose."""
    lines = docstring.splitlines()
    for index in range(len(lines) - 1):
        if lines[index].strip() == heading and set(lines[index + 1].strip()) == {"-"}:
            section: list[str] = []
            for candidate_index in range(index + 2, len(lines)):
                candidate = lines[candidate_index]
                next_line = (
                    lines[candidate_index + 1]
                    if candidate_index + 1 < len(lines)
                    else ""
                )
                if (
                    candidate.strip()
                    and next_line.strip()
                    and set(next_line.strip()) == {"-"}
                ):
                    break
                section.append(candidate)
            return "\n".join(section)
    return ""


def _public_documented_objects() -> Iterable[tuple[str, object]]:
    """Yield objects rendered on the autosummarized API pages."""
    for symbol in EXPECTED_AUTOSUMMARY_SYMBOLS:
        obj = pydoc.locate(f"tbr.{symbol}")
        if obj is None:
            continue
        documented_objects = [(symbol, obj)]
        if inspect.isclass(obj):
            documented_objects.extend(
                (f"{symbol}.{name}", member)
                for name, member in inspect.getmembers(obj)
                if not name.startswith("_")
                and (
                    inspect.isfunction(member)
                    or inspect.ismethod(member)
                    or isinstance(member, property)
                )
            )
        yield from documented_objects


def _public_returns_sections() -> Iterable[tuple[str, str]]:
    """Yield Returns sections rendered on the autosummarized API pages."""
    for name, documented_object in _public_documented_objects():
        returns = _numpy_doc_section(inspect.getdoc(documented_object) or "", "Returns")
        if returns:
            yield name, returns


def _public_documented_object(name: str) -> object:
    """Return one object from the rendered public API inventory."""
    documented_objects = dict(_public_documented_objects())
    return documented_objects[name]


def _returns_description(returns: str) -> str:
    """Normalize prose following the declared NumPy return type."""
    lines = returns.strip().splitlines()
    return " ".join(" ".join(lines[1:]).split())


def _normalized_schema_label(line: str) -> str:
    """Normalize a possible schema label for vague-entry checks."""
    label = line.strip()
    if label.startswith("-"):
        label = label[1:].strip()
    label = label.partition(":")[0].strip()
    return label.strip("`'\"").lower()


def _schema_entries(section: str) -> list[tuple[str, str]]:
    """Extract separately or jointly formatted schema names and types."""
    entries = []
    for line in section.splitlines():
        inline_match = INLINE_SCHEMA_ENTRY.match(line)
        if inline_match:
            entries.append(
                (
                    inline_match.group("name").strip(),
                    inline_match.group("type").strip(),
                )
            )
            continue
        quoted_match = QUOTED_SCHEMA_ENTRY.match(line)
        if quoted_match:
            entries.append(
                (
                    quoted_match.group("name").strip("`'\""),
                    (quoted_match.group("type") or "").lstrip(" :"),
                )
            )
    return entries


def _schema_entry_description(section: str, name: str) -> str:
    """Return normalized prose belonging to one typed schema entry."""
    lines = section.splitlines()
    entry_index = next(
        index
        for index, line in enumerate(lines)
        if any(entry_name == name for entry_name, _ in _schema_entries(line))
    )
    description = []
    for line in lines[entry_index + 1 :]:
        if _schema_entries(line):
            break
        description.append(line.strip())
    return " ".join(" ".join(description).split())


def _markdown_table_rows(source: str) -> Sequence[Sequence[str]]:
    """Parse Markdown table rows sufficiently for mapping association checks."""
    rows = []
    for line in source.splitlines():
        if not line.startswith("|"):
            continue
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        if cells and not all(set(cell) <= {"-", ":"} for cell in cells):
            rows.append(cells)
    return rows


def test_api_reference_has_exact_public_page_inventory() -> None:
    """Keep the intended 17 autosummary pages explicit and ordered."""
    source = API_REFERENCE.read_text(encoding="utf-8")

    assert _autosummary_symbols(source) == EXPECTED_AUTOSUMMARY_SYMBOLS


@pytest.mark.parametrize("symbol", EXPECTED_AUTOSUMMARY_SYMBOLS)
def test_autosummary_symbol_resolves_through_tbr(symbol: str) -> None:
    """Ensure every documented API object remains importable from ``tbr``."""
    assert pydoc.locate(f"tbr.{symbol}") is not None


def test_public_docstrings_do_not_double_escape_latex_commands() -> None:
    """Prevent raw docstrings from emitting doubled MathJax commands."""
    offenders = []
    for name, documented_object in _public_documented_objects():
        doubled_commands = re.findall(
            r"\\\\[A-Za-z]+", inspect.getdoc(documented_object) or ""
        )
        offenders.extend(f"{name}: {command}" for command in doubled_commands)

    assert not offenders, "Double-escaped LaTeX commands:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    ("symbol", "equations"),
    (
        (
            "create_tbr_summary",
            (
                "CI=estimate±t_{α/2,df}×se",
                "P(effect>threshold)=1-F_t((threshold-estimate)/se,df)",
            ),
        ),
        (
            "create_incremental_tbr_summaries",
            (
                "CI_t=estimate_t±t_{α/2,df}×se_t",
                "P(effect_t>threshold)=1-F_t((threshold-estimate_t)/se_t,df)",
            ),
        ),
        (
            "compute_interval_estimate_and_ci",
            (
                "CI=estimate±t_{α/2,df}×se",
                r"se=\sqrt{\sum_{i=start}^{end}estsd_i^2+n_{days}×σ^2}",
                r"estimate=\sum_{i=start}^{end}(y_i-pred_i)",
            ),
        ),
    ),
)
def test_validated_public_docstring_equations_are_preserved(
    symbol: str, equations: tuple[str, ...]
) -> None:
    """Keep the validated public equations unchanged during documentation work."""
    documented_object = pydoc.locate(f"tbr.{symbol}")
    assert documented_object is not None
    compact_docstring = re.sub(r"\s+", "", inspect.getdoc(documented_object) or "")

    assert all(equation in compact_docstring for equation in equations)


def test_public_returns_schemas_have_no_vague_catch_all_entries() -> None:
    """Reject known catch-all labels in public function Returns schemas."""
    offenders = []
    for symbol, returns in _public_returns_sections():
        for line in returns.splitlines():
            if _normalized_schema_label(line) in VAGUE_SCHEMA_ENTRIES:
                offenders.append(f"{symbol}: {line.strip()}")

    assert not offenders, "Vague Returns entries:\n" + "\n".join(offenders)


def test_typed_vague_schema_labels_are_normalized() -> None:
    """Ensure a prose catch-all cannot evade detection by adding a type."""
    assert _normalized_schema_label("Detailed results : Dict[str, Any]") in (
        VAGUE_SCHEMA_ENTRIES
    )


def test_public_returns_schemas_do_not_mix_typed_and_untyped_keys() -> None:
    """Require every quoted schema key to carry a type when any of them do."""
    offenders = []
    for symbol, returns in _public_returns_sections():
        entries = _schema_entries(returns)
        if any(entry_type for _, entry_type in entries):
            offenders.extend(
                f"{symbol}: {name}" for name, entry_type in entries if not entry_type
            )

    assert (
        not offenders
    ), "Untyped Returns keys mixed into typed schemas:\n" + "\n".join(offenders)


def test_public_returns_do_not_embed_definition_schemas() -> None:
    """Keep typed schemas out of Returns descriptions that Sphinx nests."""
    offenders = [
        symbol
        for symbol, returns in _public_returns_sections()
        if _schema_entries(returns)
    ]

    assert not offenders, "Schemas embedded in Returns:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "symbol",
    tuple(RESULT_CLASS_RETURN_REFERENCES) + SCHEMA_NOTES_OBJECTS,
)
def test_structured_returns_use_one_described_entry(symbol: str) -> None:
    """Keep structured Returns sections to one type and its description."""
    documented_object = _public_documented_object(symbol)
    returns = _numpy_doc_section(inspect.getdoc(documented_object) or "", "Returns")
    description = _returns_description(returns)
    declarations = [
        line.strip()
        for line in returns.strip().splitlines()
        if line.strip() and not line[:1].isspace()
    ]

    assert declarations, f"{symbol} has no declared return type"
    assert (
        len(declarations) == 1
    ), f"{symbol} has multiple top-level Returns entries: {declarations}"
    assert description, f"{symbol} has no return description"
    assert not _schema_entries(returns), f"{symbol} embeds a schema in Returns"
    assert not any(
        line.lstrip().startswith("- ") for line in returns.splitlines()
    ), f"{symbol} embeds a bullet list in Returns"


@pytest.mark.parametrize(
    ("symbol", "result_class"),
    RESULT_CLASS_RETURN_REFERENCES.items(),
)
def test_result_class_returns_link_to_public_reference(
    symbol: str, result_class: str
) -> None:
    """Link result-class returns to their dedicated public API reference."""
    documented_object = _public_documented_object(symbol)
    returns = _numpy_doc_section(inspect.getdoc(documented_object) or "", "Returns")

    assert f":class:`{result_class}`" in returns


@pytest.mark.parametrize(
    ("symbol", "expected_fields"),
    EXPECTED_SCHEMA_FIELDS.items(),
)
def test_structured_return_schemas_are_complete(
    symbol: str, expected_fields: tuple[str, ...]
) -> None:
    """Require every stable field exactly once per documented schema path."""
    documented_object = _public_documented_object(symbol)
    notes = _numpy_doc_section(inspect.getdoc(documented_object) or "", "Notes")
    documented_fields = tuple(
        name for name, field_type in _schema_entries(notes) if field_type
    )

    assert Counter(documented_fields) == Counter(expected_fields), (
        f"{symbol} schema mismatch: expected {expected_fields}, "
        f"documented {documented_fields}"
    )


def test_public_field_descriptions_do_not_embed_bullet_lists() -> None:
    """Keep nested bullet lists out of Sphinx field descriptions."""
    offenders = []
    for symbol, documented_object in _public_documented_objects():
        docstring = inspect.getdoc(documented_object) or ""
        for section_name in ("Parameters", "Attributes"):
            section = _numpy_doc_section(docstring, section_name)
            if "\n    - " in section or _schema_entries(section):
                offenders.append(f"{symbol} ({section_name})")

    assert not offenders, "Nested bullet lists in field descriptions:\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize(
    "member_name",
    ("results_", "summaries_", "summarize_incremental"),
)
def test_tbr_analysis_dataframe_returns_do_not_embed_bullet_schemas(
    member_name: str,
) -> None:
    """Keep DataFrame schemas out of Returns descriptions that Sphinx nests."""
    member = getattr(pydoc.locate("tbr.TBRAnalysis"), member_name)
    returns = _numpy_doc_section(inspect.getdoc(member) or "", "Returns")

    assert not any(line.lstrip().startswith("- ") for line in returns.splitlines())


@pytest.mark.parametrize(
    "symbol",
    ("analyze_multiple_subintervals", "create_subinterval_summary"),
)
def test_subinterval_returns_do_not_embed_definition_schemas(symbol: str) -> None:
    """Keep subinterval result schemas out of nested Returns descriptions."""
    function = pydoc.locate(f"tbr.{symbol}")
    returns = _numpy_doc_section(inspect.getdoc(function) or "", "Returns")

    assert not _schema_entries(returns)


@pytest.mark.parametrize(
    "symbol",
    (
        "validate_tbr_model",
        "diagnose_tbr_analysis",
        "check_tbr_assumptions",
        "analyze_tbr_residuals",
        "assess_tbr_performance",
        "create_tbr_diagnostic_report",
    ),
)
def test_diagnostic_returns_do_not_embed_definition_schemas(symbol: str) -> None:
    """Keep diagnostic schemas out of nested Returns descriptions."""
    function = pydoc.locate(f"tbr.{symbol}")
    returns = _numpy_doc_section(inspect.getdoc(function) or "", "Returns")

    assert not _schema_entries(returns)


def test_prediction_to_dict_documents_nested_model_parameter_keys() -> None:
    """Keep the produced model-parameter mapping individually typed."""
    result_type = pydoc.locate("tbr.core.results.TBRPredictionResult")
    notes = _numpy_doc_section(inspect.getdoc(result_type.to_dict) or "", "Notes")
    typed_names = {name for name, entry_type in _schema_entries(notes) if entry_type}

    assert {
        "alpha",
        "beta",
        "sigma",
        "var_alpha",
        "var_beta",
        "cov_alpha_beta",
        "degrees_freedom",
        "pretest_x_mean",
        "pretest_sum_x_squared_deviations",
    } <= typed_names


def test_result_se_descriptions_keep_distinct_public_semantics() -> None:
    """Distinguish summary standard error from legacy subinterval half-width."""
    summary_type = pydoc.locate("tbr.core.results.TBRSummaryResult")
    subinterval_type = pydoc.locate("tbr.core.results.TBRSubintervalResult")
    summary_notes = _numpy_doc_section(
        inspect.getdoc(summary_type.to_dict) or "", "Notes"
    )
    subinterval_notes = _numpy_doc_section(
        inspect.getdoc(subinterval_type.to_dict) or "", "Notes"
    )

    summary_se = _schema_entry_description(summary_notes, "se").lower()
    subinterval_se = _schema_entry_description(subinterval_notes, "se").lower()

    assert "standard error" in summary_se
    assert "half-width" not in summary_se
    assert "credible-interval half-width" in subinterval_se
    assert "legacy" in subinterval_se


@pytest.mark.parametrize(
    ("symbol", "schema_section"),
    (
        ("tbr.create_tbr_summary", "Notes"),
        ("tbr.create_incremental_tbr_summaries", "Notes"),
        ("tbr.functional.tbr_functions.create_tbr_summary", "Returns"),
        (
            "tbr.functional.tbr_functions.create_incremental_tbr_summaries",
            "Returns",
        ),
    ),
)
def test_summary_helpers_describe_se_as_standard_error(
    symbol: str, schema_section: str
) -> None:
    """Keep summary-helper scale terminology aligned with result objects."""
    function = pydoc.locate(symbol)
    docstring = inspect.getdoc(function) or ""
    schema = _numpy_doc_section(docstring, schema_section)
    se_description = _schema_entry_description(schema, "se").lower()

    assert "posterior standard deviation" not in docstring.lower()
    assert "standard error" in se_description


def test_methodology_uses_student_t_scale_terminology() -> None:
    """Keep cumulative uncertainty terminology precise without changing equations."""
    methodology = METHODOLOGY.read_text(encoding="utf-8").lower()

    assert "posterior standard deviation of the cumulative effect" not in methodology
    assert "posterior standard deviation (scale of the distribution)" not in methodology
    assert "credible interval at confidence level" not in methodology
    assert "standard error" in methodology
    assert "student's $t$ scale" in methodology


@pytest.mark.parametrize(
    "symbol",
    ("TBRAnalysis.results_", "core.results.TBRResults.tbr_dataframe"),
)
def test_cumsd_schema_documents_legacy_student_t_scale(symbol: str) -> None:
    """Describe ``cumsd`` by its role while retaining the legacy column name."""
    documented_object = _public_documented_object(symbol)
    notes = _numpy_doc_section(inspect.getdoc(documented_object) or "", "Notes")
    cumsd_description = _schema_entry_description(notes, "cumsd").lower()

    assert "standard error" in cumsd_description
    assert "student-t scale" in cumsd_description
    assert "legacy" in cumsd_description


def test_incremental_docstring_avoids_unsupported_sequential_claims() -> None:
    """Describe observed criteria without claiming a stopping procedure."""
    function = pydoc.locate("tbr.create_incremental_tbr_summaries")
    docstring = (inspect.getdoc(function) or "").lower()

    unsupported_claims = (
        "statistically significant",
        "optimizing test duration",
        "early stopping",
        "minimum test duration",
        "significant_days",
    )

    assert all(claim not in docstring for claim in unsupported_claims)
    assert "first observed day meeting" in docstring


def test_subinterval_docstring_defines_legacy_exclusion_flag_precisely() -> None:
    """Avoid presenting credible-interval exclusion as a hypothesis test."""
    function = pydoc.locate("tbr.create_subinterval_summary")
    docstring = (inspect.getdoc(function) or "").lower()

    assert "conservative test" not in docstring
    assert "statistical significance is determined" not in docstring
    assert "legacy interval-exclusion flag" in docstring
    assert "not a hypothesis test" in docstring


@pytest.mark.parametrize(
    "symbol",
    ("analyze_multiple_subintervals", "create_subinterval_summary"),
)
def test_subinterval_wrappers_document_propagated_input_errors(symbol: str) -> None:
    """Document errors propagated by the direct interval helper."""
    function = pydoc.locate(f"tbr.{symbol}")
    raises = _numpy_doc_section(inspect.getdoc(function) or "", "Raises")

    assert re.search(r"^KeyError$", raises, flags=re.MULTILINE)
    assert re.search(r"^IndexError$", raises, flags=re.MULTILINE)


@pytest.mark.parametrize(
    "symbol",
    (
        "tbr.core.results.TBRSummaryResult",
        "tbr.core.results.TBRSubintervalResult",
    ),
)
def test_mixed_type_to_dict_annotations_include_integer_fields(symbol: str) -> None:
    """Keep public conversion annotations consistent with returned integers."""
    result_type = pydoc.locate(symbol)
    return_hint = get_type_hints(result_type.to_dict)["return"]
    key_hint, value_hint = get_args(return_hint)

    assert get_origin(return_hint) is dict
    assert key_hint is str
    assert set(get_args(value_hint)) == {float, int}


@pytest.mark.parametrize(("symbol", "python_names"), STABLE_SYMBOL_MAPPINGS)
def test_methodology_keeps_stable_symbol_to_code_mapping(
    symbol: str, python_names: tuple[str, ...]
) -> None:
    """Keep selected scientific symbols associated with their public names."""
    rows = _markdown_table_rows(METHODOLOGY.read_text(encoding="utf-8"))
    matching_rows = [row for row in rows if row and row[0] == symbol]

    assert matching_rows, f"Missing canonical mapping row for {symbol}"
    assert any(
        all(name in " | ".join(row) for name in python_names) for row in matching_rows
    ), f"{symbol} is no longer mapped to {', '.join(python_names)}"
