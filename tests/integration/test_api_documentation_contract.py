"""Focused regression checks for the public API documentation contract."""

from __future__ import annotations

import ast
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
AUTOSUMMARY_CLASS_TEMPLATE = (
    REPOSITORY_ROOT / "docs" / "_templates" / "autosummary" / "class.rst"
)

ANNOTATION_ONLY_ATTRIBUTE_PAGES = (
    "tbr.core.results.TBRPredictionResult",
    "tbr.core.results.TBRSummaryResult",
    "tbr.core.results.TBRSubintervalResult",
    "tbr.TBRResidualStatistics",
    "tbr.TBRResidualAnalysis",
    "tbr.TBRDiagnosticReport",
)

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
    "TBRResidualStatistics",
    "TBRResidualAnalysis",
    "TBRDiagnosticReport",
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
        "precision",
        "ci_level",
        "start_day",
        "end_day",
        "n_days",
    ),
    "create_tbr_summary": SUMMARY_FIELDS,
    "create_incremental_tbr_summaries": INCREMENTAL_SUMMARY_FIELDS,
    "compute_interval_estimate_and_ci": (
        "estimate",
        "se",
        "precision",
        "lower",
        "upper",
    ),
    "analyze_multiple_subintervals": (
        "estimate",
        "se",
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
        "se",
        "precision",
        "lower",
        "upper",
        "interval_excludes_threshold",
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


def _numpy_field_description(section: str, name: str) -> str:
    """Return normalized prose for one NumPy Parameters/Attributes field."""
    lines = section.splitlines()
    declaration = re.compile(r"^(?P<names>[A-Za-z_][A-Za-z0-9_, ]*)\s*:\s*\S.*$")
    entry_index = next(
        (
            index
            for index, line in enumerate(lines)
            if (match := declaration.match(line))
            and name
            in {candidate.strip() for candidate in match.group("names").split(",")}
        ),
        None,
    )
    if entry_index is None:
        return ""
    description = []
    for line in lines[entry_index + 1 :]:
        if declaration.match(line):
            break
        description.append(line.strip())
    return " ".join(" ".join(description).split())


def _numpy_raises_description(section: str, exception: str) -> str:
    """Return normalized prose associated with one declared exception."""
    lines = section.splitlines()
    entry_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() == exception and not line[:1].isspace()
        ),
        None,
    )
    if entry_index is None:
        return ""
    description = []
    for line in lines[entry_index + 1 :]:
        if line.strip() and not line[:1].isspace():
            break
        description.append(line.strip())
    return " ".join(" ".join(description).split())


def _schema_paths(section: str) -> set[str]:
    """Return dot-separated paths represented by an indented schema."""
    paths = set()
    ancestors: list[tuple[int, str]] = []
    for line in section.splitlines():
        entries = _schema_entries(line)
        if not entries:
            continue
        indentation = len(line) - len(line.lstrip())
        while ancestors and ancestors[-1][0] >= indentation:
            ancestors.pop()
        for name, _ in entries:
            path = ".".join([ancestor for _, ancestor in ancestors] + [name])
            paths.add(path)
            ancestors.append((indentation, name))
    return paths


def _normalized_prose(text: str) -> str:
    """Normalize prose for unordered vocabulary alternatives."""
    return " ".join(re.sub(r"[^a-z0-9_]+", " ", text.casefold()).split())


def _contains_any(text: str, alternatives: tuple[str, ...]) -> bool:
    """Return whether normalized prose contains one equivalent expression."""
    normalized = f" {_normalized_prose(text)} "
    return any(
        f" {_normalized_prose(alternative)} " in normalized
        for alternative in alternatives
    )


def _paragraphs(text: str) -> tuple[str, ...]:
    """Return nonempty prose paragraphs without imposing sentence shape."""
    return tuple(
        " ".join(paragraph.split())
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    )


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
    """Keep the intended 20 autosummary pages explicit and ordered."""
    source = API_REFERENCE.read_text(encoding="utf-8")

    assert _autosummary_symbols(source) == EXPECTED_AUTOSUMMARY_SYMBOLS


def test_annotation_only_attributes_are_not_rendered_twice() -> None:
    """Suppress declared fields without hiding documented properties."""
    template = AUTOSUMMARY_CLASS_TEMPLATE.read_text(encoding="utf-8")
    mapping_match = re.search(
        r"{% set annotation_only_attributes = (?P<mapping>\{.*?\}) %}",
        template,
        flags=re.DOTALL,
    )
    result_class_directive = """.. autoclass:: {{ objname }}()
   :members:
   :exclude-members: {{ annotation_only_attributes[fullname] | join(", ") }}
   :show-inheritance:
   :class-doc-from: class"""
    public_schema_directive = """.. autoclass:: {{ objname }}
   :members:
   :exclude-members: {{ annotation_only_attributes[fullname] | join(", ") }}
   :show-inheritance:"""

    assert mapping_match is not None
    configured = ast.literal_eval(mapping_match.group("mapping"))
    expected = {
        page: tuple(pydoc.locate(page).__annotations__)
        for page in ANNOTATION_ONLY_ATTRIBUTE_PAGES
    }

    assert configured == expected
    assert result_class_directive in template
    assert public_schema_directive in template
    for page, excluded_fields in configured.items():
        documented_object = pydoc.locate(page)
        properties = {
            name
            for name, member in inspect.getmembers(documented_object)
            if isinstance(member, property)
        }
        assert properties.isdisjoint(excluded_fields)


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
                r"se=\sqrt{n_{days}×σ^2+n_{days}^2×v}",
                r"v=\mathbb{V}[\hat{\beta}_0]"
                r"+2\bar{x}\operatorname{Cov}(\hat{\beta}_0,\hat{\beta}_1)"
                r"+\bar{x}^2\mathbb{V}[\hat{\beta}_1]",
                r"estimate=\sum_{i=start}^{end}(y_i-pred_i)",
            ),
        ),
        (
            "create_subinterval_summary",
            (r"T_s=\mathrm{end\_day}-\mathrm{start\_day}+1",),
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


@pytest.mark.parametrize(
    "result_type",
    ("tbr.core.results.TBRSummaryResult", "tbr.core.results.TBRSubintervalResult"),
)
def test_result_se_and_precision_descriptions_share_public_semantics(
    result_type: str,
) -> None:
    """Document se as the standard error and precision as the half-width."""
    notes = _numpy_doc_section(
        inspect.getdoc(pydoc.locate(result_type).to_dict) or "", "Notes"
    )

    se = _schema_entry_description(notes, "se").lower()
    precision = _schema_entry_description(notes, "precision").lower()

    assert "standard error" in se
    assert "half-width" not in se
    assert "legacy" not in se
    assert "half-width" in precision


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
    assert "first observed position meeting" in docstring


def test_subinterval_docstring_defines_exclusion_flag_precisely() -> None:
    """Avoid presenting credible-interval exclusion as a hypothesis test."""
    function = pydoc.locate("tbr.create_subinterval_summary")
    docstring = (inspect.getdoc(function) or "").lower()

    assert "conservative test" not in docstring
    assert "statistical significance is determined" not in docstring
    assert "significan" not in docstring
    assert "interval_excludes_threshold" in docstring
    assert "not a hypothesis test" in docstring


@pytest.mark.parametrize(
    "symbol",
    ("analyze_multiple_subintervals", "create_subinterval_summary"),
)
def test_subinterval_wrappers_document_propagated_input_errors(symbol: str) -> None:
    """Document errors propagated by the direct interval helper."""
    function = pydoc.locate(f"tbr.{symbol}")
    raises = _numpy_doc_section(inspect.getdoc(function) or "", "Raises")

    assert re.search(r"^ValueError$", raises, flags=re.MULTILINE)
    assert re.search(r"^TypeError$", raises, flags=re.MULTILINE)
    assert not re.search(r"^KeyError$", raises, flags=re.MULTILINE)
    assert not re.search(r"^IndexError$", raises, flags=re.MULTILINE)


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


SUMMARY_SEMANTIC_SYMBOLS = (
    "TBRAnalysis.summarize_incremental",
    "TBRAnalysis.summaries_",
    "core.results.TBRResults.summary",
    "create_tbr_summary",
    "create_incremental_tbr_summaries",
)

LEGACY_SCIENTIFIC_FIELDS = {
    "alpha_beta_cov": (
        ("cov_alpha_beta",),
        ("covariance", "cov"),
        ("intercept", "alpha", "beta_0"),
        ("slope", "beta", "beta_1"),
    ),
    "thres": (
        ("threshold",),
        ("theta",),
        ("posterior",),
        ("probability", "exceedance"),
    ),
    "t_dist_df": (
        ("degrees_freedom",),
        ("student t", "t distribution"),
        ("regression", "residual", "pretest"),
    ),
}

TYPED_DICT_FIELDS = {
    "TBRResidualStatistics": ("mean", "std", "min", "max", "q25", "median", "q75"),
    "TBRResidualAnalysis": (
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
    "TBRDiagnosticReport": (
        "executive_summary",
        "overall_validity",
        "warnings_count",
        "key_findings",
        "recommendations",
        "detailed_results",
        "report_timestamp",
    ),
}

AGGREGATE_SCHEMA_PATHS = {
    "diagnose_tbr_analysis": (
        "model_validation.assumption_tests.normality_valid",
        "model_validation.goodness_of_fit.r_squared",
        "diagnostic_summary.normality_test.p_value",
        "performance_metrics.prediction_metrics.mape",
        "recommendations",
    ),
    "create_tbr_diagnostic_report": (
        "detailed_results.model_validation.assumption_tests.normality_valid",
        "detailed_results.diagnostic_summary.normality_test.p_value",
        "detailed_results.performance_metrics.prediction_metrics.mape",
        "detailed_results.recommendations",
    ),
}

POSITIONAL_CASES = (
    (
        "core.results.TBRResults.n_test_days",
        "",
        "",
        (
            ("legacy", "historical name"),
            ("test row", "time point"),
            ("count", "number", "length"),
        ),
    ),
    (
        "create_incremental_tbr_summaries",
        "Notes",
        "test_day",
        (
            ("legacy", "historical name"),
            ("test row", "time point"),
            ("one based", "starts at one"),
            ("cumulative",),
        ),
    ),
    (
        "TBRAnalysis.analyze_subinterval",
        "Parameters",
        "start_day",
        (
            ("legacy", "historical name"),
            ("test row", "time point"),
            ("position", "index"),
            ("one based", "starts at one"),
            ("inclusive", "includes the endpoint"),
        ),
    ),
    (
        "TBRAnalysis.analyze_subinterval",
        "Parameters",
        "end_day",
        (
            ("legacy", "historical name"),
            ("test row", "time point"),
            ("position", "index"),
            ("one based", "starts at one"),
            ("inclusive", "includes the endpoint"),
        ),
    ),
    (
        "core.results.TBRSubintervalResult",
        "Attributes",
        "n_days",
        (
            ("legacy", "historical name"),
            ("test row", "time point"),
            ("count", "number", "length"),
            ("inclusive", "both endpoints"),
        ),
    ),
    (
        "create_subinterval_summary",
        "Notes",
        "days",
        (
            ("legacy", "historical name"),
            ("test row", "time point"),
            ("count", "number", "length"),
            ("inclusive", "both endpoints"),
        ),
    ),
)

DIAGNOSTIC_UNIT_CASES = (
    (
        "validate_tbr_model",
        {
            "residuals": "response",
            "mse": "squared",
            "standardized_residuals": "dimensionless",
            "prediction_interval_coverage": "dimensionless",
        },
    ),
    (
        "analyze_tbr_residuals",
        {
            "residuals": "response",
            "standardized_residuals": "dimensionless",
            "residual_std": "response",
        },
    ),
    (
        "assess_tbr_performance",
        {
            "mae": "response",
            "mse": "squared",
            "interval_coverage": "dimensionless",
            "sigma": "response",
            "efficiency_score": "dimensionless",
        },
    ),
)

UNIT_CONCEPTS = {
    "response": (("response units", "units of y", "same units as y", "outcome units"),),
    "squared": (
        (
            "squared response units",
            "squared units of y",
            "response units squared",
            "outcome units squared",
        ),
    ),
    "dimensionless": (
        ("dimensionless", "unitless", "no units", "proportion", "ratio"),
    ),
}

CUMULATIVE_SEMANTIC_SYMBOLS = (
    "perform_tbr_analysis",
    "core.results.TBRResults.cumulative_effect",
    "TBRAnalysis.summarize_incremental",
    "TBRAnalysis.summaries_",
    "core.results.TBRResults.summary",
    "create_tbr_summary",
    "create_incremental_tbr_summaries",
)


def _assert_concepts(
    text: str, concept_alternatives: tuple[tuple[str, ...], ...], context: str
) -> None:
    """Require unordered concepts while accepting equivalent vocabulary."""
    missing = [
        "/".join(alternatives)
        for alternatives in concept_alternatives
        if not _contains_any(text, alternatives)
    ]
    assert not missing, f"{context} missing concept(s): {', '.join(missing)}"


def test_fit_predict_documents_retained_fitted_state() -> None:
    """Accept equivalent positive wording for the estimator's retained state."""
    docstring = (
        inspect.getdoc(_public_documented_object("TBRAnalysis.fit_predict")) or ""
    )
    sentences = re.split(r"(?<=[.!?])\s+", docstring)
    retained_state_sentences = [
        sentence
        for sentence in sentences
        if _contains_any(sentence, ("fit", "fitted"))
        and _contains_any(sentence, ("retain", "store", "keep", "leave", "remain"))
        and _contains_any(sentence, ("state", "estimator", "model"))
        and not _contains_any(sentence, ("not", "no", "without", "does not"))
    ]

    assert retained_state_sentences


def test_perform_tbr_analysis_documents_nonfinite_threshold_value_error() -> None:
    """Associate non-finite threshold rejection with its declared ValueError."""
    function = _public_documented_object("perform_tbr_analysis")
    docstring = inspect.getdoc(function) or ""
    parameters = _numpy_doc_section(docstring, "Parameters")
    raises = _numpy_doc_section(docstring, "Raises")
    threshold = _numpy_field_description(parameters, "threshold")
    value_error = _numpy_raises_description(raises, "ValueError")

    assert _contains_any(threshold, ("finite", "not nan", "not infinite"))
    assert value_error
    assert "threshold" in value_error
    assert _contains_any(value_error, ("non finite", "not finite", "nan", "infinite"))


def test_multiple_subinterval_ci_level_documents_open_unit_interval() -> None:
    """Require endpoint-excluding bounds without requiring one prose form."""
    function = _public_documented_object("analyze_multiple_subintervals")
    parameters = _numpy_doc_section(inspect.getdoc(function) or "", "Parameters")
    description = _numpy_field_description(parameters, "ci_level")
    compact = re.sub(r"\s+", "", description.casefold())

    symbolic = "(0,1)" in compact or "0<ci_level<1" in compact
    verbal = _contains_any(
        description, ("open interval", "exclusive bounds", "endpoints excluded")
    ) or (
        _contains_any(
            description, ("greater than zero", "above zero", "strictly positive")
        )
        and _contains_any(description, ("less than one", "below one"))
    )
    assert symbolic or verbal


def test_assumption_check_schema_excludes_returned_test_statistics() -> None:
    """Allow explicit negation while rejecting an unnegated return promise."""
    function = _public_documented_object("check_tbr_assumptions")
    docstring = inspect.getdoc(function) or ""
    notes = _numpy_doc_section(docstring, "Notes")
    schema_names = {name for name, entry_type in _schema_entries(notes) if entry_type}
    expected = set(EXPECTED_SCHEMA_FIELDS["check_tbr_assumptions"])

    assert schema_names == expected
    assert "validation_summary" in schema_names
    assert any(name.endswith("_valid") for name in schema_names)
    statistic_sentences = [
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+", docstring)
        if "test statistics" in sentence.casefold()
    ]
    assert statistic_sentences
    assert all(
        _contains_any(
            sentence,
            (
                "not return",
                "not include",
                "no test statistics",
                "statistics are not returned",
                "omits",
                "excludes",
            ),
        )
        for sentence in statistic_sentences
    )


@pytest.mark.parametrize("symbol", SUMMARY_SEMANTIC_SYMBOLS)
def test_summary_legacy_scientific_fields_define_meaning_and_mapping(
    symbol: str,
) -> None:
    """Protect compact, local meanings for legacy scientific output names."""
    documented_object = _public_documented_object(symbol)
    notes = _numpy_doc_section(inspect.getdoc(documented_object) or "", "Notes")

    for field, concepts in LEGACY_SCIENTIFIC_FIELDS.items():
        description = _schema_entry_description(notes, field)
        _assert_concepts(description, concepts, f"{symbol}.{field}")


@pytest.mark.parametrize(
    "symbol",
    (
        "compute_interval_estimate_and_ci",
        "analyze_multiple_subintervals",
        "create_subinterval_summary",
    ),
)
def test_subinterval_inputs_define_legacy_scientific_columns(symbol: str) -> None:
    """Define consumed covariance and degrees-of-freedom legacy columns."""
    documented_object = _public_documented_object(symbol)
    parameters = _numpy_doc_section(
        inspect.getdoc(documented_object) or "", "Parameters"
    )
    description = _numpy_field_description(parameters, "tbr_summary")

    _assert_concepts(
        description,
        (
            ("alpha_beta_cov",),
            ("cov_alpha_beta",),
            ("covariance", "cov"),
            ("intercept", "alpha", "beta_0"),
            ("slope", "beta", "beta_1"),
            ("t_dist_df",),
            ("degrees_freedom",),
            ("student t", "t distribution"),
            ("regression", "residual", "pretest"),
        ),
        f"{symbol}.tbr_summary",
    )


@pytest.mark.parametrize(
    ("symbol", "level_name", "has_alpha_parameter"),
    (
        ("create_tbr_summary", "level", True),
        ("create_incremental_tbr_summaries", "level", True),
        ("compute_interval_estimate_and_ci", "ci_level", False),
    ),
)
def test_interval_tail_alpha_defines_level_complement(
    symbol: str, level_name: str, has_alpha_parameter: bool
) -> None:
    """Require the objective tail-probability identity and disambiguation."""
    documented_object = _public_documented_object(symbol)
    docstring = inspect.getdoc(documented_object) or ""
    compact = re.sub(r"\s+", "", docstring.casefold())

    alpha_forms = ("alpha", "α")
    assert any(
        f"{alpha}=1-{level_name}" in compact or f"1-{level_name}={alpha}" in compact
        for alpha in alpha_forms
    )
    if has_alpha_parameter:
        assert _contains_any(docstring, ("regression intercept", "beta_0"))


@pytest.mark.parametrize(
    ("symbol", "legacy_name", "section_name", "field"),
    (
        ("compute_interval_estimate_and_ci", "n_days", "", ""),
        ("create_subinterval_summary", "days", "Notes", "days"),
    ),
)
def test_subinterval_t_s_is_associated_with_local_legacy_count(
    symbol: str, legacy_name: str, section_name: str, field: str
) -> None:
    """Associate T_s with n_days on compute and days on summary."""
    documented_object = _public_documented_object(symbol)
    docstring = inspect.getdoc(documented_object) or ""
    if section_name and field:
        section = _numpy_doc_section(docstring, section_name)
        association_scope = _schema_entry_description(section, field)
    else:
        association_scope = next(
            (
                paragraph
                for paragraph in _paragraphs(docstring)
                if "T_s" in paragraph and legacy_name in paragraph
            ),
            "",
        )

    assert association_scope
    assert "T_s" in association_scope
    assert legacy_name in association_scope
    compact = association_scope.casefold().replace(r"\mathrm", "").replace(r"\_", "_")
    compact = re.sub(r"[\s{}`:]", "", compact)
    assert "t_s=b-a+1" in compact or "t_s=end_day-start_day+1" in compact
    _assert_concepts(
        association_scope,
        (
            ("inclusive", "both endpoints", "including the endpoints"),
            ("test row", "time point", "test observation"),
            ("count", "number", "length"),
        ),
        f"{symbol}.{legacy_name}",
    )


@pytest.mark.parametrize(
    ("symbol", "field", "concepts"),
    (
        (
            "TBRAnalysis.results_",
            "predsd",
            (
                (
                    "coefficient uncertainty",
                    "parameter estimation uncertainty",
                    "coefficient variance",
                ),
                ("residual uncertainty", "residual variance", "residual noise"),
                ("standard deviation", "uncertainty", "variance"),
                ("combines", "includes", "incorporates", "accounts for", "from both"),
            ),
        ),
        (
            "core.results.TBRPredictionResult.to_dict",
            "control_values",
            (
                ("control", "input"),
                ("prediction",),
                ("aligned", "row for row", "same rows"),
            ),
        ),
    ),
)
def test_high_risk_quantity_descriptions_define_components_and_alignment(
    symbol: str, field: str, concepts: tuple[tuple[str, ...], ...]
) -> None:
    """Protect uncertainty composition and prediction-input alignment."""
    documented_object = _public_documented_object(symbol)
    notes = _numpy_doc_section(inspect.getdoc(documented_object) or "", "Notes")
    description = _schema_entry_description(notes, field)

    _assert_concepts(description, concepts, f"{symbol}.{field}")


@pytest.mark.parametrize(
    ("symbol", "section_name", "field", "concepts"), POSITIONAL_CASES
)
def test_representative_legacy_positions_define_test_row_semantics(
    symbol: str, section_name: str, field: str, concepts: tuple[tuple[str, ...], ...]
) -> None:
    documented_object = _public_documented_object(symbol)
    docstring = inspect.getdoc(documented_object) or ""
    if not section_name:
        description = docstring
    else:
        section = _numpy_doc_section(docstring, section_name)
        description = (
            _numpy_field_description(section, field)
            if section_name in {"Parameters", "Attributes"}
            else _schema_entry_description(section, field)
        )
    _assert_concepts(description, concepts, f"{symbol}.{field}")


@pytest.mark.parametrize("symbol", CUMULATIVE_SEMANTIC_SYMBOLS)
def test_cumulative_notation_locally_defines_delta_t_and_t(symbol: str) -> None:
    docstring = inspect.getdoc(_public_documented_object(symbol)) or ""
    scope = next(
        (
            paragraph
            for paragraph in _paragraphs(docstring)
            if _contains_any(paragraph, ("delta t",))
        ),
        "",
    )
    assert scope
    _assert_concepts(
        scope,
        (
            ("running sum", "cumulative sum"),
            ("test row", "time point", "test observation"),
            ("one based", "starts at one", "first t"),
        ),
        symbol,
    )


@pytest.mark.parametrize(("symbol", "contracts"), DIAGNOSTIC_UNIT_CASES)
def test_representative_diagnostic_quantities_define_units_and_scales(
    symbol: str, contracts: dict[str, str]
) -> None:
    documented_object = _public_documented_object(symbol)
    notes = _numpy_doc_section(inspect.getdoc(documented_object) or "", "Notes")
    for field, unit_kind in contracts.items():
        description = _schema_entry_description(notes, field)
        _assert_concepts(description, UNIT_CONCEPTS[unit_kind], f"{symbol}.{field}")


def test_report_timestamp_documents_timezone_naive_no_offset_value() -> None:
    function = _public_documented_object("create_tbr_diagnostic_report")
    notes = _numpy_doc_section(inspect.getdoc(function) or "", "Notes")
    timestamp = _schema_entry_description(notes, "report_timestamp")
    _assert_concepts(
        timestamp,
        (
            (
                "timezone naive",
                "without timezone",
                "no timezone",
                "timezone information absent",
            ),
            (
                "no offset",
                "without offset",
                "offset absent",
                "does not include an offset",
            ),
        ),
        "create_tbr_diagnostic_report.report_timestamp",
    )


def test_validate_tbr_model_documents_mape_zero_value_limitation() -> None:
    """Keep the zero and near-zero response limitation local to validation."""
    documented_object = _public_documented_object("validate_tbr_model")
    paragraphs = _paragraphs(inspect.getdoc(documented_object) or "")
    mape_scope = next(
        (paragraph for paragraph in paragraphs if "MAPE" in paragraph), ""
    )

    assert mape_scope
    _assert_concepts(
        mape_scope,
        (
            ("zero",),
            ("near zero", "close to zero", "small denominator"),
            ("undefined", "unstable", "not defined", "unreliable"),
        ),
        "validate_tbr_model MAPE limitation",
    )


def test_efficiency_score_documents_exact_coverage_component_and_range() -> None:
    """Protect the exact coverage component and its objective scale."""
    documented_object = _public_documented_object("assess_tbr_performance")
    compact = re.sub(r"\s+", "", (inspect.getdoc(documented_object) or "").casefold())

    assert "max(0,1-2*abs(interval_coverage-0.95))" in compact
    assert "[0,1]" in compact or "0<=efficiency_score<=1" in compact


@pytest.mark.parametrize(("symbol", "fields"), TYPED_DICT_FIELDS.items())
def test_standalone_typed_dict_fields_have_local_descriptions(
    symbol: str, fields: tuple[str, ...]
) -> None:
    """Require every standalone TypedDict page to explain each public field."""
    documented_object = _public_documented_object(symbol)
    attributes = _numpy_doc_section(
        inspect.getdoc(documented_object) or "", "Attributes"
    )
    missing = [
        field for field in fields if not _numpy_field_description(attributes, field)
    ]

    assert not missing, f"{symbol} missing field descriptions: {', '.join(missing)}"


@pytest.mark.parametrize(("symbol", "required_paths"), AGGREGATE_SCHEMA_PATHS.items())
def test_aggregate_diagnostics_document_local_stable_nested_paths(
    symbol: str, required_paths: tuple[str, ...]
) -> None:
    """Keep representative stable nested schemas local to aggregate pages."""
    documented_object = _public_documented_object(symbol)
    notes = _numpy_doc_section(inspect.getdoc(documented_object) or "", "Notes")
    missing = set(required_paths) - _schema_paths(notes)

    assert not missing, f"{symbol} missing nested paths: {sorted(missing)}"
