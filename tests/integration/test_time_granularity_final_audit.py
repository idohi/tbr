"""Final contracts for unit-neutral time-granularity terminology."""

import inspect
import re
from pathlib import Path

import pytest

import tbr.core.results as results_module
from tbr.core.model import TBRAnalysis
from tbr.core.results import TBRSubintervalResult
from tbr.utils.preprocessing import split_time_series_by_periods

pytestmark = pytest.mark.integration

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEGACY_IDENTIFIER_PATTERN = re.compile(
    r"\b(start_day|end_day|n_days|test_day|n_test_days|avg_daily_effect)\b"
    r"|(?<=[\"'`])days(?=[\"'`])"
)
GENERIC_DAY_PATTERN = re.compile(r"(?i)(?<![A-Za-z])days?(?![A-Za-z])|daily")
GENERIC_RUNTIME_PATTERN = re.compile(r"(?i)\bdays?\b|\bdaily\b")

EXPLICIT_DAILY_CADENCE_FILES = (
    "README.md",
    "docs/api/patterns.md",
    "docs/api/quickstart.md",
    "examples/plot_basic_analysis.py",
    "examples/plot_incremental_analysis.py",
    "examples/plot_method_chaining.py",
    "tests/mathematical/test_subinterval_variance_validation.py",
)

APPROVED_MIGRATION_LEGACY_LINE_MARKERS = (
    "start_day, end_day, ci_level",
    "(start_day, end_day, ci_level=None)",
    "`start_day`, `end_day`, `n_days`",
    "`start_day`, `end_day`, `days`",
    "`test_day`, `estimate`",
    "`TBRResults.n_test_days`",
    "`start_day` becomes",
    "`end_day` becomes",
    "`n_days` and the summary column `days`",
    "`avg_daily_effect` becomes",
    "start_day: int,",
    "end_day: int,",
    "start_day=1, end_day=7",
    "result.start_day, result.end_day, result.n_days",
    "`test_day` to `test_position`",
    'incremental["test_day"]',
    "accessing `test_day`",
    "frames with `test_day`",
    "`n_test_days` raises",
    "retain the `test_day` column",
    "`test_day` column, so recreate",
)

APPROVED_MIGRATION_GENERIC_LINE_MARKERS = (
    "day-specific names with terminology that applies to daily",
    "`days`, `estimate`",
    "| Incremental posterior keys",
    "summary column `days`",
    '`"position"` instead of `"day"`',
    'accessing `test_day`, `"day"`',
)

APPROVED_DAILY_LEGACY_LINE_MARKERS = {
    "examples/plot_basic_analysis.py": (
        "n_test_days = 45",
        "n_days = n_pretest_days + n_test_days",
        'dates = pd.date_range("2024-01-01", periods=n_days, freq="D")',
        "control = rng.normal(1000, 50, n_days)",
        "test = control + rng.normal(0, 25, n_days)",
        "for start_day in range(1, n_test_days + 1, window_length):",
        "end_day = start_day + window_length - 1",
        "start_position=start_day,",
        "end_position=end_day,",
        'print(f"Test days {start_day}-{end_day}:")',
    ),
    "examples/plot_incremental_analysis.py": (
        "n_test_days = 60",
        (
            'dates = pd.date_range("2024-01-01", '
            'periods=n_pretest_days + n_test_days, freq="D")'
        ),
        "for days_into_test in range(n_test_days):",
        (
            "ramp_factor = 1.0 + "
            "(0.02 * days_into_test / (n_test_days - 1))  # Ramp to 2%"
        ),
        'f"Scenario: Treatment effect ramps from 0% to 2% over {n_test_days}-day "',
        "for start_day in range(1, n_test_days + 1, window_length):",
        "end_day = start_day + window_length - 1",
        "result = model.analyze_subinterval(start_day, end_day)",
        '"test_days": f"{start_day}-{end_day}",',
        '"true_effect": true_daily_effect[start_day - 1 : end_day].sum(),',
        "half = n_test_days // 2",
        '("Late", half + 1, n_test_days),',
        "for day in range(window_length, n_test_days + 1, window_length):",
        'f"{late.estimate:.2f} in test days {half + 1}-{n_test_days}"',
    ),
    "examples/plot_method_chaining.py": (
        "n_test_days = 45",
        "n_days = n_pretest_days + n_test_days",
        'dates = pd.date_range("2024-01-01", periods=n_days, freq="D")',
        "control = rng.normal(1000, 50, n_days)",
        "test = control + rng.normal(0, 25, n_days)",
    ),
}

APPROVED_GENERIC_DAILY_LINE_MARKERS = {
    "README.md": ("Simulated daily data", "last test day"),
    "docs/api/patterns.md": (
        "assign(day=",
        '"day",',
        "daily aggregation",
        "daily =",
    ),
    "docs/api/quickstart.md": ("daily results",),
    "docs/mathematical_methodology.md": (
        "daily, weekly, hourly",
        "explicitly daily data",
    ),
    "examples/plot_basic_analysis.py": (
        "synthetic daily data",
        "Create daily data",
        "Total days:",
        "First pretest day",
        "First test day",
        "last test day",
        "pd.Timedelta(days=1)",
        "Pretest days:",
        "Test days:",
        "15-day windows",
        'f"Test days {start_day}-{end_day}:"',
        "first 5 days",
    ),
    "examples/plot_incremental_analysis.py": (
        "in test days",
        "{n_test_days}-day",
        "First test day",
        "last test day",
        "Pretest days:",
        "Test days:",
        "for day in",
        "iloc[day - 1]",
        'f"Test day {day:2d}: Effect',
        'xlabel="Test day"',
        "15-day windows",
        "window_length}-DAY WINDOWS",
        "15 test days",
        "window_length}-day",
        "15-day period",
        'xlabel="Test days"',
        "test days:",
        "Calculate average daily effect",
        "/ day",
        'f"Test day {day:2d}: Total',
        "Avg/day",
        "# days T",
        'f"Test day {day:2d}: Width',
        "15-day window analysis",
        "on test day 1",
        "on test day {len(incremental)}",
        "test days {start}-{end}",
        "test days 1-{half}",
        "test days {half + 1}-{n_test_days}",
    ),
    "examples/plot_method_chaining.py": (
        "Test days 1-15 effect",
        "one row per test day",
    ),
    "src/tbr/utils/preprocessing.py": ("Pretest period:",),
    "src/tbr/functional/tbr_functions.py": (
        "days since start",
        "'day': range(1, 121)",
        "time_col='day'",
        "Days 60-89",
    ),
}

GENERIC_TEST_FILES = (
    "tests/integration/test_analysis_comprehensive_validation.py",
    "tests/integration/test_analysis_framework_validation.py",
    "tests/mathematical/test_analysis_mathematical_validation.py",
    "tests/mathematical/test_core_effects_validation.py",
    "tests/mathematical/test_mathematical_correctness.py",
    "tests/unit/test_analysis_incremental.py",
    "tests/unit/test_core_effects.py",
    "tests/unit/test_core_functions.py",
    "tests/unit/test_core_posterior.py",
    "tests/unit/test_structure_validation.py",
)

PUBLIC_DAILY_DOCSTRINGS = (
    inspect.getdoc(results_module) or "",
    inspect.getdoc(TBRSubintervalResult) or "",
    inspect.getdoc(TBRAnalysis.analyze_subinterval) or "",
    inspect.getdoc(split_time_series_by_periods) or "",
)


def _iter_public_files():
    """Yield public source, documentation, example, and release-note files."""
    for root, pattern in (
        (REPOSITORY_ROOT / "src", "*.py"),
        (REPOSITORY_ROOT / "docs", "*.md"),
        (REPOSITORY_ROOT / "docs", "*.rst"),
        (REPOSITORY_ROOT / "examples", "*.py"),
    ):
        for path in root.rglob(pattern):
            # Sphinx-Gallery mirrors are ignored build products; their tracked
            # example sources are scanned below and the built gallery has a
            # separate warning-free documentation gate. Developer plans under
            # docs/to_do are not published documentation.
            if (
                path.parent != REPOSITORY_ROOT / "docs" / "auto_examples"
                and REPOSITORY_ROOT / "docs" / "to_do" not in path.parents
            ):
                yield path
    yield REPOSITORY_ROOT / "README.md"
    yield REPOSITORY_ROOT / "CHANGELOG.md"


def _legacy_identifier_is_approved(path: Path, line: str, text: str) -> bool:
    """Return whether a legacy identifier occurs in an approved context."""
    relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()

    if relative_path == "docs/migration_0_2.md":
        return any(marker in line for marker in APPROVED_MIGRATION_LEGACY_LINE_MARKERS)
    if relative_path in APPROVED_DAILY_LEGACY_LINE_MARKERS:
        return line.strip() in APPROVED_DAILY_LEGACY_LINE_MARKERS[relative_path] and (
            'freq="D"' in text or "freq='D'" in text
        )
    if relative_path == "CHANGELOG.md":
        return any(
            marker in line
            for marker in (
                "`test_day`, and incremental posterior mappings",
                "(estsd_values, n_days, sigma)",
                "Removed `TBRResults.n_test_days`",
            )
        )
    return False


def _generic_runtime_phrase_is_approved(path: Path, line: str, text: str) -> bool:
    """Return whether unit-specific wording has an explicit approved context."""
    relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()

    if relative_path == "docs/migration_0_2.md":
        return any(marker in line for marker in APPROVED_MIGRATION_GENERIC_LINE_MARKERS)
    if relative_path == "CHANGELOG.md":
        return any(
            marker in line
            for marker in (
                "Removed day-specific names",
                'of `"day"`.',
                "test day that its 59-day test",
                "last test day instead",
                "monitoring days",
                "assumes daily input",
                '"Days A-B"',
                "Day-by-day progression",
            )
        )
    if relative_path in APPROVED_GENERIC_DAILY_LINE_MARKERS:
        marker_is_approved = any(
            marker in line
            for marker in APPROVED_GENERIC_DAILY_LINE_MARKERS[relative_path]
        )
        if relative_path in {
            "README.md",
            "docs/api/patterns.md",
            "docs/api/quickstart.md",
            "examples/plot_basic_analysis.py",
            "examples/plot_incremental_analysis.py",
            "examples/plot_method_chaining.py",
            "src/tbr/utils/preprocessing.py",
        }:
            return marker_is_approved and bool(
                re.search(
                    r"pd\.date_range\([^)]*\bfreq=[\"']D[\"']",
                    text,
                    re.DOTALL,
                )
            )
        return marker_is_approved
    return False


def test_legacy_identifiers_are_confined_to_approved_public_contexts() -> None:
    """Reject legacy public names outside explicit migration contexts."""
    offenders = []
    for path in _iter_public_files():
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not LEGACY_IDENTIFIER_PATTERN.search(line):
                continue
            if not _legacy_identifier_is_approved(path, line, text):
                relative_path = path.relative_to(REPOSITORY_ROOT)
                offenders.append(f"{relative_path}:{line_number}: {line.strip()}")

    assert not offenders, "\n".join(offenders)


def test_generic_runtime_phrases_have_explicit_unit_context() -> None:
    """Reject generic day wording outside an explicit daily or migration context."""
    offenders = []
    for path in _iter_public_files():
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            matches = GENERIC_RUNTIME_PATTERN.findall(line)
            if matches and not _generic_runtime_phrase_is_approved(path, line, text):
                relative_path = path.relative_to(REPOSITORY_ROOT)
                offenders.append(f"{relative_path}:{line_number}: {', '.join(matches)}")

    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("relative_path", EXPLICIT_DAILY_CADENCE_FILES)
def test_daily_contexts_declare_daily_frequency(relative_path: str) -> None:
    """Require examples and fixtures using daily wording to declare cadence."""
    text = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    assert re.search(r"pd\.date_range\([^)]*\bfreq=[\"']D[\"']", text, re.DOTALL)


@pytest.mark.parametrize("docstring", PUBLIC_DAILY_DOCSTRINGS)
def test_daily_public_docstrings_declare_daily_frequency(docstring: str) -> None:
    """Require public docstrings using daily wording to declare daily cadence."""
    assert re.search(r"pd\.date_range\([^\n]+\bfreq=[\"']D[\"']", docstring)


@pytest.mark.parametrize("relative_path", GENERIC_TEST_FILES)
def test_unitless_contract_tests_use_unit_neutral_language(relative_path: str) -> None:
    """Reject day terminology in tests built from unitless observation rows."""
    text = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    matches = sorted(set(GENERIC_DAY_PATTERN.findall(text)))
    assert not matches, f"{relative_path}: {matches}"


def test_subinterval_contract_tests_use_unit_neutral_language() -> None:
    """Reject known stale day wording while retaining explicit daily fixtures."""
    text = (REPOSITORY_ROOT / "tests/unit/test_analysis_subinterval.py").read_text(
        encoding="utf-8"
    )
    stale_terms = (
        "single_day_analysis",
        "single day",
        "max_day",
        "invalid_days",
        "invalid day",
        "1-day, 3-day, 4-day",
        "NumPy integer days",
        "three test days",
        "numpy_integer_days",
    )

    assert not [term for term in stale_terms if term in text]


def test_unreleased_changelog_uses_positions_for_generic_bounds() -> None:
    """Describe generic NumPy bounds as positions rather than days."""
    changelog = (REPOSITORY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## [Unreleased]", maxsplit=1)[1].split(
        "## [0.", maxsplit=1
    )[0]

    assert "NumPy integer days" not in unreleased


def test_migration_guide_has_complete_before_after_tables() -> None:
    """Require compact before-and-after tables for every migrated surface."""
    guide = (REPOSITORY_ROOT / "docs/migration_0_2.md").read_text(encoding="utf-8")

    assert "### Callable signature changes" in guide
    assert "### Result and schema changes" in guide
    assert guide.count("| Surface | 0.1.x | 0.2.0 | Compatibility |") >= 2
    for route in (
        "tbr.compute_interval_estimate_and_ci",
        "tbr.analysis.compute_interval_estimate_and_ci",
        "tbr.analysis.subinterval.compute_interval_estimate_and_ci",
        "tbr.core.compute_interval_estimate_and_ci",
        "tbr.core.prediction.compute_interval_estimate_and_ci",
        "tbr.core.effects.compute_interval_estimate_and_ci",
        "tbr.functional.tbr_functions.compute_interval_estimate_and_ci",
        "TBRAnalysis.analyze_subinterval",
        "tbr.analysis.validate_subinterval_parameters",
        "create_subinterval_summary",
    ):
        assert f"`{route}`" in guide

    for signature in (
        (
            "(tbr_df, tbr_summary, intervals, ci_level=0.80, "
            "significance_threshold=0.0)"
        ),
        "(tbr_df, tbr_summary, intervals, ci_level=0.80, threshold=0.0)",
    ):
        assert signature in guide

    for ordered_schema in (
        (
            "`estimate`, `lower`, `upper`, `se`, `ci_level`, "
            "`start_day`, `end_day`, `n_days`"
        ),
        (
            "`estimate`, `lower`, `upper`, `se`, `precision`, `ci_level`, "
            "`start_position`, `end_position`, `n_observations`"
        ),
        (
            "`interval`, `start_day`, `end_day`, `days`, `estimate`, "
            "`precision`, `lower`, `upper`, `significant`, "
            "`avg_daily_effect`, `ci_level`"
        ),
        (
            "`interval`, `start_position`, `end_position`, `n_observations`, "
            "`estimate`, `se`, `precision`, `lower`, `upper`, "
            "`interval_excludes_threshold`, "
            "`average_effect_per_observation`, `ci_level`"
        ),
        (
            "`test_day`, `estimate`, `precision`, `lower`, `upper`, `se`, "
            "`level`, `thres`, `prob`, `alpha`, `beta`, `alpha_beta_cov`, "
            "`var_alpha`, `var_beta`, `sigma`, `t_dist_df`"
        ),
        (
            "`test_position`, `estimate`, `precision`, `lower`, `upper`, "
            "`se`, `level`, `thres`, `prob`, `alpha`, `beta`, "
            "`alpha_beta_cov`, `var_alpha`, `var_beta`, `sigma`, `t_dist_df`"
        ),
        (
            '`"day"`, `"estimates"`, `"standard_errors"`, `"probabilities"`, '
            '`"probability_change"`'
        ),
        (
            '`"position"`, `"estimates"`, `"standard_errors"`, '
            '`"probabilities"`, `"probability_change"`'
        ),
    ):
        assert ordered_schema in guide

    for migrated_name in (
        "start_position",
        "end_position",
        "n_observations",
        "average_effect_per_observation",
        "test_position",
        '"position"',
        "TBRResults.n_test",
    ):
        assert migrated_name in guide
