"""Consumer-style typing contract for the canonical incremental API."""

from tbr.core.results import TBRResults


def canonical_test_count_consumer(results: TBRResults) -> int:
    """Type-check the canonical test-observation count."""
    return results.n_test


def removed_test_count_name_is_rejected(results: TBRResults) -> None:
    """Require mypy to reject the removed day-specific count property."""
    _ = results.n_test_days  # type: ignore[attr-defined]
