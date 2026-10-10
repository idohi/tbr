"""
Mathematical validation tests for TBR subinterval posterior variance.

This module validates the credible intervals returned by
``TBRAnalysis.analyze_subinterval`` against independent references. The model
component of the subinterval variance is the variance of the summed
counterfactual predictions, which share the estimated coefficients.

Test Categories
---------------
1. Consistency with the full-period summary and the predsd and cumsd columns
2. Cross-validation against the statsmodels coefficient covariance matrix
3. Agreement of the batch subinterval helpers with analyze_subinterval
4. Frequentist coverage of subinterval credible intervals under simulation

Mathematical Validation
-----------------------
For a subinterval of T_s test days, the posterior variance is
V[Δ(a,b)] = T_s·σ² + aᵀ C a, where a = (T_s, Σx_t) and C is the covariance
matrix of the intercept and slope estimates. A single day reduces to the
prediction variance (predsd²), and a subinterval starting on the first test day
reduces to the cumulative variance (cumsd²).
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from tbr import TBRAnalysis
from tbr.analysis.subinterval import (
    analyze_multiple_subintervals,
    compute_interval_estimate_and_ci,
    create_subinterval_summary,
)

LEVEL = 0.90


@pytest.fixture
def fitted_model():
    """Fit a TBR model with a 30-day pretest and a 14-day test period."""
    rng = np.random.default_rng(0)
    control = rng.normal(1000, 50, size=44)
    data = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=44),
            "control": control,
            "test": 1.05 * control + rng.normal(0, 10, size=44),
        }
    )
    model = TBRAnalysis(level=LEVEL)
    model.fit(
        data,
        "date",
        "control",
        "test",
        pretest_start=pd.Timestamp("2023-01-01"),
        test_start=pd.Timestamp("2023-01-31"),
        test_end=pd.Timestamp("2023-02-14"),
    )
    return model


def _t_multiplier(model):
    """Return the two-sided Student-t multiplier for the model level."""
    dof = model.summarize().degrees_freedom
    return stats.t.ppf(1 - (1 - LEVEL) / 2, dof)


class TestSubintervalConsistency:
    """Subinterval intervals must agree with the package's own uncertainty columns."""

    def test_full_test_period_matches_summary(self, fitted_model):
        """Test that the full test period reproduces the summary interval."""
        summary = fitted_model.summarize()
        test_rows = fitted_model.results_[fitted_model.results_["period"] == 1]

        result = fitted_model.analyze_subinterval(1, len(test_rows))

        assert result.estimate == pytest.approx(summary.estimate, rel=1e-12)
        assert result.lower == pytest.approx(summary.lower, rel=1e-10)
        assert result.upper == pytest.approx(summary.upper, rel=1e-10)
        # TBRSubintervalResult.se stores the credible-interval half-width
        assert result.se == pytest.approx(summary.precision, rel=1e-10)

    def test_single_day_matches_prediction_standard_deviation(self, fitted_model):
        """Test that each single day reproduces predsd."""
        t_mult = _t_multiplier(fitted_model)
        test_rows = fitted_model.results_[fitted_model.results_["period"] == 1]
        predsd = test_rows["predsd"].to_numpy()

        for day in range(1, len(test_rows) + 1):
            result = fitted_model.analyze_subinterval(day, day)
            assert result.se / t_mult == pytest.approx(predsd[day - 1], rel=1e-10)

    def test_interval_from_first_day_matches_cumulative_standard_deviation(
        self, fitted_model
    ):
        """Test that intervals starting on day 1 reproduce cumsd."""
        t_mult = _t_multiplier(fitted_model)
        test_rows = fitted_model.results_[fitted_model.results_["period"] == 1]
        cumsd = test_rows["cumsd"].to_numpy()

        for end_day in range(1, len(test_rows) + 1):
            result = fitted_model.analyze_subinterval(1, end_day)
            assert result.se / t_mult == pytest.approx(cumsd[end_day - 1], rel=1e-10)


class TestSubintervalStatsmodelsCrossValidation:
    """Cross-validate subinterval variance against statsmodels directly."""

    @pytest.mark.parametrize(
        ("start_day", "end_day"), [(2, 2), (4, 9), (6, 14), (3, 12), (10, 11)]
    )
    def test_variance_matches_ols_coefficient_covariance(
        self, fitted_model, start_day, end_day
    ):
        """Test V[Δ(a,b)] against an OLS fit on the same pretest rows."""
        results_df = fitted_model.results_
        pretest = results_df[results_df["period"] == 0]
        test_rows = results_df[results_df["period"] == 1]

        ols = sm.OLS(
            pretest["y"].to_numpy(dtype=float),
            sm.add_constant(pretest["x"].to_numpy(dtype=float)),
        ).fit()

        x_interval = test_rows["x"].to_numpy(dtype=float)[start_day - 1 : end_day]
        n_days = len(x_interval)
        a = np.array([n_days, np.sum(x_interval)])
        expected_variance = n_days * ols.scale + a @ ols.cov_params() @ a

        result = fitted_model.analyze_subinterval(start_day, end_day)
        actual_variance = (result.se / _t_multiplier(fitted_model)) ** 2

        assert actual_variance == pytest.approx(expected_variance, rel=1e-10)

    @pytest.mark.parametrize(("start_day", "end_day"), [(2, 2), (4, 9), (1, 14)])
    def test_returned_standard_error_matches_ols_coefficient_covariance(
        self, fitted_model, start_day, end_day
    ):
        """Test the returned se against OLS and its relation to the half-width."""
        results_df = fitted_model.results_
        pretest = results_df[results_df["period"] == 0]
        test_rows = results_df[results_df["period"] == 1]

        ols = sm.OLS(
            pretest["y"].to_numpy(dtype=float),
            sm.add_constant(pretest["x"].to_numpy(dtype=float)),
        ).fit()

        x_interval = test_rows["x"].to_numpy(dtype=float)[start_day - 1 : end_day]
        n_days = len(x_interval)
        a = np.array([n_days, np.sum(x_interval)])
        expected_se = np.sqrt(n_days * ols.scale + a @ ols.cov_params() @ a)

        result = compute_interval_estimate_and_ci(
            results_df, fitted_model.summaries_, start_day, end_day, ci_level=LEVEL
        )

        assert result["se"] == pytest.approx(expected_se, rel=1e-10)
        assert result["precision"] == pytest.approx(
            _t_multiplier(fitted_model) * result["se"], rel=1e-12
        )
        assert result["precision"] > result["se"]


class TestSubintervalBatchHelpers:
    """Batch subinterval helpers must agree with TBRAnalysis.analyze_subinterval."""

    INTERVALS = [(1, 1), (1, 7), (4, 9), (8, 14), (1, 14)]

    def test_analyze_multiple_subintervals_matches_model(self, fitted_model):
        """Test analyze_multiple_subintervals against analyze_subinterval."""
        results = analyze_multiple_subintervals(
            fitted_model.results_,
            fitted_model.summaries_,
            self.INTERVALS,
            ci_level=LEVEL,
        )

        for (start_day, end_day), batch in zip(self.INTERVALS, results):
            single = fitted_model.analyze_subinterval(start_day, end_day)
            assert batch["estimate"] == pytest.approx(single.estimate, rel=1e-12)
            assert batch["precision"] == pytest.approx(single.se, rel=1e-12)
            assert batch["lower"] == pytest.approx(single.lower, rel=1e-12)
            assert batch["upper"] == pytest.approx(single.upper, rel=1e-12)

    def test_create_subinterval_summary_matches_model(self, fitted_model):
        """Test create_subinterval_summary against analyze_subinterval."""
        summary = create_subinterval_summary(
            fitted_model.results_,
            fitted_model.summaries_,
            self.INTERVALS,
            ci_level=LEVEL,
        )

        assert len(summary) == len(self.INTERVALS)
        for row, (start_day, end_day) in zip(
            summary.itertuples(index=False), self.INTERVALS
        ):
            single = fitted_model.analyze_subinterval(start_day, end_day)
            assert row.estimate == pytest.approx(single.estimate, rel=1e-12)
            assert row.precision == pytest.approx(single.se, rel=1e-12)
            assert row.lower == pytest.approx(single.lower, rel=1e-12)
            assert row.upper == pytest.approx(single.upper, rel=1e-12)


class TestSubintervalCoverage:
    """Frequentist coverage of subinterval credible intervals."""

    @pytest.mark.slow
    def test_credible_interval_coverage_under_null(self):
        """Test that 90% subinterval intervals cover the true zero effect ~90%."""
        rng = np.random.default_rng(20261006)
        n_pretest, n_test, n_sims = 12, 20, 1000
        start_day, end_day = 5, 20
        dates = pd.date_range("2024-01-01", periods=n_pretest + n_test)

        covered = np.zeros(n_sims, dtype=bool)
        covered_residual_only = np.zeros(n_sims, dtype=bool)
        for i in range(n_sims):
            # Control level shifts in the test period, so coefficient
            # uncertainty is a large part of the subinterval variance
            control = np.concatenate(
                [rng.normal(100, 10, n_pretest), rng.normal(130, 10, n_test)]
            )
            test = 5.0 + 1.0 * control + rng.normal(0, 5, n_pretest + n_test)
            data = pd.DataFrame({"date": dates, "control": control, "test": test})

            model = TBRAnalysis(level=LEVEL, test_end_inclusive=True)
            model.fit(
                data,
                "date",
                "control",
                "test",
                pretest_start=dates[0],
                test_start=dates[n_pretest],
                test_end=dates[-1],
            )
            result = model.analyze_subinterval(start_day, end_day)
            covered[i] = result.lower <= 0.0 <= result.upper

            summary = model.summarize()
            t_mult = stats.t.ppf(1 - (1 - LEVEL) / 2, summary.degrees_freedom)
            residual_only_half_width = (
                t_mult * np.sqrt(end_day - start_day + 1) * summary.sigma
            )
            covered_residual_only[i] = abs(result.estimate) <= residual_only_half_width

        # Binomial standard error at 0.90 with 1000 draws is about 0.0095
        assert abs(covered.mean() - LEVEL) < 0.03
        # Omitting the model component must be detectably overconfident
        assert covered_residual_only.mean() < LEVEL - 0.10
