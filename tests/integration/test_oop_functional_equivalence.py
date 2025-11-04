"""
Cross-validation tests between OOP API (TBRAnalysis) and functional API (perform_tbr_analysis).

This module validates that the TBRAnalysis class produces mathematically identical
results to the perform_tbr_analysis() functional API across all configurations
and scenarios.

Test Categories
---------------
1. Basic Equivalence: Simple scenarios validating identical results
2. Configuration Variations: Different parameter combinations
3. Edge Cases: Minimal data, extreme values, boundary conditions
4. Time Column Types: datetime64[ns], int64, float64
5. Result Components: DataFrame structure, summary fields, parameters
6. Mathematical Accuracy: Precision validation at machine epsilon
"""

import numpy as np
import pandas as pd
import pytest

from tbr import TBRAnalysis
from tbr.functional import perform_tbr_analysis


class TestBasicEquivalence:
    """Test basic equivalence between OOP and functional APIs."""

    def test_basic_analysis_equivalence(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test basic TBR analysis produces identical results."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
            level=sample_analysis_parameters["level"],
            threshold=sample_analysis_parameters["threshold"],
        )

        # OOP API
        model = TBRAnalysis(
            level=sample_analysis_parameters["level"],
            threshold=sample_analysis_parameters["threshold"],
        )
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Extract functional results
        func_df = functional_results.tbr_dataframe()
        func_summary = functional_results.summary()

        # Compare DataFrames
        pd.testing.assert_frame_equal(
            model.results_, func_df, check_dtype=True, rtol=1e-14
        )

        # Compare summaries
        pd.testing.assert_frame_equal(
            model.summaries_, func_summary, check_dtype=True, rtol=1e-14
        )

    def test_final_estimate_equivalence(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test final treatment effect estimate is identical."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
            level=0.80,
            threshold=0.0,
        )
        func_estimate = functional_results.summary().iloc[-1]["estimate"]

        # OOP API
        model = TBRAnalysis()
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )
        oop_estimate = model.final_effect

        # Validate equivalence at machine precision
        assert np.isclose(oop_estimate, func_estimate, rtol=1e-15, atol=0)

    def test_credible_interval_equivalence(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test credible intervals are identical."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
            level=0.90,
            threshold=0.0,
        )
        func_summary = functional_results.summary().iloc[-1]

        # OOP API
        model = TBRAnalysis(level=0.90)
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )
        oop_summary = model.summarize()

        # Compare CI bounds
        assert np.isclose(oop_summary.lower, func_summary["lower"], rtol=1e-15)
        assert np.isclose(oop_summary.upper, func_summary["upper"], rtol=1e-15)


class TestConfigurationVariations:
    """Test equivalence across different configuration parameters."""

    @pytest.mark.parametrize("level", [0.80, 0.90, 0.95, 0.99])
    def test_different_confidence_levels(
        self, sample_time_series_data, sample_analysis_parameters, level
    ):
        """Test equivalence with different confidence levels."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
            level=level,
            threshold=0.0,
        )

        # OOP API
        model = TBRAnalysis(level=level)
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Compare summaries
        pd.testing.assert_frame_equal(
            model.summaries_,
            functional_results.summary(),
            check_dtype=True,
            rtol=1e-14,
        )

    @pytest.mark.parametrize("threshold", [0.0, 10.0, -5.0, 100.0])
    def test_different_thresholds(
        self, sample_time_series_data, sample_analysis_parameters, threshold
    ):
        """Test equivalence with different probability thresholds."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
            level=0.80,
            threshold=threshold,
        )
        func_prob = functional_results.summary().iloc[-1]["prob"]

        # OOP API
        model = TBRAnalysis(threshold=threshold)
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )
        oop_prob = model.summarize().prob

        # Posterior probabilities should be identical
        assert np.isclose(oop_prob, func_prob, rtol=1e-15)

    def test_test_end_inclusive_equivalence(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test equivalence with test_end_inclusive parameter."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_start"],  # Same day
            level=0.80,
            threshold=0.0,
            test_end_inclusive=True,
        )

        # OOP API
        model = TBRAnalysis(test_end_inclusive=True)
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_start"],  # Same day
        )

        # Results should be identical
        pd.testing.assert_frame_equal(
            model.results_,
            functional_results.tbr_dataframe(),
            check_dtype=True,
            rtol=1e-14,
        )


class TestEdgeCases:
    """Test equivalence in edge case scenarios."""

    def test_minimal_test_period(self, minimal_valid_data):
        """Test equivalence with minimal test period (1 day)."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=minimal_valid_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-09"),
            test_end=pd.Timestamp("2023-01-10"),
            level=0.80,
            threshold=0.0,
        )

        # OOP API
        model = TBRAnalysis()
        model.fit(
            data=minimal_valid_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-01-09"),
            test_end=pd.Timestamp("2023-01-10"),
        )

        # Should produce identical results
        pd.testing.assert_frame_equal(
            model.results_,
            functional_results.tbr_dataframe(),
            check_dtype=True,
            rtol=1e-14,
        )

    def test_large_scale_values(self):
        """Test equivalence with large-scale data values."""
        # Create data with large values
        np.random.seed(789)
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=50),
                "control": np.random.normal(1000000, 50000, 50),
                "test": np.random.normal(1050000, 55000, 50),
            }
        )

        params = {
            "time_col": "date",
            "control_col": "control",
            "test_col": "test",
            "pretest_start": pd.Timestamp("2023-01-01"),
            "test_start": pd.Timestamp("2023-02-01"),
            "test_end": pd.Timestamp("2023-02-19"),
        }

        # Functional API
        functional_results = perform_tbr_analysis(
            data=data, level=0.80, threshold=0.0, **params
        )

        # OOP API
        model = TBRAnalysis()
        model.fit(data=data, **params)

        # Compare final estimates (should be close despite large scales)
        func_estimate = functional_results.summary().iloc[-1]["estimate"]
        oop_estimate = model.final_effect
        assert np.isclose(func_estimate, oop_estimate, rtol=1e-14)

    def test_small_scale_values(self):
        """Test equivalence with small-scale data values."""
        # Create data with small values
        np.random.seed(456)
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=50),
                "control": np.random.normal(0.1, 0.01, 50),
                "test": np.random.normal(0.105, 0.011, 50),
            }
        )

        params = {
            "time_col": "date",
            "control_col": "control",
            "test_col": "test",
            "pretest_start": pd.Timestamp("2023-01-01"),
            "test_start": pd.Timestamp("2023-02-01"),
            "test_end": pd.Timestamp("2023-02-19"),
        }

        # Functional API
        functional_results = perform_tbr_analysis(
            data=data, level=0.80, threshold=0.0, **params
        )

        # OOP API
        model = TBRAnalysis()
        model.fit(data=data, **params)

        # Compare estimates
        func_estimate = functional_results.summary().iloc[-1]["estimate"]
        oop_estimate = model.final_effect
        assert np.isclose(func_estimate, oop_estimate, rtol=1e-12)


class TestTimeColumnTypes:
    """Test equivalence across different time column types."""

    def test_datetime_time_column(self, sample_time_series_data):
        """Test equivalence with datetime64[ns] time column."""
        # Already datetime - standard case
        assert sample_time_series_data["date"].dtype == "datetime64[ns]"

        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-03-02"),
            test_end=pd.Timestamp("2023-04-10"),
            level=0.80,
            threshold=0.0,
        )

        # OOP API
        model = TBRAnalysis()
        model.fit(
            data=sample_time_series_data,
            time_col="date",
            control_col="control",
            test_col="test",
            pretest_start=pd.Timestamp("2023-01-01"),
            test_start=pd.Timestamp("2023-03-02"),
            test_end=pd.Timestamp("2023-04-10"),
        )

        # Results identical
        pd.testing.assert_frame_equal(
            model.results_, functional_results.tbr_dataframe(), rtol=1e-14
        )

    def test_integer_time_column(self):
        """Test equivalence with integer time column."""
        # Create data with integer time
        np.random.seed(111)
        data = pd.DataFrame(
            {
                "day": np.arange(1, 51),
                "control": np.random.normal(100, 10, 50),
                "test": np.random.normal(105, 11, 50),
            }
        )

        # Functional API
        functional_results = perform_tbr_analysis(
            data=data,
            time_col="day",
            control_col="control",
            test_col="test",
            pretest_start=1,
            test_start=31,
            test_end=50,
            level=0.80,
            threshold=0.0,
        )

        # OOP API
        model = TBRAnalysis()
        model.fit(
            data=data,
            time_col="day",
            control_col="control",
            test_col="test",
            pretest_start=1,
            test_start=31,
            test_end=50,
        )

        # Compare results
        pd.testing.assert_frame_equal(
            model.results_, functional_results.tbr_dataframe(), rtol=1e-14
        )

    def test_float_time_column(self):
        """Test equivalence with float time column."""
        # Create data with float time
        np.random.seed(222)
        data = pd.DataFrame(
            {
                "time": np.linspace(0.0, 10.0, 50),
                "control": np.random.normal(100, 10, 50),
                "test": np.random.normal(105, 11, 50),
            }
        )

        # Functional API
        functional_results = perform_tbr_analysis(
            data=data,
            time_col="time",
            control_col="control",
            test_col="test",
            pretest_start=0.0,
            test_start=6.0,
            test_end=10.0,
            level=0.80,
            threshold=0.0,
        )

        # OOP API
        model = TBRAnalysis()
        model.fit(
            data=data,
            time_col="time",
            control_col="control",
            test_col="test",
            pretest_start=0.0,
            test_start=6.0,
            test_end=10.0,
        )

        # Compare results
        pd.testing.assert_frame_equal(
            model.results_, functional_results.tbr_dataframe(), rtol=1e-14
        )


class TestResultComponents:
    """Test equivalence of individual result components."""

    def test_regression_parameters_equivalence(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test regression parameters (alpha, beta, sigma) are identical."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
            level=0.80,
            threshold=0.0,
        )
        func_params = {
            "alpha": functional_results.alpha,
            "beta": functional_results.beta,
            "sigma": functional_results.sigma,
        }

        # OOP API
        model = TBRAnalysis()
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Compare each parameter
        assert np.isclose(model.params_["alpha"], func_params["alpha"], rtol=1e-15)
        assert np.isclose(model.params_["beta"], func_params["beta"], rtol=1e-15)
        assert np.isclose(model.params_["sigma"], func_params["sigma"], rtol=1e-15)

    def test_variance_parameters_equivalence(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test variance/covariance parameters are identical."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
            level=0.80,
            threshold=0.0,
        )
        func_summary = functional_results.summary().iloc[0]

        # OOP API
        model = TBRAnalysis()
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Compare variance parameters
        assert np.isclose(
            model.params_["var_alpha"], func_summary["var_alpha"], rtol=1e-15
        )
        assert np.isclose(
            model.params_["var_beta"], func_summary["var_beta"], rtol=1e-15
        )
        assert np.isclose(
            model.params_["cov_alpha_beta"], func_summary["alpha_beta_cov"], rtol=1e-15
        )

    def test_dataframe_structure_equivalence(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test TBR DataFrame has identical structure and columns."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
            level=0.80,
            threshold=0.0,
        )
        func_df = functional_results.tbr_dataframe()

        # OOP API
        model = TBRAnalysis()
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Check structure
        assert list(model.results_.columns) == list(func_df.columns)
        assert len(model.results_) == len(func_df)
        assert model.results_.dtypes.equals(func_df.dtypes)

    def test_incremental_summaries_equivalence(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test incremental summaries are identical."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
            level=0.80,
            threshold=0.0,
        )

        # OOP API
        model = TBRAnalysis()
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Compare all incremental summaries
        pd.testing.assert_frame_equal(
            model.summaries_,
            functional_results.summary(),
            check_dtype=True,
            rtol=1e-14,
        )


class TestMathematicalAccuracy:
    """Test mathematical accuracy and precision of equivalence."""

    def test_machine_precision_equivalence(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test results match at machine precision (epsilon)."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
            level=0.80,
            threshold=0.0,
        )

        # OOP API
        model = TBRAnalysis()
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )

        # Test at machine epsilon for float64
        func_estimate = functional_results.estimate
        oop_estimate = model.final_effect

        # Should match within machine epsilon
        assert np.abs(func_estimate - oop_estimate) <= np.finfo(np.float64).eps * max(
            abs(func_estimate), abs(oop_estimate)
        )

    def test_numerical_stability_equivalence(self):
        """Test numerical stability in both implementations."""
        # Create potentially problematic data
        np.random.seed(999)
        n = 100
        control_data = np.random.normal(1000, 0.001, n)  # Very low variance
        test_data = control_data + np.random.normal(0.1, 0.0001, n)

        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n),
                "control": control_data,
                "test": test_data,
            }
        )

        params = {
            "time_col": "date",
            "control_col": "control",
            "test_col": "test",
            "pretest_start": pd.Timestamp("2023-01-01"),
            "test_start": pd.Timestamp("2023-03-02"),
            "test_end": pd.Timestamp("2023-04-10"),
        }

        # Functional API
        functional_results = perform_tbr_analysis(
            data=data, level=0.80, threshold=0.0, **params
        )

        # OOP API
        model = TBRAnalysis()
        model.fit(data=data, **params)

        # Both should handle this stably and produce identical results
        pd.testing.assert_frame_equal(
            model.summaries_,
            functional_results.summary(),
            check_dtype=True,
            rtol=1e-12,
        )

    def test_all_summary_fields_equivalence(
        self, sample_time_series_data, sample_analysis_parameters
    ):
        """Test all 15 summary fields are mathematically identical."""
        # Functional API
        functional_results = perform_tbr_analysis(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
            level=0.85,
            threshold=5.0,
        )
        func_summary = functional_results.summary().iloc[-1]

        # OOP API
        model = TBRAnalysis(level=0.85, threshold=5.0)
        model.fit(
            data=sample_time_series_data,
            time_col=sample_analysis_parameters["time_col"],
            control_col=sample_analysis_parameters["control_col"],
            test_col=sample_analysis_parameters["test_col"],
            pretest_start=sample_analysis_parameters["pretest_start"],
            test_start=sample_analysis_parameters["test_start"],
            test_end=sample_analysis_parameters["test_end"],
        )
        oop_summary = model.summarize()

        # Compare all fields
        fields = [
            ("estimate", oop_summary.estimate),
            ("precision", oop_summary.precision),
            ("lower", oop_summary.lower),
            ("upper", oop_summary.upper),
            ("se", oop_summary.se),
            ("level", oop_summary.level),
            ("threshold", oop_summary.threshold),
            ("prob", oop_summary.prob),
            ("alpha", oop_summary.alpha),
            ("beta", oop_summary.beta),
            ("sigma", oop_summary.sigma),
            ("var_alpha", oop_summary.var_alpha),
            ("var_beta", oop_summary.var_beta),
            ("cov_alpha_beta", oop_summary.cov_alpha_beta),
            ("degrees_freedom", oop_summary.degrees_freedom),
        ]

        for field_name, oop_value in fields:
            # Map OOP field names to functional DataFrame column names
            if field_name == "threshold":
                func_value = func_summary["thres"]
            elif field_name == "cov_alpha_beta":
                func_value = func_summary["alpha_beta_cov"]
            elif field_name == "degrees_freedom":
                func_value = int(func_summary["t_dist_df"])
            else:
                func_value = func_summary[field_name]

            # Compare values with high precision
            assert np.isclose(
                oop_value, func_value, rtol=1e-15
            ), f"{field_name} mismatch: OOP={oop_value}, Functional={func_value}"
