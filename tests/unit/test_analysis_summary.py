"""
Unit tests for tbr.analysis.summary module.

This module tests the analysis summary functionality including TBR summary
creation and incremental summary generation. All tests validate mathematical
correctness, backward compatibility with functional implementation, and
proper error handling.

Test Categories
---------------
1. TBR Summary Creation - create_tbr_summary() function
2. Incremental Summaries - create_incremental_tbr_summaries() function
3. Input Validation - Error handling and edge cases
4. Mathematical Properties - Statistical correctness validation
5. Backward Compatibility - Cross-validation with functional implementation
6. Integration Testing - Module imports and workflow integration
"""


import pandas as pd
import pytest

from tbr.analysis.summary import create_incremental_tbr_summaries, create_tbr_summary
from tbr.functional.tbr_functions import (
    create_incremental_tbr_summaries as functional_create_incremental_tbr_summaries,
)
from tbr.functional.tbr_functions import (
    create_tbr_summary as functional_create_tbr_summary,
)


class TestCreateTbrSummary:
    """Test cases for create_tbr_summary function."""

    def test_basic_summary_creation(self):
        """Test basic TBR summary creation with valid inputs."""
        # Create test TBR dataframe
        tbr_data = {
            "period": [0, 0, 0, 1, 1, 1],
            "cumdif": [0, 0, 0, 10.5, 15.2, 20.8],
            "cumsd": [0, 0, 0, 5.1, 7.3, 8.9],
        }
        tbr_df = pd.DataFrame(tbr_data)

        # Test parameters
        alpha = 50.2
        beta = 0.95
        sigma = 25.3
        var_alpha = 100.5
        var_beta = 0.001
        cov_alpha_beta = -0.05
        degrees_freedom = 43
        level = 0.80
        threshold = 0.0

        # Create summary
        summary = create_tbr_summary(
            tbr_dataframe=tbr_df,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            var_alpha=var_alpha,
            var_beta=var_beta,
            cov_alpha_beta=cov_alpha_beta,
            degrees_freedom=degrees_freedom,
            level=level,
            threshold=threshold,
        )

        # Validate structure
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1  # Single row summary

        # Validate required columns
        expected_cols = [
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
        ]
        for col in expected_cols:
            assert col in summary.columns

        # Validate values
        assert summary["estimate"].iloc[0] == 20.8  # Final cumdif
        assert summary["se"].iloc[0] == 8.9  # Final cumsd
        assert summary["level"].iloc[0] == level
        assert summary["thres"].iloc[0] == threshold
        assert summary["alpha"].iloc[0] == alpha
        assert summary["beta"].iloc[0] == beta
        assert summary["sigma"].iloc[0] == sigma
        assert summary["t_dist_df"].iloc[0] == degrees_freedom

        # Validate probability is between 0 and 1
        assert 0 <= summary["prob"].iloc[0] <= 1

        # Validate credible interval relationship
        assert summary["lower"].iloc[0] < summary["upper"].iloc[0]
        assert summary["precision"].iloc[0] > 0

    def test_summary_mathematical_properties(self):
        """Test mathematical properties of TBR summary."""
        # Create test data with known properties
        tbr_data = {
            "period": [0, 0, 1, 1],
            "cumdif": [0, 0, 100.0, 200.0],
            "cumsd": [0, 0, 10.0, 15.0],
        }
        tbr_df = pd.DataFrame(tbr_data)

        # Create summary with 95% confidence level
        summary = create_tbr_summary(
            tbr_dataframe=tbr_df,
            alpha=50.0,
            beta=1.0,
            sigma=20.0,
            var_alpha=25.0,
            var_beta=0.01,
            cov_alpha_beta=0.0,
            degrees_freedom=30,
            level=0.95,
            threshold=0.0,
        )

        # Test mathematical relationships
        estimate = summary["estimate"].iloc[0]
        precision = summary["precision"].iloc[0]
        lower = summary["lower"].iloc[0]
        upper = summary["upper"].iloc[0]

        # Credible interval should be symmetric around estimate
        assert abs((lower + upper) / 2 - estimate) < 1e-10

        # Precision should be half the interval width
        interval_width = upper - lower
        assert abs(precision - interval_width / 2) < 1e-10

        # Bounds should be estimate ± precision
        assert abs(lower - (estimate - precision)) < 1e-10
        assert abs(upper - (estimate + precision)) < 1e-10

    def test_backward_compatibility_with_functional(self):
        """Test 100% backward compatibility with functional implementation."""
        # Create identical test data
        tbr_data = {
            "period": [0, 0, 0, 1, 1, 1, 1],
            "cumdif": [0, 0, 0, 5.2, 12.8, 18.5, 25.1],
            "cumsd": [0, 0, 0, 3.1, 6.7, 9.2, 11.8],
        }
        tbr_df = pd.DataFrame(tbr_data)

        # Test parameters
        params = {
            "tbr_dataframe": tbr_df,
            "alpha": 45.7,
            "beta": 0.92,
            "sigma": 22.1,
            "var_alpha": 85.3,
            "var_beta": 0.0015,
            "cov_alpha_beta": -0.08,
            "degrees_freedom": 38,
            "level": 0.85,
            "threshold": 10.0,
        }

        # Create summaries using both implementations
        analysis_summary = create_tbr_summary(**params)
        functional_summary = functional_create_tbr_summary(**params)

        # Validate identical results
        pd.testing.assert_frame_equal(
            analysis_summary, functional_summary, check_dtype=True, check_exact=True
        )

    def test_input_validation_errors(self):
        """Test proper error handling for invalid inputs."""
        # Valid base parameters
        valid_params = {
            "alpha": 50.0,
            "beta": 1.0,
            "sigma": 20.0,
            "var_alpha": 25.0,
            "var_beta": 0.01,
            "cov_alpha_beta": 0.0,
            "degrees_freedom": 30,
            "level": 0.80,
            "threshold": 0.0,
        }

        # Test empty dataframe
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="TBR dataframe cannot be empty"):
            create_tbr_summary(tbr_dataframe=empty_df, **valid_params)

        # Test missing required columns
        incomplete_df = pd.DataFrame({"period": [0, 1], "cumdif": [0, 10]})
        with pytest.raises(ValueError, match="Missing required columns"):
            create_tbr_summary(tbr_dataframe=incomplete_df, **valid_params)

        # Test invalid level
        valid_df = pd.DataFrame({"period": [0, 1], "cumdif": [0, 10], "cumsd": [0, 5]})
        with pytest.raises(ValueError, match="Level must be between 0 and 1"):
            create_tbr_summary(
                tbr_dataframe=valid_df,
                level=1.5,
                **{k: v for k, v in valid_params.items() if k != "level"},
            )

        # Test invalid degrees of freedom
        with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
            create_tbr_summary(
                tbr_dataframe=valid_df,
                degrees_freedom=-5,
                **{k: v for k, v in valid_params.items() if k != "degrees_freedom"},
            )

        # Test invalid sigma
        with pytest.raises(ValueError, match="Sigma must be positive"):
            create_tbr_summary(
                tbr_dataframe=valid_df,
                sigma=-10.0,
                **{k: v for k, v in valid_params.items() if k != "sigma"},
            )

        # Test no test period data
        no_test_df = pd.DataFrame({"period": [0, 0], "cumdif": [0, 5], "cumsd": [0, 2]})
        with pytest.raises(ValueError, match="No test period data found"):
            create_tbr_summary(tbr_dataframe=no_test_df, **valid_params)

    def test_edge_cases_mathematical(self):
        """Test edge cases with mathematical validation."""
        # Test with zero threshold (common case)
        tbr_data = {
            "period": [0, 1, 1],
            "cumdif": [0, 50.0, 100.0],
            "cumsd": [0, 10.0, 15.0],
        }
        tbr_df = pd.DataFrame(tbr_data)

        summary = create_tbr_summary(
            tbr_dataframe=tbr_df,
            alpha=0.0,
            beta=1.0,
            sigma=10.0,
            var_alpha=1.0,
            var_beta=0.001,
            cov_alpha_beta=0.0,
            degrees_freedom=50,
            level=0.90,
            threshold=0.0,
        )

        # With positive estimate and zero threshold, prob should be > 0.5
        assert summary["prob"].iloc[0] > 0.5

        # Test with negative estimate
        tbr_data_neg = {
            "period": [0, 1, 1],
            "cumdif": [0, -20.0, -40.0],
            "cumsd": [0, 8.0, 12.0],
        }
        tbr_df_neg = pd.DataFrame(tbr_data_neg)

        summary_neg = create_tbr_summary(
            tbr_dataframe=tbr_df_neg,
            alpha=0.0,
            beta=1.0,
            sigma=10.0,
            var_alpha=1.0,
            var_beta=0.001,
            cov_alpha_beta=0.0,
            degrees_freedom=50,
            level=0.90,
            threshold=0.0,
        )

        # With negative estimate and zero threshold, prob should be < 0.5
        assert summary_neg["prob"].iloc[0] < 0.5


class TestCreateIncrementalTbrSummaries:
    """Test cases for create_incremental_tbr_summaries function."""

    def test_basic_incremental_summaries(self):
        """Test basic incremental summaries creation."""
        # Create test TBR dataframe with multiple test days
        tbr_data = {
            "period": [0, 0, 0, 1, 1, 1, 1, 1],
            "cumdif": [0, 0, 0, 5.0, 12.0, 18.0, 25.0, 30.0],
            "cumsd": [0, 0, 0, 3.0, 5.5, 7.2, 8.8, 10.1],
        }
        tbr_df = pd.DataFrame(tbr_data)

        # Create incremental summaries
        summaries = create_incremental_tbr_summaries(
            tbr_dataframe=tbr_df,
            alpha=50.0,
            beta=0.95,
            sigma=20.0,
            var_alpha=100.0,
            var_beta=0.001,
            cov_alpha_beta=-0.05,
            degrees_freedom=40,
            level=0.80,
            threshold=0.0,
        )

        # Validate structure
        assert isinstance(summaries, pd.DataFrame)
        assert len(summaries) == 5  # 5 test period rows

        # Validate test_day column exists and is sequential
        assert "test_day" in summaries.columns
        expected_days = list(range(1, 6))
        actual_days = summaries["test_day"].tolist()
        assert actual_days == expected_days

        # Validate estimates are increasing (cumulative effect)
        estimates = summaries["estimate"].tolist()
        assert all(estimates[i] <= estimates[i + 1] for i in range(len(estimates) - 1))

        # Validate final summary matches last row
        assert summaries["estimate"].iloc[-1] == 30.0
        assert summaries["se"].iloc[-1] == 10.1

    def test_incremental_mathematical_consistency(self):
        """Test mathematical consistency of incremental summaries."""
        # Create test data
        tbr_data = {
            "period": [0, 0, 1, 1, 1, 1],
            "cumdif": [0, 0, 10.0, 25.0, 35.0, 50.0],
            "cumsd": [0, 0, 5.0, 8.0, 10.0, 12.0],
        }
        tbr_df = pd.DataFrame(tbr_data)

        summaries = create_incremental_tbr_summaries(
            tbr_dataframe=tbr_df,
            alpha=0.0,
            beta=1.0,
            sigma=15.0,
            var_alpha=10.0,
            var_beta=0.01,
            cov_alpha_beta=0.0,
            degrees_freedom=25,
            level=0.95,
            threshold=0.0,
        )

        # Test mathematical properties for each row
        for _i, row in summaries.iterrows():
            # Credible interval should be symmetric
            estimate = row["estimate"]
            lower = row["lower"]
            upper = row["upper"]
            precision = row["precision"]

            # Interval symmetry
            assert abs((lower + upper) / 2 - estimate) < 1e-10

            # Precision relationship
            assert abs(precision - (upper - lower) / 2) < 1e-10

            # Probability bounds
            assert 0 <= row["prob"] <= 1

    def test_backward_compatibility_incremental(self):
        """Test backward compatibility with functional implementation."""
        # Create test data
        tbr_data = {
            "period": [0, 0, 1, 1, 1],
            "cumdif": [0, 0, 8.5, 17.2, 26.8],
            "cumsd": [0, 0, 4.1, 6.9, 9.3],
        }
        tbr_df = pd.DataFrame(tbr_data)

        params = {
            "tbr_dataframe": tbr_df,
            "alpha": 42.3,
            "beta": 0.88,
            "sigma": 18.7,
            "var_alpha": 75.2,
            "var_beta": 0.0012,
            "cov_alpha_beta": -0.06,
            "degrees_freedom": 35,
            "level": 0.75,
            "threshold": 5.0,
        }

        # Create summaries using both implementations
        analysis_summaries = create_incremental_tbr_summaries(**params)
        functional_summaries = functional_create_incremental_tbr_summaries(**params)

        # Validate identical results
        pd.testing.assert_frame_equal(
            analysis_summaries, functional_summaries, check_dtype=True, check_exact=True
        )

    def test_incremental_input_validation(self):
        """Test input validation for incremental summaries."""
        # Valid parameters
        valid_params = {
            "alpha": 50.0,
            "beta": 1.0,
            "sigma": 20.0,
            "var_alpha": 25.0,
            "var_beta": 0.01,
            "cov_alpha_beta": 0.0,
            "degrees_freedom": 30,
            "level": 0.80,
            "threshold": 0.0,
        }

        # Test empty dataframe
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="TBR dataframe cannot be empty"):
            create_incremental_tbr_summaries(tbr_dataframe=empty_df, **valid_params)

        # Test no test period data
        no_test_df = pd.DataFrame({"period": [0, 0], "cumdif": [0, 5], "cumsd": [0, 2]})
        with pytest.raises(ValueError, match="No test period data found"):
            create_incremental_tbr_summaries(tbr_dataframe=no_test_df, **valid_params)


class TestAnalysisSummaryIntegration:
    """Test integration and module-level functionality."""

    def test_module_imports(self):
        """Test that all functions can be imported correctly."""
        # Test direct imports
        # Test main package imports

        # Test module-level imports
        from tbr.analysis import (
            create_incremental_tbr_summaries as analysis_create_incremental,
        )
        from tbr.analysis import create_tbr_summary as analysis_create_tbr_summary
        from tbr.analysis.summary import (
            create_incremental_tbr_summaries,
            create_tbr_summary,
        )

        # Validate all imports reference the same functions
        assert create_tbr_summary == analysis_create_tbr_summary
        assert create_incremental_tbr_summaries == analysis_create_incremental

    def test_lazy_loading_functionality(self):
        """Test that lazy loading works correctly."""
        # This test ensures lazy loading doesn't break functionality
        import tbr.analysis

        # Create test data
        tbr_data = {
            "period": [0, 1, 1],
            "cumdif": [0, 10.0, 20.0],
            "cumsd": [0, 5.0, 8.0],
        }
        tbr_df = pd.DataFrame(tbr_data)

        # Test lazy-loaded function works
        summary = tbr.analysis.create_tbr_summary(
            tbr_dataframe=tbr_df,
            alpha=50.0,
            beta=1.0,
            sigma=20.0,
            var_alpha=25.0,
            var_beta=0.01,
            cov_alpha_beta=0.0,
            degrees_freedom=30,
            level=0.80,
            threshold=0.0,
        )

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1

    def test_workflow_integration(self):
        """Test integration with complete TBR workflow."""
        # Create realistic test data
        tbr_data = {
            "period": [0, 0, 0, 1, 1, 1, 1],
            "cumdif": [0, 0, 0, 15.2, 28.7, 42.1, 55.8],
            "cumsd": [0, 0, 0, 6.3, 9.8, 12.5, 15.1],
        }
        tbr_df = pd.DataFrame(tbr_data)

        # Model parameters from realistic regression
        model_params = {
            "alpha": 48.7,
            "beta": 0.94,
            "sigma": 22.3,
            "var_alpha": 89.2,
            "var_beta": 0.0018,
            "cov_alpha_beta": -0.07,
            "degrees_freedom": 42,
            "level": 0.80,
            "threshold": 0.0,
        }

        # Create both summary types
        summary = create_tbr_summary(tbr_dataframe=tbr_df, **model_params)
        incremental = create_incremental_tbr_summaries(
            tbr_dataframe=tbr_df, **model_params
        )

        # Validate workflow consistency
        # Final incremental summary should match overall summary
        final_incremental = incremental.iloc[-1]

        # Key statistics should match
        assert abs(summary["estimate"].iloc[0] - final_incremental["estimate"]) < 1e-10
        assert abs(summary["se"].iloc[0] - final_incremental["se"]) < 1e-10
        assert abs(summary["prob"].iloc[0] - final_incremental["prob"]) < 1e-10

        # Both should have same model parameters
        assert summary["alpha"].iloc[0] == final_incremental["alpha"]
        assert summary["beta"].iloc[0] == final_incremental["beta"]
        assert summary["sigma"].iloc[0] == final_incremental["sigma"]


class TestAnalysisSummaryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_test_day_summaries(self):
        """Test summaries with only one test day."""
        tbr_data = {
            "period": [0, 0, 1],
            "cumdif": [0, 0, 25.0],
            "cumsd": [0, 0, 8.0],
        }
        tbr_df = pd.DataFrame(tbr_data)

        # Test both summary types
        summary = create_tbr_summary(
            tbr_dataframe=tbr_df,
            alpha=50.0,
            beta=1.0,
            sigma=20.0,
            var_alpha=25.0,
            var_beta=0.01,
            cov_alpha_beta=0.0,
            degrees_freedom=30,
            level=0.80,
            threshold=0.0,
        )

        incremental = create_incremental_tbr_summaries(
            tbr_dataframe=tbr_df,
            alpha=50.0,
            beta=1.0,
            sigma=20.0,
            var_alpha=25.0,
            var_beta=0.01,
            cov_alpha_beta=0.0,
            degrees_freedom=30,
            level=0.80,
            threshold=0.0,
        )

        # Should have single row for both
        assert len(summary) == 1
        assert len(incremental) == 1

        # Should have identical results
        assert summary["estimate"].iloc[0] == incremental["estimate"].iloc[0]
        assert summary["se"].iloc[0] == incremental["se"].iloc[0]

    def test_extreme_confidence_levels(self):
        """Test with extreme confidence levels."""
        tbr_data = {
            "period": [0, 1, 1],
            "cumdif": [0, 10.0, 20.0],
            "cumsd": [0, 5.0, 8.0],
        }
        tbr_df = pd.DataFrame(tbr_data)

        # Test very high confidence level (99.9%)
        summary_high = create_tbr_summary(
            tbr_dataframe=tbr_df,
            alpha=50.0,
            beta=1.0,
            sigma=20.0,
            var_alpha=25.0,
            var_beta=0.01,
            cov_alpha_beta=0.0,
            degrees_freedom=30,
            level=0.999,
            threshold=0.0,
        )

        # Test very low confidence level (1%)
        summary_low = create_tbr_summary(
            tbr_dataframe=tbr_df,
            alpha=50.0,
            beta=1.0,
            sigma=20.0,
            var_alpha=25.0,
            var_beta=0.01,
            cov_alpha_beta=0.0,
            degrees_freedom=30,
            level=0.01,
            threshold=0.0,
        )

        # High confidence should have wider intervals
        assert summary_high["precision"].iloc[0] > summary_low["precision"].iloc[0]

        # Both should have valid probabilities
        assert 0 <= summary_high["prob"].iloc[0] <= 1
        assert 0 <= summary_low["prob"].iloc[0] <= 1
