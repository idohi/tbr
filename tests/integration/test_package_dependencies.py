"""
Integration tests for package dependencies and installation.

These tests verify that the package can be imported and used with only
the runtime dependencies specified in pyproject.toml, without requiring
development dependencies.

This catches dependency specification bugs before PyPI release.
"""


import pytest


class TestRuntimeDependencies:
    """Test that all public APIs work with only runtime dependencies."""

    def test_main_package_import(self):
        """Test that main package imports successfully."""
        import tbr

        assert hasattr(tbr, "__version__")
        assert hasattr(tbr, "__author__")
        assert hasattr(tbr, "__license__")

    def test_oop_api_imports(self):
        """Test that OOP API classes can be imported."""
        from tbr import TBRAnalysis
        from tbr.core import TBRPredictionResult, TBRSubintervalResult, TBRSummaryResult

        # Verify classes are importable
        assert TBRAnalysis is not None
        assert TBRSummaryResult is not None
        assert TBRPredictionResult is not None
        assert TBRSubintervalResult is not None

    def test_functional_api_imports(self):
        """Test that functional API can be imported."""
        from tbr import perform_tbr_analysis

        assert perform_tbr_analysis is not None

    def test_core_module_imports(self):
        """Test that all core modules can be imported."""
        from tbr import core

        # Test key exports
        assert hasattr(core, "TBRAnalysis")
        assert hasattr(core, "TBRSummaryResult")
        assert hasattr(core, "TBRPredictionResult")

    def test_analysis_module_imports(self):
        """Test that analysis modules can be imported."""
        from tbr import analysis

        # Test key exports
        assert hasattr(analysis, "create_tbr_summary")
        assert hasattr(analysis, "create_incremental_tbr_summaries")

    def test_utils_module_imports(self):
        """Test that utils modules can be imported."""
        from tbr import utils

        # Test key exports exist
        assert hasattr(utils, "TBRError")
        assert hasattr(utils, "ConvergenceError")

    def test_oop_api_instantiation(self):
        """Test that TBRAnalysis can be instantiated."""
        from tbr import TBRAnalysis

        model = TBRAnalysis(level=0.90)
        assert model is not None
        assert not model.fitted_

    def test_result_objects_instantiation(self):
        """Test that result objects can be created."""

        from tbr.core import TBRSummaryResult

        # Create a minimal TBRSummaryResult
        summary = TBRSummaryResult(
            estimate=100.0,
            lower=90.0,
            upper=110.0,
            se=5.0,
            prob=0.95,
            precision=10.0,
            level=0.90,
            threshold=0.0,
            alpha=1.0,
            beta=2.0,
            sigma=2.0,
            var_alpha=0.1,
            var_beta=0.1,
            cov_alpha_beta=0.01,
            degrees_freedom=10,
        )
        assert summary is not None
        assert summary.estimate == 100.0


class TestAllRequiredDependencies:
    """Test that all required dependencies are importable."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "pandas",
            "numpy",
            "scipy",
            "statsmodels",
            "lazy_loader",
            "psutil",  # Critical: this was missing!
        ],
    )
    def test_required_dependency_present(self, module_name: str):
        """Test that each required runtime dependency can be imported."""
        try:
            __import__(module_name)
        except ImportError as e:
            pytest.fail(
                f"Required runtime dependency '{module_name}' cannot be imported. "
                f"This suggests it's missing from pyproject.toml dependencies. "
                f"Error: {e}"
            )


class TestNoDevelopmentDependencies:
    """Test that development dependencies are not required for basic functionality."""

    @pytest.mark.parametrize(
        "module_name,reason",
        [
            ("pytest", "Testing framework - should be dev-only"),
            ("black", "Code formatter - should be dev-only"),
            ("ruff", "Linter - should be dev-only"),
            ("mypy", "Type checker - should be dev-only"),
        ],
    )
    def test_dev_dependency_not_required(self, module_name: str, reason: str):
        """
        Test that basic imports work without dev dependencies.

        Note: This test will pass in dev environment (dev deps present).
        The real test is in CI when installing from wheel.
        """
        # Try importing tbr without the dev dependency
        # This is informational - the real test is in CI
        try:
            import tbr  # noqa: F401

            # If we can import tbr, that's good
            # (even if dev deps are present in current environment)
        except ImportError as e:
            # If import fails and it's because of a dev dependency, that's a bug
            if module_name.lower() in str(e).lower():
                pytest.fail(
                    f"TBR package import failed due to development dependency '{module_name}'. "
                    f"{reason}. This dependency should not be required at runtime. "
                    f"Error: {e}"
                )
            else:
                # Different import error, re-raise
                raise


class TestPackageMetadata:
    """Test that package metadata is correct."""

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        import re

        import tbr

        # Should match semver or pre-release (e.g., 0.1.0, 0.1.0rc1, 0.1.0b1)
        version_pattern = r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$"
        assert re.match(
            version_pattern, tbr.__version__
        ), f"Version '{tbr.__version__}' doesn't match semver pattern"

    def test_license_specified(self):
        """Test that license is specified."""
        import tbr

        assert tbr.__license__, "Package license should be specified"
        assert "Apache" in tbr.__license__, "Expected Apache license"

    def test_author_specified(self):
        """Test that author is specified."""
        import tbr

        assert tbr.__author__, "Package author should be specified"


class TestBasicFunctionality:
    """Test that basic functionality works with runtime dependencies only."""

    def test_model_can_be_configured(self):
        """Test that model can be configured with different parameters."""
        from tbr import TBRAnalysis

        # Test various configurations
        model1 = TBRAnalysis(level=0.90)
        assert model1.level == 0.90

        model2 = TBRAnalysis(level=0.95, threshold=10.0)
        assert model2.level == 0.95
        assert model2.threshold == 10.0

    def test_model_string_representation(self):
        """Test that model has proper string representation."""
        from tbr import TBRAnalysis

        model = TBRAnalysis(level=0.90)
        str_repr = str(model)
        assert "TBRAnalysis" in str_repr
        assert "not fitted" in str_repr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
