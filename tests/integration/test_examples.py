"""Enforcement gate for the example scripts in ``examples/``.

The example scripts are the first code many users run, but the rest of the
test suite never executes them. Every script is run here, so an example that
stops working fails the build instead of reaching users.
"""

import runpy
from pathlib import Path

import matplotlib
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg", force=True)

pytestmark = pytest.mark.integration

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
_EXAMPLE_SCRIPTS = sorted(_EXAMPLES_DIR.glob("*.py"))
_EXPECTED_FIGURES_BY_SCRIPT = {
    "plot_basic_analysis.py": ("prediction_fig",),
    "plot_incremental_analysis.py": ("cumulative_fig", "subinterval_fig"),
    "plot_method_chaining.py": ("precision_fig",),
}

# Guards against the glob silently finding nothing and the gate passing
# vacuously.
_MINIMUM_EXPECTED_EXAMPLES = 3


def test_example_scripts_are_collected():
    """The gate finds the example scripts."""
    assert len(_EXAMPLE_SCRIPTS) >= _MINIMUM_EXPECTED_EXAMPLES


@pytest.mark.parametrize("script", _EXAMPLE_SCRIPTS, ids=lambda path: path.name)
def test_example_script_runs(script, tmp_path, monkeypatch):
    """Each example script runs and creates its intended figures."""
    from matplotlib import pyplot as plt

    # The examples write their exports to the current directory.
    monkeypatch.chdir(tmp_path)
    try:
        namespace = runpy.run_path(str(script), run_name="__main__")
        for figure_name in _EXPECTED_FIGURES_BY_SCRIPT[script.name]:
            assert isinstance(namespace.get(figure_name), Figure)
    finally:
        plt.close("all")
