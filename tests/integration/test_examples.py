"""Enforcement gate for the example scripts in ``examples/``.

The example scripts are the first code many users run, but the rest of the
test suite never executes them. Every script is run here, so an example that
stops working fails the build instead of reaching users.
"""

import runpy
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
_EXAMPLE_SCRIPTS = sorted(_EXAMPLES_DIR.glob("*.py"))

# Guards against the glob silently finding nothing and the gate passing
# vacuously.
_MINIMUM_EXPECTED_EXAMPLES = 3


def test_example_scripts_are_collected():
    """The gate finds the example scripts."""
    assert len(_EXAMPLE_SCRIPTS) >= _MINIMUM_EXPECTED_EXAMPLES


@pytest.mark.parametrize("script", _EXAMPLE_SCRIPTS, ids=lambda path: path.name)
def test_example_script_runs(script, tmp_path, monkeypatch):
    """Each example script runs to completion."""
    # The examples write their exports to the current directory.
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name="__main__")
