"""Enforcement gate for the code examples embedded in TBR docstrings.

Docstring examples are published verbatim to the API reference and shipped to
users, but the rest of the test suite never executes them. This module closes
that gap: every ``>>>`` example in the package is executed here, so an example
that stops working fails the build instead of reaching the documentation site.

The gate is deliberately hybrid. Examples with no expected output are checked
for executability only. Examples that publish expected output are checked
against that output, because those values become part of the public
documentation contract.

There are two deliberate exceptions:

* An example whose expected output is a ``Traceback`` block must raise, and the
  exception type and message must match (honoring ``# doctest: +ELLIPSIS``).
  Without this, a deliberate demonstration of an error would be
  indistinguishable from an accidental failure.
* An example marked ``# doctest: +SKIP`` is not executed. That marker is
  reserved for examples that genuinely cannot run, such as the ones that write
  files; it is never used to silence an example that is simply wrong.
"""

import doctest
import importlib
import io
import pkgutil
import textwrap
import traceback
import warnings
from contextlib import redirect_stdout
from typing import Dict, List, Optional

import numpy as np
import pytest

import tbr

pytestmark = pytest.mark.integration

_CHECKER = doctest.OutputChecker()

# Guards against the collection logic silently finding nothing and the gate
# passing vacuously. Well below the current count, so ordinary additions and
# removals of examples do not require touching this number.
_MINIMUM_EXPECTED_EXAMPLES = 100


def _iter_modules():
    """Yield the ``tbr`` package and every module inside it."""
    yield tbr
    for info in pkgutil.walk_packages(tbr.__path__, prefix=tbr.__name__ + "."):
        yield importlib.import_module(info.name)


def _collect_doctests() -> List[doctest.DocTest]:
    """Return every docstring in the package that contains at least one example."""
    finder = doctest.DocTestFinder(exclude_empty=True)
    collected: Dict[str, doctest.DocTest] = {}
    for module in _iter_modules():
        for test in finder.find(module):
            if test.examples:
                collected.setdefault(test.name, test)
    return [collected[name] for name in sorted(collected)]


def _expected_exception(want: str) -> Optional[str]:
    """Return the exception line an example expects, or None if it expects no raise.

    The return value is the final ``<ExceptionType>: <message>`` line of the
    expected ``Traceback`` block, matching what ``format_exception_only``
    produces for a real exception.
    """
    if not want.lstrip().startswith("Traceback (most recent call last)"):
        return None
    for line in reversed(want.strip().splitlines()):
        stripped = line.strip()
        if not stripped or line.startswith((" ", "\t")) or stripped == "...":
            continue
        if stripped.startswith(("Traceback", "File ")):
            continue
        return stripped
    return None


def _describe(test: doctest.DocTest, example: doctest.Example) -> str:
    """Locate an example by file and line for a readable failure message."""
    line = (test.lineno or 0) + example.lineno + 1
    return f"{test.name} ({test.filename}:{line})\n    >>> {example.source.strip()}"


def _option_flags(example: doctest.Example) -> int:
    """Return doctest option flags enabled for an individual example."""
    flags = 0
    for option, enabled in example.options.items():
        if enabled:
            flags |= option
    return flags


_DOCTESTS = _collect_doctests()


def test_examples_were_collected() -> None:
    """Fail loudly if example discovery breaks, rather than passing vacuously."""
    assert len(_DOCTESTS) >= _MINIMUM_EXPECTED_EXAMPLES, (
        f"Only {len(_DOCTESTS)} docstrings with examples were discovered, which "
        f"is fewer than the {_MINIMUM_EXPECTED_EXAMPLES} expected. Example "
        f"collection is probably broken, so the gate is not actually checking "
        f"anything."
    )


@pytest.mark.parametrize("test", _DOCTESTS, ids=lambda test: test.name)
def test_docstring_examples_execute(test: doctest.DocTest) -> None:
    """Execute every runnable example and verify output when it is documented."""
    namespace = dict(test.globs)
    np.random.seed(0)

    for example in test.examples:
        if example.options.get(doctest.SKIP, False):
            continue

        expected = _expected_exception(example.want)
        where = _describe(test, example)
        flags = _option_flags(example)

        try:
            # Some examples legitimately warn (small-sample statistics,
            # unavailable system metrics), and the suite otherwise turns
            # warnings into errors.
            stdout = io.StringIO()
            with warnings.catch_warnings(), redirect_stdout(stdout):
                warnings.simplefilter("ignore")
                code = compile(example.source, test.filename or "<docstring>", "single")
                exec(code, namespace)
        except Exception as exc:
            actual = traceback.format_exception_only(type(exc), exc)[-1].strip()
            if expected is None:
                raise AssertionError(f"Example raised {actual}\n  in {where}") from exc
            if not _CHECKER.check_output(expected + "\n", actual + "\n", flags):
                raise AssertionError(
                    f"Example raised the wrong exception\n"
                    f"  expected: {expected}\n"
                    f"  actual:   {actual}\n"
                    f"  in {where}"
                ) from exc
        else:
            if expected is not None:
                raise AssertionError(
                    f"Example documents an exception that was never raised\n"
                    f"  expected: {expected}\n"
                    f"  in {where}"
                )
            actual_output = stdout.getvalue()
            if example.want and not _CHECKER.check_output(
                example.want, actual_output, flags
            ):
                diff = _CHECKER.output_difference(example, actual_output, flags)
                raise AssertionError(
                    f"Example output did not match documented output\n"
                    f"  in {where}\n{diff}"
                )


def _synthetic(docstring: str) -> doctest.DocTest:
    """Build a throwaway DocTest from literal docstring text."""
    return doctest.DocTestParser().get_doctest(
        textwrap.dedent(docstring), {}, "synthetic", "<synthetic>", 0
    )


class TestGateDetectsBrokenExamples:
    """A gate that cannot fail is worse than no gate, so prove that it can.

    These cover every verdict the gate can reach, and keep a later refactor from
    quietly turning it into a no-op.
    """

    def test_failing_example_is_reported(self) -> None:
        with pytest.raises(AssertionError, match="ZeroDivisionError"):
            test_docstring_examples_execute(_synthetic(">>> 1 / 0\n"))

    def test_documented_output_is_accepted(self) -> None:
        test_docstring_examples_execute(
            _synthetic(
                """
                >>> print("Effect: 12.34")
                Effect: 12.34
                """
            )
        )

    def test_wrong_documented_output_is_reported(self) -> None:
        with pytest.raises(AssertionError, match="output did not match"):
            test_docstring_examples_execute(
                _synthetic(
                    """
                    >>> print("Effect: 12.80")
                    Effect: 12.34
                    """
                )
            )

    def test_documented_exception_is_accepted(self) -> None:
        test_docstring_examples_execute(
            _synthetic(
                """
                >>> raise ValueError("boom")
                Traceback (most recent call last):
                    ...
                ValueError: boom
                """
            )
        )

    def test_wrong_exception_is_reported(self) -> None:
        with pytest.raises(AssertionError, match="wrong exception"):
            test_docstring_examples_execute(
                _synthetic(
                    """
                    >>> raise ValueError("boom")
                    Traceback (most recent call last):
                        ...
                    ValueError: something else entirely
                    """
                )
            )

    def test_exception_that_never_happens_is_reported(self) -> None:
        with pytest.raises(AssertionError, match="never raised"):
            test_docstring_examples_execute(
                _synthetic(
                    """
                    >>> 1 + 1
                    Traceback (most recent call last):
                        ...
                    ValueError: boom
                    """
                )
            )

    def test_skip_marker_is_honored(self) -> None:
        test_docstring_examples_execute(_synthetic(">>> 1 / 0  # doctest: +SKIP\n"))
