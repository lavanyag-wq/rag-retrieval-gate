"""Error types and exit codes, because "no" and "cannot answer" are different.

A gate that exits 1 both when it finds a regression and when it crashes trains
people to ignore it. The codes here keep three outcomes apart: the tool did its
job and the answer was no (REGRESSION), the tool did its job and the evidence
cannot support any answer (UNDERPOWERED), and the tool failed to do its job
(ERROR). CI pipelines can then block on 1, warn on 2, and page on 3.
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_UNDERPOWERED = 2
EXIT_ERROR = 3


class GateError(Exception):
    """Base error for this package. Maps to EXIT_ERROR at the CLI boundary."""


class UsageError(GateError):
    """The caller asked for something invalid. Refused at the boundary, before work starts."""


class DataError(GateError):
    """An input file is missing, malformed, or inconsistent with its own metadata."""
