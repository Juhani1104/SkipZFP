"""Shared test setup."""

import pytest

from skipzfp import query


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_default_gap: run with the library's default merge gap"
    )


@pytest.fixture(autouse=True)
def adjacent_reads_by_default(request):
    """Merge adjacent blocks only, unless a test asks for the real default.

    The test arrays are small, so the library's 256 KB default gap would merge
    most queries into a read per chunk and leave the many-small-reads path
    untested. Tests that set merge_gap_blocks themselves are not affected. It
    uses its own MonkeyPatch, so a test's monkeypatch.undo() keeps it.
    """
    if "real_default_gap" in request.keywords:
        yield
    else:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(query, "DEFAULT_MERGE_GAP_BYTES", 0)
            yield
