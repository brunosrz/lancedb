# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

"""Re-entrancy behavior of the shared background event loop.

The sync API delegates to `LOOP.run(...)`, which moves the coroutine onto the
background loop thread. Anything that thread pulls -- notably the first item of
a generator handed to `Table.add()` / `create_table()` -- therefore runs *on
that thread*, and a nested sync lancedb call from there would wait on the only
thread able to complete it. See https://github.com/lancedb/lancedb/issues/2107.
"""

import subprocess
import sys
from unittest import mock

import pytest
from lancedb.background_loop import LOOP, BackgroundEventLoop


def test_run_from_the_loop_thread_raises_instead_of_deadlocking():
    """The guard itself: re-entering `run()` from the loop thread must raise.

    Written so a regression fails rather than hangs -- if the guard were
    removed, the inner call would block the loop thread forever and this test
    would never return.
    """

    async def outer():
        async def inner():
            return 1

        return LOOP.run(inner())

    with pytest.raises(RuntimeError, match="background event loop was re-entered"):
        LOOP.run(outer())


def test_guard_tolerates_a_partially_constructed_loop():
    """The guard must not assume `_start()` has run.

    Existing tests build a `BackgroundEventLoop` with `__init__` patched out
    and set only `.loop`, so reading `.thread` unconditionally would turn a
    working call into an `AttributeError`. They also pass a non-coroutine,
    which has no `close()`.
    """
    loop = BackgroundEventLoop.__new__(BackgroundEventLoop)
    loop.loop = mock.MagicMock()
    assert not hasattr(loop, "thread")

    sentinel = object()
    with mock.patch("asyncio.run_coroutine_threadsafe") as run_threadsafe:
        run_threadsafe.return_value.result.return_value = sentinel
        # Reaches the normal path instead of raising, and tolerates a
        # non-coroutine argument on the way.
        assert loop.run(None) is sentinel


def test_generator_that_does_not_call_lancedb_still_works(mem_db):
    """The guard must be surgical.

    Only the *first* item of a generator is pulled on the loop thread; the rest
    are pulled from Rust on Tokio blocking threads, where nested `LOOP.run` is
    legitimate. A generator that never re-enters lancedb must be unaffected.
    """
    table = mem_db.create_table("plain_gen", data=[{"id": 0}])

    def gen():
        for i in range(1, 4):
            yield [{"id": i}]

    table.add(gen())

    assert table.count_rows() == 4


_REENTRANT_REPRO = """
import sys
import tempfile

import lancedb

with tempfile.TemporaryDirectory() as tmp:
    db = lancedb.connect(tmp)
    rows = [{"vector": [1.0, 2.0], "id": 1}]
    t1 = db.create_table("t1", data=rows)
    t2 = db.create_table("t2", data=rows)

    def gen():
        for _ in range(3):
            yield t1.search([1.0, 2.0]).limit(1).to_arrow()

    try:
        t2.add(gen())
    except RuntimeError as exc:
        if "background event loop was re-entered" in str(exc):
            print("RAISED")
            sys.exit(0)
        raise
    print("NO_ERROR")
    sys.exit(1)
"""


def test_reentrant_add_raises_rather_than_hanging(tmp_path):
    """End-to-end regression for #2107.

    Runs in a child process this test can actually time out and kill: a
    regression here deadlocks, and an in-process watchdog could report that but
    not escape it, so pytest would hang regardless of what it detected.
    """
    script = tmp_path / "_reentrant_repro.py"
    script.write_text(_REENTRANT_REPRO)

    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=120.0,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"deadlock: re-entrant add() never returned "
            f"(stdout={exc.stdout!r} stderr={exc.stderr!r})",
            pytrace=False,
        )

    assert result.returncode == 0, (
        f"expected a clear RuntimeError, got stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert "RAISED" in result.stdout
