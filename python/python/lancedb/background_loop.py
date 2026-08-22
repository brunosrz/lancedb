# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

import asyncio
import concurrent.futures
import inspect
import os
import threading
import warnings


class BackgroundEventLoop:
    """
    A background event loop that can run futures.

    Used to bridge sync and async code, without messing with users event loops.
    """

    def __init__(self):
        self._start()

    def _start(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever,
            name="LanceDBBackgroundEventLoop",
            daemon=True,
        )
        self.thread.start()

    def run(self, future):
        # `getattr` rather than `self.thread`: the attribute only exists once
        # `_start()` has run, and callers may hold a partially constructed
        # instance. `current_thread()` is never None, so a missing attribute
        # simply means "not the loop thread" and falls through.
        if threading.current_thread() is getattr(self, "thread", None):
            # `run_coroutine_threadsafe` needs this loop to make progress, but
            # this *is* the thread that drives it, so `.result()` below would
            # wait on work only this thread could do. Waiting would hang
            # forever, so fail loudly instead.
            #
            # Closing the coroutine first keeps Python from also emitting
            # "coroutine was never awaited", which under `-W error` would
            # replace the explanation below with a far less useful error.
            # Guarded because `run()` is not statically restricted to
            # coroutines, and only coroutines have `close()`.
            if inspect.iscoroutine(future):
                future.close()
            raise RuntimeError(
                "lancedb's background event loop was re-entered: a synchronous "
                "lancedb call was made from inside another one. The usual cause "
                "is a generator passed to Table.add() or create_table() that "
                "itself queries lancedb, e.g. "
                "`yield other_table.search(...).to_arrow()` -- the generator's "
                "first item is pulled on the background loop thread, so the "
                "nested call would deadlock. Either materialize the nested "
                "results first (`rows = [t1.search(...).to_arrow() for ...]` "
                "then `t2.add(rows)`), or use the async API and await the "
                "nested query outside the generator."
            )
        concurrent_future = asyncio.run_coroutine_threadsafe(future, self.loop)
        try:
            return concurrent_future.result()
        except BaseException:
            concurrent_future.cancel()
            raise


LOOP = BackgroundEventLoop()


def _new_embedding_executor() -> concurrent.futures.ThreadPoolExecutor:
    return concurrent.futures.ThreadPoolExecutor(thread_name_prefix="lancedb-embedding")


# Embedding functions can block for a long time -- a heavy local model or an
# HTTP request to a remote embeddings API. Running them on asyncio's default
# executor lets them starve the unrelated blocking I/O that shares that pool,
# so they get a dedicated one. See
# https://github.com/lancedb/lancedb/issues/3310.
_EMBEDDING_EXECUTOR = _new_embedding_executor()


def embedding_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the executor dedicated to running blocking embedding calls."""
    return _EMBEDDING_EXECUTOR


_FORK_WARNED = False


def _reset_after_fork():
    # Threads do not survive fork(), so the asyncio loop in LOOP.thread is
    # dead in the child. Re-initialize the singleton in place so existing
    # `from .background_loop import LOOP` references in other modules see
    # the new state. The Rust-side tokio runtime is reset analogously by a
    # pthread_atfork hook installed in the _lancedb extension.
    LOOP._start()
    # The embedding executor's worker threads are dead in the child as well.
    # Replace it with a fresh pool (threads are spawned lazily, so this is
    # cheap); we don't shut down the old one, since joining its dead workers
    # could hang.
    global _EMBEDDING_EXECUTOR
    _EMBEDDING_EXECUTOR = _new_embedding_executor()
    global _FORK_WARNED
    if not _FORK_WARNED:
        _FORK_WARNED = True
        warnings.warn(
            "lancedb fork support is experimental: the internal async "
            "runtime has been reset in the forked child, but a small chance "
            "of deadlock remains if other state was mid-operation at fork "
            "time. The 'forkserver' or 'spawn' multiprocessing start method "
            "is likely a safer alternative.",
            RuntimeWarning,
            stacklevel=2,
        )


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
