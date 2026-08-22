# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

"""Fork safety of `OAuthProvider`'s refresh lock.

`_refresh_token_if_needed` holds the lock across the user's `token_fetcher()`
call, and the Rust side invokes the provider from a tokio worker thread. So the
thread holding the lock at fork time is not the thread that forked, and a
`threading.Lock` held by a thread that does not exist in the child can never be
released there.
"""

import gc
import os
import subprocess
import sys
import weakref

import pytest
from lancedb.remote.header import (
    _LIVE_OAUTH_PROVIDERS,
    OAuthProvider,
    _reset_locks_after_fork,
)


def _provider() -> OAuthProvider:
    return OAuthProvider(lambda: {"access_token": "t", "expires_in": 3600})


def test_fork_handler_replaces_the_refresh_lock():
    """The handler is an ordinary function, so its effect is assertable
    without forking -- which also means this runs on Windows, where the
    handler itself is never registered."""
    provider = _provider()
    original = provider._refresh_lock

    _reset_locks_after_fork()

    assert provider._refresh_lock is not original
    assert not provider._refresh_lock.locked()


def test_fork_handler_replaces_a_lock_that_is_currently_held():
    """The case that matters: in a child, the lock is inherited locked and
    the thread that would release it does not exist. Replacing it must not
    depend on being able to acquire it first."""
    provider = _provider()
    provider._refresh_lock.acquire()
    held = provider._refresh_lock

    _reset_locks_after_fork()

    assert provider._refresh_lock is not held
    assert not provider._refresh_lock.locked()
    # The abandoned lock is still held; nothing tried to release it.
    assert held.locked()


def test_registry_does_not_keep_providers_alive():
    """Registration must not turn every provider into a leak."""
    provider = _provider()
    ref = weakref.ref(provider)
    assert any(p is provider for p in _LIVE_OAUTH_PROVIDERS)

    del provider
    gc.collect()

    # Asserting on the weakref rather than scanning the registry keeps this
    # independent of whatever providers other tests happen to leave behind.
    assert ref() is None


_FORK_REPRO = """
import os
import sys
import threading
import time

from lancedb.remote.header import OAuthProvider

provider = OAuthProvider(lambda: {"access_token": "t", "expires_in": 3600})

# Hold the lock from a thread that will not survive the fork, mirroring the
# real shape: the Rust side calls get_headers() on a tokio worker while the
# user's token_fetcher() is mid-request.
holding = threading.Event()

def hold_forever():
    provider._refresh_lock.acquire()
    holding.set()
    time.sleep(300)

threading.Thread(target=hold_forever, daemon=True).start()
holding.wait(timeout=10)

pid = os.fork()
if pid == 0:
    # Child: the inherited lock is held by a thread that does not exist here.
    # Without the at-fork handler this blocks forever.
    try:
        provider.get_headers()
    except BaseException:
        os._exit(2)
    os._exit(0)

_, status = os.waitpid(pid, 0)
sys.exit(0 if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else 1)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork()")
def test_child_can_refresh_after_forking_while_the_lock_is_held(tmp_path):
    """End-to-end: fork with the lock held by a thread that vanishes, then
    use the provider in the child.

    Run in a subprocess with a timeout this test can enforce -- a regression
    deadlocks the child, and waiting on it in-process would hang pytest.
    """
    script = tmp_path / "_fork_repro.py"
    script.write_text(_FORK_REPRO)

    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"child deadlocked on the inherited refresh lock "
            f"(stdout={exc.stdout!r} stderr={exc.stderr!r})",
            pytrace=False,
        )

    assert result.returncode == 0, (
        f"child failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
