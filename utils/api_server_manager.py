"""
utils/api_server_manager.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ensures the Mock ERP REST API server is running before any agent that
depends on it is constructed.

Usage (add near the top of any entry-point, before agents are built):

    from utils.api_server_manager import ensure_api_server
    ensure_api_server()          # blocks only until the server is ready (≤10 s)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# One module-level sentinel so repeated calls within the same process are free.
_SERVER_CONFIRMED_RUNNING: bool = False

# Project root is two levels up from this file (utils/api_server_manager.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_HOST = "127.0.0.1"
_PORT = 8000
_HEALTH_URL = f"http://{_HOST}:{_PORT}/health"
_MAX_WAIT_SECONDS = 15
_POLL_INTERVAL = 0.5


def _is_server_up() -> bool:
    """Return True if the /health endpoint responds with HTTP 200."""
    try:
        # Use urllib so we have zero extra dependencies (requests may not be present)
        import urllib.request
        with urllib.request.urlopen(_HEALTH_URL, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _start_server() -> subprocess.Popen:
    """Launch run_server.py as a detached background process."""
    run_server = _PROJECT_ROOT / "mock_erp_api" / "run_server.py"

    if not run_server.exists():
        raise FileNotFoundError(
            f"Cannot find run_server.py at: {run_server}\n"
            "Make sure the mock_erp_api package is present."
        )

    # On Windows, CREATE_NEW_PROCESS_GROUP keeps the child alive after the
    # parent exits and prevents Ctrl-C from being forwarded to it.
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        [sys.executable, str(run_server)],
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )
    return proc


def is_api_server_up() -> bool:
    """Public helper — returns True if the /health endpoint responds with HTTP 200."""
    return _is_server_up()


def ensure_api_server(*, verbose: bool = True) -> None:
    """
    Check whether the Mock ERP REST API is reachable.  If not, start it and
    wait until it is ready (up to ``_MAX_WAIT_SECONDS`` seconds).

    Parameters
    ----------
    verbose:
        When True (default) print status messages to stdout.  Set to False to
        suppress all output (useful inside Streamlit where print goes nowhere).

    Raises
    ------
    RuntimeError
        If the server does not become healthy within the timeout window.
    """
    global _SERVER_CONFIRMED_RUNNING  # noqa: PLW0603

    # Fast path — already confirmed in this process
    if _SERVER_CONFIRMED_RUNNING:
        return

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # 1. Already running? (e.g. user started it manually, or another process did)
    if _is_server_up():
        _log(f"[API server] Already running at {_HEALTH_URL}")
        _SERVER_CONFIRMED_RUNNING = True
        return

    # 2. Not running — launch it
    _log("[API server] Not detected - starting mock_erp_api/run_server.py ...")
    _start_server()

    # 3. Poll until healthy or timeout
    deadline = time.monotonic() + _MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        if _is_server_up():
            _log(f"[API server] Ready at {_HEALTH_URL} (OK)")
            _SERVER_CONFIRMED_RUNNING = True
            return
        time.sleep(_POLL_INTERVAL)

    # 4. Timeout — raise so the caller knows something is wrong
    raise RuntimeError(
        f"[API server] Server did not become healthy within {_MAX_WAIT_SECONDS}s.\n"
        f"  Health endpoint: {_HEALTH_URL}\n"
        "  Check that uvicorn and fastapi are installed, and that port "
        f"{_PORT} is not already in use by another application."
    )