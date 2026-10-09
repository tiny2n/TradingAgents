"""Run a CLI subprocess with a deadline, an output cap and process-group cleanup.

Ported from TradingAgents-KR (llm_clients/process_runner.py). No shell is
involved: the command is an argv list.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time


class CLIOutputLimitError(RuntimeError):
    """The subprocess wrote more than its output allowance."""


def _kill_process_group(process) -> None:
    try:
        if os.name == "posix":
            # The child runs in its own session, never the caller's group.
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)


def run_cli_subprocess(cmd, *, input: str, timeout: float, cwd=None, env=None,
                       max_output_bytes: int = 4_000_000,
                       watched_output_path: str | None = None) -> subprocess.CompletedProcess:
    """Run ``cmd`` with ``input`` on stdin and return its decoded stdout and stderr.

    Stdin comes from a temporary file, so the child sees EOF without a
    half-written pipe. Output goes to temporary files checked every 50 ms against
    ``max_output_bytes`` (and ``watched_output_path``, a file the CLI writes its
    answer to). On timeout, overflow or completion every process in the child's
    group is killed, so a CLI that spawns helpers leaves nothing behind.
    """
    if timeout <= 0:
        raise subprocess.TimeoutExpired(cmd, timeout)
    deadline = time.monotonic() + timeout
    with (tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stdout,
          tempfile.TemporaryFile() as stderr):
        stdin.write(input.encode("utf-8"))
        stdin.seek(0)
        process = subprocess.Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr,
                                   cwd=cwd, env=env, start_new_session=os.name == "posix")
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                try:
                    process.wait(timeout=min(remaining, 0.05))
                    completed = True
                except subprocess.TimeoutExpired:
                    completed = False
                sizes = [os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size]
                if watched_output_path and os.path.exists(watched_output_path):
                    sizes.append(os.stat(watched_output_path).st_size)
                if any(size > max_output_bytes for size in sizes):
                    raise CLIOutputLimitError("CLI output exceeded its byte budget")
                if completed:
                    break
            stdout.seek(0)
            stderr.seek(0)
            out = stdout.read(max_output_bytes + 1).decode("utf-8", errors="replace")
            err = stderr.read(max_output_bytes + 1).decode("utf-8", errors="replace")
            return subprocess.CompletedProcess(cmd, process.returncode, out, err)
        finally:
            _kill_process_group(process)
