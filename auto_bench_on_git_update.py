#!/usr/bin/env python3
"""Poll origin every minute and run the WeLMv4 prefill-Attention NPU benchmark.

Put this file in the root of the ``testfiles`` Git repository, then run:

    python auto_bench_on_git_update.py

The process must be started from the Python/Conda environment that can run the
benchmark. Press Ctrl+C to stop it.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


REMOTE = "origin"
BENCHMARK_SCRIPT = "bench_welmv4_prefill_attention_npu.py"
RESULT_DIR = "welmv4_prefill_attention_results"
ERROR_LOG = "welmv4_prefill_attention_run_error.log"
DEFAULT_INTERVAL_SECONDS = 60
AUTO_COMMIT_MARKER = "Auto-Benchmark: true"


class GitCommandError(RuntimeError):
    """Raised when a required Git command fails."""


@dataclass(frozen=True)
class PendingPush:
    branch: str
    base_sha: str
    commit_sha: str
    created_commit: bool


REPO = Path(__file__).resolve().parent
BENCHMARK_PATH = REPO / BENCHMARK_SCRIPT
RESULT_PATH = REPO / RESULT_DIR
ERROR_PATH = REPO / ERROR_LOG


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def run_command(
    command: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=REPO,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture_output,
        check=False,
    )
    if check and result.returncode != 0:
        rendered = shlex.join(command)
        detail = (result.stderr or result.stdout or "no error output").strip()
        raise GitCommandError(f"command failed ({result.returncode}): {rendered}\n{detail}")
    return result


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command(["git", *args], check=check)


def git_text(*args: str) -> str:
    return git(*args).stdout.strip()


def current_branch() -> str:
    branch = git_text("symbolic-ref", "--quiet", "--short", "HEAD")
    if not branch:
        raise GitCommandError("HEAD is detached; check out a branch before starting the monitor")
    return branch


def fetch(branch: str) -> None:
    git("fetch", "--quiet", REMOTE, branch)


def remote_sha(branch: str) -> str:
    return git_text("rev-parse", f"refs/remotes/{REMOTE}/{branch}")


def is_ancestor(older: str, newer: str) -> bool:
    return git("merge-base", "--is-ancestor", older, newer, check=False).returncode == 0


def pull_if_updated(branch: str) -> str | None:
    """Fast-forward to a newer remote commit and return its SHA."""
    fetch(branch)
    local = git_text("rev-parse", "HEAD")
    remote = remote_sha(branch)

    if local == remote:
        return None

    if is_ancestor(local, remote):
        log(f"remote update found: {local[:12]} -> {remote[:12]}")
        git("pull", "--ff-only", REMOTE, branch)
        pulled = git_text("rev-parse", "HEAD")
        log(f"git pull completed at {pulled[:12]}")
        return pulled

    if is_ancestor(remote, local):
        log("local branch is ahead of origin; no new remote commit to run")
        return None

    raise GitCommandError(
        "local and remote branches have diverged; resolve them manually before restarting"
    )


def benchmark_command(device: str, output_dir: Path) -> list[str]:
    python_executable = os.environ.get("BENCH_PYTHON", sys.executable)
    return [
        python_executable,
        BENCHMARK_SCRIPT,
        "--suite",
        "remote",
        "--mode",
        "both",
        "--device",
        device,
        "--capture-ir",
        "on",
        "--capture-profile",
        "on",
        "--capture-msprof-op",
        "on",
        "--output-dir",
        str(output_dir),
    ]


def write_error_log(command: Sequence[str], returncode: int, output: str, reason: str) -> None:
    content = (
        f"time: {now()}\n"
        f"command: {shlex.join(command)}\n"
        f"return_code: {returncode}\n"
        f"reason: {reason}\n\n"
        "===== stdout + stderr =====\n"
        f"{output}"
    )
    if output and not output.endswith("\n"):
        content += "\n"

    temporary_path = ERROR_PATH.with_suffix(ERROR_PATH.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, ERROR_PATH)


def remove_result_dir() -> None:
    resolved = RESULT_PATH.resolve()
    if resolved.parent != REPO or resolved.name != RESULT_DIR:
        raise RuntimeError(f"refusing to remove unexpected result path: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def run_benchmark(device: str) -> bool:
    """Run once and preserve every valid result, including failed/regressed runs."""
    remove_result_dir()

    returncode = -1
    captured = ""
    launch_error = ""
    with tempfile.TemporaryDirectory(prefix="welm_attn_result_", dir=REPO) as tmp:
        staging = Path(tmp)
        command = benchmark_command(device, staging)
        log(f"starting benchmark: {shlex.join(command)}")
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stream:
            try:
                result = subprocess.run(
                    command,
                    cwd=REPO,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                returncode = result.returncode
            except OSError as exc:
                launch_error = f"could not start benchmark: {exc}"
            finally:
                stream.seek(0)
                captured = stream.read()

        staging_manifest = staging / "result.json"
        manifest_status = ""
        if staging_manifest.is_file() and staging_manifest.stat().st_size > 0:
            try:
                manifest = json.loads(staging_manifest.read_text(encoding="utf-8"))
                if isinstance(manifest, dict):
                    manifest_status = str(manifest.get("status", ""))
            except (OSError, json.JSONDecodeError):
                manifest_status = ""
        if manifest_status in {"PASS", "FAIL", "PERF_REGRESSION", "ERROR"}:
            # Copy out of TemporaryDirectory explicitly.  Moving the staging
            # root left some failed runs without their CSV/IR/profile tree on
            # the remote worker; copytree makes artifact retention independent
            # of TemporaryDirectory cleanup semantics.
            # A freshly loaded benchmark child can also publish this directory
            # itself for compatibility with an older long-running monitor.
            # Replace that complete copy instead of raising EEXIST here.
            remove_result_dir()
            shutil.copytree(staging, RESULT_PATH)
        if returncode == 0 and manifest_status == "PASS" and not launch_error:
            ERROR_PATH.unlink(missing_ok=True)
            log(f"benchmark succeeded; generated {RESULT_DIR}")
            return True

    if launch_error:
        reason = launch_error
    elif returncode != 0:
        reason = (
            f"benchmark exited with status {returncode}; "
            f"manifest_status={manifest_status or 'missing'}"
        )
    else:
        reason = (
            "benchmark exited successfully but result.json was missing or not PASS; "
            f"manifest_status={manifest_status or 'missing'}"
        )
    write_error_log(command, returncode, captured, reason)
    log(f"benchmark failed; details written to {ERROR_LOG}")
    return False


def artifact_status_paths() -> list[str]:
    changed: list[str] = []
    for path in (RESULT_DIR, ERROR_LOG):
        if git("status", "--porcelain", "--", path).stdout.strip():
            changed.append(path)
    return changed


def commit_artifacts(branch: str, base_sha: str, succeeded: bool) -> PendingPush:
    changed = artifact_status_paths()
    created_commit = False

    if changed:
        git("add", "-A", "--", *changed)
        result_word = "success" if succeeded else "failure"
        message = (
            f"chore: update NPU benchmark {result_word}\n\n"
            f"Benchmark-Base: {base_sha}\n"
            f"{AUTO_COMMIT_MARKER}"
        )
        # --only prevents unrelated staged files from entering this commit.
        git("commit", "--only", "-m", message, "--", *changed)
        created_commit = True
        log(f"committed benchmark artifacts: {', '.join(changed)}")
    else:
        log("benchmark artifacts are unchanged; no new commit was needed")

    return PendingPush(
        branch=branch,
        base_sha=base_sha,
        commit_sha=git_text("rev-parse", "HEAD"),
        created_commit=created_commit,
    )


def restore_artifacts_from_head() -> None:
    """Discard only generated artifacts, leaving all unrelated files untouched."""
    for name, path in ((RESULT_DIR, RESULT_PATH), (ERROR_LOG, ERROR_PATH)):
        tracked = git("ls-files", "--error-unmatch", "--", name, check=False).returncode == 0
        if tracked:
            git("restore", "--source=HEAD", "--staged", "--worktree", "--", name)
        else:
            git("reset", "--quiet", "HEAD", "--", name, check=False)
            if path.is_dir():
                remove_result_dir()
            else:
                path.unlink(missing_ok=True)


def rewind_own_commit(pending: PendingPush) -> None:
    """Remove only the auto-commit made by this process after a push race."""
    if not pending.created_commit:
        return
    ref = f"refs/heads/{pending.branch}"
    git("update-ref", ref, pending.base_sha, pending.commit_sha)
    restore_artifacts_from_head()
    log("remote changed before push; discarded the stale auto-commit")


def try_push(pending: PendingPush) -> tuple[PendingPush | None, bool]:
    """Return (pending_push, retry_remote_immediately)."""
    result = git(
        "push",
        REMOTE,
        f"HEAD:refs/heads/{pending.branch}",
        check=False,
    )
    if result.returncode == 0:
        log(f"git push {REMOTE} completed")
        return None, False

    detail = (result.stderr or result.stdout or "no error output").strip()
    log(f"git push failed: {detail}")

    # Distinguish a concurrent remote update from a temporary network/auth error.
    try:
        fetch(pending.branch)
        latest_remote = remote_sha(pending.branch)
    except GitCommandError as exc:
        log(f"could not verify remote after push failure; will retry in one minute: {exc}")
        return pending, False

    if latest_remote != pending.base_sha:
        rewind_own_commit(pending)
        return None, True

    log("remote did not move; retaining the local result commit and retrying push in one minute")
    return pending, False


def detect_interrupted_pending_push(branch: str) -> PendingPush | None:
    """Recover an auto-commit left behind if this monitor was restarted."""
    message = git_text("log", "-1", "--format=%B")
    if AUTO_COMMIT_MARKER not in message:
        return None

    commit_sha = git_text("rev-parse", "HEAD")
    parent_result = git("rev-parse", "HEAD^", check=False)
    if parent_result.returncode != 0:
        return None

    base_sha = parent_result.stdout.strip()
    fetch(branch)
    latest_remote = remote_sha(branch)
    # It was already pushed if the remote equals it or contains it as an ancestor.
    if latest_remote == commit_sha or is_ancestor(commit_sha, latest_remote):
        return None

    log("found an unpushed automatic benchmark commit from an earlier monitor run")
    return PendingPush(branch, base_sha, commit_sha, True)


def validate_repository() -> None:
    if git("rev-parse", "--is-inside-work-tree", check=False).stdout.strip() != "true":
        raise GitCommandError(f"{REPO} is not inside a Git working tree")
    repository_root = Path(git_text("rev-parse", "--show-toplevel")).resolve()
    if repository_root != REPO:
        raise GitCommandError(
            f"put this monitor in the repository root: expected {repository_root}, got {REPO}"
        )
    if not BENCHMARK_PATH.is_file():
        raise FileNotFoundError(f"benchmark script not found: {BENCHMARK_PATH}")
    if git("remote", "get-url", REMOTE, check=False).returncode != 0:
        raise GitCommandError(f"Git remote {REMOTE!r} is not configured")
    for key in ("user.name", "user.email"):
        if not git("config", "--get", key, check=False).stdout.strip():
            raise GitCommandError(
                f"Git {key} is not configured; set it before automatic commits are made"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="poll interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="check once and exit; useful for testing the setup",
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help=(
            "benchmark the current synchronized HEAD once at startup, then "
            "continue monitoring"
        ),
    )
    parser.add_argument(
        "--device",
        default="npu:5",
        help="NPU device passed to the benchmark (default: npu:5)",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    return args


def main() -> int:
    args = parse_args()
    validate_repository()
    branch = current_branch()
    log(
        f"monitoring {REMOTE}/{branch} every {args.interval:g} seconds; "
        f"benchmark={BENCHMARK_SCRIPT}, device={args.device}, "
        f"artifact={RESULT_DIR}"
    )

    pending: PendingPush | None = detect_interrupted_pending_push(branch)
    # Recovering an already-produced pending result takes precedence; do not
    # immediately benchmark the automatic result commit itself after pushing.
    force_run = args.run_now and pending is None
    while True:
        retry_immediately = False
        try:
            if pending is not None:
                pending, retry_immediately = try_push(pending)
            else:
                base_sha = pull_if_updated(branch)
                if force_run and base_sha is None:
                    local = git_text("rev-parse", "HEAD")
                    remote = remote_sha(branch)
                    if local != remote:
                        raise GitCommandError(
                            "--run-now requires local HEAD to equal origin; "
                            "resolve or push the local-only commits first"
                        )
                    base_sha = local
                    log(f"--run-now selected current HEAD {base_sha[:12]}")
                force_run = False
                if base_sha is not None:
                    succeeded = run_benchmark(args.device)

                    # Do not publish a result produced from a commit that is no longer current.
                    fetch(branch)
                    if remote_sha(branch) != base_sha:
                        restore_artifacts_from_head()
                        log("remote changed during the benchmark; rerunning on the newest commit")
                        retry_immediately = True
                    else:
                        pending = commit_artifacts(branch, base_sha, succeeded)
                        pending, retry_immediately = try_push(pending)
        except (GitCommandError, OSError) as exc:
            log(f"cycle error: {exc}")

        if retry_immediately:
            continue
        if args.once:
            return 0 if pending is None else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("stopped by user")
        raise SystemExit(130)
