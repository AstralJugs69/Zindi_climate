from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import artifacts, gitops, state
from .config import artifact_handle, branch, is_kaggle, repo_root
from .data import hydrate
from .stages import STAGES, get as get_stage


def _install(root: Path) -> None:
    req = root / "requirements-kaggle.txt"
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-e", str(root)]
    if req.exists():
        cmd += ["-r", str(req)]
    subprocess.run(cmd, check=True, cwd=root)


def _maybe_restore(root: Path) -> None:
    handle = artifact_handle()
    if not handle:
        return
    try:
        artifacts.restore(root, handle)
    except Exception as exc:
        print(f"Checkpoint restore skipped: {exc}")


def _maybe_snapshot(root: Path, notes: str) -> None:
    handle = artifact_handle()
    if not handle:
        return
    try:
        artifacts.snapshot(root, handle, notes)
    except Exception as exc:
        print(f"Checkpoint upload failed: {exc}")


def cmd_bootstrap(args: argparse.Namespace) -> int:
    root = repo_root()
    if not args.skip_update and gitops.is_repo(root) and gitops.remote_url(root):
        gitops.update(root, args.branch or branch())
    _install(root)
    hydrate(root, Path(args.input_root))
    if not args.skip_restore:
        _maybe_restore(root)
    state.save(root, status="ready", git_commit=gitops.current_commit(root))
    print(f"Ready: {root}")
    print("Next: crhp status  |  crhp run baseline  |  crhp resume")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    root = repo_root()
    gitops.update(root, args.branch or branch())
    if not args.no_install:
        _install(root)
    state.save(root, git_commit=gitops.current_commit(root))
    return 0


def cmd_hydrate(args: argparse.Namespace) -> int:
    manifest = hydrate(repo_root(), Path(args.input_root))
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = repo_root()
    payload = {
        "root": str(root),
        "kaggle": is_kaggle(),
        "git": gitops.is_repo(root),
        "branch": gitops.current_branch(root) if gitops.is_repo(root) else None,
        "commit": gitops.current_commit(root),
        "remote": gitops.remote_url(root) if gitops.is_repo(root) else None,
        "artifact_dataset": artifact_handle(),
        "state": state.load(root),
        "stages": sorted(STAGES),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_stage(root: Path, stage_name: str, *, auto_snapshot: bool) -> int:
    command = get_stage(stage_name)
    state.save(
        root,
        status="running",
        stage=stage_name,
        resume_command=["crhp", "run", stage_name, "--no-update"],
        git_commit=gitops.current_commit(root),
    )
    try:
        result = subprocess.run(command, cwd=root)
    except KeyboardInterrupt:
        state.save(root, status="interrupted", stage=stage_name)
        raise

    if result.returncode != 0:
        state.save(root, status="failed", stage=stage_name, returncode=result.returncode)
        return result.returncode

    state.save(root, status="completed", stage=stage_name, returncode=0)
    if auto_snapshot:
        _maybe_snapshot(root, f"completed {stage_name}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = repo_root()
    if not args.no_update and gitops.is_repo(root) and gitops.remote_url(root):
        gitops.update(root, branch())
    hydrate(root, Path(args.input_root))
    return _run_stage(root, args.stage, auto_snapshot=not args.no_snapshot)


def cmd_resume(args: argparse.Namespace) -> int:
    root = repo_root()
    if gitops.is_repo(root) and gitops.remote_url(root):
        gitops.update(root, args.branch or branch())
    hydrate(root, Path(args.input_root))
    _maybe_restore(root)

    saved = state.load(root)
    status = saved.get("status")
    stage_name = saved.get("stage")
    if status in {"running", "failed", "interrupted"} and stage_name in STAGES:
        print(f"Resuming interrupted stage: {stage_name}")
        return _run_stage(root, stage_name, auto_snapshot=not args.no_snapshot)

    print("No interrupted stage found. Workspace is synchronized and ready.")
    if stage_name:
        print(f"Last stage: {stage_name} ({status})")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    root = repo_root()
    gitops.sync(root, args.message, args.branch or branch())
    state.save(root, git_commit=gitops.current_commit(root))
    if args.snapshot:
        handle = artifact_handle()
        if not handle:
            raise RuntimeError("Set CRHP_ARTIFACT_DATASET before using --snapshot")
        artifacts.snapshot(root, handle, args.message)
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    handle = args.handle or artifact_handle()
    if not handle:
        raise RuntimeError("Set CRHP_ARTIFACT_DATASET or pass --handle")
    artifacts.snapshot(repo_root(), handle, args.notes)
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    handle = args.handle or artifact_handle()
    if not handle:
        raise RuntimeError("Set CRHP_ARTIFACT_DATASET or pass --handle")
    artifacts.restore(repo_root(), handle)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crhp", description="Kaggle-first competition control plane")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bootstrap", help="Update, install, hydrate data, and restore state")
    b.add_argument("--branch")
    b.add_argument("--input-root", default="/kaggle/input")
    b.add_argument("--skip-update", action="store_true")
    b.add_argument("--skip-restore", action="store_true")
    b.set_defaults(func=cmd_bootstrap)

    u = sub.add_parser("update", help="Pull latest code and refresh the editable install")
    u.add_argument("--branch")
    u.add_argument("--no-install", action="store_true")
    u.set_defaults(func=cmd_update)

    h = sub.add_parser("hydrate", help="Discover attached Kaggle competition files")
    h.add_argument("--input-root", default="/kaggle/input")
    h.set_defaults(func=cmd_hydrate)

    s = sub.add_parser("status", help="Show code/data/runtime state")
    s.set_defaults(func=cmd_status)

    r = sub.add_parser("run", help="Update then run a named competition stage")
    r.add_argument("stage", choices=sorted(STAGES))
    r.add_argument("--input-root", default="/kaggle/input")
    r.add_argument("--no-update", action="store_true")
    r.add_argument("--no-snapshot", action="store_true")
    r.set_defaults(func=cmd_run)

    rs = sub.add_parser("resume", help="Update, restore, and resume an interrupted stage")
    rs.add_argument("--branch")
    rs.add_argument("--input-root", default="/kaggle/input")
    rs.add_argument("--no-snapshot", action="store_true")
    rs.set_defaults(func=cmd_resume)

    sy = sub.add_parser("sync", help="Pull/rebase, commit local code edits, and push")
    sy.add_argument("-m", "--message", required=True)
    sy.add_argument("--branch")
    sy.add_argument("--snapshot", action="store_true")
    sy.set_defaults(func=cmd_sync)

    sn = sub.add_parser("snapshot", help="Persist models/submissions/state to a Kaggle Dataset")
    sn.add_argument("--handle")
    sn.add_argument("--notes", default="competition checkpoint")
    sn.set_defaults(func=cmd_snapshot)

    re = sub.add_parser("restore", help="Restore the latest Kaggle Dataset checkpoint")
    re.add_argument("--handle")
    re.set_defaults(func=cmd_restore)
    return p


def main() -> None:
    args = parser().parse_args()
    try:
        raise SystemExit(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

