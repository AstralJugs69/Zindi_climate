from __future__ import annotations

import os
from pathlib import Path

from .auth import github_auth_header
from .process import capture, run


class GitError(RuntimeError):
    pass


def _auth_prefix() -> list[str]:
    header = github_auth_header()
    return ["git", "-c", f"http.extraHeader={header}"] if header else ["git"]


def is_repo(root: Path) -> bool:
    return (root / ".git").exists()


def current_branch(root: Path) -> str:
    return capture(["git", "branch", "--show-current"], cwd=root) or "main"


def current_commit(root: Path) -> str | None:
    try:
        return capture(["git", "rev-parse", "HEAD"], cwd=root)
    except Exception:
        return None


def remote_url(root: Path, remote: str = "origin") -> str | None:
    try:
        return capture(["git", "remote", "get-url", remote], cwd=root)
    except Exception:
        return None


def clone(repo_url: str, destination: Path, branch: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = _auth_prefix() + [
        "clone",
        "--branch",
        branch,
        "--single-branch",
        repo_url,
        str(destination),
    ]
    run(cmd)


def update(root: Path, branch: str | None = None) -> None:
    if not is_repo(root):
        raise GitError(f"{root} is not a Git repository")
    target = branch or current_branch(root)
    prefix = _auth_prefix()
    run(prefix + ["fetch", "origin", target], cwd=root)
    # Autostash protects small local edits while still making update deterministic.
    run(prefix + ["pull", "--rebase", "--autostash", "origin", target], cwd=root)


def sync(root: Path, message: str, branch: str | None = None) -> None:
    if not is_repo(root):
        raise GitError(f"{root} is not a Git repository")

    target = branch or current_branch(root)
    # Rebase first so a long-running Kaggle session does not blindly overwrite newer work.
    update(root, target)
    run(["git", "add", "-A"], cwd=root)

    status = capture(["git", "status", "--porcelain"], cwd=root)
    if status:
        run(["git", "commit", "-m", message], cwd=root)
    else:
        print("No code changes to commit.")

    prefix = _auth_prefix()
    run(prefix + ["push", "origin", target], cwd=root)


def set_clean_remote(root: Path, repo_url: str) -> None:
    """Store only a token-free URL in .git/config."""
    existing = remote_url(root)
    if existing:
        run(["git", "remote", "set-url", "origin", repo_url], cwd=root)
    else:
        run(["git", "remote", "add", "origin", repo_url], cwd=root)

