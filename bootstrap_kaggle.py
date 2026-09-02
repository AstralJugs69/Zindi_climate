"""Standalone Kaggle bootstrapper.

This file intentionally uses only the Python standard library so it can be
downloaded and executed before the project itself exists in /kaggle/working.
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_DEST = Path("/kaggle/working/climate-risk-health-prediction")


def secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        value = UserSecretsClient().get_secret(name)
        return value.strip() if value else None
    except Exception:
        return None


def git_prefix() -> list[str]:
    token = secret("GITHUB_TOKEN") or secret("GH_TOKEN")
    if not token:
        return ["git"]
    raw = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return ["git", "-c", f"http.extraHeader=AUTHORIZATION: Basic {raw}"]


def run(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap the CRHP repo on Kaggle")
    p.add_argument("--repo-url", default=os.getenv("CRHP_REPO_URL"))
    p.add_argument("--branch", default=os.getenv("CRHP_BRANCH", "main"))
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--input-root", default="/kaggle/input")
    args = p.parse_args()

    if not args.repo_url:
        raise SystemExit("Pass --repo-url or set CRHP_REPO_URL")

    dest = args.dest.resolve()
    if (dest / ".git").exists():
        run(git_prefix() + ["fetch", "origin", args.branch], cwd=dest)
        run(
            git_prefix()
            + ["pull", "--rebase", "--autostash", "origin", args.branch],
            cwd=dest,
        )
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(
            git_prefix()
            + [
                "clone",
                "--branch",
                args.branch,
                "--single-branch",
                args.repo_url,
                str(dest),
            ]
        )

    # Keep the remote clean: auth is supplied only per Git command, never in the URL.
    run(["git", "remote", "set-url", "origin", args.repo_url], cwd=dest)

    req = dest / "requirements-kaggle.txt"
    install = [sys.executable, "-m", "pip", "install", "-q", "-e", str(dest)]
    if req.exists():
        install += ["-r", str(req)]
    run(install, cwd=dest)

    env = os.environ.copy()
    env["CRHP_ROOT"] = str(dest)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "climateops.cli",
            "bootstrap",
            "--skip-update",
            "--input-root",
            args.input_root,
        ],
        cwd=dest,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()

