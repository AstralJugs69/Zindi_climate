from __future__ import annotations

import base64
import os


def secret(name: str) -> str | None:
    """Read a secret without requiring Kaggle outside Kaggle.

    Precedence: environment variable, then Kaggle's UserSecretsClient.
    The value is never printed by this module.
    """
    value = os.getenv(name)
    if value:
        return value.strip()

    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        value = UserSecretsClient().get_secret(name)
        return value.strip() if value else None
    except Exception:
        return None


def github_auth_header() -> str | None:
    token = secret("GITHUB_TOKEN") or secret("GH_TOKEN")
    if not token:
        return None
    raw = f"x-access-token:{token}".encode("utf-8")
    return "AUTHORIZATION: Basic " + base64.b64encode(raw).decode("ascii")

