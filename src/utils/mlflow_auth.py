"""Shared MLflow/DagShub authentication utilities."""

import os
from pathlib import Path
from typing import List


def load_env_token(
    token_env_key: str = "DAGSHUB_TOKEN",
    env_file_paths: List[str] | None = None,
) -> str:
    """Load a token from environment variable or .env file.

    Args:
        token_env_key: Environment variable name to check first.
        env_file_paths: List of .env file paths to search as fallback.

    Returns:
        The token string.

    Raises:
        ValueError: If the token is not found anywhere.
    """
    token = os.getenv(token_env_key, "").strip()
    if token:
        return token

    if env_file_paths is None:
        env_file_paths = [".env", "/app/.env"]

    for env_path in env_file_paths:
        p = Path(env_path)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == token_env_key:
                token = value.strip().strip('"').strip("'")
                if token:
                    return token

    raise ValueError(f"{token_env_key} not found. Set it as an environment variable or in .env file.")


def setup_mlflow(
    tracking_uri: str,
    username: str,
    token_env_key: str = "DAGSHUB_TOKEN",
    env_file_paths: List[str] | None = None,
):
    """Configure MLflow tracking URI and authentication.

    Returns:
        The mlflow module (for convenience).
    """
    import mlflow

    token = load_env_token(token_env_key, env_file_paths)
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow
