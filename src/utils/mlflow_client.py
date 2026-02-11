import importlib
import sys
from pathlib import Path
from typing import List


def load_external_mlflow():
    """
    Load pip-installed mlflow package and avoid local shadowing from src/mlflow.
    """
    current_dir = str(Path(__file__).resolve().parent)
    parent_src_dir = str(Path(__file__).resolve().parents[1])
    removed_paths: List[str] = []

    for candidate in (current_dir, parent_src_dir):
        if candidate in sys.path:
            sys.path.remove(candidate)
            removed_paths.append(candidate)

    local_mod = sys.modules.get("mlflow")
    if local_mod is not None:
        mod_file = getattr(local_mod, "__file__", "")
        if mod_file and str(Path(mod_file).resolve()).startswith(parent_src_dir):
            del sys.modules["mlflow"]

    external_mlflow = importlib.import_module("mlflow")

    for candidate in reversed(removed_paths):
        sys.path.insert(0, candidate)

    return external_mlflow
