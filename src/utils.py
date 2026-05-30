"""
src/utils.py
------------
Shared utility helpers used across the project.
"""

from __future__ import annotations

import os
import sys
import warnings


def suppress_warnings() -> None:
    """Suppress all Python warnings globally (call once at notebook start)."""
    warnings.filterwarnings("ignore")


def add_src_to_path(project_root: str | None = None) -> None:
    """Prepend *project_root/src* to ``sys.path`` so notebook imports work.

    Call this at the top of ``notebooks/analysis.ipynb`` when running
    inside the ``notebooks/`` subdirectory:

    .. code-block:: python

        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path("..").resolve()))
        from src.utils import add_src_to_path
        add_src_to_path()

    Parameters
    ----------
    project_root:
        Absolute path to the project root directory.  When ``None``
        (default) the function assumes it is being imported from
        ``project-root/src/`` and walks two levels up.
    """
    if project_root is None:
        # __file__ = project_root/src/utils.py  →  two .parent calls
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added '{project_root}' to sys.path")


def ensure_dirs(*dirs: str) -> None:
    """Create each directory in *dirs* (including parents) if absent.

    Parameters
    ----------
    *dirs:
        Any number of directory paths.
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)