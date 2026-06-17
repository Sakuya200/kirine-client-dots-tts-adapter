"""Load this script project under the alias ``kirine_dots_tts``.

The script project directory ``<src-model>/dots_tts/`` shares the package
name ``dots_tts`` with the official ``dots.tts`` runtime library that
``download.py`` installs into the environment via ``pip install -e``.

The library's own modules use absolute imports (``from dots_tts.xxx import
...``) everywhere, so ``dots_tts`` *must* keep resolving to the installed
library — it cannot be aliased away.  To avoid the collision we instead
load this script project as a separate package under the alias
``kirine_dots_tts``.  Script-project helper modules (``common``, ``params``,
``dataset``, ``training_common``) are then imported as
``kirine_dots_tts.xxx``, while ``dots_tts`` continues to point at the
installed library.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ALIAS = "kirine_dots_tts"


def setup() -> None:
    """Register this script project as the ``kirine_dots_tts`` package."""
    if ALIAS in sys.modules:
        return

    pkg_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        ALIAS,
        pkg_dir / "__init__.py",
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load script project as {ALIAS!r}")

    package = importlib.util.module_from_spec(spec)
    sys.modules[ALIAS] = package
    spec.loader.exec_module(package)
