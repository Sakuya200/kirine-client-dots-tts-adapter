from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_root_on_path() -> Path:
    """Ensure the ``src-model/`` directory is on ``sys.path``.

    This allows scripts inside ``dots_tts/`` to import the installed
    ``dots_tts`` runtime package as well as sibling model packages.
    """
    src_root = Path(__file__).resolve().parents[1]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)
    return src_root


ensure_src_root_on_path()
