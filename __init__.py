"""kirine-client dots_tts script project.

This directory is loaded under the alias ``kirine_dots_tts`` (see
``_bootstrap.py``) so that it does not shadow the official ``dots_tts``
runtime library installed in the environment via ``pip install -e``.

Script-project helper modules (``common``, ``params``, ``dataset``,
``training_common``) are imported as ``kirine_dots_tts.xxx``.  Imports of
``dots_tts.xxx`` always resolve to the installed library.
"""
