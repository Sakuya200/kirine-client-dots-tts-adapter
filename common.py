from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch


def current_python_executable() -> str:
    """Return the path to the current Python interpreter."""
    return sys.executable


def run_subprocess(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
) -> None:
    """Run a subprocess command, raising on failure."""
    print(f"[dots_tts] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def resolve_model_path(
    base_model_path: str,
    model_name: str,
) -> str:
    """Resolve a model path to a local directory or HuggingFace repo id.

    If ``base_model_path`` points to an existing directory containing
    ``model_name``, return that local path. Otherwise return the HF id.
    """
    candidate = Path(base_model_path).expanduser().resolve() / model_name
    if candidate.exists():
        return str(candidate)
    # Fall back to HuggingFace repo id pattern
    return f"rednote-hilab/{model_name}"


def load_runtime(
    model_path: str,
    *,
    precision: str = "bfloat16",
    optimize: bool = False,
    max_generate_length: int = 500,
) -> Any:
    """Load a :class:`dots_tts.runtime.DotsTtsRuntime` from a pretrained path.

    Args:
        model_path: Local directory or HuggingFace repo id
            (e.g. ``"rednote-hilab/dots.tts-soar"``).
        precision: One of ``"bfloat16"``, ``"float32"``, ``"float16"``.
        optimize: Enable ``torch.compile`` acceleration.
        max_generate_length: Maximum audio patch count for generation.

    Returns:
        An initialized :class:`DotsTtsRuntime` instance.
    """
    from dots_tts.runtime import DotsTtsRuntime

    resolved = Path(model_path).expanduser().resolve()
    print(f"[dots_tts] Loading runtime from: {resolved}")
    print(f"[dots_tts]   precision={precision} optimize={optimize}")

    runtime = DotsTtsRuntime.from_pretrained(
        str(resolved),
        precision=precision,
        optimize=optimize,
        max_generate_length=max_generate_length,
    )
    print(f"[dots_tts] Runtime loaded. sample_rate={runtime.sample_rate}")
    return runtime


def save_generated_audio(
    output_path: str,
    audio: torch.Tensor,
    sample_rate: int,
) -> None:
    """Save generated audio tensor to a WAV file.

    Args:
        output_path: Destination ``.wav`` path.
        audio: Audio waveform tensor (any shape — will be squeezed and
            moved to CPU float).
        sample_rate: Sample rate in Hz (typically 48000 for dots_tts).
    """
    waveform = audio.detach().float().cpu().squeeze().numpy()
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), waveform, sample_rate)
    duration_s = waveform.shape[-1] / sample_rate if waveform.ndim >= 1 else 0.0
    print(
        f"[dots_tts] Audio saved: {out} "
        f"({waveform.shape[-1]} samples, {duration_s:.2f}s @ {sample_rate} Hz)"
    )
