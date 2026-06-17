from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch


def ensure_src_root_on_path() -> None:
    """Ensure the installed ``dots_tts`` Python package can be imported.

    The official dots.tts library is a dependency installed in the venv.
    This function is a no-op when the package is already importable.
    """
    try:
        import dots_tts  # noqa: F401
    except ImportError:
        raise ImportError(
            "dots_tts Python package is not installed. "
            "Please ensure it is installed in your environment "
            "(e.g. via 'pip install -e <dots.tts repo>')."
        ) from None


def load_runtime(
    model_path: str,
    *,
    precision: str = "float32",
    optimize: bool = False,
    max_generate_length: int = 500,
) -> Any:
    """Load a :class:`dots_tts.runtime.DotsTtsRuntime` for inference.

    This wraps the official :meth:`DotsTtsRuntime.from_pretrained` constructor.

    Args:
        model_path: Local directory containing a pretrained checkpoint
            (e.g. ``"<models>/dots.tts-soar"``).
        precision: ``"bfloat16"``, ``"float32"``, or ``"float16"``.
        optimize: Enable ``torch.compile`` acceleration.
        max_generate_length: Maximum audio patch count (≈audio length).

    Returns:
        An initialized :class:`DotsTtsRuntime` ready for ``generate()`` calls.
    """
    from dots_tts.runtime import DotsTtsRuntime

    resolved = Path(model_path).expanduser().resolve()
    print(f"[dots_tts] Loading runtime from: {resolved}")
    print(f"[dots_tts]   precision={precision} optimize={optimize}")

    runtime = DotsTtsRuntime.from_pretrained(
        model_name_or_path=str(resolved),
        precision=precision,
        optimize=optimize,
        max_generate_length=max_generate_length,
    )
    print(f"[dots_tts] Runtime loaded. sample_rate={runtime.sample_rate}")
    return runtime


def load_model_for_training(
    model_path: str,
) -> tuple[Any, Any]:
    """Load a :class:`dots_tts.models.dots_tts.model.DotsTtsModel` for training.

    Returns ``(model, tokenizer)`` where ``model`` is the pretrained
    :class:`DotsTtsModel` (in eval mode, on CPU) and ``tokenizer`` is the
    model's tokenizer.

    Args:
        model_path: Local directory containing a pretrained checkpoint
            (e.g. ``"<models>/dots.tts-base"``).
    """
    from dots_tts.models.dots_tts.model import DotsTtsModel

    resolved = Path(model_path).expanduser().resolve()
    print(f"[dots_tts] Loading training model from: {resolved}")

    model = DotsTtsModel.from_pretrained(model_name_or_path=str(resolved))
    tokenizer = model.tokenizer

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"[dots_tts] Model loaded.")
    print(f"[dots_tts]   Total params: {total_params:,}")
    print(f"[dots_tts]   Trainable params: {trainable_params:,}")
    print(f"[dots_tts]   Sample rate: {model.config.vocoder.sample_rate}")
    return model, tokenizer


def save_generated_audio(
    output_path: str,
    audio: torch.Tensor,
    sample_rate: int,
) -> None:
    """Save an audio tensor to a WAV file.

    Args:
        output_path: Destination ``.wav`` file path.
        audio: Audio waveform tensor (any shape — squeezed to 1-D).
        sample_rate: Sample rate in Hz.
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


def detect_device() -> str:
    """Return the best available torch device string.

    Returns ``"cuda"`` if CUDA is available, otherwise ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
