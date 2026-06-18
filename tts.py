from __future__ import annotations

import argparse

import _bootstrap
_bootstrap.setup()

from kirine_dots_tts.common import load_runtime, save_generated_audio  # noqa: E402
from kirine_dots_tts.params import load_tts_params  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dots_tts text-to-speech inference."
    )
    parser.add_argument(
        "--params-file",
        dest="params_file",
        type=str,
        required=True,
        help="Path to a JSON params file produced by the kirine-client UI.",
    )
    return parser.parse_args(argv)


def generate_audio(args: argparse.Namespace) -> None:
    """Run TTS generation and write the output WAV file."""
    text = (args.text or "").strip()
    if not text:
        raise ValueError("Text cannot be empty.")

    num_steps = getattr(args, "num_steps", None) or 10
    guidance_scale = getattr(args, "guidance_scale", None) or 1.2
    speaker_scale = getattr(args, "speaker_scale", None) or 1.5
    language = (getattr(args, "language", None) or "").strip() or None

    print(f"[dots_tts] TTS text len={len(text)}")
    print(
        f"[dots_tts]   num_steps={num_steps} "
        f"guidance_scale={guidance_scale} "
        f"speaker_scale={speaker_scale} "
        f"language={language}"
    )

    runtime = load_runtime(
        args.init_model_path,
        precision=getattr(args, "precision", "float32") or "float32",
        optimize=bool(getattr(args, "optimize", False)),
        max_generate_length=int(getattr(args, "max_generate_length", 500) or 500),
    )

    result = runtime.generate(
        text=text,
        num_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        speaker_scale=float(speaker_scale),
        language=language,
    )

    save_generated_audio(args.output_path, result["audio"], result["sample_rate"])


def main(argv: list[str] | None = None) -> None:
    cli_args = parse_args(argv)
    params = load_tts_params(cli_args.params_file)
    generate_audio(params.to_namespace())


if __name__ == "__main__":
    main()
