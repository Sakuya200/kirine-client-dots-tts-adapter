from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap
_bootstrap.setup()

from kirine_dots_tts.common import load_runtime, save_generated_audio  # noqa: E402
from kirine_dots_tts.params import load_voice_clone_params  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dots_tts voice cloning inference."
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
    """Run voice cloning generation and write the output WAV file."""
    text = (args.text or "").strip()
    if not text:
        raise ValueError("Text cannot be empty.")

    prompt_audio_path = (
        str(Path(args.prompt_audio_path).expanduser().resolve())
        if getattr(args, "prompt_audio_path", None)
        else None
    )
    if prompt_audio_path:
        prompt_audio = Path(prompt_audio_path)
        if not prompt_audio.exists():
            raise FileNotFoundError(
                f"Reference audio file not found: {prompt_audio}"
            )
    else:
        raise ValueError(
            "Voice cloning requires a prompt_audio_path (reference audio)."
        )

    prompt_text = (getattr(args, "prompt_text", None) or "").strip() or None

    num_steps = getattr(args, "num_steps", None) or 10
    guidance_scale = getattr(args, "guidance_scale", None) or 1.2
    speaker_scale = getattr(args, "speaker_scale", None) or 1.5
    language = (getattr(args, "language", None) or "").strip() or None

    print(f"[dots_tts] Voice clone text len={len(text)}")
    print(f"[dots_tts]   prompt_audio={prompt_audio_path}")
    print(f"[dots_tts]   prompt_text={prompt_text}")
    print(
        f"[dots_tts]   num_steps={num_steps} "
        f"guidance_scale={guidance_scale} "
        f"speaker_scale={speaker_scale} "
        f"language={language}"
    )

    mode_label = (
        "continuation cloning" if prompt_text
        else "x-vector-only cloning"
    )
    print(f"[dots_tts] Clone mode: {mode_label}")

    runtime = load_runtime(
        args.init_model_path,
        precision=getattr(args, "precision", "float32") or "float32",
        optimize=bool(getattr(args, "optimize", False)),
        max_generate_length=int(getattr(args, "max_generate_length", 500) or 500),
    )

    generate_kwargs: dict = {
        "text": text,
        "num_steps": int(num_steps),
        "guidance_scale": float(guidance_scale),
        "speaker_scale": float(speaker_scale),
        "language": language,
        "prompt_audio_path": prompt_audio_path,
    }
    if prompt_text:
        generate_kwargs["prompt_text"] = prompt_text

    result = runtime.generate(**generate_kwargs)

    save_generated_audio(args.output_path, result["audio"], result["sample_rate"])


def main(argv: list[str] | None = None) -> None:
    cli_args = parse_args(argv)
    params = load_voice_clone_params(cli_args.params_file)
    generate_audio(params.to_namespace())


if __name__ == "__main__":
    main()
