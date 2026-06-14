from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import torch
from torch.utils.data import Dataset


class DotsTtsTrainingDataset(Dataset):
    """Dataset that reads a JSONL manifest and produces samples for training.

    Each line in the manifest is a JSON object:

    .. code-block:: json

        {
            "fid": "sample-0001",
            "audio": "/abs/path/to/audio.wav",
            "text": "hello world",
            "language": "en"
        }

    The ``audio`` field must be an absolute path to a WAV file.
    The ``text`` field is the transcription.
    The ``language`` field is optional (defaults to auto-detect).

    Each ``__getitem__`` returns a raw dict with fid, audio_path, and text.
    Tokenization and feature extraction happen in the training pipeline
    (via :meth:`DotsTtsModel.prepare_training_batch`).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Training manifest not found: {self.manifest_path}"
            )
        self.sample_rate = sample_rate
        self._entries: list[dict[str, Any]] = []

        with self.manifest_path.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                for key in ("fid", "audio", "text"):
                    if key not in entry:
                        raise ValueError(
                            f"Missing required field '{key}' in manifest "
                            f"line {line_num}: {stripped[:120]}"
                        )
                audio_path = Path(entry["audio"]).expanduser().resolve()
                if not audio_path.exists():
                    raise FileNotFoundError(
                        f"Audio file not found: {audio_path} "
                        f"(manifest line {line_num})"
                    )
                entry["audio"] = str(audio_path)
                self._entries.append(entry)

        if not self._entries:
            raise ValueError(
                f"Training manifest has no valid entries: {self.manifest_path}"
            )
        print(
            f"[dots_tts] Loaded {len(self._entries)} training samples "
            f"from {self.manifest_path}"
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a raw sample dict with ``fid``, ``audio``, ``text``.

        The actual tokenization and feature extraction is done by the
        training pipeline (via :meth:`DotsTtsModel.prepare_training_batch`
        and :class:`PadCollator`).
        """
        return self._entries[idx]


class DotsTtsInMemoryDataset(Dataset):
    """Dataset that loads audio waveforms into memory.

    Preloads all audio files and tokenizes text upfront so training
    batches only need padding/collation. This is the recommended format
    for single-GPU fine-tuning scenarios where dataset fits in RAM.

    Each ``__getitem__`` returns a dict compatible with the official
    :class:`dots_tts.data.collator.PadCollator`.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: Any,
        *,
        sample_rate: int = 48000,
    ) -> None:
        import numpy as np

        from dots_tts.data.pipelines.tokenizing import build_generation_schedule
        from dots_tts.data.pipelines.tts_pipeline import DEFAULT_TRAIN_TEMPLATE
        from dots_tts.utils.audio import high_quality_resample

        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Training manifest not found: {self.manifest_path}"
            )
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self._samples: list[dict[str, Any]] = []

        # Required tokens for schedule encoding
        self._boa_token_id = tokenizer.encode("<|audio_start|>")[0]
        self._eoa_token_id = tokenizer.encode("<|audio_end|>")[0]

        with self.manifest_path.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                for key in ("fid", "audio", "text"):
                    if key not in entry:
                        raise ValueError(
                            f"Missing required field '{key}' at line {line_num}"
                        )

                audio_path = Path(entry["audio"]).expanduser().resolve()
                if not audio_path.exists():
                    raise FileNotFoundError(
                        f"Audio file not found: {audio_path} (line {line_num})"
                    )

                # Load and resample audio
                audio, orig_sr = librosa.load(str(audio_path), sr=None, mono=True)
                audio = high_quality_resample(
                    torch.from_numpy(audio).unsqueeze(0),
                    orig_sr=orig_sr,
                    target_sr=sample_rate,
                )
                audio = audio.squeeze(0)  # (samples,)

                text = entry["text"].strip()
                language = entry.get("language", None)

                # Build generation schedule tokens to get audio/text token counts
                schedule_spec = build_generation_schedule(
                    text=text,
                    tokenizer=tokenizer,
                    template=DEFAULT_TRAIN_TEMPLATE,
                    max_audio_tokens=500,
                )
                schedule_ids = schedule_spec["schedule_ids"]

                # Find audio/text token boundaries
                num_text_tokens = 0
                num_audio_tokens = 0
                in_audio = False
                for tid in schedule_ids:
                    if tid == self._boa_token_id:
                        in_audio = True
                        continue
                    if tid == self._eoa_token_id:
                        in_audio = False
                        continue
                    if in_audio:
                        num_audio_tokens += 1
                    else:
                        num_text_tokens += 1

                # Build input_ids — encoded full schedule
                # Build labels — copy of input_ids (standard LM training)
                # Build loss_mask — 1.0 only for audio region, 0.0 elsewhere
                input_ids_list = list(schedule_ids)
                labels_list = list(schedule_ids)
                loss_mask_list = []
                in_audio_mask = False
                for tid in schedule_ids:
                    if tid == self._boa_token_id:
                        in_audio_mask = True
                        loss_mask_list.append(0.0)
                    elif tid == self._eoa_token_id:
                        loss_mask_list.append(1.0)
                        in_audio_mask = False
                    elif in_audio_mask:
                        loss_mask_list.append(1.0)
                    else:
                        loss_mask_list.append(0.0)

                sample_dict = {
                    "fid": entry["fid"],
                    "source_name": entry.get("source", "jsonl_manifest"),
                    "input_ids": input_ids_list,
                    "labels": labels_list,
                    "loss_mask": loss_mask_list,
                    "sample": audio.unsqueeze(0),  # (1, samples)
                    "sample_length": int(audio.shape[-1]),
                    "num_text_tokens": num_text_tokens,
                    "num_audio_tokens": num_audio_tokens,
                    "language": language,
                }

                # Also store text for tokenizer debug output
                sample_dict["_text"] = text

                self._samples.append(sample_dict)

        if not self._samples:
            raise ValueError(
                f"Training manifest has no valid entries: {self.manifest_path}"
            )

        total_duration = sum(
            s["sample_length"] / sample_rate for s in self._samples
        )
        print(
            f"[dots_tts] Loaded {len(self._samples)} samples "
            f"(total audio: {total_duration:.1f}s) "
            f"from {self.manifest_path}"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._samples[idx]
