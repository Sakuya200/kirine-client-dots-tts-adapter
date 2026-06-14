from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class DotsTtsDataset(Dataset):
    """Dataset that reads a JSONL manifest of training samples.

    Each line in the manifest is a JSON object with the following fields:

    .. code-block:: json

        {"fid": "sample-0001", "audio": "/abs/path/to/audio.wav", "text": "hello world"}

    The ``audio`` field must be an absolute path to a WAV file.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Training manifest not found: {self.manifest_path}"
            )
        self._entries: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                # Validate required fields
                for key in ("fid", "audio", "text"):
                    if key not in entry:
                        raise ValueError(
                            f"Missing required field '{key}' in manifest "
                            f"line: {stripped[:120]}"
                        )
                # Validate audio path exists
                audio_path = Path(entry["audio"]).expanduser().resolve()
                entry["audio"] = str(audio_path)
                self._entries.append(entry)

        if not self._entries:
            raise ValueError(
                f"Training manifest is empty: {self.manifest_path}"
            )
        print(
            f"[dots_tts] Loaded {len(self._entries)} samples from "
            f"{self.manifest_path}"
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._entries[idx]

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
        """Simple collate that groups items into lists.

        The actual tokenization and batching is handled inside the
        dots_tts training pipeline (``DotsTtsCore.forward``).
        """
        return {
            "fid": [item["fid"] for item in batch],
            "audio": [item["audio"] for item in batch],
            "text": [item["text"] for item in batch],
        }
