from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from torch.utils.data import DataLoader

from dots_tts.dataset import DotsTtsDataset


@dataclass
class TrainingRuntimeOptions:
    """Options controlling accelerator and model loading configuration."""

    is_cpu: bool
    accelerator_kwargs: dict[str, Any]
    model_load_kwargs: dict[str, Any]
    mode_label: str


@dataclass
class TrainingPipelineContext:
    """All objects created during training pipeline initialization."""

    runtime: TrainingRuntimeOptions
    accelerator: Any
    model: Any  # DotsTtsModel / DotsTtsCore
    optimizer: Any
    scheduler: Any | None
    train_dataloader: Any
    train_data: Any  # DotsTtsDataset


def is_cpu_device(device: str) -> bool:
    """Return True when ``device`` indicates CPU-only execution."""
    return (device or "").strip().lower() in {"cpu", ""}


def add_common_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register CLI arguments shared by all training modes."""
    parser.add_argument(
        "--params-file", dest="params_file", type=str, required=True,
        help="Path to a JSON params file produced by the kirine-client UI.",
    )
    return parser


def enable_gradient_checkpointing(model: Any) -> None:
    """Enable gradient checkpointing on the model if supported.

    For dots_tts, this applies to the internal LLM backbone (Qwen2).
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()


def disable_use_cache_for_training(model: Any) -> None:
    """Disable KV-cache for the internal LLM during training."""
    core = getattr(model, "core", None)
    if core is None:
        return
    llm = getattr(core, "llm", None)
    if llm is not None:
        config = getattr(llm, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False


def load_training_dependencies() -> SimpleNamespace:
    """Lazy-load heavy training dependencies.

    Returns a namespace with ``torch``, ``Accelerator``, ``AdamW``,
    and the installed ``dots_tts`` package.
    """
    import torch
    from accelerate import Accelerator

    deps = SimpleNamespace(
        torch=torch,
        Accelerator=Accelerator,
    )
    # Best-effort: import AdamW from transformers if available
    try:
        from torch.optim import AdamW
        deps.AdamW = AdamW
    except ImportError:
        from transformers import AdamW
        deps.AdamW = AdamW

    return deps


def build_train_dataloader(
    args: argparse.Namespace,
    *,
    manifest_path: str | None = None,
) -> tuple[DotsTtsDataset, DataLoader]:
    """Build a :class:`DataLoader` from a JSONL training manifest.

    The manifest path is resolved from ``args.train_manifest`` or from the
    ``manifest_path`` keyword argument.

    Returns ``(dataset, dataloader)``.
    """
    resolved_manifest = Path(
        manifest_path or getattr(args, "train_manifest", "")
    ).expanduser().resolve()

    if not resolved_manifest.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {resolved_manifest}"
        )

    dataset = DotsTtsDataset(str(resolved_manifest))
    dataloader = DataLoader(
        dataset,
        batch_size=int(getattr(args, "batch_size", None) or 1),
        shuffle=True,
        collate_fn=DotsTtsDataset.collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    return dataset, dataloader


def build_runtime_options(
    args: argparse.Namespace,
    torch_module: Any,
) -> TrainingRuntimeOptions:
    """Build :class:`TrainingRuntimeOptions` from CLI args."""
    mixed_precision = (getattr(args, "mixed_precision", None) or "bf16").strip()
    gradient_accumulation_steps = int(
        getattr(args, "gradient_accumulation_steps", None) or 1
    )
    logging_dir = getattr(args, "logging_dir", None)

    accelerator_kwargs: dict[str, Any] = {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "log_with": "tensorboard",
    }
    if logging_dir:
        accelerator_kwargs["project_dir"] = logging_dir

    if is_cpu_device(getattr(args, "device", "cpu")):
        return TrainingRuntimeOptions(
            is_cpu=True,
            accelerator_kwargs={
                **accelerator_kwargs,
                "mixed_precision": "no",
                "cpu": True,
            },
            model_load_kwargs={
                "precision": "float32",
            },
            mode_label="full fine-tune (CPU)",
        )

    return TrainingRuntimeOptions(
        is_cpu=False,
        accelerator_kwargs={
            **accelerator_kwargs,
            "mixed_precision": mixed_precision,
        },
        model_load_kwargs={
            "precision": "bfloat16",
        },
        mode_label="full fine-tune",
    )


def build_optimizer(
    model: Any,
    args: argparse.Namespace,
    deps: SimpleNamespace,
) -> Any:
    """Build an AdamW optimizer for training."""
    learning_rate = float(getattr(args, "learning_rate", None) or 2e-5)
    weight_decay = float(getattr(args, "weight_decay", None) or 0.1)
    return deps.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(
    optimizer: Any,
    args: argparse.Namespace,
    num_training_steps: int,
) -> Any | None:
    """Build a learning rate scheduler with warmup."""
    warmup_steps = getattr(args, "warmup_steps", None)
    warmup_ratio = getattr(args, "warmup_ratio", None)

    if warmup_steps is not None and int(warmup_steps) > 0:
        num_warmup_steps = int(warmup_steps)
    elif warmup_ratio is not None:
        num_warmup_steps = int(float(warmup_ratio) * num_training_steps)
    else:
        num_warmup_steps = 0

    import torch
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps and num_warmup_steps > 0:
            return float(current_step) / float(max(num_warmup_steps, 1))
        progress = float(current_step - num_warmup_steps) / float(
            max(num_training_steps - num_warmup_steps, 1)
        )
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    return LambdaLR(optimizer, lr_lambda)
