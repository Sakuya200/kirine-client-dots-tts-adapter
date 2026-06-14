from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from dots_tts.data.collator import PadCollator
from dots_tts.training import losses as loss_ops


# ---------------------------------------------------------------------------
# Training State
# ---------------------------------------------------------------------------


@dataclass
class TrainProgress:
    """Minimal progress counters (mirrors dots_tts.training.utils.TrainProgress)."""

    global_step: int = 0
    epoch: int = 0
    total_tokens: int = 0
    audio_tokens: int = 0
    text_tokens: int = 0


@dataclass
class TrainingContext:
    """All objects created during training pipeline initialization."""

    accelerator: Accelerator
    model: Any  # wrapped DotsTtsModel
    unwrapped_model: Any  # unwrapped DotsTtsModel
    tokenizer: Any
    optimizer: AdamW
    scheduler: Any
    train_loader: DataLoader
    progress: TrainProgress = field(default_factory=TrainProgress)
    max_train_steps: int = 0
    gradient_accumulation_steps: int = 1
    sample_rate: int = 48000


# ---------------------------------------------------------------------------
# CLI Helpers
# ---------------------------------------------------------------------------


def add_common_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the ``--params-file`` argument shared by all training entry points."""
    parser.add_argument(
        "--params-file",
        dest="params_file",
        type=str,
        required=True,
        help="Path to a JSON params file produced by the kirine-client UI.",
    )
    return parser


# ---------------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------------


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.1,
) -> AdamW:
    """Build an AdamW optimizer that only optimizes parameters requiring gradients."""
    return AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
) -> Any:
    """Build a cosine learning-rate scheduler with linear warmup."""
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


def build_train_dataloader(
    dataset: torch.utils.data.Dataset,
    tokenizer: Any,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    """Build a :class:`DataLoader` with the official dots.tts :class:`PadCollator`.

    The ``dataset`` must yield items compatible with ``PadCollator``
    (i.e. dicts with ``input_ids``, ``labels``, ``loss_mask``, ``sample``,
    ``sample_length``, ``num_text_tokens``, ``num_audio_tokens``, ``fid``).
    """
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=PadCollator(tokenizer),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Accelerator
# ---------------------------------------------------------------------------


def build_accelerator(
    *,
    gradient_accumulation_steps: int = 1,
    output_dir: str = "",
    mixed_precision: str = "no",
    cpu: bool = False,
    max_checkpoints_to_keep: int = 3,
) -> Accelerator:
    """Build an HF :class:`Accelerator` for training.

    Args:
        gradient_accumulation_steps: Number of steps to accumulate before
            an optimizer step.
        output_dir: Directory for TensorBoard logs and checkpoints.
        mixed_precision: ``"bf16"``, ``"fp16"``, or ``"no"``.
        cpu: If ``True``, runs entirely on CPU.
        max_checkpoints_to_keep: Max recent checkpoints to retain on disk.
    """
    if cpu:
        mixed_precision = "no"

    project_config = ProjectConfiguration(
        project_dir=output_dir,
        total_limit=int(max_checkpoints_to_keep),
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    return Accelerator(
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
        step_scheduler_with_optimizer=False,
        cpu=cpu,
    )


# ---------------------------------------------------------------------------
# Gradient Checkpointing (LLM backbone)
# ---------------------------------------------------------------------------


def disable_use_cache_for_training(model: torch.nn.Module) -> None:
    """Disable KV-cache in the underlying LLM backbone during training."""
    core = getattr(model, "core", None)
    if core is None:
        return
    llm = getattr(core, "llm", None)
    if llm is not None:
        config = getattr(llm, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Enable gradient checkpointing if supported by the model."""
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


# ---------------------------------------------------------------------------
# Loss helpers (delegate to official dots_tts.training.losses)
# ---------------------------------------------------------------------------


def compute_loss(
    loss_terms: dict[str, Any],
    global_normalizers: dict[str, float],
    ddp_world_size: int = 1,
    gradient_accumulation_steps: int = 1,
) -> torch.Tensor:
    """Compute the training loss from model outputs.

    Delegates to :func:`dots_tts.training.losses.compute_gradient_loss`.
    """
    from dots_tts.config import loss as loss_config_module

    # Use a minimal loss config — only total loss matters for
    # kirine-client fine-tuning.
    loss_config = loss_config_module.LossConfig()
    return loss_ops.compute_gradient_loss(
        loss_terms,
        global_normalizers=global_normalizers,
        loss_config=loss_config,
        ddp_world_size=int(ddp_world_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
    )
