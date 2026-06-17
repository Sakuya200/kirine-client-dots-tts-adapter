from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator

import _bootstrap
_bootstrap.setup()

from kirine_dots_tts.common import load_model_for_training  # noqa: E402
from kirine_dots_tts.dataset import DotsTtsInMemoryDataset  # noqa: E402
from kirine_dots_tts.params import load_training_params  # noqa: E402
from kirine_dots_tts.training_common import (  # noqa: E402
    TrainProgress,
    add_common_training_args,
    build_accelerator,
    build_optimizer,
    build_scheduler,
    build_train_dataloader,
    compute_loss,
    disable_use_cache_for_training,
    enable_gradient_checkpointing,
)  # noqa: E402

CHECKPOINT_MAX_SHARD_SIZE = "1GB"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Run dots_tts fine-tuning via kirine-client.",
    )
    parser = add_common_training_args(parser)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _save_training_checkpoint(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: str,
    epoch: int,
    global_step: int,
) -> None:
    """Save a fine-tuned checkpoint using the model's native ``save_pretrained``."""
    if not accelerator.is_main_process:
        return

    from huggingface_hub import save_torch_state_dict

    save_dir = Path(output_dir) / f"checkpoint-epoch-{epoch:03d}-step-{global_step:08d}"
    tmp_dir = save_dir.with_name(f"{save_dir.name}.tmp")
    model_dir = tmp_dir / "model"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(str(model_dir))

        state_dict = {}
        for key, value in unwrapped.state_dict().items():
            state_dict[key] = value.detach().cpu()

        save_torch_state_dict(
            state_dict,
            str(model_dir),
            max_shard_size=CHECKPOINT_MAX_SHARD_SIZE,
            safe_serialization=True,
            is_main_process=accelerator.is_main_process,
            shared_tensors_to_discard=getattr(
                unwrapped, "_tied_weights_keys", None
            ),
        )

        torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")

        if save_dir.exists():
            shutil.rmtree(save_dir)
        tmp_dir.rename(save_dir)

        accelerator.print(f"[dots_tts] Checkpoint saved: {save_dir}")
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Training Initialization
# ---------------------------------------------------------------------------


def _initialize_training(args: argparse.Namespace) -> dict[str, Any]:
    """Initialize model, dataset, optimizer, scheduler, and accelerator.

    Returns a dict with all objects needed for the training loop.
    """
    # 1. Load params from JSON
    params = load_training_params(args.params_file)
    ns = params.to_namespace()

    init_model_path = ns.init_model_path
    output_model_path = ns.output_model_path
    input_jsonl = getattr(ns, "input_jsonl", getattr(ns, "train_manifest", ""))
    batch_size = int(getattr(ns, "batch_size", 1))
    learning_rate = float(getattr(ns, "lr", 2e-5))
    num_epochs = int(getattr(ns, "num_epochs", 5))
    gradient_accumulation_steps = int(
        getattr(ns, "gradient_accumulation_steps", 1)
    )
    device = getattr(ns, "device", "cuda")
    logging_dir = getattr(ns, "logging_dir", output_model_path)
    enable_grad_ckpt = bool(
        getattr(ns, "enable_gradient_checkpointing", False)
    )
    warmup_steps = int(getattr(ns, "warmup_steps", 100))
    grad_clip_norm = float(getattr(ns, "gradient_clip_norm", 1.0))
    log_interval = int(getattr(ns, "log_interval", 10))
    is_cpu = (device or "cpu").strip().lower() == "cpu"
    mixed_precision = "bf16" if not is_cpu else "no"

    print(f"[dots_tts] Training configuration:")
    print(f"[dots_tts]   init_model_path={init_model_path}")
    print(f"[dots_tts]   output_model_path={output_model_path}")
    print(f"[dots_tts]   input_jsonl={input_jsonl}")
    print(f"[dots_tts]   batch_size={batch_size}")
    print(f"[dots_tts]   lr={learning_rate}")
    print(f"[dots_tts]   num_epochs={num_epochs}")
    print(f"[dots_tts]   gradient_accumulation_steps={gradient_accumulation_steps}")
    print(f"[dots_tts]   device={device} is_cpu={is_cpu}")
    print(f"[dots_tts]   enable_gradient_checkpointing={enable_grad_ckpt}")

    # 2. Load model
    model, tokenizer = load_model_for_training(init_model_path)
    sample_rate = int(model.config.vocoder.sample_rate)

    # 3. Gradient checkpointing
    if enable_grad_ckpt:
        disable_use_cache_for_training(model)
        enable_gradient_checkpointing(model)

    # 4. Load dataset
    dataset = DotsTtsInMemoryDataset(
        input_jsonl,
        tokenizer=tokenizer,
        sample_rate=sample_rate,
    )

    # 5. Build DataLoader
    train_loader: torch.utils.data.DataLoader = build_train_dataloader(
        dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=0,
    )

    num_batches_per_epoch = len(train_loader)
    max_train_steps = num_epochs * num_batches_per_epoch

    # 6. Build optimizer & scheduler
    optimizer = build_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=0.1,
    )
    scheduler = build_scheduler(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    # 7. Build accelerator
    accelerator = build_accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir=logging_dir,
        mixed_precision=mixed_precision,
        cpu=is_cpu,
    )

    # 8. Prepare with accelerator
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    unwrapped_model = accelerator.unwrap_model(model)

    accelerator.print(
        f"[dots_tts] Training initialized: "
        f"batches/epoch={num_batches_per_epoch} "
        f"epochs={num_epochs} "
        f"total_steps={max_train_steps}"
    )

    return {
        "accelerator": accelerator,
        "model": model,
        "unwrapped_model": unwrapped_model,
        "tokenizer": tokenizer,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "dataset": dataset,
        "num_epochs": num_epochs,
        "num_batches_per_epoch": num_batches_per_epoch,
        "max_train_steps": max_train_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "output_model_path": output_model_path,
        "grad_clip_norm": grad_clip_norm,
        "log_interval": log_interval,
    }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> None:
    """Execute the full fine-tuning pipeline.

    This is the main training orchestration function:
    1. Initialize model, dataset, optimizer, scheduler, accelerator.
    2. Run the training loop for the configured number of epochs.
    3. Save epoch checkpoints.
    """
    ctx = _initialize_training(args)

    accelerator: Accelerator = ctx["accelerator"]
    model = ctx["model"]
    unwrapped_model = ctx["unwrapped_model"]
    optimizer = ctx["optimizer"]
    scheduler = ctx["scheduler"]
    train_loader = ctx["train_loader"]
    num_epochs: int = ctx["num_epochs"]
    num_batches_per_epoch: int = ctx["num_batches_per_epoch"]
    max_train_steps: int = ctx["max_train_steps"]
    gradient_accumulation_steps: int = ctx["gradient_accumulation_steps"]
    output_model_path: str = ctx["output_model_path"]
    grad_clip_norm: float = ctx["grad_clip_norm"]
    log_interval: int = ctx["log_interval"]

    progress = TrainProgress()
    global_step = 0

    # Track per-epoch loss for progress reporting
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        epoch_start_time = time.perf_counter()

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                # Prepare batch for the model
                prepared = unwrapped_model.prepare_training_batch(batch)

                # Move to device
                prepared = _move_to_device(prepared, accelerator.device)

                # Forward pass produces loss_terms dict
                loss_terms = model(prepared)

                # Compute global normalizers from loss masks
                from dots_tts.training.losses import (
                    collapse_loss_masks,
                    to_host_named_scalars,
                    sum_named_scalars_across_ranks,
                )

                local_normalizers = to_host_named_scalars(
                    collapse_loss_masks(prepared["loss_masks"])
                )
                global_normalizers = sum_named_scalars_across_ranks(
                    local_normalizers,
                )

                # Compute gradient-scaled loss
                loss = compute_loss(
                    loss_terms,
                    global_normalizers=global_normalizers,
                    ddp_world_size=int(accelerator.num_processes),
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), grad_clip_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            global_step += 1

            loss_value = loss.detach().item()
            epoch_loss_sum += loss_value
            epoch_loss_count += 1

            if step % log_interval == 0:
                lr_val = scheduler.get_last_lr()[0]
                accelerator.print(
                    f"[dots_tts] Epoch {epoch} | Step {step}/{num_batches_per_epoch} | "
                    f"Loss: {loss_value:.4f} | LR: {lr_val:.2e}"
                )

        epoch_elapsed = time.perf_counter() - epoch_start_time
        epoch_avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        accelerator.print(
            f"[dots_tts] Epoch {epoch} complete. "
            f"Avg loss: {epoch_avg_loss:.4f} | "
            f"Time: {epoch_elapsed:.1f}s | "
            f"Steps: {global_step}/{max_train_steps}"
        )

        # Save checkpoint after each epoch
        if accelerator.is_main_process:
            _save_training_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                output_dir=output_model_path,
                epoch=epoch,
                global_step=global_step,
            )

    # Save final checkpoint
    if accelerator.is_main_process:
        accelerator.print(
            f"[dots_tts] Training complete. "
            f"Total steps: {global_step}. "
            f"Saving final model..."
        )
        unwrapped_model.save_pretrained(
            str(Path(output_model_path) / "final")
        )
        accelerator.print(
            f"[dots_tts] Final model saved to "
            f"{Path(output_model_path) / 'final'}"
        )

    accelerator.print("[dots_tts] Training completed successfully.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move nested tensors/dicts/dataclasses to ``device``."""
    from dataclasses import fields, is_dataclass

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                f.name: _move_to_device(getattr(value, f.name), device)
                for f in fields(value)
            }
        )
    return value


# ---------------------------------------------------------------------------
# Entry Points
# ---------------------------------------------------------------------------


def train(argv: list[str] | None = None) -> None:
    """CLI entry point — parse args, load params, run training."""
    cli_args = parse_args(argv)
    run_training(cli_args)


if __name__ == "__main__":
    train()
