from __future__ import annotations

import argparse
import shutil
from types import SimpleNamespace
from typing import Any

from dots_tts import ensure_src_root_on_path

ensure_src_root_on_path()

from dots_tts.common import load_runtime  # noqa: E402
from dots_tts.dataset import DotsTtsDataset  # noqa: E402
from dots_tts.params import load_training_params  # noqa: E402
from dots_tts.training_common import (  # noqa: E402
    TrainingPipelineContext,
    add_common_training_args,
    build_optimizer,
    build_runtime_options,
    build_scheduler,
    build_train_dataloader,
    disable_use_cache_for_training,
    enable_gradient_checkpointing,
    load_training_dependencies,
)

CHECKPOINT_MAX_SHARD_SIZE = "1GB"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dots_tts inline fine-tuning.",
    )
    parser = add_common_training_args(parser)
    parser.add_argument(
        "--init-model-path", dest="init_model_path", type=str, required=True,
        help="Path to pretrained dots_tts model checkpoint.",
    )
    parser.add_argument(
        "--output-model-path", dest="output_model_path", type=str, required=True,
        help="Directory where fine-tuned checkpoints will be saved.",
    )
    parser.add_argument(
        "--train-manifest", dest="train_manifest", type=str, required=True,
        help="Path to a JSONL training manifest file.",
    )
    parser.add_argument(
        "--num-epochs", dest="num_epochs", type=int, default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=1,
        help="Batch size per device (default: 1).",
    )
    parser.add_argument(
        "--learning-rate", dest="learning_rate", type=float, default=2e-5,
        help="Learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--weight-decay", dest="weight_decay", type=float, default=0.1,
        help="Weight decay (default: 0.1).",
    )
    parser.add_argument(
        "--warmup-steps", dest="warmup_steps", type=int, default=100,
        help="Number of warmup steps (default: 100).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", dest="gradient_accumulation_steps",
        type=int, default=1,
        help="Gradient accumulation steps (default: 1).",
    )
    parser.add_argument(
        "--mixed-precision", dest="mixed_precision", type=str, default="bf16",
        help="Mixed precision mode: 'no', 'fp16', 'bf16' (default: 'bf16').",
    )
    parser.add_argument(
        "--device", dest="device", type=str, default="cuda",
        help="Device: 'cuda', 'cpu' (default: 'cuda').",
    )
    parser.add_argument(
        "--logging-dir", dest="logging_dir", type=str, default=None,
        help="TensorBoard logging directory.",
    )
    parser.add_argument(
        "--enable-gradient-checkpointing", dest="enable_gradient_checkpointing",
        action="store_true", default=False,
        help="Enable gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--gradient-clip-norm", dest="gradient_clip_norm", type=float, default=1.0,
        help="Max gradient norm for clipping (default: 1.0).",
    )
    return parser.parse_args(argv)


def save_training_checkpoint(
    args: argparse.Namespace,
    accelerator: Any,
    model: Any,
    epoch: int,
) -> None:
    """Save a fine-tuned model checkpoint using the HuggingFace Hub utility."""
    from huggingface_hub import save_torch_state_dict

    output_dir = f"{args.output_model_path}/checkpoint-epoch-{epoch}"

    # Copy config files from the initial model path.
    shutil.copytree(args.init_model_path, output_dir, dirs_exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {}
    for key, value in unwrapped_model.state_dict().items():
        state_dict[key] = value.detach().to("cpu")

    save_torch_state_dict(
        state_dict,
        output_dir,
        max_shard_size=CHECKPOINT_MAX_SHARD_SIZE,
        safe_serialization=True,
        is_main_process=accelerator.is_main_process,
        shared_tensors_to_discard=getattr(
            unwrapped_model, "_tied_weights_keys", None
        ),
    )


def initialize_training_pipeline(
    args: argparse.Namespace,
    dependencies: SimpleNamespace | None = None,
) -> TrainingPipelineContext:
    """Initialize all objects needed for the training loop.

    Loads the pretrained dots_tts model, wraps it with HF Accelerator,
    builds the DataLoader, optimizer, and scheduler.
    """
    deps = dependencies or load_training_dependencies()
    runtime = build_runtime_options(args, deps.torch)
    accelerator = deps.Accelerator(**runtime.accelerator_kwargs)

    # Load the dots_tts runtime and extract its internal core/model.
    dots_runtime = load_runtime(
        args.init_model_path,
        precision=runtime.model_load_kwargs.get("precision", "bfloat16"),
    )

    # Access the internal model for training.
    # DotsTtsRuntime wraps a DotsTtsModel which holds the core.
    model: Any = getattr(dots_runtime, "model", None)
    if model is None:
        raise AttributeError(
            "DotsTtsRuntime does not expose a 'model' attribute. "
            "Cannot extract model for training."
        )

    if getattr(args, "enable_gradient_checkpointing", False):
        disable_use_cache_for_training(model)
        enable_gradient_checkpointing(model)

    train_data, train_dataloader = build_train_dataloader(args)

    num_epochs = int(getattr(args, "num_epochs", None) or 5)
    num_batches_per_epoch = len(train_dataloader)
    num_training_steps = num_epochs * num_batches_per_epoch

    optimizer = build_optimizer(model, args, deps)
    scheduler = build_scheduler(optimizer, args, num_training_steps)

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    accelerator.print(
        f"[dots_tts] Training initialized: {runtime.mode_label}"
    )
    accelerator.print(
        f"[dots_tts]   batches/epoch={num_batches_per_epoch} "
        f"epochs={num_epochs} total_steps={num_training_steps}"
    )

    return TrainingPipelineContext(
        runtime=runtime,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        train_data=train_data,
    )


def run_training(
    args: argparse.Namespace,
    dependencies: SimpleNamespace | None = None,
) -> None:
    """Execute the full fine-tuning pipeline.

    This function orchestrates the end-to-end training process:
    1. Initialize the model, optimizer, scheduler, and DataLoader.
    2. Run the training loop across all epochs.
    3. Save the final checkpoint.

    The training loop uses the dots_tts model's native ``forward``,
    which internally computes the flow-matching loss.
    """
    deps = dependencies or load_training_dependencies()
    context = initialize_training_pipeline(args, deps)

    accelerator = context.accelerator
    model = context.model
    optimizer = context.optimizer
    scheduler = context.scheduler
    train_dataloader = context.train_dataloader

    num_epochs = int(getattr(args, "num_epochs", None) or 5)
    gradient_clip_norm = float(
        getattr(args, "gradient_clip_norm", None) or 1.0
    )
    log_interval = int(getattr(args, "log_interval", None) or 10)

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # dots_tts forward pass: the model internally computes
                # the flow-matching loss on audio + text pairs.
                # The collate_fn returns {"fid", "audio", "text"} —
                # the model expects audio paths / waveforms and text
                # strings, and internally runs encode → LLM → acoustic.
                outputs = model(
                    audio=batch["audio"],
                    text=batch["text"],
                )
                loss = outputs["loss"]

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if step % log_interval == 0:
                lr_val = (
                    scheduler.get_last_lr()[0]
                    if scheduler is not None
                    else float(getattr(args, "learning_rate", None) or 2e-5)
                )
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | "
                    f"Loss: {loss.item():.4f} | LR: {lr_val:.2e}"
                )

        accelerator.print(
            f"[dots_tts] Epoch {epoch} complete. "
            f"Steps: {global_step}"
        )

        if accelerator.is_main_process:
            save_training_checkpoint(args, accelerator, model, epoch)

    accelerator.print("[dots_tts] Training complete.")


def train(argv: list[str] | None = None) -> None:
    """CLI entry point for training."""
    cli_args = parse_args(argv)
    params = load_training_params(cli_args.params_file)
    run_training(params.to_namespace())


if __name__ == "__main__":
    train()
