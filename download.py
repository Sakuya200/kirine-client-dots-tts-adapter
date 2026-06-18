import argparse
from pathlib import Path
import subprocess
import shutil
import sys
import time

DOTS_TTS_INFERENCE_MODEL_PATH = "rednote-hilab/dots.tts-soar"
DOTS_TTS_INFERENCE_MODEL_NAME = "dots.tts-soar"
DOTS_TTS_TRAINING_MODEL_PATH = "rednote-hilab/dots.tts-base"
DOTS_TTS_TRAINING_MODEL_NAME = "dots.tts-base"

DEFAULT_REPO_BRANCH = "main"

def _emit(message: str, *, stderr: bool = False) -> None:
    print(message, file=sys.stderr if stderr else sys.stdout)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", dest="base_model", type=str, required=True)
    parser.add_argument("--model-version", dest="model_version", type=str, required=True)
    parser.add_argument("--target-root-dir", dest="target_root_dir", type=str, required=True)
    parser.add_argument("--log-path", dest="log_path", type=str, required=False)
    parser.add_argument("--task-log-file", dest="task_log_file", type=str, required=False)
    parser.add_argument(
        "--repo-url",
        dest="repo_url",
        type=str,
        default="https://github.com/rednote-hilab/dots.tts",
    )
    parser.add_argument(
        "--repo-branch",
        dest="repo_branch",
        type=str,
        default=DEFAULT_REPO_BRANCH,
        help="Git branch to fetch before checking out fixed revision.",
    )
    parser.add_argument(
        "--asset-source",
        dest="asset_source",
        choices=["HF", "ModelScope"],
        default="ModelScope",
        help="Where to download pretrained assets from (matches upstream install.ps1).",
    )
    return parser.parse_args(argv)


def _clone_repo(
    repo_url: str,
    branch: str,
    destination: Path,
) -> Path:
    """Clone a git repository with retry logic and timeout handling."""
    git_bin = shutil.which("git")
    if git_bin is None:
        raise SystemExit(
            "Custom download requires git for automatic clone. "
            "Install git and make sure it is available in PATH."
        )

    max_retries = 3
    retry_delay = 5  # seconds
    timeout = 300  # seconds (5 minutes)
    
    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(
                [
                    git_bin,
                    "clone",
                    "--branch",
                    branch,
                    "--single-branch",
                    repo_url,
                    str(destination),
                ],
                check=True,
                timeout=timeout,
            )
            return destination
        except subprocess.TimeoutExpired:
            error_msg = f"Git clone timed out after {timeout}s (attempt {attempt}/{max_retries})"
            if attempt < max_retries:
                print(f"⚠️  {error_msg}. Retrying in {retry_delay}s...", file=sys.stderr)
                time.sleep(retry_delay)
                retry_delay *= 2  # exponential backoff
            else:
                raise SystemExit(
                    f"❌ Git clone failed: {error_msg}.\n"
                    f"Please check your network connection and try again.\n"
                    f"Alternatively, you can:\n"
                    f"  1. Use a proxy: git config --global http.proxy <proxy_url>\n"
                    f"  2. Retry later if GitHub is unavailable"
                )
        except subprocess.CalledProcessError as e:
            error_msg = f"Git clone failed with exit code {e.returncode} (attempt {attempt}/{max_retries})"
            if attempt < max_retries:
                print(f"⚠️  {error_msg}. Retrying in {retry_delay}s...", file=sys.stderr)
                # Clean up partial clone if any
                if destination.exists():
                    shutil.rmtree(destination)
                time.sleep(retry_delay)
                retry_delay *= 2  # exponential backoff
            else:
                raise SystemExit(
                    f"❌ {error_msg}.\n"
                    f"Error details: {e}\n"
                    f"Common causes:\n"
                    f"  - Network connectivity issues (RPC failed, connection reset)\n"
                    f"  - GitHub is temporarily unavailable\n"
                    f"  - Firewall/proxy blocking connection\n"
                    f"  - SSH key issues (if using SSH)\n"
                    f"\n"
                    f"Solutions:\n"
                    f"  1. Check your network connection\n"
                    f"  2. Try again in a few moments\n"
                    f"  3. Use a proxy if behind firewall: git config --global http.proxy <proxy_url>\n"
                    f"  4. Retry later if GitHub service is unstable"
                )
                
def _mount_project(target_root: Path, target_dir: Path, base_model: str) -> None:
    # target_root is <src-model>/base-models, conda env lives at <src-model>/<base_model>/conda_env
    conda_env_path = target_root.parent / base_model / "conda_env"
    pip_in_conda = conda_env_path / "Scripts" / "pip.exe"

    if pip_in_conda.exists():
        # 该模型由于依赖 pynini，该依赖在windows上安装较为麻烦，因此不使用 venv，
        # 而是直接使用 conda 环境来安装依赖。使用 --prefix 指定 conda 环境路径，
        # 避免 -n 按名称查找时找不到非标准路径下的环境。
        try:
            # 进行项目挂载之前需要先安装 pynini
            subprocess.run(
                ["conda", "install", "--prefix", str(conda_env_path), "-y", "-c", "conda-forge", "pynini"],
                check=True,
                shell=True
            )
            
            subprocess.run(
                ["conda", "run", "--prefix", str(conda_env_path), "pip", "install", "-e", str(target_dir)],
                check=True,
                shell=True
            )
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                f"❌ Failed to mount project using conda environment: {e}\n"
                f"Please ensure you have set up the conda environment correctly and try again."
            )
        return

    # 如果 conda 环境不存在，检查是否误用了 venv
    pip_in_venv = target_root.parent / "dots_tts" / "venv" / "Scripts" / "pip.exe"
    if pip_in_venv.exists():
        raise SystemExit(
            f"❌ Detected pip executable at {pip_in_venv}, but dots_tts setup is configured "
            f"to use conda environment for dependency management on Windows.\n"
            f"Please ensure you have set up the conda environment correctly and try again.\n"
        )

    raise SystemExit(
        f"❌ Cannot mount project: pip executable not found at expected path: {pip_in_conda}\n"
        f"Please ensure you have set up the conda environment correctly and that pip is available."
    )
    
def _download_models(target_dir: Path, asset_source: str) -> None:
    models_dir = target_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 下载 base 以及 inference 模型文件到 models 目录下，供后续训练和推理使用
    try:
        if asset_source == "HF":
            _emit("Downloading models from Hugging Face...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=DOTS_TTS_INFERENCE_MODEL_PATH,
                local_dir=str(models_dir / DOTS_TTS_INFERENCE_MODEL_NAME)
            )
            snapshot_download(
                repo_id=DOTS_TTS_TRAINING_MODEL_PATH,
                local_dir=str(models_dir / DOTS_TTS_TRAINING_MODEL_NAME)
            )
        elif asset_source == "ModelScope":
            _emit("Downloading models from ModelScope...")
            from modelscope.hub.snapshot_download import snapshot_download
            # 下载推理模型
            snapshot_download(
                model_id=DOTS_TTS_INFERENCE_MODEL_PATH, 
                local_dir=str(models_dir / DOTS_TTS_INFERENCE_MODEL_NAME)
            )
            # 下载训练模型
            snapshot_download(
                model_id=DOTS_TTS_TRAINING_MODEL_PATH, 
                local_dir=str(models_dir / DOTS_TTS_TRAINING_MODEL_NAME)
            )
        else:
            raise ValueError(f"Unsupported asset source: {asset_source}")
    except ValueError:
        raise
    except Exception as e:
        _emit(f"⚠️  Model download failed: {e}", stderr=True)
        raise SystemExit(
            f"❌ Failed to download models from {asset_source}.\n"
            f"Please check your network connection and try again.\n"
            f"Alternatively, you can manually download the models from the respective platform and place them in the 'models' directory."
        )

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    target_root = Path(args.target_root_dir).expanduser().resolve()
    target_dir = target_root / args.base_model
    target_root.mkdir(parents=True, exist_ok=True)

    if not target_dir.exists():
        try:
            _emit(
            f"📥 Cloning dots_tts directly into target directory: {target_dir}",
            )
            _clone_repo(
                args.repo_url,
                args.repo_branch,
                target_dir,
            )
            _emit("✓ Clone completed successfully")
        except Exception as e:
            raise SystemExit(
                f"❌ Setup failed after direct clone: dots_tts runtime is incomplete, target directory: {target_dir}\n"
                f"\n\n"
                f"Troubleshooting:\n"
                f"  1. Ensure you have sufficient disk space\n"
                f"  2. Check your network connection\n"
                f"  3. Manually download from: {args.repo_url}\n"
                f"  4. Try a different asset source via --asset-source HF-Mirror\n"
                f"  5. Verify selected model-scale/asset-version points to available checkpoints"
            )
    else:
        _emit(
            f"✓ dots_tts checkout already exists at {target_dir}; skip clone",
        )

    _mount_project(target_root, target_dir, args.base_model)
    _download_models(target_dir, args.asset_source)
    
    _emit(f"✅ dots_tts is ready at {target_dir}")
    



if __name__ == "__main__":
    main(sys.argv[1:])
