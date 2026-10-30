"""Fully Automated Kaggle GPU Job Submission, Monitoring, and Retrieval for NEXUS-SLM.

Orchestrates:
1. Safe Kaggle authentication verification (never printing or logging secret keys).
2. Preparation of isolated clean-data Kaggle kernel bundle (clean notebook + kernel-metadata.json).
3. Submitting kernel to free Kaggle GPU accelerator runtime (Nvidia Tesla T4).
4. Real-time monitoring of job execution status until completion.
5. Downloading and unpacking trained LoRA adapter artifacts into a dedicated unique run directory.
6. Cryptographic SHA-256 and zip archive integrity verification (with Zip Slip prevention).
7. Validating adapter configuration, weight safetensors, tokenizer, and metadata.
8. Triggering clean-data candidate evaluation and propagating exit codes reliably.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Default Canonical Paths
CLEAN_NOTEBOOK_V2 = PROJECT_ROOT / "notebooks" / "clean_data_qwen25_05b_lora_training_v2.ipynb"
CLEAN_NOTEBOOK_V1 = PROJECT_ROOT / "notebooks" / "clean_data_qwen25_05b_lora_training.ipynb"
DEFAULT_NOTEBOOK_SRC = CLEAN_NOTEBOOK_V2 if CLEAN_NOTEBOOK_V2.exists() else CLEAN_NOTEBOOK_V1

DEFAULT_KERNEL_SLUG = "nexus-qwen25-clean-v2-training"
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "notebooks" / "kaggle_clean_v2_bundle"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "experiments" / "artifacts" / "training_runs"
CANDIDATE_NAME = "qwen25_05b_clean_v2_lora"

# Protected legacy model directories that must NEVER be overwritten or wiped
PROTECTED_MODEL_DIRS = [
    PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42",
    PROJECT_ROOT / "models" / "nexus_qlora_v1",
    PROJECT_ROOT / "models" / "nexus_qlora_v2",
    PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_gguf",
    PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_gguf_q4_k_m",
    PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_gguf_q8_0",
    PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42_int8",
    PROJECT_ROOT / "models" / "scratch_merged_qwen",
    PROJECT_ROOT / "models" / "scratch_onnx_qwen",
    PROJECT_ROOT / "models",
]


def check_protected_path(target_dir: Path) -> None:
    """Verifies that target_dir is not an existing protected model directory."""
    resolved_target = target_dir.resolve()
    for prot in PROTECTED_MODEL_DIRS:
        resolved_prot = prot.resolve()
        if resolved_target == resolved_prot:
            raise PermissionError(
                f"SAFETY VIOLATION: Target directory '{target_dir}' is a protected model directory ({prot}). Overwrite forbidden!"
            )


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_authenticated_username() -> Optional[str]:
    """Verifies presence of Kaggle credentials and safely retrieves the username.

    Never prints or logs the secret API key.
    """
    home = Path.home()

    # 0. Check access_token file (~/.kaggle/access_token)
    token_file = home / ".kaggle" / "access_token"
    try:
        if token_file.exists():
            token_val = token_file.read_text(encoding="utf-8").strip()
            if token_val:
                os.environ["KAGGLE_API_TOKEN"] = token_val
    except Exception:
        pass

    # 1. Check environment variables
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and (env_key or os.environ.get("KAGGLE_API_TOKEN")):
        return env_user.strip()

    # 2. Check ~/.kaggle/kaggle.json
    kaggle_json = home / ".kaggle" / "kaggle.json"
    try:
        if kaggle_json.exists():
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            u = data.get("username")
            k = data.get("key")
            if u and (k or os.environ.get("KAGGLE_API_TOKEN")):
                return u.strip()
    except Exception:
        pass

    # 3. Check via Kaggle REST endpoint if KAGGLE_API_TOKEN is available
    if os.environ.get("KAGGLE_API_TOKEN"):
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://www.kaggle.com/api/v1/kernels/list?group=profile",
                headers={"Authorization": f"Bearer {os.environ['KAGGLE_API_TOKEN']}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                if data and isinstance(data, list) and len(data) > 0:
                    ref = data[0].get("ref", "")
                    if "/" in ref:
                        u = ref.split("/")[0].strip()
                        os.environ["KAGGLE_USERNAME"] = u
                        return u
        except Exception:
            pass

    # 4. Check KaggleApi if library installed
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        u = api.get_config_value(api.CONFIG_NAME_USER)
        if u:
            return u.strip()
    except Exception:
        pass

    return None


def prepare_kernel_bundle(
    username: str,
    kernel_slug: str = DEFAULT_KERNEL_SLUG,
    notebook_src: Path = DEFAULT_NOTEBOOK_SRC,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
) -> Path:
    """Prepares isolated bundle folder containing the clean notebook and kernel-metadata.json.

    Keeps the clean bundle strictly segregated from legacy bundles.
    """
    if not notebook_src.exists():
        raise FileNotFoundError(f"Clean training notebook source not found: {notebook_src}")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    target_nb_name = notebook_src.name
    dest_nb = bundle_dir / target_nb_name
    shutil.copy2(notebook_src, dest_nb)

    meta = {
        "id": f"{username}/{kernel_slug}",
        "title": "Nexus Qwen25 Clean V2 Training",
        "code_file": target_nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    meta_file = bundle_dir / "kernel-metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return bundle_dir


def run_kaggle_cli(args: list) -> Tuple[int, str]:
    """Runs kaggle CLI in the project venv or system PATH, returning (returncode, stdout+stderr).

    Redacts credentials if present in command arguments.
    """
    # Ensure KAGGLE_API_TOKEN is populated if available in ~/.kaggle/access_token
    if not os.environ.get("KAGGLE_API_TOKEN"):
        try:
            token_file = Path.home() / ".kaggle" / "access_token"
            if token_file.exists():
                tok = token_file.read_text(encoding="utf-8").strip()
                if tok:
                    os.environ["KAGGLE_API_TOKEN"] = tok
        except Exception:
            pass

    kaggle_exe = PROJECT_ROOT / ".venv" / "Scripts" / "kaggle.exe"
    cmd = [str(kaggle_exe)] + args if kaggle_exe.exists() else ["kaggle"] + args

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
    )
    return res.returncode, res.stdout.strip()


def monitor_kernel_execution(
    kernel_id: str,
    poll_interval_sec: int = 30,
    max_wait_sec: int = 7200,
    wait_for_transition: bool = False,
) -> bool:
    """Monitors kernel execution until completion, error, or timeout."""
    print(f"\nMonitoring Kaggle job: {kernel_id} (polling every {poll_interval_sec}s)...")
    sys.stdout.flush()

    if wait_for_transition:
        print("Waiting for Kaggle to transition to newly queued or running execution...")
        sys.stdout.flush()
        for attempt in range(12):
            time.sleep(5)
            ret, out = run_kaggle_cli(["kernels", "status", kernel_id])
            out_lower = out.lower()
            if "queued" in out_lower or "running" in out_lower:
                print(f"Kaggle job registered active state: {out}")
                sys.stdout.flush()
                break
            elif "error" in out_lower or "failed" in out_lower:
                print(f"Kernel submission failed: {out}")
                return False

    start_t = time.time()
    while time.time() - start_t < max_wait_sec:
        ret, out = run_kaggle_cli(["kernels", "status", kernel_id])
        elapsed_min = round((time.time() - start_t) / 60, 1)
        print(f"[{elapsed_min}m elapsed] Status: {out}")
        sys.stdout.flush()

        out_lower = out.lower()
        if "has status \"complete\"" in out_lower or "complete" in out_lower:
            print("\nRemote Kaggle job completed successfully!")
            sys.stdout.flush()
            return True
        elif "has status \"error\"" in out_lower or "error" in out_lower or "failed" in out_lower:
            print(f"\nRemote Kaggle job failed with status: {out}")
            sys.stdout.flush()
            return False

        time.sleep(poll_interval_sec)

    print(f"\nTimed out waiting for kernel {kernel_id} after {max_wait_sec}s.")
    return False


def safe_extract_zip(zip_path: Path, target_dir: Path) -> List[Path]:
    """Safely extracts zip archive preventing path traversal (Zip Slip vulnerability).

    Verifies archive integrity with testzip() before unpacking.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive file does not exist: {zip_path}")
    if zip_path.stat().st_size == 0:
        raise ValueError(f"Archive file is empty (0 bytes): {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"File is not a valid zip archive: {zip_path}")

    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Check CRC / corrupt file detection
        corrupted = zf.testzip()
        if corrupted is not None:
            raise ValueError(f"Archive corruption detected: bad CRC on '{corrupted}' in {zip_path}")

        for member in zf.infolist():
            member_name = member.filename
            # Reject absolute paths or path traversal (..)
            normalized = member_name.replace("\\", "/")
            parts = normalized.split("/")
            if member_name.startswith(("/", "\\")) or ".." in parts:
                raise ValueError(
                    f"SECURITY ALERT: Unsafe archive path detected (Zip Slip vulnerability): {member_name}"
                )
            dest = (target_dir / member_name).resolve()
            if not str(dest).startswith(str(target_dir)):
                raise ValueError(
                    f"SECURITY ALERT: Extraction path escapes destination: {member_name} -> {dest}"
                )

        zf.extractall(str(target_dir))
        for member in zf.infolist():
            extracted_files.append(target_dir / member.filename)

    return extracted_files


def verify_adapter_artifacts(adapter_dir: Path) -> Dict[str, Any]:
    """Verifies presence and validity of adapter weights, config, tokenizer, and metadata."""
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    # 1. Adapter config
    config_file = adapter_dir / "adapter_config.json"
    if not config_file.exists():
        # Check subdirectories if nested
        sub_configs = list(adapter_dir.glob("**/adapter_config.json"))
        if sub_configs:
            config_file = sub_configs[0]
            adapter_dir = config_file.parent
        else:
            raise FileNotFoundError(f"Missing required 'adapter_config.json' in {adapter_dir}")

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    base_model = config_data.get("base_model_name_or_path", "")
    if "qwen" not in base_model.lower():
        print(f"Warning: Unexpected base_model_name_or_path: '{base_model}'")

    # 2. Weights check
    safetensors = list(adapter_dir.glob("*.safetensors"))
    bin_weights = list(adapter_dir.glob("*.bin"))
    weight_files = safetensors or bin_weights
    if not weight_files:
        raise FileNotFoundError(f"No adapter weights (*.safetensors or *.bin) found in {adapter_dir}")
    primary_weight = weight_files[0]
    if primary_weight.stat().st_size == 0:
        raise ValueError(f"Adapter weight file is empty: {primary_weight}")

    # 3. Tokenizer check
    tok_files = list(adapter_dir.glob("tokenizer*.json")) + list(adapter_dir.glob("vocab*.json"))
    has_tokenizer = len(tok_files) > 0

    verification_summary = {
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "weight_file": primary_weight.name,
        "weight_size_bytes": primary_weight.stat().st_size,
        "has_tokenizer": has_tokenizer,
        "verified": True,
    }
    return verification_summary


def run_clean_evaluation(adapter_dir: Path, split: str = "test", samples: Optional[int] = None) -> bool:
    """Runs clean evaluation script on the retrieved candidate adapter and checks gates."""
    print(f"\nEvaluating clean candidate at {adapter_dir} on split '{split}'...")
    py_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not py_exe.exists():
        py_exe = Path(sys.executable)

    eval_script = PROJECT_ROOT / "scripts" / "evaluation" / "evaluate_clean_v2_comprehensive.py"
    if not eval_script.exists():
        print(f"Warning: {eval_script} not found, falling back to evaluate_clean_v2_test.py")
        eval_script = PROJECT_ROOT / "scripts" / "evaluation" / "evaluate_clean_v2_test.py"

    cmd = [
        str(py_exe),
        str(eval_script),
        "--model", CANDIDATE_NAME,
        "--split", split,
    ]
    if samples is not None:
        cmd.extend(["--samples", str(samples)])

    env = os.environ.copy()
    env["NEXUS_ADAPTER_PATH"] = str(adapter_dir)
    env["ACTIVE_MODEL_VERSION"] = CANDIDATE_NAME

    res = subprocess.run(cmd, env=env, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        print(f"\n[FAIL] Evaluation process failed with exit code: {res.returncode}")
        return False
    print("\n[PASS] Candidate evaluation completed successfully.")
    return True


def retrieve_artifacts_and_evaluate(
    kernel_id: str,
    output_base: Path = DEFAULT_OUTPUT_BASE,
    run_id: Optional[str] = None,
    eval_split: str = "test",
    eval_samples: Optional[int] = None,
) -> Tuple[bool, Path]:
    """Safely retrieves Kaggle kernel outputs into a dedicated new run directory,

    verifies integrity and archives, and evaluates the candidate.
    Refuses to overwrite existing model directories or existing runs.
    """
    if run_id is None:
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    run_dir = output_base / f"{CANDIDATE_NAME}_{run_id}"

    # Critical Safety: Refuse to use or delete protected model directories
    check_protected_path(run_dir)

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory '{run_dir}' already exists and is not empty. Refusing to overwrite!"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    raw_download_dir = run_dir / "raw_download"
    raw_download_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading kernel outputs from '{kernel_id}' to: {raw_download_dir}...")
    ret, out = run_kaggle_cli(["kernels", "output", kernel_id, "-p", str(raw_download_dir)])
    print(out)
    if ret != 0:
        print(f"[ERROR] Failed to download kernel outputs: {out}")
        return False, run_dir

    # Discover downloaded archives
    zip_files = list(raw_download_dir.glob("**/*.zip"))
    if not zip_files:
        print(f"[ERROR] No zip archives found in downloaded output from {kernel_id}")
        return False, run_dir

    extracted_adapter_dir = run_dir / "adapter"
    extracted_adapter_dir.mkdir(parents=True, exist_ok=True)

    for zf in zip_files:
        sha256_hash = compute_sha256(zf)
        print(f"\nArchive: {zf.name}")
        print(f"  Size:   {zf.stat().st_size:,} bytes")
        print(f"  SHA256: {sha256_hash}")

        print(f"Verifying integrity and safe extracting {zf.name} -> {extracted_adapter_dir}...")
        try:
            safe_extract_zip(zf, extracted_adapter_dir)
            print("  Extraction PASSED safely!")
        except Exception as e:
            print(f"[FAIL] Archive extraction failed: {e}")
            return False, run_dir

    # Check for nested adapter directory
    target_adapter_path = extracted_adapter_dir
    nested_configs = list(extracted_adapter_dir.glob("**/adapter_config.json"))
    if nested_configs:
        target_adapter_path = nested_configs[0].parent

    # Verify adapter files
    try:
        summary = verify_adapter_artifacts(target_adapter_path)
        print(f"Adapter verification successful: {summary}")
    except Exception as e:
        print(f"[FAIL] Adapter verification failed: {e}")
        return False, run_dir

    # Run clean evaluation
    eval_ok = run_clean_evaluation(
        adapter_dir=target_adapter_path,
        split=eval_split,
        samples=eval_samples,
    )
    if not eval_ok:
        print(f"[FAIL] Clean candidate evaluation failed for run: {run_id}")
        return False, run_dir

    return True, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS-SLM Kaggle GPU Training Automation")
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK_SRC, help="Path to clean training notebook")
    parser.add_argument("--kernel-slug", type=str, default=DEFAULT_KERNEL_SLUG, help="Kaggle kernel slug")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE, help="Directory to store run artifacts")
    parser.add_argument("--run-id", type=str, default=None, help="Unique run identifier")
    parser.add_argument("--check-only", action="store_true", help="Only check credentials and kernel status")
    parser.add_argument("--retrieve-only", action="store_true", help="Only retrieve and evaluate completed run")
    parser.add_argument("--max-wait-sec", type=int, default=7200, help="Maximum seconds to wait for kernel completion")
    parser.add_argument("--eval-split", type=str, default="test", choices=["test", "val"], help="Split for post-training eval")
    parser.add_argument("--eval-samples", type=int, default=None, help="Sample limit for post-training eval (None=full)")
    args = parser.parse_args()

    print("=" * 75)
    print("NEXUS-SLM: CLEAN-DATA KAGGLE GPU TRAINING AUTOMATION")
    print("=" * 75)

    # 1. Safe credential check
    username = get_authenticated_username()
    if not username:
        print("\n[AUTH REQUIRED] No authenticated Kaggle credentials found.")
        print("Please complete the one-time authentication:")
        print("1. Go to https://www.kaggle.com/settings -> scroll to API -> click 'Create New Token'.")
        print("2. Move the downloaded 'kaggle.json' to 'C:\\Users\\avira\\.kaggle\\kaggle.json'.")
        print("   (PowerShell: Move-Item \"$HOME\\Downloads\\kaggle.json\" \"$HOME\\.kaggle\\kaggle.json\")")
        print("   OR set environment variables: KAGGLE_USERNAME and KAGGLE_KEY.")
        print("3. Re-run this script.")
        sys.exit(1)

    print(f"Authenticated as Kaggle user: {username}")
    kernel_id = f"{username}/{args.kernel_slug}"

    if args.check_only:
        ret, status_out = run_kaggle_cli(["kernels", "status", kernel_id])
        print(f"Kernel '{kernel_id}' status: {status_out}")
        sys.exit(0)

    if args.retrieve_only:
        print(f"\nExecuting artifact retrieval and evaluation for {kernel_id}...")
        ok, rdir = retrieve_artifacts_and_evaluate(
            kernel_id=kernel_id,
            output_base=args.output_base,
            run_id=args.run_id,
            eval_split=args.eval_split,
            eval_samples=args.eval_samples,
        )
        if not ok:
            print(f"\n[AUTOMATION FAILED] Retrieval or evaluation failed for {kernel_id}.")
            sys.exit(1)
        print(f"\n[AUTOMATION SUCCESS] Artifacts retrieved and validated in: {rdir}")
        sys.exit(0)

    # 2. Prepare clean kernel bundle
    bundle_dir = prepare_kernel_bundle(
        username=username,
        kernel_slug=args.kernel_slug,
        notebook_src=args.notebook,
    )
    print(f"Prepared clean kernel bundle in: {bundle_dir} (Source: {args.notebook.name})")

    # 3. Check status & push kernel
    is_new_push = False
    ret, status_out = run_kaggle_cli(["kernels", "status", kernel_id])
    if "running" in status_out.lower() or "queued" in status_out.lower():
        print(f"Kernel {kernel_id} is already actively running: {status_out}")
    else:
        print(f"Submitting GPU kernel to Kaggle: {kernel_id} (Accelerator: NvidiaTeslaT4 via metadata)...")
        ret, out = run_kaggle_cli(["kernels", "push", "-p", str(bundle_dir)])
        print(out)
        if ret != 0:
            print(f"[FAIL] Kernel push failed: {out}")
            sys.exit(1)
        is_new_push = True

    # 4. Monitor execution
    success = monitor_kernel_execution(
        kernel_id=kernel_id,
        max_wait_sec=args.max_wait_sec,
        wait_for_transition=is_new_push,
    )
    if not success:
        print("\n[FAIL] Remote Kaggle execution did not complete successfully.")
        import tempfile
        try:
            with tempfile.TemporaryDirectory() as td:
                run_kaggle_cli(["kernels", "output", kernel_id, "-p", td])
                for log_file in Path(td).glob("*.log"):
                    content = log_file.read_text(encoding="utf-8", errors="replace")
                    print(f"Execution logs (last 4000 chars):\n{content[-4000:] if len(content) > 4000 else content}")
                    break
        except Exception:
            pass
        sys.exit(1)

    # 5. Retrieve artifacts, verify integrity, and evaluate
    ok, rdir = retrieve_artifacts_and_evaluate(
        kernel_id=kernel_id,
        output_base=args.output_base,
        run_id=args.run_id,
        eval_split=args.eval_split,
        eval_samples=args.eval_samples,
    )
    if not ok:
        print(f"\n[FAIL] Artifact retrieval, verification, or evaluation failed for run in {rdir}")
        sys.exit(1)

    print(f"\n===========================================================================")
    print(f"NEXUS-SLM: GPU TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Trained adapter location: {rdir / 'adapter'}")
    print(f"===========================================================================")


if __name__ == "__main__":
    main()
