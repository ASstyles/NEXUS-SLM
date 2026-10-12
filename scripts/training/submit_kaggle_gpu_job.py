"""Fully Automated Kaggle GPU Job Submission, Monitoring, and Retrieval for NEXUS-SLM.

Orchestrates:
1. Safe Kaggle authentication verification (without printing secret keys).
2. Preparation of Kaggle kernel bundle (notebook + kernel-metadata.json).
3. Pushing kernel to Kaggle GPU accelerator runtime.
4. Real-time monitoring of job execution status until completion.
5. Downloading and unpacking trained LoRA adapter artifacts.
6. Triggering candidate evaluation against all 6 acceptance gates.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

KAGGLE_JOB_DIR = PROJECT_ROOT / "notebooks" / "kaggle_job"
NOTEBOOK_SRC = PROJECT_ROOT / "notebooks" / "nexus_qwen25_05b_lora_training.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42"


def get_authenticated_username() -> Optional[str]:
    """
    Verifies presence of Kaggle credentials and safely retrieves the username.
    Never prints or logs the secret key.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        u = api.get_config_value(api.CONFIG_NAME_USER)
        if u:
            return u.strip()
    except Exception:
        pass

    # 1. Check environment variables
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return env_user.strip()

    # 2. Check ~/.kaggle/kaggle.json
    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            u = data.get("username")
            k = data.get("key")
            if u and k:
                return u.strip()
        except Exception:
            return None

    return None


def prepare_kernel_bundle(username: str, kernel_slug: str = "nexus-qwen25-05b-lora-training") -> Path:
    """Prepares isolated folder containing notebook and kernel-metadata.json."""
    KAGGLE_JOB_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(NOTEBOOK_SRC, KAGGLE_JOB_DIR / "nexus_qwen25_05b_lora_training.ipynb")

    meta = {
        "id": f"{username}/{kernel_slug}",
        "title": "Nexus Qwen25 05b Lora Training",
        "code_file": "nexus_qwen25_05b_lora_training.ipynb",
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

    meta_file = KAGGLE_JOB_DIR / "kernel-metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return KAGGLE_JOB_DIR


def run_kaggle_cli(args: list) -> Tuple[int, str]:
    """Runs kaggle CLI in the current venv and returns (returncode, stdout+stderr)."""
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
    max_wait_sec: int = 1800,
    wait_for_transition: bool = False,
) -> bool:
    """Monitors kernel execution until completion, error, or timeout."""
    print(f"\nMonitoring Kaggle job: {kernel_id} (polling every {poll_interval_sec}s)...")
    sys.stdout.flush()

    if wait_for_transition:
        print("Waiting for Kaggle to transition to newly queued or running execution...")
        sys.stdout.flush()
        for attempt in range(12):  # wait up to 60 seconds
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
            print("\nJob completed successfully!")
            sys.stdout.flush()
            return True
        elif "has status \"error\"" in out_lower or "error" in out_lower or "failed" in out_lower:
            print(f"\nJob failed with status: {out}")
            sys.stdout.flush()
            return False

        time.sleep(poll_interval_sec)

    print(f"\nTimed out waiting for kernel {kernel_id} after {max_wait_sec}s.")
    return False


def retrieve_artifacts_and_evaluate(kernel_id: str) -> bool:
    """Downloads artifacts, unzips them to models/qwen25_05b_lora_seed42, and triggers evaluation."""
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading kernel outputs from {kernel_id} to {OUTPUT_DIR}...")
    ret, out = run_kaggle_cli(["kernels", "output", kernel_id, "-p", str(OUTPUT_DIR)])
    print(out)

    # Find any zip archive in OUTPUT_DIR recursively
    zip_files = list(OUTPUT_DIR.glob("**/*.zip"))
    for zf in zip_files:
        print(f"Extracting archive {zf} into {OUTPUT_DIR}...")
        try:
            shutil.unpack_archive(str(zf), str(OUTPUT_DIR))
            print(f"Successfully extracted {zf}")
        except Exception as e:
            print(f"Warning: Failed to extract {zf}: {e}")

    # If extracted files are nested inside a subfolder, bring them up to OUTPUT_DIR
    for sub in OUTPUT_DIR.glob("**/adapter_config.json"):
        if sub.parent != OUTPUT_DIR:
            print(f"Found nested adapter directory at {sub.parent}, copying files to {OUTPUT_DIR}...")
            for item in sub.parent.iterdir():
                if item.is_file():
                    shutil.copy2(item, OUTPUT_DIR / item.name)
            break

    # Run evaluation script
    eval_script = PROJECT_ROOT / "scripts" / "evaluation" / "evaluate_qwen_candidate.py"
    py_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    print("\nRunning candidate evaluation against acceptance gates...")
    sub = subprocess.run(
        [str(py_exe), str(eval_script), "--adapter-dir", str(OUTPUT_DIR)],
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return sub.returncode == 0


def main():
    print("=" * 75)
    print("NEXUS-SLM: KAGGLE GPU TRAINING AUTOMATION")
    print("=" * 75)

    username = get_authenticated_username()
    if not username:
        print("\n[AUTH REQUIRED] No authenticated Kaggle credentials found.")
        print("Please complete the one-time authentication:")
        print("1. Go to https://www.kaggle.com/settings -> scroll to API -> click 'Create New Token'.")
        print("2. Move the downloaded 'kaggle.json' to 'C:\\Users\\avira\\.kaggle\\kaggle.json'.")
        print("   (PowerShell: Move-Item \"$HOME\\Downloads\\kaggle.json\" \"$HOME\\.kaggle\\kaggle.json\")")
        print("3. Re-run this script.")
        sys.exit(1)

    print(f"Authenticated as Kaggle user: {username}")
    kernel_slug = "nexus-qwen25-05b-lora-training"
    kernel_id = f"{username}/{kernel_slug}"

    # 1. Prepare bundle
    bundle_dir = prepare_kernel_bundle(username=username, kernel_slug=kernel_slug)
    print(f"Prepared kernel bundle in: {bundle_dir}")

    # 2. Check if already running / push kernel
    is_new_push = False
    ret, status_out = run_kaggle_cli(["kernels", "status", kernel_id])
    if "running" in status_out.lower() or "queued" in status_out.lower():
        print(f"Kernel {kernel_id} is already active: {status_out}")
    else:
        print(f"Submitting GPU kernel to Kaggle: {kernel_id} with accelerator NvidiaTeslaT4...")
        ret, out = run_kaggle_cli(["kernels", "push", "-p", str(bundle_dir), "--accelerator", "NvidiaTeslaT4"])
        print(out)
        if ret != 0:
            print(f"Kernel push failed: {out}")
            sys.exit(1)
        is_new_push = True

    # 3. Monitor
    success = monitor_kernel_execution(kernel_id=kernel_id, wait_for_transition=is_new_push)
    if not success:
        print("Kaggle execution did not complete successfully.")
        # Print logs
        ret, logs = run_kaggle_cli(["kernels", "logs", kernel_id])
        print(f"Execution logs:\n{logs}")
        sys.exit(1)

    # 4. Retrieve & Evaluate
    retrieve_artifacts_and_evaluate(kernel_id=kernel_id)
    print("\nAutomation completed successfully!")


if __name__ == "__main__":
    main()
