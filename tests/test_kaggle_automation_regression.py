"""Regression tests for Kaggle GPU Job Submission, Retrieval, and Safeguards.

Proves:
1. The clean notebook is selected (legacy notebook rejected).
2. Legacy model directories remain strictly untouched and protected.
3. Existing artifacts cannot be overwritten (unique run directory enforced).
4. Evaluation failures cause automation failure (exit code propagation).
5. Invalid, corrupted, incomplete downloads and Zip-Slip attacks are rejected.
"""

import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.training.submit_kaggle_gpu_job as kaggle_job
from scripts.training.submit_kaggle_gpu_job import (
    DEFAULT_KERNEL_SLUG,
    DEFAULT_NOTEBOOK_SRC,
    PROTECTED_MODEL_DIRS,
    check_protected_path,
    prepare_kernel_bundle,
    retrieve_artifacts_and_evaluate,
    safe_extract_zip,
    verify_adapter_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_clean_notebook_is_selected():
    """Verify clean-data notebook is selected and legacy notebook is discarded."""
    assert DEFAULT_NOTEBOOK_SRC.exists(), f"Notebook source must exist: {DEFAULT_NOTEBOOK_SRC}"
    assert "clean_data" in DEFAULT_NOTEBOOK_SRC.name, (
        f"Selected notebook must be clean-data notebook, got: {DEFAULT_NOTEBOOK_SRC.name}"
    )
    assert DEFAULT_NOTEBOOK_SRC.name != "nexus_qwen25_05b_lora_training.ipynb", (
        "Legacy defective notebook must NOT be selected"
    )
    assert DEFAULT_KERNEL_SLUG == "nexus-qwen25-clean-v2-training"


def test_prepare_kernel_bundle_segregation():
    """Verify bundle contains clean notebook and isolated kernel metadata."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path = Path(tmp_dir) / "test_bundle"
        result_dir = prepare_kernel_bundle(
            username="test_user",
            kernel_slug="nexus-qwen25-clean-v2-training",
            notebook_src=DEFAULT_NOTEBOOK_SRC,
            bundle_dir=bundle_path,
        )

        assert result_dir.exists()
        meta_file = result_dir / "kernel-metadata.json"
        assert meta_file.exists()

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["id"] == "test_user/nexus-qwen25-clean-v2-training"
        assert meta["code_file"] == DEFAULT_NOTEBOOK_SRC.name
        assert meta["machine_shape"] == "NvidiaTeslaT4"
        assert meta["enable_gpu"] is True
        assert meta["is_private"] is True


def test_legacy_model_directories_remain_untouched():
    """Verify legacy model directories cannot be targeted or wiped."""
    legacy_qwen = PROJECT_ROOT / "models" / "qwen25_05b_lora_seed42"
    legacy_v1 = PROJECT_ROOT / "models" / "nexus_qlora_v1"
    models_root = PROJECT_ROOT / "models"

    with pytest.raises(PermissionError) as exc_qwen:
        check_protected_path(legacy_qwen)
    assert "SAFETY VIOLATION" in str(exc_qwen.value)

    with pytest.raises(PermissionError) as exc_v1:
        check_protected_path(legacy_v1)
    assert "SAFETY VIOLATION" in str(exc_v1.value)

    with pytest.raises(PermissionError) as exc_root:
        check_protected_path(models_root)
    assert "SAFETY VIOLATION" in str(exc_root.value)


def test_existing_artifacts_cannot_be_overwritten():
    """Verify unique run directory invariant refusing to overwrite existing run."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_base = Path(tmp_dir)
        run_id = "test_run_001"
        existing_run_dir = output_base / f"qwen25_05b_clean_v2_lora_{run_id}"
        existing_run_dir.mkdir(parents=True)
        (existing_run_dir / "previous_artifact.txt").write_text("critical prior data")

        with pytest.raises(FileExistsError) as exc_info:
            retrieve_artifacts_and_evaluate(
                kernel_id="test_user/test-kernel",
                output_base=output_base,
                run_id=run_id,
            )
        assert "already exists and is not empty" in str(exc_info.value)
        # Ensure prior data was NOT destroyed
        assert (existing_run_dir / "previous_artifact.txt").read_text() == "critical prior data"


def test_safe_extract_zip_rejects_empty_archive():
    """Verify empty download zip is rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        empty_zip = Path(tmp_dir) / "empty.zip"
        empty_zip.touch()
        target_dir = Path(tmp_dir) / "extracted"

        with pytest.raises(ValueError) as exc:
            safe_extract_zip(empty_zip, target_dir)
        assert "empty (0 bytes)" in str(exc.value)


def test_safe_extract_zip_rejects_corrupted_archive():
    """Verify corrupted non-zip bytes are rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        corrupted = Path(tmp_dir) / "corrupt.zip"
        corrupted.write_bytes(b"not a real zip archive header")
        target_dir = Path(tmp_dir) / "extracted"

        with pytest.raises(ValueError) as exc:
            safe_extract_zip(corrupted, target_dir)
        assert "not a valid zip archive" in str(exc.value)


def test_safe_extract_zip_prevents_zip_slip():
    """Verify Zip-Slip directory traversal attacks are detected and blocked."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        malicious_zip = Path(tmp_dir) / "malicious.zip"
        with zipfile.ZipFile(malicious_zip, "w") as zf:
            zf.writestr("../../evil_escape.txt", "exploit payload")

        target_dir = Path(tmp_dir) / "safe_sandbox"

        with pytest.raises(ValueError) as exc:
            safe_extract_zip(malicious_zip, target_dir)
        assert "Zip Slip" in str(exc.value)
        assert not (Path(tmp_dir) / "evil_escape.txt").exists()


def test_verify_adapter_artifacts_rejects_incomplete():
    """Verify post-extraction verification catches missing config, missing weights, or 0-byte files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        adapter_dir = Path(tmp_dir)

        # Missing adapter_config.json
        with pytest.raises(FileNotFoundError) as exc_cfg:
            verify_adapter_artifacts(adapter_dir)
        assert "adapter_config.json" in str(exc_cfg.value)

        # Create valid adapter_config.json
        cfg_file = adapter_dir / "adapter_config.json"
        cfg_file.write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}))

        # Missing weights
        with pytest.raises(FileNotFoundError) as exc_weights:
            verify_adapter_artifacts(adapter_dir)
        assert "No adapter weights" in str(exc_weights.value)

        # Empty weights file (0 bytes)
        empty_weights = adapter_dir / "adapter_model.safetensors"
        empty_weights.touch()
        with pytest.raises(ValueError) as exc_empty_w:
            verify_adapter_artifacts(adapter_dir)
        assert "Adapter weight file is empty" in str(exc_empty_w.value)

        # Valid non-empty weights
        empty_weights.write_bytes(b"VALID_WEIGHT_BYTES" * 100)
        summary = verify_adapter_artifacts(adapter_dir)
        assert summary["verified"] is True
        assert summary["weight_file"] == "adapter_model.safetensors"


def test_evaluation_failure_causes_automation_failure():
    """Verify evaluation failure causes retrieve_artifacts_and_evaluate to return False."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_base = Path(tmp_dir)
        run_id = "test_run_fail"

        # Create a mock valid downloaded zip
        download_zip_path = Path(tmp_dir) / "mock_download.zip"
        with zipfile.ZipFile(download_zip_path, "w") as zf:
            zf.writestr("adapter_config.json", json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}))
            zf.writestr("adapter_model.safetensors", b"dummy_weights" * 50)
            zf.writestr("tokenizer.json", "{}")

        # Mock run_kaggle_cli to copy mock_download.zip into the raw_download dir
        def mock_kaggle(args):
            if "output" in args:
                target_p = Path(args[args.index("-p") + 1])
                target_p.mkdir(parents=True, exist_ok=True)
                shutil.copy2(download_zip_path, target_p / "output.zip")
                return 0, "Download OK"
            return 0, "OK"

        with patch.object(kaggle_job, "run_kaggle_cli", side_effect=mock_kaggle):
            with patch.object(kaggle_job, "run_clean_evaluation", return_value=False):
                ok, rdir = retrieve_artifacts_and_evaluate(
                    kernel_id="test_user/test-kernel",
                    output_base=output_base,
                    run_id=run_id,
                )
                assert ok is False, "Expected retrieve_artifacts_and_evaluate to return False on eval failure"


def test_get_authenticated_username_detection():
    """Verify safe retrieval of Kaggle username from access_token or environment."""
    from scripts.training.submit_kaggle_gpu_job import get_authenticated_username

    with patch.dict(os.environ, {"KAGGLE_USERNAME": "test_env_user", "KAGGLE_KEY": "dummy_secret"}):
        user = get_authenticated_username()
        assert user == "test_env_user"


