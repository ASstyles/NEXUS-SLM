"""Model registry for versioning, metadata tracking, and deployment status."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Tracks model versions (e.g. nexus-v0.1, nexus-v1.0), SHA256 hashes, evaluation metrics, and deployment status."""

    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        if not self.registry_path.exists():
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def _calculate_dir_hash(self, directory_path: Path) -> str:
        """Calculates a deterministic SHA256 hash of directory files."""
        hasher = hashlib.sha256()
        if not directory_path.exists():
            fallback_h = hashlib.sha256(str(directory_path).encode()).hexdigest()
            return f"sha256_{fallback_h}"

        for file_path in sorted(directory_path.glob("**/*")):
            if file_path.is_file():
                hasher.update(file_path.name.encode())
                try:
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            hasher.update(chunk)
                except Exception:
                    pass
        return f"sha256_{hasher.hexdigest()}"

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns all registered model versions."""
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def get_model(self, version: str) -> Optional[Dict[str, Any]]:
        """Retrieves record for specific version tag."""
        models = self.list_models()
        for m in models:
            if m["version"] == version:
                return m
        return None

    def register_model(
        self,
        version: str,
        base_model_name: str,
        adapter_path: str,
        dataset_version: str = "dataset_v1",
        quantization: str = "4bit",
        training_config: Optional[Dict[str, Any]] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        deployment_status: str = "STAGED",  # "ACTIVE", "STAGED", "ARCHIVED"
    ) -> Dict[str, Any]:
        """Registers or updates a model version in the registry."""
        models = self.list_models()
        p = Path(adapter_path)
        model_hash = self._calculate_dir_hash(p)

        record = {
            "version": version,
            "base_model": base_model_name,
            "adapter_path": str(p),
            "model_hash": model_hash,
            "dataset_version": dataset_version,
            "quantization": quantization,
            "training_config": training_config or {},
            "evaluation_results": evaluation_results or {},
            "deployment_status": deployment_status,
            "registered_at": datetime.utcnow().isoformat(),
        }

        # If setting to ACTIVE, demote other active models
        if deployment_status == "ACTIVE":
            for m in models:
                if m.get("deployment_status") == "ACTIVE" and m["version"] != version:
                    m["deployment_status"] = "ARCHIVED"

        # Replace existing or append
        existing_idx = next((i for i, m in enumerate(models) if m["version"] == version), None)
        if existing_idx is not None:
            models[existing_idx] = record
        else:
            models.append(record)

        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(models, f, indent=2)

        logger.info(f"Model {version} registered with status={deployment_status} (Hash: {model_hash[:10]}...)")
        return record

    def set_active_version(self, version: str) -> bool:
        """Promotes a registered version to ACTIVE."""
        models = self.list_models()
        found = False
        for m in models:
            if m["version"] == version:
                m["deployment_status"] = "ACTIVE"
                found = True
            elif m.get("deployment_status") == "ACTIVE":
                m["deployment_status"] = "ARCHIVED"

        if found:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(models, f, indent=2)
            logger.info(f"Set model version {version} to ACTIVE.")
            return True
        return False
