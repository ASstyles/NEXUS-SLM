"""Environment reproducibility and standalone dependency verification suite."""

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

CORE_DEPENDENCIES = [
    "torch",
    "transformers",
    "peft",
    "fastapi",
    "uvicorn",
    "pydantic",
    "sklearn",
    "numpy",
    "psutil",
    "httpx",
]


def test_dependency_imports() -> Dict[str, Any]:
    logger.info("Verifying core dependency imports from environment...")
    import_results = {}
    for pkg in CORE_DEPENDENCIES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "installed")
            import_results[pkg] = {"status": "SUCCESS", "version": version}
        except ImportError as e:
            import_results[pkg] = {"status": "FAILED", "error": str(e)}
    return import_results


def test_dataset_independent_inference() -> Dict[str, Any]:
    """Tests that the model and API can triage an arbitrary unseen ticket without reading any dataset files."""
    logger.info("Executing dataset-independent inference...")
    from fastapi.testclient import TestClient
    from api.main import app

    unseen_ticket = "I bought a lifetime plan during Black Friday but my account shows free trial expired."
    with TestClient(app) as client:
        t0 = time.perf_counter()
        resp = client.post("/predict", json={"ticket": unseen_ticket})
        latency = (time.perf_counter() - t0) * 1000

        if resp.status_code == 200:
            data = resp.json()
            return {
                "status": "SUCCESS",
                "status_code": resp.status_code,
                "latency_ms": round(latency, 2),
                "prediction": {
                    "category": data.get("category"),
                    "priority": data.get("priority"),
                    "language": data.get("language"),
                    "requires_human": data.get("requires_human"),
                    "confidence": data.get("confidence"),
                }
            }
        else:
            return {
                "status": "FAILED",
                "status_code": resp.status_code,
                "error": resp.text,
            }


def test_health_check() -> Dict[str, Any]:
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        resp = client.get("/health")
        return {
            "status": "SUCCESS" if resp.status_code == 200 else "FAILED",
            "data": resp.json(),
        }


def main():
    dep_results = test_dependency_imports()
    health_results = test_health_check()
    inf_results = test_dataset_independent_inference()

    all_deps_ok = all(v["status"] == "SUCCESS" for v in dep_results.values())
    health_ok = health_results["status"] == "SUCCESS"
    inf_ok = inf_results["status"] == "SUCCESS"

    overall_pass = all_deps_ok and health_ok and inf_ok

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "dependencies": dep_results,
        "health_check": health_results,
        "dataset_independent_inference": inf_results,
    }

    out_file = Path("experiments/artifacts/reproducibility_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Reproducibility audit saved to {out_file}")
    print(f"\nReproducibility Audit: {report['overall_status']}")
    for k, v in dep_results.items():
        print(f"  {k}: {v.get('version', v.get('status'))}")


if __name__ == "__main__":
    main()
