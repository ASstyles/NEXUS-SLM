"""Experiment tracking integration with MLflow and local run registry."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Manages experiment runs, logging parameters, metrics, and persisting metadata."""

    def __init__(
        self,
        experiment_name: str = "nexus-slm-triage",
        tracking_uri: str = "./mlruns",
        local_registry_path: str = "./experiments/runs.json",
    ):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "sqlite:///mlruns.db"
        self.local_registry_path = Path(local_registry_path)
        self.local_registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_run_id: Optional[str] = None
        self.mlflow_available = False

        self._init_mlflow()
        self._ensure_local_registry()

    def _init_mlflow(self) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_available = True
            logger.info(f"MLflow initialized: URI='{self.tracking_uri}', Exp='{self.experiment_name}'")
        except Exception as e:
            logger.warning(f"MLflow initialization skipped: {e}. Using local JSON tracking.")
            self.mlflow_available = False

    def _ensure_local_registry(self) -> None:
        if not self.local_registry_path.exists():
            with open(self.local_registry_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Starts an experiment run and returns run_id."""
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{run_name}"
        self.active_run_id = run_id

        if self.mlflow_available:
            try:
                import mlflow
                active = mlflow.start_run(run_name=run_name, tags=tags)
                self.active_run_id = active.info.run_id
            except Exception as e:
                logger.warning(f"Failed to start MLflow run: {e}")

        logger.info(f"Started experiment run: '{run_name}' (ID: {self.active_run_id})")
        return self.active_run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Logs hyperparameters and configuration values."""
        if self.mlflow_available:
            try:
                import mlflow
                # Flatten complex values
                flat_params = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in params.items()}
                mlflow.log_params(flat_params)
            except Exception as e:
                logger.warning(f"MLflow log_params failed: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Logs quantitative evaluation and performance metrics."""
        if self.mlflow_available:
            try:
                import mlflow
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"MLflow log_metrics failed: {e}")

    def end_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        status: str = "FINISHED",
    ) -> Dict[str, Any]:
        """Finalizes run and saves persistent record into local registry."""
        if self.mlflow_available:
            try:
                import mlflow
                mlflow.end_run(status=status)
            except Exception as e:
                logger.warning(f"MLflow end_run failed: {e}")

        record = {
            "run_id": self.active_run_id,
            "run_name": run_name,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "params": params,
            "metrics": metrics,
        }

        # Update local runs.json
        runs: List[Dict[str, Any]] = []
        if self.local_registry_path.exists():
            try:
                with open(self.local_registry_path, "r", encoding="utf-8") as f:
                    runs = json.load(f)
            except Exception:
                runs = []

        runs.append(record)
        with open(self.local_registry_path, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)

        logger.info(f"Experiment run {run_name} recorded in {self.local_registry_path}")
        return record

    def list_runs(self) -> List[Dict[str, Any]]:
        """Returns all recorded experiment runs."""
        if not self.local_registry_path.exists():
            return []
        try:
            with open(self.local_registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
