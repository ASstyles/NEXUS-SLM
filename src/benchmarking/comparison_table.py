"""Final benchmark comparison generator for Base SLM vs NEXUS-SLM vs Large LLM."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from src.benchmarking.cost_analysis import CostCalculator


class BenchmarkComparisonTable:
    """
    Generates side-by-side benchmark comparison matrices across Base SLM, NEXUS-SLM, and Large LLMs.
    Strict rule: Only populates values from actual measurements. Missing values are displayed as '—'.
    """

    def __init__(
        self,
        base_results_path: Optional[str] = "experiments/baseline/baseline_results.json",
        nexus_results_path: Optional[str] = None,
        large_llm_results_path: Optional[str] = None,
    ):
        self.base_path = Path(base_results_path) if base_results_path else None
        self.nexus_path = Path(nexus_results_path) if nexus_results_path else None
        self.large_path = Path(large_llm_results_path) if large_llm_results_path else None

    def _load_json(self, path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if path and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _find_latest_evaluation(self, prefix: str = "eval_") -> Optional[Dict[str, Any]]:
        eval_dir = Path("experiments/evaluations")
        if not eval_dir.exists():
            return None
        matching = sorted(eval_dir.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matching:
            return self._load_json(matching[0])
        return None

    def generate_matrix(self) -> Dict[str, Any]:
        """Loads actual benchmark measurements and structures the comparison matrix."""
        base_data = self._load_json(self.base_path)
        nexus_data = self._load_json(self.nexus_path) or self._find_latest_evaluation("eval_")
        large_data = self._load_json(self.large_path)

        def _fmt(val: Any, suffix: str = "", default: str = "-") -> str:
            if val is None or val == "":
                return default
            if isinstance(val, float):
                return f"{val:.4f}{suffix}" if val < 1.0 else f"{val:.1f}{suffix}"
            return f"{val}{suffix}"

        cost_report = CostCalculator.generate_comparison_report()
        cost_1k = cost_report["cost_per_1k_summary"]

        rows = [
            {
                "metric": "Macro-F1",
                "base_slm": _fmt(base_data.get("macro_f1") if base_data else None),
                "nexus_slm": _fmt(nexus_data.get("macro_f1") if nexus_data else None),
                "large_llm": _fmt(large_data.get("macro_f1") if large_data else None),
            },
            {
                "metric": "Weighted-F1",
                "base_slm": _fmt(base_data.get("weighted_f1") if base_data else None),
                "nexus_slm": _fmt(nexus_data.get("weighted_f1") if nexus_data else None),
                "large_llm": _fmt(large_data.get("weighted_f1") if large_data else None),
            },
            {
                "metric": "Category Accuracy",
                "base_slm": _fmt(base_data.get("category_accuracy") if base_data else None),
                "nexus_slm": _fmt(nexus_data.get("category_accuracy") if nexus_data else None),
                "large_llm": _fmt(large_data.get("category_accuracy") if large_data else None),
            },
            {
                "metric": "Priority Accuracy",
                "base_slm": _fmt(base_data.get("priority_accuracy") if base_data else None),
                "nexus_slm": _fmt(nexus_data.get("priority_accuracy") if nexus_data else None),
                "large_llm": _fmt(large_data.get("priority_accuracy") if large_data else None),
            },
            {
                "metric": "JSON Validity",
                "base_slm": _fmt(base_data.get("json_validity_rate") if base_data else None),
                "nexus_slm": _fmt(nexus_data.get("json_validity_rate") if nexus_data else None),
                "large_llm": _fmt(large_data.get("json_validity_rate") if large_data else None),
            },
            {
                "metric": "P50 Latency (ms)",
                "base_slm": _fmt(base_data.get("latency_p50_ms") if base_data else None, " ms"),
                "nexus_slm": _fmt(nexus_data.get("latency_p50_ms") if nexus_data else None, " ms"),
                "large_llm": _fmt(large_data.get("latency_p50_ms") if large_data else None, " ms"),
            },
            {
                "metric": "P95 Latency (ms)",
                "base_slm": _fmt(base_data.get("latency_p95_ms") if base_data else None, " ms"),
                "nexus_slm": _fmt(nexus_data.get("latency_p95_ms") if nexus_data else None, " ms"),
                "large_llm": _fmt(large_data.get("latency_p95_ms") if large_data else None, " ms"),
            },
            {
                "metric": "P99 Latency (ms)",
                "base_slm": _fmt(base_data.get("latency_p99_ms") if base_data else None, " ms"),
                "nexus_slm": _fmt(nexus_data.get("latency_p99_ms") if nexus_data else None, " ms"),
                "large_llm": _fmt(large_data.get("latency_p99_ms") if large_data else None, " ms"),
            },
            {
                "metric": "VRAM / Memory",
                "base_slm": _fmt(base_data.get("vram_mb") if base_data else None, " MB"),
                "nexus_slm": _fmt(nexus_data.get("vram_mb") if nexus_data else None, " MB"),
                "large_llm": "Cloud API (0 MB local)",
            },
            {
                "metric": "Cost / 1K Reqs (est)",
                "base_slm": f"${cost_1k['self_hosted_nexus_slm_usd']:.4f}",
                "nexus_slm": f"${cost_1k['self_hosted_nexus_slm_usd']:.4f}",
                "large_llm": f"${cost_1k['large_llm_tier1_usd']:.4f}",
            },
            {
                "metric": "Local Inference",
                "base_slm": "Yes",
                "nexus_slm": "Yes",
                "large_llm": "No (Cloud API)",
            },
        ]

        return {
            "columns": ["Metric", "Base SLM", "NEXUS-SLM", "Large LLM"],
            "rows": rows,
            "status": {
                "base_measured": bool(base_data),
                "nexus_measured": bool(nexus_data),
                "large_measured": bool(large_data),
            },
        }

    def render_markdown(self) -> str:
        """Renders GitHub-flavored markdown table."""
        matrix = self.generate_matrix()
        md = "| Metric | Base SLM | NEXUS-SLM | Large LLM |\n"
        md += "|---|---:|---:|---:|\n"
        for r in matrix["rows"]:
            md += f"| {r['metric']} | {r['base_slm']} | {r['nexus_slm']} | {r['large_llm']} |\n"
        return md


if __name__ == "__main__":
    table = BenchmarkComparisonTable()
    print(table.render_markdown())
