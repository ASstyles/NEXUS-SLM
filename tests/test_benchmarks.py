"""Unit tests for Cost Calculator and Large LLM Adapter."""

import pytest
from src.benchmarking.cost_analysis import CostCalculator
from src.benchmarking.large_llm_adapter import LargeLLMAdapter
from src.benchmarking.comparison_table import BenchmarkComparisonTable


def test_cost_calculator():
    report = CostCalculator.generate_comparison_report()
    assert "pricing_assumptions" in report
    assert "cost_per_1k_summary" in report
    assert report["cost_per_1k_summary"]["self_hosted_nexus_slm_usd"] < report["cost_per_1k_summary"]["large_llm_tier1_usd"]


def test_benchmark_table_markdown():
    table = BenchmarkComparisonTable()
    md = table.render_markdown()
    assert "| Metric | Base SLM | NEXUS-SLM | Large LLM |" in md
    assert "Macro-F1" in md
    assert "P50 Latency" in md


def test_large_llm_adapter_unconfigured():
    adapter = LargeLLMAdapter(provider="openai", api_key="")
    assert adapter.is_available() is False
    with pytest.raises(RuntimeError) as exc_info:
        adapter.predict("Test query")
    assert "NOT RUN" in str(exc_info.value)
