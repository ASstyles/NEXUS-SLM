"""CLI script to generate benchmark comparisons and economic cost projections."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.benchmarking.comparison_table import BenchmarkComparisonTable
from src.benchmarking.cost_analysis import CostCalculator


def main():
    print("\n" + "=" * 70)
    print("NEXUS-SLM OFFICIAL BENCHMARK MATRIX (Base SLM vs NEXUS-SLM vs Large LLM)")
    print("=" * 70)
    table = BenchmarkComparisonTable()
    print(table.render_markdown())

    print("\n" + "=" * 70)
    print("COST & ECONOMIC PROJECTION ANALYSIS:")
    print("=" * 70)
    cost_report = CostCalculator.generate_comparison_report()
    summary = cost_report["cost_per_1k_summary"]
    print(f"Self-Hosted NEXUS-SLM (Marginal / 1K Reqs): ${summary['self_hosted_nexus_slm_usd']:.4f}")
    print(f"Large LLM Commercial Tier 1 (Cost / 1K Reqs):   ${summary['large_llm_tier1_usd']:.4f}")
    print(f"Large LLM Commercial Premium (Cost / 1K Reqs):  ${summary['large_llm_premium_usd']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
