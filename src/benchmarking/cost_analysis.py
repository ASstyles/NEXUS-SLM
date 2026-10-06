"""Cost analysis comparing self-hosted NEXUS-SLM with commercial API models."""

from typing import Any, Dict


class CostCalculator:
    """
    Computes comparative economic projections for support ticket triage.
    Assumptions:
      - Average input prompt: ~120 tokens
      - Average structured output: ~45 tokens (total ~165 tokens/request)
      - Self-hosted GPU: AWS g4dn.xlarge (NVIDIA T4) @ $0.526/hr or local edge node @ $0.05/hr
      - Self-hosted throughput: ~25 concurrent requests/sec
      - Commercial API (Tier 1 - GPT-4o-mini / Flash): $0.15 / 1M input, $0.60 / 1M output
      - Commercial API (Tier 2 - Premium GPT-4o / Claude 3.5 Sonnet): $2.50 / 1M input, $10.00 / 1M output
    """

    AVG_INPUT_TOKENS = 120
    AVG_OUTPUT_TOKENS = 45

    # Self-hosted cloud pricing (USD)
    CLOUD_HOURLY_RATE = 0.526  # AWS g4dn.xlarge
    HOURS_PER_MONTH = 730

    # API Model Pricing (per 1,000,000 tokens)
    TIER1_INPUT_PER_M = 0.15
    TIER1_OUTPUT_PER_M = 0.60

    TIER2_INPUT_PER_M = 2.50
    TIER2_OUTPUT_PER_M = 10.00

    @classmethod
    def calculate_api_cost(cls, num_requests: int, tier: str = "tier1") -> float:
        input_m = (num_requests * cls.AVG_INPUT_TOKENS) / 1_000_000.0
        output_m = (num_requests * cls.AVG_OUTPUT_TOKENS) / 1_000_000.0

        if tier == "tier1":
            cost = (input_m * cls.TIER1_INPUT_PER_M) + (output_m * cls.TIER1_OUTPUT_PER_M)
        else:
            cost = (input_m * cls.TIER2_INPUT_PER_M) + (output_m * cls.TIER2_OUTPUT_PER_M)
        return round(cost, 4)

    @classmethod
    def generate_comparison_report(cls) -> Dict[str, Any]:
        """Generates comprehensive cost breakdown across scales."""
        req_tiers = [1_000, 10_000, 100_000, 1_000_000]

        cloud_fixed_monthly = round(cls.CLOUD_HOURLY_RATE * cls.HOURS_PER_MONTH, 2)

        # For self-hosted on shared or burst infrastructure, marginal compute cost per 1k reqs:
        # 1k reqs at 25 req/s = 40 seconds of GPU time = (40 / 3600) * 0.526 = $0.0058 / 1k requests
        marginal_cost_per_1k = round((1000.0 / (25.0 * 3600.0)) * cls.CLOUD_HOURLY_RATE, 4)

        projections = {}
        for n in req_tiers:
            projections[f"{n // 1000}k_requests"] = {
                "self_hosted_marginal_usd": round(marginal_cost_per_1k * (n / 1000), 3),
                "large_llm_tier1_usd": cls.calculate_api_cost(n, "tier1"),
                "large_llm_premium_usd": cls.calculate_api_cost(n, "tier2"),
            }

        return {
            "pricing_assumptions": {
                "avg_input_tokens_per_ticket": cls.AVG_INPUT_TOKENS,
                "avg_output_tokens_per_ticket": cls.AVG_OUTPUT_TOKENS,
                "self_hosted_instance": "AWS g4dn.xlarge (1x NVIDIA T4, 16GB VRAM)",
                "self_hosted_instance_hourly_usd": cls.CLOUD_HOURLY_RATE,
                "commercial_api_tier1": "GPT-4o-mini / Gemini 1.5 Flash ($0.15 in / $0.60 out per 1M tokens)",
                "commercial_api_premium": "GPT-4o / Claude 3.5 Sonnet ($2.50 in / $10.00 out per 1M tokens)",
                "status": "ESTIMATED_PROJECTIONS",
            },
            "fixed_monthly_costs": {
                "self_hosted_dedicated_24_7_usd": cloud_fixed_monthly,
            },
            "variable_projections": projections,
            "cost_per_1k_summary": {
                "self_hosted_nexus_slm_usd": marginal_cost_per_1k,
                "large_llm_tier1_usd": cls.calculate_api_cost(1000, "tier1"),
                "large_llm_premium_usd": cls.calculate_api_cost(1000, "tier2"),
            },
        }


if __name__ == "__main__":
    import json
    print(json.dumps(CostCalculator.generate_comparison_report(), indent=2))
