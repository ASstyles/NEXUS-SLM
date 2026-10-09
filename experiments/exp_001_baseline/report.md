# Experiment 001: Base SLM Baseline Evaluation

- **Model**: `HuggingFaceTB/SmolLM-135M-Instruct` (134,515,008 parameters)
- **Execution Mode**: Zero-Shot Base Model (No Fine-Tuning)
- **Dataset**: `dataset_v1` Locked Test Set (42 examples)
- **Date**: 2026-10-06T12:12:52.418203

## Measured Metrics

| Metric | Measured Value |
|---|---:|
| **Macro-F1** | 0.0000 |
| **Weighted-F1** | 0.0000 |
| **Category Accuracy** | 0.00% |
| **JSON Validity Rate** | 0.00% |
| **Exact Match Accuracy** | 0.00% |
| **P50 Latency** | 4888.22 ms |
| **P95 Latency** | 5396.91 ms |
| **Throughput** | 0.2 req/s |
| **Peak RAM** | 13196.75 MB |
| **Model Size** | 513.13 MB |

## Empirical Observations
1. In zero-shot mode without fine-tuning, the base SLM struggles significantly with adhering to structured JSON outputs.
2. The model frequently emits conversational text or python code snippets (`def support_ticket(...)`) rather than valid JSON.
