"""Inference engine for NEXUS-SLM with latency tracking and confidence scoring."""

import time
from typing import Any, Dict, Optional, Tuple
import torch
from src.data.schemas import TriagePrediction
from src.inference.parser import OutputParser
from src.models.prompt_formatter import PromptFormatter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """Runs structured support ticket triage inference with real-time latency and confidence."""

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_version: str = "nexus-v1.0",
        max_new_tokens: int = 60,
        temperature: float = 0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_version = model_version
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.parser = OutputParser()

    def _compute_confidence(self, scores: Tuple[torch.Tensor, ...]) -> Optional[float]:
        """
        Computes mean token confidence across generated tokens using softmax probabilities.
        Returns None when scores are missing or cannot be computed.
        """
        if not scores:
            return None
        try:
            probs = []
            for step_logits in scores:
                step_prob = torch.softmax(step_logits[0], dim=-1)
                max_prob = torch.max(step_prob).item()
                probs.append(max_prob)
            if probs:
                return round(sum(probs) / len(probs), 4)
        except Exception as e:
            logger.warning(f"Confidence computation fallback: {e}")
        return None

    def predict(self, text: str) -> TriagePrediction:
        """
        Processes a single customer support ticket and returns structured triage prediction.
        Measures exact wall-clock latency.
        """
        start_time = time.perf_counter()

        # Handle case where model is not initialized (e.g. unit test or pre-load test)
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "InferenceEngine model is not loaded. Load a model before executing inference."
            )

        # 1. Format ChatML prompt
        prompt = PromptFormatter.format_prompt(text, tokenizer=self.tokenizer)

        if hasattr(self.model, "create_completion"):
            import math
            can_logprobs = getattr(self.model, "logits_all", getattr(self.model, "_logits_all", False))
            res = self.model(
                prompt,
                max_tokens=self.max_new_tokens,
                stop=["<|im_end|>", "<|endoftext|>"],
                temperature=self.temperature if self.temperature > 0 else 0.0,
                logprobs=1 if can_logprobs else None,
            )
            choice = res["choices"][0]
            raw_text = choice["text"].strip()
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            if can_logprobs and choice.get("logprobs"):
                token_logprobs = choice.get("logprobs", {}).get("token_logprobs", [])
                probs = [math.exp(lp) for lp in token_logprobs if lp is not None]
                confidence = round(sum(probs) / len(probs), 4) if probs else None
            else:
                confidence = None
        else:
            # 2. Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(self.model, "parameters"):
                try:
                    device = next(self.model.parameters()).device
                except (StopIteration, TypeError):
                    device = getattr(self.model, "device", torch.device("cpu"))
            else:
                device = getattr(self.model, "device", torch.device("cpu"))
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_length = inputs["input_ids"].shape[1]

            # 3. Model Generation
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=(self.temperature > 0),
                    temperature=self.temperature if self.temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # 4. Extract generated tokens
            generated_tokens = output.sequences[0][prompt_length:]
            raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            confidence = self._compute_confidence(output.scores)

        # 5. Parse and strictly validate JSON
        is_valid, parsed_data, err = self.parser.parse_and_validate(raw_text)

        if not is_valid or parsed_data is None:
            logger.warning(f"Structured output parse error: {err}. Raw output: '{raw_text}'")
            # Never fabricate missing fields; record invalid state
            return TriagePrediction(
                category="general_query",  # Safe triage fallback
                priority="medium",
                language="english",
                requires_human=True,  # Always escalate on parse failure
                confidence=None,
                latency_ms=latency_ms,
                model_version=self.model_version,
                raw_output=raw_text,
                is_valid=False,
                error_message=err,
            )

        return TriagePrediction(
            category=parsed_data["category"],
            priority=parsed_data["priority"],
            language=parsed_data["language"],
            requires_human=parsed_data["requires_human"],
            confidence=confidence,
            latency_ms=latency_ms,
            model_version=self.model_version,
            raw_output=raw_text,
            is_valid=True,
            raw_parsed=parsed_data,
        )
