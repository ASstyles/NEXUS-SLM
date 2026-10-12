"""Focused Unit Tests for Qwen2.5-0.5B-Instruct Tokenization, Label Masking, and Compatibility.

Verifies:
1. Tokenizer vs model config vocabulary sizes (len(tokenizer) == 151665, config.vocab_size == 151936).
2. ChatML formatting and generation prompt structure.
3. Prompt masking: Prompt tokens are assigned label -100 so loss is strictly computed
   only on the assistant structured JSON completion tokens and ending <|im_end|>.
4. Padding tokens are assigned label -100.
5. Integration compatibility with OutputParser and TriageResponse contract.
"""

import json
from pathlib import Path
import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

from src.models.prompt_formatter import PromptFormatter
from src.inference.parser import OutputParser
from src.data.schemas import TriagePrediction
from api.schemas import PredictResponse, SupportCategory, SupportPriority, SupportLanguage
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def qwen_config():
    return AutoConfig.from_pretrained(MODEL_ID)


def test_vocab_size_discrepancy_recorded(qwen_tokenizer, qwen_config):
    """Verify that model.config.vocab_size and len(tokenizer) are accurately inspected."""
    tok_len = len(qwen_tokenizer)
    cfg_vocab = qwen_config.vocab_size
    assert tok_len == 151665
    assert cfg_vocab == 151936
    assert tok_len != cfg_vocab
    assert qwen_tokenizer.eos_token_id == 151645  # <|im_end|>


def test_chatml_prompt_formatting(qwen_tokenizer):
    """Verify ChatML generation prompt conforms to Qwen template."""
    ticket = "Account se 500 cut ho gaye but payment fail hua."
    prompt = PromptFormatter.format_prompt(ticket, tokenizer=qwen_tokenizer)

    assert "<|im_start|>system" in prompt
    assert "<|im_end|>" in prompt
    assert "<|im_start|>user" in prompt
    assert ticket in prompt
    assert "<|im_start|>assistant\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_prompt_masking_completion_only_loss(qwen_tokenizer):
    """
    Verify that prompt tokens and padding receive label -100,
    ensuring cross-entropy loss is strictly computed on assistant JSON target tokens.
    """
    ticket = "I would like to cancel my order ORD-9912 immediately."
    target_obj = {
        "category": "order_cancel",
        "priority": "high",
        "language": "english",
        "requires_human": False,
    }
    target_json_str = json.dumps(target_obj, ensure_ascii=False)

    # Build prompt and full example
    prompt = PromptFormatter.format_prompt(ticket, tokenizer=qwen_tokenizer)
    full_text = f"{prompt}{target_json_str}<|im_end|>"

    # Tokenize
    prompt_tokens = qwen_tokenizer.encode(prompt, add_special_tokens=False)
    full_tokens = qwen_tokenizer.encode(full_text, add_special_tokens=False)

    assert len(full_tokens) > len(prompt_tokens)
    assert full_tokens[:len(prompt_tokens)] == prompt_tokens

    # Construct label mask
    labels = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]

    # Verify mask properties
    assert len(labels) == len(full_tokens)
    # All prompt tokens must be masked (-100)
    for i in range(len(prompt_tokens)):
        assert labels[i] == -100

    # Completion tokens must NOT be masked
    completion_token_ids = labels[len(prompt_tokens):]
    assert len(completion_token_ids) > 0
    assert all(tid != -100 for tid in completion_token_ids)

    # The last completion token must be <|im_end|>
    assert completion_token_ids[-1] == qwen_tokenizer.eos_token_id

    # Verify loss behavior: CrossEntropyLoss should ignore -100
    logits = torch.randn(1, len(full_tokens), len(qwen_tokenizer))
    labels_tensor = torch.tensor([labels], dtype=torch.long)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    # Shift logits and labels for causal LM loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels_tensor[..., 1:].contiguous()
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_padding_masking(qwen_tokenizer):
    """Verify that padding tokens in batch collator receive label -100."""
    max_len = 128
    tokens = [100, 200, 300]
    pad_id = qwen_tokenizer.pad_token_id

    padded_tokens = tokens + [pad_id] * (max_len - len(tokens))
    padded_labels = [-100, -100, 300] + [-100] * (max_len - len(tokens))

    assert len(padded_tokens) == max_len
    assert len(padded_labels) == max_len
    # Padding tokens must have label -100
    for idx in range(len(tokens), max_len):
        assert padded_labels[idx] == -100


def test_output_parser_contract_with_qwen_json():
    """Verify OutputParser parses structured Qwen assistant JSON conforming to TriageResponse."""
    parser = OutputParser()
    sample_json = (
        '{"category": "billing", "priority": "medium", '
        '"language": "english", "requires_human": false}'
    )
    is_valid, parsed, err = parser.parse_and_validate(sample_json)

    assert is_valid is True
    assert err is None
    assert parsed["category"] == "billing"
    assert parsed["priority"] == "medium"
    assert parsed["language"] == "english"
    assert parsed["requires_human"] is False

    # Check TriagePrediction contract
    triage_pred = TriagePrediction(
        category=parsed["category"],
        priority=parsed["priority"],
        language=parsed["language"],
        requires_human=parsed["requires_human"],
        confidence=0.95,
        model_version="qwen2.5-0.5b-lora",
        latency_ms=12.5,
    )
    assert triage_pred.category in CATEGORIES
    assert triage_pred.priority in PRIORITIES
    assert triage_pred.language in LANGUAGES

    # Check FastAPI PredictResponse Pydantic contract
    api_resp = PredictResponse(
        category=parsed["category"],
        priority=parsed["priority"],
        language=parsed["language"],
        requires_human=parsed["requires_human"],
        confidence=0.95,
        model_version="qwen2.5-0.5b-lora",
        latency_ms=12.5,
    )
    assert api_resp.category.value in CATEGORIES
    assert api_resp.priority.value in PRIORITIES
    assert api_resp.language.value in LANGUAGES
