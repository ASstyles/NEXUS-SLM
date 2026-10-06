"""Model and Tokenizer loader supporting FP16, INT8, and 4-bit QLoRA."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from src.utils.hardware import get_hardware_info
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Loads base SLMs, quantizes weights, and attaches PEFT LoRA adapters."""

    @staticmethod
    def load_tokenizer(
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code: bool = True,
    ) -> Any:
        """Loads and configures the Hugging Face tokenizer."""
        from transformers import AutoTokenizer

        logger.info(f"Loading tokenizer from: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def load_model(
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        quantization: str = "4bit",  # "4bit", "8bit", "fp16", "fp32"
        device: str = "auto",
        trust_remote_code: bool = True,
    ) -> Tuple[Any, Any]:
        """
        Loads base model with optional quantization and LoRA adapter.
        Returns:
            (model, tokenizer)
        """
        import torch
        from transformers import AutoModelForCausalLM

        tokenizer = ModelLoader.load_tokenizer(model_name_or_path, trust_remote_code)
        hw_info = get_hardware_info()
        cuda_avail = hw_info["cuda_available"]

        # Determine torch dtype and quantization configuration
        bnb_config = None
        torch_dtype = torch.float32

        if cuda_avail and device != "cpu":
            torch_dtype = torch.float16

            if quantization == "4bit" and hw_info["bitsandbytes_available"]:
                from transformers import BitsAndBytesConfig
                logger.info("Configuring 4-bit NormalFloat4 (NF4) quantization...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif quantization == "8bit" and hw_info["bitsandbytes_available"]:
                from transformers import BitsAndBytesConfig
                logger.info("Configuring 8-bit Int8 quantization...")
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "fp16":
                logger.info("Loading in half-precision FP16 (no quantization)...")
        else:
            logger.info("CUDA or bitsandbytes unavailable; falling back to CPU execution.")
            device = "cpu"
            torch_dtype = torch.float32

        logger.info(f"Loading base model '{model_name_or_path}' with dtype={torch_dtype}...")
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        elif cuda_avail and device != "cpu":
            model_kwargs["device_map"] = device
        else:
            model_kwargs["device_map"] = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        # Attach LoRA adapter if provided
        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            logger.info(f"Attaching LoRA adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()

        return model, tokenizer
