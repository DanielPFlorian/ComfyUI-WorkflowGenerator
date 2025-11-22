"""
HuggingFace model loader using transformers with GPU layer management.
"""

import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..model_manager import auto_detect_tokenizer_path
from .device_utils import (
    get_device_for_model,
    get_dtype_for_model,
)


class HuggingFaceModelWrapper:
    """Wrapper for HuggingFace models using transformers with GPU layer management."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None = None,
        dtype: str | torch.dtype = "auto",
        device_preference: str = "auto",
        max_memory: dict | None = None,
        attn_implementation: str = "auto",
        **kwargs,
    ):
        """
        Initialize HuggingFace model wrapper.

        Args:
            model_path: Path to HuggingFace model directory
            tokenizer_path: Optional separate tokenizer path (defaults to model_path)
            dtype: "auto", "fp16", "bf16", "fp32", "fp8", or torch.dtype
            device_preference: "auto", "cuda", or "cpu"
            max_memory: Dict for device_map memory limits (e.g., {0: "10GB", "cpu": "30GB"})
            attn_implementation: "auto", "flash_attention_2", or "sdpa"
            **kwargs: Additional arguments for AutoModelForCausalLM.from_pretrained
        """
        logging.info(f"Loading HuggingFace model: {os.path.basename(model_path)}")
        load_start = time.time()

        device = get_device_for_model(device_preference)
        logging.debug(f"Device preference: {device_preference}, selected: {device}")

        if isinstance(dtype, str):
            if dtype == "auto" and device_preference == "auto":
                torch_dtype = get_dtype_for_model("fp16", device)
            else:
                torch_dtype = get_dtype_for_model(dtype, device)
        else:
            torch_dtype = dtype

        logging.debug(f"Using dtype: {torch_dtype}")

        if attn_implementation == "auto":
            try:
                import flash_attn  # noqa: F401

                if device == "cuda" and torch.cuda.is_available():
                    attn_impl = "flash_attention_2"
                    logging.debug("Auto-detected attention: flash_attention_2")
                else:
                    attn_impl = "sdpa"
                    logging.debug("Auto-detected attention: sdpa")
            except ImportError:
                attn_impl = "sdpa"
                logging.debug("Auto-detected attention: sdpa")
        else:
            attn_impl = attn_implementation
            if attn_impl == "flash_attention_2" and (device != "cuda" or not torch.cuda.is_available()):
                logging.warning("flash_attention_2 requires NVIDIA CUDA, but CUDA is not available. Falling back to sdpa.")
                attn_impl = "sdpa"
            logging.debug(f"Using attention: {attn_impl}")

        load_kwargs = {
            "dtype": torch_dtype,
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
        }

        if device_preference == "cpu":
            load_kwargs["device_map"] = {"": "cpu"}
            logging.debug("Using CPU only")
        elif device == "cuda":
            load_kwargs["device_map"] = "auto"
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
                logging.debug("Using CUDA with device_map='auto' and max_memory limits")
            else:
                logging.debug("Using CUDA with device_map='auto' (automatic GPU/CPU split)")
        else:
            load_kwargs["device_map"] = {"": "cpu"}
            logging.debug("Using CPU only")

        quantization_enabled = False
        if "quantization_config" in kwargs:
            quantization_enabled = True
            logging.debug("8-bit quantization enabled")

        load_kwargs.update(kwargs)

        if quantization_enabled:
            logging.info("Quantizing model to 8-bit (this may take a few minutes for large models)...")

        if tokenizer_path is None:
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
            model_path = os.path.normpath(model_path)
            if model_path.endswith(".gguf"):
                raise ValueError(
                    f"Invalid model path for HuggingFace format: {model_path}\nHuggingFace models must be directories, not .gguf files.\n"
                )
            tokenizer_path = auto_detect_tokenizer_path(model_path, model_format="huggingface")
            logging.debug(f"Auto-detected tokenizer path: {tokenizer_path}")

        tokenizer_load_path = tokenizer_path if tokenizer_path else model_path

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        load_time = time.time() - load_start
        logging.info(f"HuggingFace model loaded in {load_time:.2f}s")

        logging.debug(f"Loading tokenizer from: {tokenizer_load_path}")
        tokenizer_start = time.time()
        tokenizer_path_abs = os.path.abspath(tokenizer_load_path) if not os.path.isabs(tokenizer_load_path) else tokenizer_load_path

        try:
            from huggingface_hub.utils import HFValidationError
        except ImportError:
            HFValidationError = Exception

        if os.path.exists(tokenizer_path_abs) and os.path.isdir(tokenizer_path_abs):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_abs, trust_remote_code=True, local_files_only=True)
            except (HFValidationError, Exception) as e:
                if isinstance(e, HFValidationError) or "Repo id must use alphanumeric" in str(e):
                    logging.warning(f"Tokenizer path validation error, retrying with explicit local path handling: {e}")
                else:
                    logging.warning(f"Failed to load tokenizer with local_files_only=True: {e}")
                logging.warning("Trying without local_files_only...")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_abs, trust_remote_code=True)
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True)
            except (HFValidationError, Exception) as e:
                if (isinstance(e, HFValidationError) or "Repo id must use alphanumeric" in str(e)) and (
                    os.path.sep in tokenizer_load_path
                    or (os.name == "nt" and len(tokenizer_load_path) > 1 and tokenizer_load_path[1] == ":")
                ):
                    raise ValueError(
                        f"Tokenizer path does not exist: {tokenizer_path_abs}\n"
                        f"Please ensure the tokenizer directory exists at this location."
                    ) from e
                raise
        tokenizer_time = time.time() - tokenizer_start
        logging.debug(f"Tokenizer loaded in {tokenizer_time:.2f}s")

        # Fix for "The attention mask and the pad token id were not set" warning
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.debug("Tokenizer pad_token was None, set to eos_token")

        if hasattr(self.model, "device"):
            self.device = self.model.device
        elif hasattr(self.model, "hf_device_map"):
            self.device = torch.device(device) if device == "cuda" else torch.device("cpu")
        else:
            self.device = next(self.model.parameters()).device

        try:
            if torch.cuda.is_available() and device == "cuda":
                vram_used = torch.cuda.memory_allocated() / 1024**3
                logging.debug(f"GPU memory allocated: {vram_used:.2f} GB")
        except Exception:
            pass

    def generate(
        self, input_ids, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True, **kwargs
    ):
        """
        Generate text using HuggingFace model.

        Args:
            input_ids: Token IDs tensor
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters (seed will be extracted and used separately)

        Returns:
            Generated token IDs tensor
        """
        seed = kwargs.pop("seed", None)

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if torch.is_tensor(input_ids):
            if isinstance(self.device, dict):
                device = list(self.device.values())[0] if self.device else torch.device("cpu")
            else:
                device = self.device
            input_ids = input_ids.to(device)

        # Create attention mask if not provided
        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is None:
            # Assuming input_ids contains only valid tokens (no padding), create full mask
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )

        return generated_ids

    def __del__(self):
        """Cleanup method for VRAM management."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
