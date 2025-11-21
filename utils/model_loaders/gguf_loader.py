"""
GGUF model loader using llama-cpp-python with GPU layer management.
"""

import logging
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from ..model_manager import auto_detect_tokenizer_path
from .device_utils import (
    calculate_optimal_gpu_layers,
    get_available_vram,
    get_device_for_model,
)


class GGUFModelWrapper:
    """Wrapper for GGUF models using llama-cpp-python with GPU layer management."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None = None,
        n_gpu_layers: int | str = -1,
        device_preference: str = "auto",
        **kwargs,
    ):
        """
        Initialize GGUF model wrapper.

        Args:
            model_path: Path to GGUF model file
            tokenizer_path: Path to HuggingFace tokenizer directory (auto-detected if None)
            n_gpu_layers: Number of GPU layers (-1 for all, 0 for CPU only, or 1+ for specific count).
                         If value exceeds model's total layers, all layers will be used (no error).
            device_preference: "auto", "cuda", or "cpu"
            **kwargs: Additional arguments for llama-cpp-python Llama initialization (including context_size)
        """
        if tokenizer_path is None:
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
            model_path = os.path.normpath(model_path)
            tokenizer_path = auto_detect_tokenizer_path(model_path, model_format="gguf")
            logging.debug(f"Auto-detected tokenizer path: {tokenizer_path}")
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "This package must be installed manually:\n"
                "  • CPU only: pip install llama-cpp-python\n"
                "  • CUDA support: pip install llama-cpp-python[cuda]\n"
                "Note: CUDA support may require compilation (CMake, Visual Studio on Windows, or build tools on Linux/Mac). "
                "See README.md for detailed installation instructions."
            ) from None

        logging.info(f"Loading GGUF model: {os.path.basename(model_path)}")
        load_start = time.time()

        device = get_device_for_model(device_preference)
        cuda_available = device == "cuda" and torch.cuda.is_available()

        logging.debug(f"Device preference: {device_preference}, selected: {device}")
        if cuda_available:
            logging.debug(f"GPU: {torch.cuda.get_device_name(0)}")

        context_size = kwargs.pop("context_size", 4096)

        # If user explicitly requested CPU, force n_gpu_layers=0 regardless of other settings
        if device_preference == "cpu":
            n_gpu_layers = 0
            logging.debug("Device preference is CPU, forcing CPU-only execution")

        if isinstance(n_gpu_layers, str) and n_gpu_layers.lower() == "auto":
            if cuda_available:
                available_vram = get_available_vram()
                logging.debug(f"Auto GPU layers: Available VRAM: {available_vram:.2f} GB")

                try:
                    model_size_gb = os.path.getsize(model_path) / (1024**3)
                    logging.debug(f"Auto GPU layers: Model size: {model_size_gb:.2f} GB")
                except Exception:
                    model_size_gb = 4.0
                    logging.debug(f"Auto GPU layers: Could not determine model size, using estimate: {model_size_gb:.2f} GB")

                optimal_layers = calculate_optimal_gpu_layers(
                    model_size_gb=model_size_gb, available_vram_gb=available_vram, context_size=context_size
                )

                if optimal_layers == -1:
                    logging.debug("Auto GPU layers: Using all GPU layers (model fits in VRAM)")
                    n_gpu_layers = -1
                elif optimal_layers == 0:
                    logging.info("Auto GPU layers: Not enough VRAM, using CPU only")
                    n_gpu_layers = 0
                else:
                    logging.debug(f"Auto GPU layers: Using {optimal_layers} layers on GPU")
                    n_gpu_layers = optimal_layers
            else:
                logging.debug("Auto GPU layers: CUDA not available, using CPU only")
                n_gpu_layers = 0
        else:
            n_gpu_layers = int(n_gpu_layers)
            if n_gpu_layers == -1:
                logging.debug("Using all GPU layers")
            elif n_gpu_layers == 0:
                logging.debug("Using CPU only")
            else:
                logging.debug(f"Using {n_gpu_layers} layers on GPU")

        use_mmap = kwargs.pop("use_mmap", True)
        use_mlock = kwargs.pop("use_mlock", False)
        n_batch = kwargs.pop("n_batch", 512)
        n_threads = kwargs.pop("n_threads", None)

        llama_kwargs = {
            "n_ctx": context_size,
            "verbose": False,
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
            "n_batch": n_batch,
        }

        if n_threads is not None:
            llama_kwargs["n_threads"] = n_threads

        llama_kwargs["n_gpu_layers"] = n_gpu_layers
        llama_kwargs.update(kwargs)

        # Check if llama-cpp-python was built with CUDA support
        try:
            import llama_cpp

            if hasattr(llama_cpp, "__version__"):
                logging.debug(f"llama-cpp-python version: {llama_cpp.__version__}")
        except Exception:
            pass

        self.model = Llama(model_path=model_path, **llama_kwargs)
        load_time = time.time() - load_start

        if cuda_available and n_gpu_layers > 0:
            try:
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    logging.debug(f"GPU memory allocated: {gpu_memory:.2f} GB")
                    if gpu_memory > 0.1:
                        logging.debug("Model appears to be using GPU")
                    else:
                        logging.debug("Low GPU memory usage - model may be running on CPU")
            except Exception:
                pass

        logging.info(f"GGUF model loaded in {load_time:.2f}s")

        # Use HuggingFace tokenizer even though GGUF models can have embedded tokenizers
        # Reason: Our code relies on apply_chat_template() which is HuggingFace-specific
        logging.debug(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer_start = time.time()
        tokenizer_path_abs = os.path.abspath(tokenizer_path) if not os.path.isabs(tokenizer_path) else tokenizer_path

        try:
            from huggingface_hub.utils import HFValidationError
        except ImportError:
            HFValidationError = Exception

        if os.path.exists(tokenizer_path_abs) and os.path.isdir(tokenizer_path_abs):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_abs, trust_remote_code=True, local_files_only=True)
            except (HFValidationError, Exception) as e:
                # If local_files_only fails, try without it (might be a cached HuggingFace model)
                if isinstance(e, HFValidationError) or "Repo id must use alphanumeric" in str(e):
                    # Transformers is still validating the path - use absolute path with explicit local_files_only
                    logging.warning(f"Tokenizer path validation error, retrying with explicit local path handling: {e}")
                else:
                    logging.warning(f"Failed to load tokenizer with local_files_only=True: {e}")
                logging.warning("Trying without local_files_only...")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_abs, trust_remote_code=True)
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            except (HFValidationError, Exception) as e:
                if (isinstance(e, HFValidationError) or "Repo id must use alphanumeric" in str(e)) and (
                    os.path.sep in tokenizer_path or (os.name == "nt" and len(tokenizer_path) > 1 and tokenizer_path[1] == ":")
                ):
                    raise ValueError(
                        f"Tokenizer path does not exist or is invalid: {tokenizer_path_abs}\n"
                        f"Please ensure the tokenizer directory exists at this location.\n"
                        f"Expected location: {os.path.dirname(model_path)}/workflow-generator/ (or similar based on model name)"
                    ) from e
                raise
        tokenizer_time = time.time() - tokenizer_start
        logging.debug(f"Tokenizer loaded in {tokenizer_time:.2f}s")

        self.device = device

    def generate(
        self, input_ids, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True, **kwargs
    ):
        """
        Generate text using GGUF model.

        Args:
            input_ids: Token IDs from HuggingFace tokenizer (tensor or list)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs tensor (full sequence: input + generated)
        """
        if torch.is_tensor(input_ids):
            input_ids_list = input_ids.cpu().tolist()
        elif isinstance(input_ids, np.ndarray):
            input_ids_list = input_ids.tolist()
        else:
            input_ids_list = input_ids

        # Handle batch dimension (llama-cpp-python expects single sequence)
        if isinstance(input_ids_list[0], list):
            input_ids_list = input_ids_list[0]

        output = self.model(input_ids_list, max_tokens=max_new_tokens, temperature=temperature, top_p=top_p, stop=[], **kwargs)

        if isinstance(output, dict) and "choices" in output:
            generated_text = output["choices"][0]["text"]
        elif hasattr(output, "choices"):
            generated_text = output.choices[0].text
        else:
            generated_text = str(output)

        generated_token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
        full_token_ids = input_ids_list + generated_token_ids
        full_tensor = torch.tensor([full_token_ids], dtype=torch.long)

        return full_tensor

    def __del__(self):
        """Cleanup method for VRAM management."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
