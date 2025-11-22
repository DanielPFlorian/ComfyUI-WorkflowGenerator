"""
Device, dtype, and VRAM utility functions for model loading.
Shared utilities for both GGUF and HuggingFace model loaders.
"""

import logging

import torch

# Try to import ComfyUI model management for device detection
try:
    from comfy import model_management

    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    logging.warning("ComfyUI model_management not available, using basic device detection")


def calculate_optimal_gpu_layers(model_size_gb: float, available_vram_gb: float, reserve_gb: float = 2.0, context_size: int = 4096) -> int:
    """
    Calculate optimal number of GPU layers based on available VRAM.

    Args:
        model_size_gb: Total model size in GB
        available_vram_gb: Available VRAM in GB
        reserve_gb: VRAM to reserve for other operations (default: 2.0 GB)
        context_size: Context window size (used to estimate KV cache size)

    Returns:
        Number of layers to load on GPU (-1 for all, 0 for CPU only, or specific count)
    """
    if available_vram_gb <= 0:
        return 0  # CPU only

    # 1. Reserve VRAM for system + KV cache
    # Rough KV cache estimate: ~1-2GB for 4096 context, scales linearly with context
    # Formula: 2 * 2 * layers * hidden_size * context_len (conservative estimate)
    # Using a simpler heuristic: 0.5GB per 4096 tokens of context
    kv_cache_gb = (context_size / 4096.0) * 0.5
    effective_reserve = reserve_gb + kv_cache_gb

    usable_vram = available_vram_gb - effective_reserve
    if usable_vram <= 0:
        return 0

    if model_size_gb <= usable_vram:
        return -1  # Fits fully

    # 2. Estimate per-layer size dynamically
    # Assume ~40-80 layers for 7B-14B models.
    # If we don't know layer count, estimating based on file size is better than fixed 100MB
    # Qwen2.5-14B has 48 layers. Qwen2.5-7B has 28 layers.
    # Average layers approx 40 for safe estimation.
    estimated_layers = 40
    layer_size_gb = model_size_gb / estimated_layers

    # 3. Calculate layers that fit
    max_layers = int(usable_vram / layer_size_gb)
    return max(0, max_layers)


def get_available_vram() -> float:
    """
    Get available VRAM in GB using ComfyUI's memory management if available.

    Returns:
        Available VRAM in GB
    """
    if COMFYUI_AVAILABLE:
        try:
            dev = model_management.get_torch_device()
            free_memory = model_management.get_free_memory(dev)
            return free_memory / (1024**3)
        except Exception as e:
            logging.warning(f"Failed to get VRAM from ComfyUI: {e}")

    if torch.cuda.is_available():
        try:
            free_memory, _ = torch.cuda.mem_get_info(0)
            return free_memory / (1024**3)
        except Exception as e:
            logging.warning(f"Failed to get VRAM from torch: {e}")

    return 0.0


def get_device_for_model(device_preference: str = "auto") -> str:
    """
    Get device string for model loading.

    Args:
        device_preference: "auto", "cuda", or "cpu"

    Returns:
        Device string ("cuda" or "cpu")
    """
    if device_preference == "cpu":
        return "cpu"

    if device_preference == "cuda" or device_preference == "auto":
        if COMFYUI_AVAILABLE:
            try:
                dev = model_management.get_torch_device()
                if hasattr(dev, "type"):
                    if dev.type == "cuda":
                        return "cuda"
                    elif dev.type == "cpu":
                        return "cpu"
            except Exception:
                pass

        # Fallback to torch detection
        if torch.cuda.is_available():
            return "cuda"

    return "cpu"


def get_dtype_for_model(dtype_preference: str = "auto", device: str = "cuda") -> torch.dtype:
    """
    Get dtype for HuggingFace model loading.

    Args:
        dtype_preference: "auto", "fp16", "bf16", "fp32", or "fp8"
        device: Device string ("cuda" or "cpu")

    Returns:
        torch.dtype
    """
    if dtype_preference == "fp32":
        return torch.float32

    if dtype_preference == "fp16":
        return torch.float16

    if dtype_preference == "bf16":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if dtype_preference == "fp8":
        # FP8 is CUDA-only and requires specific hardware support
        if device == "cpu":
            logging.debug("FP8 is not supported on CPU, falling back to fp16")
            return torch.float16

        # FP8 requires CUDA and specific PyTorch builds with FP8 support
        if not torch.cuda.is_available():
            logging.debug("FP8 requires CUDA, but CUDA is not available, falling back to fp16")
            return torch.float16

        if COMFYUI_AVAILABLE:
            try:
                float8_types = model_management.get_supported_float8_types()
                if float8_types:
                    selected_type = float8_types[0]
                    # Verify FP8 storage is actually available by testing set_default_dtype
                    # This is what transformers will do internally, so we catch the error early
                    try:
                        original_dtype = torch.get_default_dtype()
                        torch.set_default_dtype(selected_type)
                        torch.set_default_dtype(original_dtype)
                        type_name = str(selected_type).replace("torch.", "")
                        logging.debug(f"Using FP8: {type_name}")
                        return selected_type
                    except (TypeError, RuntimeError) as e:
                        # FP8 dtype exists but storage is not available
                        logging.debug(f"FP8 storage not available: {e}, falling back to fp16")
                        return torch.float16
            except Exception as e:
                logging.debug(f"Failed to get FP8 types: {e}, falling back to fp16")
        logging.debug("FP8 not available, falling back to fp16")
        return torch.float16

    if COMFYUI_AVAILABLE and device == "cuda":
        try:
            model_management.get_torch_device()
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        except Exception:
            pass

    # Default: fp16 for CUDA, fp32 for CPU
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_model(
    model_format: str,
    model_path: str,
    tokenizer_path: str | None = None,
    n_gpu_layers: int | str = -1,
    dtype: str | torch.dtype = "auto",
    device_preference: str = "auto",
    max_memory: dict | None = None,
    attn_implementation: str = "auto",
    context_size: int = 4096,
    **kwargs,
):
    """
    Unified model loader that supports both GGUF and HuggingFace formats.

    Args:
        model_format: "gguf" or "huggingface"
        model_path: Path to model file (GGUF) or directory (HuggingFace)
        tokenizer_path: Path to tokenizer directory (auto-detected if None)
        n_gpu_layers: For GGUF: "auto" (calculate optimal), -1 for all, 0 for CPU only, or 1+ for specific count.
                     If value exceeds model's total layers, all layers will be used (no error).
        dtype: For HuggingFace: "auto", "fp16", "bf16", "fp32", "fp8", or torch.dtype
        device_preference: "auto", "cuda", or "cpu"
        max_memory: For HuggingFace: Dict for device_map memory limits
        attn_implementation: For HuggingFace: "auto", "flash_attention_2", or "sdpa"
        context_size: For GGUF: Context window size
        **kwargs: Additional arguments passed to model wrapper initialization

    Returns:
        Model wrapper instance (GGUFModelWrapper or HuggingFaceModelWrapper)
    """
    import os

    from ..model_manager import auto_detect_tokenizer_path

    if model_format.lower() == "gguf":
        from .gguf_loader import GGUFModelWrapper

        if tokenizer_path is None:
            # Ensure model_path is absolute and normalized before auto-detection
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
            model_path = os.path.normpath(model_path)
            tokenizer_path = auto_detect_tokenizer_path(model_path, model_format="gguf")
            logging.info(f"Auto-detected tokenizer path for GGUF model: {tokenizer_path}")
        return GGUFModelWrapper(
            model_path, tokenizer_path, n_gpu_layers=n_gpu_layers, device_preference=device_preference, context_size=context_size, **kwargs
        )

    elif model_format.lower() == "huggingface":
        from .huggingface_loader import HuggingFaceModelWrapper

        return HuggingFaceModelWrapper(
            model_path,
            tokenizer_path,
            dtype=dtype,
            device_preference=device_preference,
            max_memory=max_memory,
            attn_implementation=attn_implementation,
            **kwargs,
        )

    else:
        raise ValueError(f"Unsupported model_format: {model_format}. Supported formats: 'gguf', 'huggingface'")
