import logging

try:
    import psutil
except ImportError:
    psutil = None


def get_n_threads_options() -> list[str]:
    """
    Generate options for n_threads Combo input.
    Returns: List of strings ["All", "Auto"] + integer options up to CPU count
    """
    try:
        if psutil is not None:
            cpu_count = psutil.cpu_count(logical=True)
        else:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()

        options = ["All", "Auto"]
        options.extend([str(i) for i in range(1, cpu_count + 1)])
        return options
    except (OSError, RuntimeError, AttributeError, Exception) as e:
        logging.warning(f"Could not detect CPU count, using fallback options: {e}")
        return ["All", "Auto"] + [str(i) for i in range(1, 33)]


def parse_n_threads(value: str) -> int | None:
    """
    Parse n_threads Combo value to integer or None.

    Args:
        value: String from Combo input ("All", "Auto", or integer string)

    Returns:
        int for "All" or specific count, None for "Auto"
    """
    if value == "All":
        try:
            if psutil is not None:
                return psutil.cpu_count(logical=True)
            else:
                import multiprocessing

                return multiprocessing.cpu_count()
        except Exception:
            return 32
    elif value == "Auto":
        return None
    else:
        try:
            return int(value)
        except (ValueError, TypeError):
            return None


def check_llm_lazy_status(
    use_llm_refinement: bool = False,
    refine_model_path: str | None = None,
    max_new_tokens: int | None = None,
    device_preference: str | None = None,
    n_gpu_layers: int | None = None,
    dtype: str | None = None,
    attn_implementation: str | None = None,
    context_size: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    # Additional parameters that might be present
    refine_max_new_tokens: int | None = None,
    refine_device_preference: str | None = None,
    refine_n_gpu_layers: int | None = None,
    refine_dtype: str | None = None,
    refine_attn_implementation: str | None = None,
    refine_context_size: int | None = None,
    refine_temperature: float | None = None,
    refine_top_p: float | None = None,
    refine_auto_gpu_layers: bool | None = None,
    auto_gpu_layers: bool | None = None,
    **kwargs,
) -> list[str]:
    """
    Determine which lazy inputs need to be evaluated for LLM refinement.
    Handles both standard parameter names (NodeValidator) and prefixed names (Pipeline).

    Returns:
        List of input names that need to be evaluated.
    """
    needed = []

    if not use_llm_refinement:
        return needed

    def check_add(val, name):
        if val is None:
            needed.append(name)

    # Detect if we're in pipeline context (any prefixed parameter is not None)
    is_pipeline_context = any(
        [
            refine_max_new_tokens is not None,
            refine_device_preference is not None,
            refine_n_gpu_layers is not None,
            refine_dtype is not None,
            refine_attn_implementation is not None,
            refine_context_size is not None,
            refine_temperature is not None,
            refine_top_p is not None,
            refine_auto_gpu_layers is not None,
        ]
    )

    if is_pipeline_context:
        # Pipeline node context - check prefixed names
        check_add(refine_model_path, "refine_model_path")
        check_add(refine_max_new_tokens, "refine_max_new_tokens")
        check_add(refine_device_preference, "refine_device_preference")
        check_add(refine_n_gpu_layers, "refine_n_gpu_layers")
        check_add(refine_dtype, "refine_dtype")
        check_add(refine_attn_implementation, "refine_attn_implementation")
        check_add(refine_context_size, "refine_context_size")
        check_add(refine_temperature, "refine_temperature")
        check_add(refine_top_p, "refine_top_p")
        check_add(refine_auto_gpu_layers, "refine_auto_gpu_layers")
    else:
        # NodeValidator node context - check standard names
        check_add(refine_model_path, "refine_model_path")
        check_add(max_new_tokens, "max_new_tokens")
        check_add(device_preference, "device_preference")
        check_add(n_gpu_layers, "n_gpu_layers")
        check_add(dtype, "dtype")
        check_add(attn_implementation, "attn_implementation")
        check_add(context_size, "context_size")
        check_add(temperature, "temperature")
        check_add(top_p, "top_p")
        check_add(seed, "seed")
        check_add(auto_gpu_layers, "auto_gpu_layers")

    return needed
