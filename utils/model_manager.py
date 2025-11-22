"""
Model manager for resolving model paths and detecting devices.
Integrates with ComfyUI's model management system.
"""

import logging
import os
from pathlib import Path

# Try to import ComfyUI folder_paths for model path resolution
try:
    import folder_paths

    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    logging.warning("ComfyUI folder_paths not available, using basic path resolution")

# Try to import ComfyUI model management for device detection
try:
    from comfy import model_management

    COMFYUI_MODEL_MGMT = True
except ImportError:
    COMFYUI_MODEL_MGMT = False


def get_comfyui_base_path() -> Path | None:
    """
    Get the base path of the ComfyUI installation.

    Returns:
        Path to ComfyUI base directory or None if not found
    """
    # Common locations:
    # 1. Current working directory (if running from ComfyUI root)
    # 2. Parent of custom_nodes directory
    # 3. Environment variable
    current_path = Path.cwd()

    # Case 1: running from inside ComfyUI directory
    if (current_path / "main.py").exists() and (current_path / "nodes.py").exists():
        return current_path

    # Case 2: running from parent directory (portable version root)
    # Check for "ComfyUI" directory
    if (current_path / "ComfyUI").exists() and (current_path / "ComfyUI" / "main.py").exists():
        return current_path / "ComfyUI"
        
    # Check for "comfy" directory (unlikely but possible)
    if (current_path / "comfy").exists() and (current_path / "comfy" / "main.py").exists():
        return current_path / "comfy"

    # Case 3: Check parents (if running from a subdir)
    for parent in current_path.parents:
        if (parent / "main.py").exists() and (parent / "nodes.py").exists():
            return parent
        if (parent / "ComfyUI").exists() and (parent / "ComfyUI" / "main.py").exists():
            return parent / "ComfyUI"

    comfyui_path = os.environ.get("COMFYUI_PATH")
    if comfyui_path:
        return Path(comfyui_path)

    return None


def resolve_model_path(model_path: str, model_type: str = "llm", fallback_to_absolute: bool = True) -> str:
    """
    Resolve a model path relative to ComfyUI's model directories.

    Args:
        model_path: Model path (can be relative or absolute)
        model_type: Type of model ("llm", "gguf", etc.)
        fallback_to_absolute: If True, return absolute path if not found in ComfyUI dirs

    Returns:
        Resolved absolute path to the model
    """
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path

    comfyui_base = get_comfyui_base_path()
    if comfyui_base:
        for base_dir in ["models/LLM", f"models/{model_type}"]:
            # Handle subdirectory paths (e.g., "GGUF/llama-joycaption-beta-one-hf-llava.IQ4_XS.gguf")
            full_path = comfyui_base / base_dir / model_path
            if full_path.exists():
                return str(full_path)

            # Also try with just the filename (for backward compatibility)
            filename = os.path.basename(model_path)
            full_path = comfyui_base / base_dir / filename
            if full_path.exists():
                return str(full_path)

        full_path = comfyui_base / model_path
        if full_path.exists():
            return str(full_path)

    if COMFYUI_AVAILABLE:
        try:
            if model_path.endswith(".gguf"):
                model_dirs = folder_paths.get_folder_paths("unet_gguf")
                if not model_dirs:
                    model_dirs = folder_paths.get_folder_paths("clip_gguf")
                if not model_dirs:
                    model_dirs = folder_paths.get_folder_paths("llm")
                
                # If "llm" path wasn't found via get_folder_paths, try accessing the registered paths directly
                if not model_dirs and "llm" in folder_paths.folder_names_and_paths:
                     model_dirs = folder_paths.folder_names_and_paths["llm"][0]

                if model_dirs:
                    for model_dir in model_dirs:
                        full_path = os.path.join(model_dir, model_path)
                        if os.path.exists(full_path):
                            return full_path

                        filename = os.path.basename(model_path)
                        full_path = os.path.join(model_dir, filename)
                        if os.path.exists(full_path):
                            return full_path

            model_dirs = folder_paths.get_folder_paths("llm")
            # If "llm" path wasn't found via get_folder_paths, try accessing the registered paths directly
            if not model_dirs and "llm" in folder_paths.folder_names_and_paths:
                    model_dirs = folder_paths.folder_names_and_paths["llm"][0]
            
            if model_dirs:
                for model_dir in model_dirs:
                    full_path = os.path.join(model_dir, model_path)
                    if os.path.exists(full_path):
                        return full_path

                    filename = os.path.basename(model_path)
                    full_path = os.path.join(model_dir, filename)
                    if os.path.exists(full_path):
                        return full_path
        except Exception as e:
            logging.warning(f"Failed to resolve path using folder_paths: {e}")

    # Manual fallback (always try this if folder_paths didn't find it)
    comfyui_base = get_comfyui_base_path()
    if comfyui_base:
        llm_dir = comfyui_base / "models" / "LLM"
        if llm_dir.exists():
            # Try exact path in LLM dir
            full_path = llm_dir / model_path
            if full_path.exists():
                return str(full_path)

            # Try filename in LLM dir
            filename = os.path.basename(model_path)
            full_path = llm_dir / filename
            if full_path.exists():
                return str(full_path)
            
            # Try recursive search in LLM dir for GGUF files
            if model_path.endswith(".gguf"):
                for root, _, files in os.walk(llm_dir):
                    if filename in files:
                        return os.path.join(root, filename)

    if fallback_to_absolute:
        abs_path = os.path.abspath(model_path)
        if os.path.exists(abs_path):
            return abs_path
        # Only return absolute path if it actually exists, otherwise return the original path
        # or try to return the most likely expected path (in models/LLM) so the error message is clearer
        if comfyui_base:
             expected_path = comfyui_base / "models" / "LLM" / model_path
             return str(expected_path)
        return abs_path

    return model_path


def auto_detect_tokenizer_path(model_path: str, model_format: str = "gguf") -> str:
    """
    Automatically detect tokenizer path based on model path.

    For GGUF models: Tokenizer is typically in a directory with the same name
    as the model (without .gguf extension) in the same parent directory.
    Example: models/LLM/workflow-generator-q8_0.gguf -> models/LLM/workflow-generator/

    For HuggingFace models: Tokenizer is in the same directory as the model.
    Example: models/LLM/Qwen2.5-7B-Instruct/ -> models/LLM/Qwen2.5-7B-Instruct/

    Args:
        model_path: Resolved absolute path to the model file or directory
        model_format: "gguf" or "huggingface"

    Returns:
        Resolved absolute path to the tokenizer directory
    """
    import os

    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    model_path = os.path.normpath(model_path)
    model_path_obj = Path(model_path).resolve()

    if model_format.lower() == "gguf":
        # For GGUF: tokenizer is in a directory with the same name (without .gguf)
        # Example: workflow-generator-q8_0.gguf -> workflow-generator/
        model_name = model_path_obj.stem
        # Remove quantization suffix if present (e.g., "workflow-generator-q8_0" -> "workflow-generator")
        # Common suffixes: -q8_0, -q4_0, -q4_1, -q5_0, -q5_1, -q6_0, -f16, -f32
        import re

        model_name_clean = re.sub(r"-[qf]\d+[_\d]*$", "", model_name)

        parent_dir = model_path_obj.parent
        tokenizer_dir = parent_dir / model_name_clean

        if tokenizer_dir.exists() and tokenizer_dir.is_dir():
            tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
            if any((tokenizer_dir / f).exists() for f in tokenizer_files):
                return str(tokenizer_dir)

        # Fallback: try with full model name (including quantization suffix)
        tokenizer_dir = parent_dir / model_name
        if tokenizer_dir.exists() and tokenizer_dir.is_dir():
            tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
            if any((tokenizer_dir / f).exists() for f in tokenizer_files):
                return str(tokenizer_dir)

        # If not found, return the expected path anyway (will be handled by error handling)
        return str(tokenizer_dir)

    else:
        if model_path_obj.is_dir():
            return str(model_path_obj)
        else:
            # Model path is a file, tokenizer should be in the same directory
            # This shouldn't happen for HuggingFace, but handle it anyway
            return str(model_path_obj.parent)


def get_device_info() -> tuple[str, bool]:
    """
    Get device information using ComfyUI's model management.

    Returns:
        Tuple of (device_string, cuda_available)
    """
    if COMFYUI_MODEL_MGMT:
        try:
            dev = model_management.get_torch_device()
            if hasattr(dev, "type"):
                if dev.type == "cuda":
                    return "cuda", True
                elif dev.type == "cpu":
                    return "cpu", False
                elif dev.type == "mps":
                    return "mps", False
        except Exception as e:
            logging.warning(f"Failed to get device from ComfyUI: {e}")

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", True
    except ImportError:
        pass

    return "cpu", False


def register_llm_folder_type():
    """
    Register "llm" folder type in ComfyUI's folder_paths.
    This allows us to use folder_paths.get_filename_list("llm") and folder_paths.get_full_path("llm", ...).
    """
    if not COMFYUI_AVAILABLE:
        logging.warning("ComfyUI folder_paths not available, cannot register LLM folder type")
        return

    try:
        if "llm" in folder_paths.folder_names_and_paths:
            return

        comfyui_base = get_comfyui_base_path()
        if comfyui_base:
            llm_dir = comfyui_base / "models" / "LLM"
            if llm_dir.exists():
                # folder_names_and_paths structure: {key: (list_of_paths, set_of_extensions)}
                # For LLM, we want to accept both .gguf files and directories (no extension filter)
                folder_paths.folder_names_and_paths["llm"] = ([str(llm_dir)], set())
                logging.info(f"Registered LLM folder type: {llm_dir}")
            else:
                logging.warning(f"LLM directory not found: {llm_dir}")
        else:
            try:
                # Look for existing model directories to infer the models path
                if "checkpoints" in folder_paths.folder_names_and_paths:
                    checkpoints_paths, _ = folder_paths.folder_names_and_paths["checkpoints"]
                    if checkpoints_paths:
                        # models/LLM should be sibling to models/checkpoints
                        models_dir = Path(checkpoints_paths[0]).parent
                        llm_dir = models_dir / "LLM"
                        if llm_dir.exists():
                            folder_paths.folder_names_and_paths["llm"] = ([str(llm_dir)], set())
                            logging.info(f"Registered LLM folder type: {llm_dir}")
                            return
            except Exception as e:
                logging.warning(f"Failed to infer LLM directory from existing folders: {e}")

            logging.warning("Could not determine ComfyUI base path, LLM folder type not registered")
    except Exception as e:
        logging.error(f"Failed to register LLM folder type: {e}")


def get_llm_model_list() -> list[str]:
    """
    Get all models from ComfyUI/models/LLM/ (both .gguf files and directories).
    Recursively scans subdirectories for .gguf files.
    Returns only model names (not full paths).

    Returns:
        List of model names (e.g., ["workflow-generator-q8_0.gguf", "Qwen2.5-7B-Instruct"])
    """
    model_names = []

    def scan_directory(directory: Path, prefix: str = ""):
        """
        Recursively scan directory for .gguf files and HuggingFace model directories.

        Args:
            directory: Directory to scan
            prefix: Prefix to add to model names (for subdirectories)
        """
        if not directory.exists():
            return

        try:
            for item in directory.iterdir():
                if item.is_file() and item.suffix == ".gguf":
                    if prefix:
                        model_names.append(f"{prefix}{item.name}")
                    else:
                        model_names.append(item.name)
                elif item.is_dir():
                    has_gguf = False
                    try:
                        has_gguf = any(f.suffix == ".gguf" for f in item.iterdir() if f.is_file())
                    except (PermissionError, OSError) as e:
                        logging.debug(f"Cannot read directory {item}: {e}")
                        continue

                    if has_gguf:
                        # Directory contains .gguf files - scan recursively and add files with prefix
                        sub_prefix = f"{item.name}/" if prefix else f"{item.name}/"
                        scan_directory(item, sub_prefix)
                    else:
                        # Directory is likely a HuggingFace model - add directory name
                        if prefix:
                            model_names.append(f"{prefix}{item.name}")
                        else:
                            model_names.append(item.name)
        except (PermissionError, OSError) as e:
            logging.warning(f"Cannot scan directory {directory}: {e}")

    # Always try to get models, even if folder_paths isn't registered
    comfyui_base = get_comfyui_base_path()
    
    if not COMFYUI_AVAILABLE:
        if comfyui_base:
            llm_dir = comfyui_base / "models" / "LLM"
            scan_directory(llm_dir)
        return sorted(model_names)

    try:
        # First try using folder_paths if registered
        if "llm" in folder_paths.folder_names_and_paths:
            llm_paths, _ = folder_paths.folder_names_and_paths["llm"]
            for llm_path in llm_paths:
                llm_dir = Path(llm_path)
                scan_directory(llm_dir)
        
        # Always also try direct path as fallback (in case folder_paths wasn't registered)
        if comfyui_base:
            llm_dir = comfyui_base / "models" / "LLM"
            scan_directory(llm_dir)
    except Exception as e:
        logging.warning(f"Failed to get LLM model list: {e}")
        # Fallback to direct path scan
        if comfyui_base:
            llm_dir = comfyui_base / "models" / "LLM"
            scan_directory(llm_dir)

    return sorted(model_names)


def get_workflow_generator_models() -> list[str]:
    """
    Filter models suitable for workflow generator.
    Includes both GGUF files and HuggingFace directories.

    Returns:
        List of model names suitable for workflow generator
    """
    all_models = get_llm_model_list()
    # For now, return all models (both GGUF and HuggingFace work for workflow generator)
    # In the future, we could filter by naming patterns if needed
    return all_models


def get_refine_agent_models() -> list[str]:
    """
    Filter models suitable for refine_agent (node validator).
    Includes both GGUF files and HuggingFace directories.

    Returns:
        List of model names suitable for refine_agent
    """
    all_models = get_llm_model_list()
    # For now, return all models (both GGUF and HuggingFace work for refine_agent)
    # In the future, we could filter by naming patterns if needed
    return all_models


def get_embedding_models() -> list[str]:
    """
    Filter embedding models (SentenceTransformer/HuggingFace directories only).
    Excludes .gguf files as embedding models are typically HuggingFace/SentenceTransformer.

    Returns:
        List of embedding model directory names
    """
    all_models = get_llm_model_list()
    embedding_models = [m for m in all_models if not m.endswith(".gguf")]
    return embedding_models


def detect_model_format(model_name: str) -> str:
    """
    Auto-detect model format from model name.

    Args:
        model_name: Model name (e.g., "workflow-generator-q8_0.gguf" or "Qwen2.5-7B-Instruct")

    Returns:
        "gguf" if model ends with .gguf, "huggingface" if it's a directory
    """
    if model_name.endswith(".gguf"):
        return "gguf"
    else:
        return "huggingface"


def get_model_full_path(model_name: str) -> str:
    """
    Get full path to a model using ComfyUI's folder_paths.

    Args:
        model_name: Model name (e.g., "workflow-generator-q8_0.gguf" or "Qwen2.5-7B-Instruct")

    Returns:
        Full absolute path to the model file or directory
    """
    from pathlib import Path

    if not COMFYUI_AVAILABLE:
        return resolve_model_path(model_name, model_type="llm")

    try:
        if "llm" in folder_paths.folder_names_and_paths:
            # Check for exact match first to avoid fuzzy matching issues
            llm_paths, _ = folder_paths.folder_names_and_paths["llm"]
            for llm_path in llm_paths:
                exact_path = Path(llm_path) / model_name
                if exact_path.exists() and (
                    (not model_name.endswith(".gguf") and exact_path.is_dir()) or
                    (model_name.endswith(".gguf") and exact_path.is_file())
                ):
                    return str(exact_path)

            return folder_paths.get_full_path("llm", model_name)
        else:
            return resolve_model_path(model_name, model_type="llm")
    except Exception as e:
        logging.warning(f"Failed to get full path using folder_paths: {e}, falling back to resolve_model_path")
        return resolve_model_path(model_name, model_type="llm")
