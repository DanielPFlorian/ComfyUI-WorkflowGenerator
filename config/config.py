"""
Configuration management for ComfyUI-WorkflowGenerator nodes.
Builds configuration objects from node inputs and default values.
"""

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..utils.model_manager import resolve_model_path

# Models are in ComfyUI/models/LLM/, catalog files are in custom node catalog/
DEFAULT_CONFIG = {
    "workflow_generator": {
        "model_format": "gguf",
        "model_path": "models/LLM/workflow-generator-q8_0.gguf",
        "tokenizer_path": "models/LLM/workflow-generator",
        "max_new_tokens": 8192,
        "top_p": 0.7,
        "temperature": 0.95,
        "allow_primitive_nodes": False,
    },
    "node_validator": {
        "local": False,
        "model_format": "gguf",
        "model_path": "models/LLM/Qwen2.5-7B-Instruct-q8_0.gguf",
        "tokenizer_path": "models/LLM/Qwen2.5-7B-Instruct",
        "k": 5,
        "max_new_tokens": 4096,
    },
    "node_base": {
        "catalog_directory": "catalog",
        "embedding_model_path": "models/LLM/paraphrase-multilingual-MiniLM-L12-v2",
    },
}


def get_custom_node_path() -> Path:
    """
    Get the path to the ComfyUI-WorkflowGenerator custom node directory.

    Returns:
        Path to custom node directory
    """
    current_file = Path(__file__).resolve()

    # config/config.py -> ComfyUI-WorkflowGenerator/
    if current_file.parent.name == "config":
        return current_file.parent.parent

    return Path.cwd()


def resolve_config_path(path: str, base_path: Path | None = None) -> str:
    """
    Resolve a configuration path relative to the custom node directory or ComfyUI root.

    Args:
        path: Path string (can be relative or absolute)
            - If starts with "custom_nodes/", resolves relative to ComfyUI root
            - Otherwise, resolves relative to custom node directory
        base_path: Base path for resolution (default: custom node directory)

    Returns:
        Resolved absolute path
    """
    if base_path is None:
        base_path = get_custom_node_path()

    if os.path.isabs(path) and os.path.exists(path):
        return path

    if path.startswith("custom_nodes/"):
        from ..utils.model_manager import get_comfyui_base_path

        comfyui_base = get_comfyui_base_path()
        if comfyui_base:
            full_path = comfyui_base / path
            if full_path.exists():
                return str(full_path)
            return str(full_path.resolve())

    # Check if path is just "catalog" or relative path inside custom node dir
    full_path = base_path / path
    if full_path.exists():
        return str(full_path)

    # Check if it's in the catalog directory (convenience for just passing filename)
    catalog_path = base_path / "catalog" / path
    if catalog_path.exists():
        return str(catalog_path)

    # Fallback: return absolute path even if it doesn't exist
    return str(full_path.resolve())


class ConfigBuilder:
    """
    Builder class for creating configuration objects from node inputs.
    """

    def __init__(self, base_config: dict | None = None):
        """
        Initialize ConfigBuilder.

        Args:
            base_config: Base configuration dictionary (default: DEFAULT_CONFIG)
        """
        if base_config is None:
            base_config = DEFAULT_CONFIG.copy()

        self.base_config = base_config
        self.custom_node_path = get_custom_node_path()

    def build_workflow_generator_config(
        self,
        model_format: str = "gguf",
        model_path: str = "",
        tokenizer_path: str = "",
        max_new_tokens: int = 8192,
        top_p: float = 0.7,
        temperature: float = 0.95,
        allow_primitive_nodes: bool = False,
        **kwargs,
    ) -> DictConfig:
        """
        Build configuration for WorkflowGenerator.

        Args:
            model_format: Model format ("gguf" or "huggingface")
            model_path: Path to model file/directory
            tokenizer_path: Path to tokenizer directory
            max_new_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            temperature: Temperature for sampling
            allow_primitive_nodes: Whether to include primitive nodes
            **kwargs: Additional configuration overrides

        Returns:
            OmegaConf DictConfig object
        """
        config = self.base_config["workflow_generator"].copy()

        if model_format:
            config["model_format"] = model_format
        if model_path:
            config["model_path"] = model_path
        # Only set tokenizer_path if explicitly provided (not None)
        # If None, remove it from config so auto-detection happens
        if tokenizer_path is not None:
            if tokenizer_path:  # Non-empty string
                config["tokenizer_path"] = tokenizer_path
            else:
                # Empty string means remove from config (use auto-detection)
                config.pop("tokenizer_path", None)
        else:
            # None means remove from config (use auto-detection)
            config.pop("tokenizer_path", None)
        if max_new_tokens is not None:
            config["max_new_tokens"] = max_new_tokens
        if top_p is not None:
            config["top_p"] = top_p
        if temperature is not None:
            config["temperature"] = temperature
        if allow_primitive_nodes is not None:
            config["allow_primitive_nodes"] = allow_primitive_nodes

        config.update(kwargs)

        # Model/tokenizer paths are in ComfyUI models/, not custom node dir
        if "model_path" in config:
            config["model_path"] = resolve_model_path(config["model_path"], model_type="llm")
        if "tokenizer_path" in config:
            tokenizer_path_val = config["tokenizer_path"]
            if not os.path.isabs(tokenizer_path_val):
                resolved = resolve_model_path(tokenizer_path_val, model_type="llm")
                if os.path.exists(resolved) and os.path.isdir(resolved):
                    config["tokenizer_path"] = resolved
                else:
                    from ..utils.model_manager import get_comfyui_base_path

                    comfyui_base = get_comfyui_base_path()
                    if comfyui_base:
                        full_path = comfyui_base / tokenizer_path_val
                        if full_path.exists():
                            config["tokenizer_path"] = str(full_path)
                        else:
                            config["tokenizer_path"] = str(full_path.resolve())
                    else:
                        config["tokenizer_path"] = os.path.abspath(tokenizer_path_val)
            else:
                config["tokenizer_path"] = os.path.normpath(os.path.abspath(tokenizer_path_val))

        return OmegaConf.create({"workflow_generator": config})

    def build_node_validator_config(
        self,
        local: bool = False,
        model_format: str = "gguf",
        model_path: str = "",
        tokenizer_path: str = "",
        k: int = 5,
        max_new_tokens: int = 4096,
        **kwargs,
    ) -> DictConfig:
        """
        Build configuration for NodeValidator.

        Args:
            local: Whether to use local model
            model_format: Model format ("gguf" or "huggingface")
            model_path: Path to model file/directory
            tokenizer_path: Path to tokenizer directory
            k: Number of similar nodes to consider
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional configuration overrides

        Returns:
            OmegaConf DictConfig object
        """
        config = self.base_config["node_validator"].copy()

        if local is not None:
            config["local"] = local
        if model_format:
            config["model_format"] = model_format
        if model_path:
            config["model_path"] = model_path
        # Only set tokenizer_path if explicitly provided (not None)
        # If None, remove it from config so auto-detection happens
        if tokenizer_path is not None:
            if tokenizer_path:  # Non-empty string
                config["tokenizer_path"] = tokenizer_path
            else:
                # Empty string means remove from config (use auto-detection)
                config.pop("tokenizer_path", None)
        else:
            # None means remove from config (use auto-detection)
            config.pop("tokenizer_path", None)
        if k is not None:
            config["k"] = k
        if max_new_tokens is not None:
            config["max_new_tokens"] = max_new_tokens

        config.update(kwargs)

        # Model/tokenizer paths are in ComfyUI models/, not custom node dir
        if "model_path" in config:
            config["model_path"] = resolve_model_path(config["model_path"], model_type="llm")
        if "tokenizer_path" in config:
            tokenizer_path_val = config["tokenizer_path"]
            if not os.path.isabs(tokenizer_path_val):
                resolved = resolve_model_path(tokenizer_path_val, model_type="llm")
                if os.path.exists(resolved) and os.path.isdir(resolved):
                    config["tokenizer_path"] = resolved
                else:
                    from ..utils.model_manager import get_comfyui_base_path

                    comfyui_base = get_comfyui_base_path()
                    if comfyui_base:
                        full_path = comfyui_base / tokenizer_path_val
                        if full_path.exists():
                            config["tokenizer_path"] = str(full_path)
                        else:
                            config["tokenizer_path"] = str(full_path.resolve())
                    else:
                        config["tokenizer_path"] = os.path.abspath(tokenizer_path_val)
            else:
                config["tokenizer_path"] = os.path.normpath(os.path.abspath(tokenizer_path_val))

        return OmegaConf.create({"node_validator": config})

    def build_node_base_config(self, catalog_directory: str = "", embedding_model_path: str | None = None, **kwargs) -> DictConfig:
        """
        Build configuration for NodeBase.

        Args:
            catalog_directory: Directory containing node_list.json and node_info.json
            embedding_model_path: Path to embedding model (optional - only needed for semantic search)
            **kwargs: Additional configuration overrides

        Returns:
            OmegaConf DictConfig object
        """
        config = self.base_config["node_base"].copy()

        if catalog_directory:
            config["catalog_directory"] = catalog_directory

        if embedding_model_path is not None:
            if embedding_model_path:
                config["embedding_model_path"] = embedding_model_path
            else:
                config.pop("embedding_model_path", None)

        config.update(kwargs)

        if "catalog_directory" in config:
            catalog_dir = resolve_config_path(config["catalog_directory"], self.custom_node_path)
            config["node_list_path"] = os.path.join(catalog_dir, "node_list.json")
            config["node_info_db_path"] = os.path.join(catalog_dir, "node_info.json")
            config.pop("catalog_directory", None)
        else:
            config["node_list_path"] = resolve_config_path("catalog/node_list.json", self.custom_node_path)
            config["node_info_db_path"] = resolve_config_path("catalog/node_info.json", self.custom_node_path)

        if "embedding_model_path" in config:
            from ..utils.model_manager import resolve_model_path

            config["embedding_model_path"] = resolve_model_path(config["embedding_model_path"], model_type="llm")

        return OmegaConf.create({"node_base": config})

    def build_full_config(
        self,
        workflow_generator_kwargs: dict | None = None,
        node_validator_kwargs: dict | None = None,
        node_base_kwargs: dict | None = None,
        workflow_builder_kwargs: dict | None = None,
    ) -> DictConfig:
        """
        Build full configuration for all generators.

        Args:
            workflow_generator_kwargs: WorkflowGenerator configuration overrides
            node_validator_kwargs: NodeValidator configuration overrides
            node_base_kwargs: NodeBase configuration overrides

        Returns:
            OmegaConf DictConfig object with all generator configurations
        """
        workflow_generator_cfg = self.build_workflow_generator_config(**(workflow_generator_kwargs or {}))
        node_validator_cfg = self.build_node_validator_config(**(node_validator_kwargs or {}))
        node_base_cfg = self.build_node_base_config(**(node_base_kwargs or {}))

        full_config = OmegaConf.merge(workflow_generator_cfg, node_validator_cfg, node_base_cfg)

        return full_config
