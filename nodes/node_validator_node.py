"""
NodeValidatorNode: Validates and corrects node names in workflow diagrams.
Step 2 of the ComfyUI-WorkflowGenerator pipeline (optional but recommended).
"""

import json
import logging
import random
from typing import Any

import numpy as np
import torch
from comfy_api.latest import io
from omegaconf import DictConfig

from ..config.config import ConfigBuilder
from ..generators.node_validator import NodeBase, NodeValidator
from ..utils.model_manager import (
    detect_model_format,
    get_embedding_models,
    get_model_full_path,
    get_refine_agent_models,
)
from ..utils.node_utils import check_llm_lazy_status, get_n_threads_options, parse_n_threads

# Generate options list at module import time (safe fallback if detection fails)
_N_THREADS_OPTIONS = get_n_threads_options()


class NodeValidatorNode(io.ComfyNode):
    """
    NodeValidator: Validates and corrects node names in workflow diagrams.
    Uses semantic search and optionally an LLM to fix invalid node references.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NodeValidator",
            category="WorkflowGenerator",
            display_name="2. NodeValidator (Optional)",
            description=(
                "Step 2 (Optional) - Validates and corrects node names.\n"
                "Uses semantic search and optionally an LLM to fix invalid node references.\n"
                "Recommended to ensure workflow generation succeeds if Step 1 produces errors."
            ),
            inputs=[
                io.String.Input(
                    "instruction",
                    multiline=True,
                    default="",
                    tooltip="Optional context for LLM refinement. Connect 'instruction' output from Step 1 or add your own instructions to guide refinement.",
                ),
                io.String.Input("workflow_edges", force_input=True, tooltip="Input diagram from Step 1 (WorkflowGenerator)."),
                io.Boolean.Input(
                    "use_llm_refinement",
                    default=False,
                    tooltip="Enable LLM-based correction for better accuracy (slower). False uses only semantic search.",
                ),
                io.Combo.Input(
                    "refine_model_path",
                    options=get_refine_agent_models(),  # Called dynamically in define_schema() - updates on frontend reload
                    default="Qwen2.5-7B-Instruct-q8_0.gguf",
                    lazy=True,
                    tooltip="LLM model for refinement (if enabled).",
                ),
                io.Combo.Input(
                    "embedding_model_path",
                    options=get_embedding_models(),  # Called dynamically in define_schema() - updates on frontend reload
                    default="paraphrase-multilingual-MiniLM-L12-v2",
                    tooltip="Embedding model for semantic search.",
                ),
                io.String.Input("catalog_directory", default="catalog", tooltip="Directory containing node catalog files."),
                io.Combo.Input(
                    "dtype",
                    options=["auto", "fp16", "bf16", "fp32", "fp8"],
                    default="auto",
                    lazy=True,
                    tooltip="Data type for refinement model (HuggingFace).",
                ),
                io.Combo.Input(
                    "device_preference",
                    options=["auto", "cuda", "cpu"],
                    default="auto",
                    lazy=True,
                    tooltip="Device preference for refinement model.",
                ),
                io.Combo.Input(
                    "attn_implementation",
                    options=["auto", "flash_attention_2", "sdpa"],
                    default="auto",
                    lazy=True,
                    tooltip="Attention implementation (HuggingFace).",
                ),
                io.Boolean.Input("auto_gpu_layers", default=True, lazy=True, tooltip="Auto-calculate GPU layers based on available VRAM (prevents OOM errors) (GGUF)."),
                io.Int.Input("n_gpu_layers", default=-1, min=-1, max=1000, lazy=True, tooltip="Number of GPU layers (GGUF). -1 for all."),
                io.Int.Input("max_new_tokens", default=4096, min=1, max=16384, lazy=True, tooltip="Max tokens for refinement."),
                io.Int.Input("context_size", default=4096, min=512, max=32768, lazy=True, tooltip="Context window size (GGUF)."),
                io.Boolean.Input("use_mmap", default=True, lazy=True, tooltip="Use memory-mapped loading (GGUF)."),
                io.Boolean.Input("use_mlock", default=False, lazy=True, tooltip="Lock memory to prevent swapping (GGUF)."),
                io.Int.Input("n_batch", default=512, min=32, max=2048, lazy=True, tooltip="Batch size for prompt processing (GGUF)."),
                io.Combo.Input(
                    "n_threads", options=_N_THREADS_OPTIONS, default="Auto", lazy=True, tooltip="CPU threads for inference (GGUF)."
                ),
                io.Float.Input("temperature", default=0.95, min=0.0, max=2.0, step=0.01, lazy=True, tooltip="Sampling temperature."),
                io.Int.Input("top_k", default=5, min=1, max=20, tooltip="Number of similar nodes to consider."),
                io.Float.Input("top_p", default=0.7, min=0.0, max=1.0, step=0.01, lazy=True, tooltip="Top-p sampling."),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,  # 2**32 - 1 (4294967295) - maximum valid seed value
                    control_after_generate=True,
                    lazy=True,
                    tooltip="Random seed.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="workflow_edges (refined)", tooltip="Refined workflow diagram with corrected names."),
                io.String.Output(display_name="llm_prompts (debug)", tooltip="Prompts sent to LLM (debug)."),
                io.String.Output(display_name="candidate_nodes (debug)", tooltip="Candidate nodes found (debug)."),
            ],
        )

    @classmethod
    def validate_inputs(cls, instruction: str = "", catalog_directory: str = "", embedding_model_path: str = "", **kwargs) -> bool | str:
        """
        Validate constant inputs before execution.

        Note: Only constant inputs (widget values) are available here.
        Connected inputs (like workflow_edges) are not available and should be validated in execute().

        Returns:
            True if valid, or error message string if invalid
        """
        # workflow_edges is force_input=True (connection-only), so it's not available here
        # instruction can be constant or connected - only validate if it's a constant
        # Other validation happens in execute() when upstream nodes have run
        return True

    @classmethod
    def check_lazy_status(
        cls,
        use_llm_refinement: bool = False,
        refine_model_path: str = None,
        max_new_tokens: int = None,
        device_preference: str = None,
        n_gpu_layers: int | None = None,
        dtype: str = None,
        attn_implementation: str = None,
        context_size: int | None = None,
        temperature: float = None,
        top_p: float = None,
        seed: int = None,
        **kwargs,
    ) -> list[str]:
        """
        Determine which lazy inputs need to be evaluated.

        Returns:
            List of input names that need to be evaluated.
            LLM-related inputs are only needed if use_llm_refinement is True.
        """
        return check_llm_lazy_status(
            use_llm_refinement=use_llm_refinement,
            refine_model_path=refine_model_path,
            max_new_tokens=max_new_tokens,
            device_preference=device_preference,
            n_gpu_layers=n_gpu_layers,
            dtype=dtype,
            attn_implementation=attn_implementation,
            context_size=context_size,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            **kwargs,
        )

    @classmethod
    def fingerprint_inputs(cls, workflow_edges: str = "", seed: int = 0, use_llm_refinement: bool = False, **kwargs) -> Any:
        """
        Fingerprint inputs to determine if node should re-execute.

        Returns:
            Tuple of relevant inputs (no hashing needed - ComfyUI compares with !=)
        """
        # Include seed only if using LLM refinement (LLM is stochastic)
        # Semantic search is deterministic, so seed doesn't matter when use_llm_refinement=False
        if use_llm_refinement:
            return (workflow_edges, seed)
        else:
            return (workflow_edges,)

    @classmethod
    def execute(
        cls,
        workflow_edges: str,
        instruction: str = "",
        catalog_directory: str = "catalog",
        embedding_model_path: str = "",
        use_llm_refinement: bool = False,
        refine_model_path: str | None = None,
        top_k: int = 5,
        max_new_tokens: int | None = None,
        device_preference: str | None = None,
        auto_gpu_layers: bool | None = True,
        n_gpu_layers: int | None = None,
        dtype: str | None = None,
        attn_implementation: str | None = None,
        context_size: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        use_mmap: bool | None = None,
        use_mlock: bool | None = None,
        n_batch: int | None = None,
        n_threads: str | None = None,
        seed: int | None = None,
    ) -> io.NodeOutput:
        """Execute NodeValidator to validate and correct node names."""
        try:
            logging.info("WorkflowGenerator NodeValidator: Starting diagram refinement")

            if not workflow_edges or not workflow_edges.strip():
                raise ValueError(
                    "workflow_edges input is required - connect the 'workflow_edges' output from Step 1 (WorkflowGenerator). "
                    "If Step 1 failed, check its error message (e.g., missing llama-cpp-python)."
                )

            if use_llm_refinement and seed is not None:
                max_seed = 2**32 - 1
                if seed < 0 or seed > max_seed:
                    seed = seed % (max_seed + 1)

                random.seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                torch.manual_seed(seed)

            embedding_model_full_path = get_model_full_path(embedding_model_path) if embedding_model_path else None

            refine_model_full_path = None
            refine_model_format = None
            if use_llm_refinement:
                if refine_model_path is None:
                    refine_model_path = "models/LLM/Qwen2.5-7B-Instruct-q8_0.gguf"
                refine_model_full_path = get_model_full_path(refine_model_path)
                refine_model_format = detect_model_format(refine_model_path)

                if max_new_tokens is None:
                    max_new_tokens = 4096
                if device_preference is None:
                    device_preference = "auto"
                if auto_gpu_layers is None:
                    auto_gpu_layers = False
                if n_gpu_layers is None:
                    n_gpu_layers = -1
                if dtype is None:
                    dtype = "auto"
                if attn_implementation is None:
                    attn_implementation = "auto"
                if temperature is None:
                    temperature = 0.95
                if top_p is None:
                    top_p = 0.7
                if seed is None:
                    seed = 0
                if context_size is None:
                    context_size = 4096
                if use_mmap is None:
                    use_mmap = True
                if use_mlock is None:
                    use_mlock = False
                if n_batch is None:
                    n_batch = 512
                if n_threads is None:
                    n_threads = "Auto"
                n_threads_parsed = parse_n_threads(n_threads)

                if auto_gpu_layers:
                    n_gpu_layers = "auto"

            try:
                diagram_obj = json.loads(workflow_edges)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"workflow_edges contains invalid JSON: {e}\n"
                    "The workflow_edges must be a valid JSON string (typically a list of edges). "
                    "If you're connecting from Step 1 (WorkflowGenerator), ensure Step 1 executed successfully. "
                    "If entering manually, ensure the JSON is properly formatted."
                ) from e

            config_builder = ConfigBuilder()
            node_base_cfg = config_builder.build_node_base_config(
                catalog_directory=catalog_directory, embedding_model_path=embedding_model_full_path
            )

            refine_cfg = config_builder.build_node_validator_config(
                local=use_llm_refinement,
                model_format=refine_model_format if use_llm_refinement else "gguf",
                model_path=refine_model_full_path if use_llm_refinement else "",
                tokenizer_path=None,  # Auto-detected in model loader
                k=top_k,
                max_new_tokens=max_new_tokens if use_llm_refinement else 4096,
            )

            cfg = DictConfig({"node_base": node_base_cfg.node_base, "node_validator": refine_cfg.node_validator})

            node_base = NodeBase(cfg)

            node_validator = NodeValidator(
                node_base,
                cfg,
                n_gpu_layers=n_gpu_layers if use_llm_refinement else -1,
                dtype=dtype if use_llm_refinement else "auto",
                device_preference=device_preference if use_llm_refinement else "auto",
                attn_implementation=attn_implementation if use_llm_refinement else "auto",
                context_size=context_size if use_llm_refinement else 4096,
                temperature=temperature if use_llm_refinement else 0.95,
                top_p=top_p if use_llm_refinement else 0.7,
                use_mmap=use_mmap if use_llm_refinement else True,
                use_mlock=use_mlock if use_llm_refinement else False,
                n_batch=n_batch if use_llm_refinement else 512,
                n_threads=n_threads_parsed if use_llm_refinement else None,
            )

            try:
                refine_seed = seed if use_llm_refinement else None
                refine_prompt = instruction if instruction else ""
                refined_diagram, debug_info = node_validator.refine_diagram(diagram_obj, prompt=refine_prompt, seed=refine_seed)

                refined_diagram_json = json.dumps(refined_diagram, indent=2)
                llm_prompts_json = json.dumps(debug_info.get("llm_prompts", []), indent=2)
                candidate_nodes_json = json.dumps(debug_info.get("candidate_nodes", {}), indent=2)

                logging.info("WorkflowGenerator NodeValidator: Diagram refined successfully")
                return io.NodeOutput(refined_diagram_json, llm_prompts_json, candidate_nodes_json)

            finally:
                node_validator.cleanup()

        except Exception as e:
            logging.error(f"WorkflowGenerator NodeValidator error: {e}", exc_info=True)
            raise
