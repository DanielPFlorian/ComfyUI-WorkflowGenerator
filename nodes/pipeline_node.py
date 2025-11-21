"""
WGPipelineNode: Complete ComfyUI-WorkflowGenerator workflow generation pipeline.
Executes WorkflowGenerator → NodeValidator → WorkflowBuilder in sequence.
"""

import hashlib
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
from ..generators.workflow_builder import WorkflowBuilder
from ..generators.workflow_generator import WorkflowGenerator
from ..utils.file_utils import save_workflow_json
from ..utils.model_manager import (
    detect_model_format,
    get_embedding_models,
    get_model_full_path,
    get_refine_agent_models,
    get_workflow_generator_models,
)
from ..utils.node_utils import check_llm_lazy_status

try:
    import folder_paths

    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    logging.warning("ComfyUI folder_paths not available")

# Internal cache for built workflows in pipeline (keyed by refined_diagram + catalog_directory hash)
_pipeline_workflow_cache: dict[str, str] = {}


class WGPipelineNode(io.ComfyNode):
    """
    Pipeline: Complete ComfyUI-WorkflowGenerator workflow generation pipeline.
    Executes WorkflowGenerator → NodeValidator → WorkflowBuilder in sequence.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WG_Pipeline",
            category="WorkflowGenerator",
            display_name="Workflow Generator Pipeline",
            description=(
                "Complete Pipeline - Executes steps 1-3 sequentially:\n"
                "1. WorkflowGenerator -> Creates initial diagram\n"
                "2. NodeValidator -> Validates/corrects node names\n"
                "3. Build and Save -> Converts to JSON\n"
                "Convenience node for the full process."
            ),
            inputs=[
                # WorkflowGenerator inputs (Step 1) - in same order as WorkflowGeneratorNode
                io.String.Input("instruction", multiline=True, tooltip="Description of the desired ComfyUI workflow."),
                io.Combo.Input(
                    "model_path",
                    options=get_workflow_generator_models(),  # Called dynamically in define_schema() - updates on frontend reload
                    default="workflow-generator-q8_0.gguf",
                    tooltip="Model file (GGUF) or directory (HuggingFace).",
                ),
                io.Combo.Input(
                    "dtype", options=["auto", "fp16", "bf16", "fp32", "fp8"], default="auto", tooltip="Data type (HuggingFace only)."
                ),
                io.Combo.Input(
                    "device_preference", options=["auto", "cuda", "cpu"], default="auto", tooltip="Device preference (cuda/cpu)."
                ),
                io.Combo.Input(
                    "attn_implementation",
                    options=["auto", "flash_attention_2", "sdpa"],
                    default="auto",
                    tooltip="Attention implementation (HuggingFace).",
                ),
                io.Boolean.Input("auto_gpu_layers", default=True, tooltip="Auto-calculate GPU layers based on available VRAM (prevents OOM errors) (GGUF)."),
                io.Int.Input("n_gpu_layers", default=-1, min=-1, max=1000, tooltip="Number of GPU layers (GGUF). -1 for all."),
                io.Int.Input("max_new_tokens", default=8192, min=1, max=32768, tooltip="Max tokens to generate."),
                io.Int.Input("context_size", default=4096, min=512, max=131072, tooltip="Context window size (GGUF)."),
                io.Float.Input("temperature", default=0.95, min=0.0, max=2.0, step=0.01, tooltip="Sampling temperature."),
                io.Float.Input("top_p", default=0.7, min=0.0, max=1.0, step=0.01, tooltip="Top-p sampling."),
                io.Boolean.Input(
                    "allow_primitive_nodes",
                    default=False,
                    tooltip="Include primitive nodes in diagram. False removes them for cleaner workflows.",
                ),
                # NodeValidator inputs (Step 2) - in same order as NodeValidatorNode
                io.Boolean.Input("refine_enabled", default=True, tooltip="Enable Step 2 (NodeValidator)."),
                io.Boolean.Input(
                    "use_llm_refinement", default=False, tooltip="Enable LLM-based node correction (slower but more accurate)."
                ),
                io.Combo.Input(
                    "refine_model_path",
                    options=get_refine_agent_models(),  # Called dynamically in define_schema() - updates on frontend reload
                    default="Qwen2.5-7B-Instruct-q8_0.gguf",
                    lazy=True,
                    tooltip="LLM model for refinement.",
                ),
                io.Combo.Input(
                    "embedding_model_path",
                    options=get_embedding_models(),  # Called dynamically in define_schema() - updates on frontend reload
                    default="paraphrase-multilingual-MiniLM-L12-v2",
                    tooltip="Embedding model for semantic search.",
                ),
                io.String.Input("catalog_directory", default="catalog", tooltip="Directory containing node catalog files."),
                io.Combo.Input(
                    "refine_dtype",
                    options=["auto", "fp16", "bf16", "fp32", "fp8"],
                    default="auto",
                    lazy=True,
                    tooltip="Data type for refinement model (HuggingFace).",
                ),
                io.Combo.Input(
                    "refine_device_preference",
                    options=["auto", "cuda", "cpu"],
                    default="auto",
                    lazy=True,
                    tooltip="Device preference for refinement model.",
                ),
                io.Combo.Input(
                    "refine_attn_implementation",
                    options=["auto", "flash_attention_2", "sdpa"],
                    default="auto",
                    lazy=True,
                    tooltip="Attention implementation for refinement model.",
                ),
                io.Boolean.Input(
                    "refine_auto_gpu_layers", default=False, lazy=True, tooltip="Auto-calculate GPU layers for refinement model (GGUF)."
                ),
                io.Int.Input(
                    "refine_n_gpu_layers",
                    default=-1,
                    min=-1,
                    max=1000,
                    lazy=True,
                    tooltip="GPU layers for refinement model (GGUF). -1 for all.",
                ),
                io.Int.Input("refine_max_new_tokens", default=4096, min=1, max=16384, lazy=True, tooltip="Max tokens for refinement."),
                io.Int.Input(
                    "refine_context_size", default=4096, min=512, max=32768, lazy=True, tooltip="Context window size for refinement model (GGUF)."
                ),
                io.Float.Input(
                    "refine_temperature", default=0.95, min=0.0, max=2.0, step=0.01, lazy=True, tooltip="Temperature for refinement."
                ),
                io.Int.Input("top_k", default=5, min=1, max=20, tooltip="Number of similar nodes to consider."),
                io.Float.Input("refine_top_p", default=0.7, min=0.0, max=1.0, step=0.01, lazy=True, tooltip="Top-p for refinement."),
                # WorkflowBuilder inputs (Step 3) - in same order as WorkflowBuilderNode
                io.Boolean.Input("save_workflow", default=True, tooltip="Save workflow to file."),
                io.String.Input("filename_prefix", default="generated_workflow", tooltip="Filename prefix or path for saved workflow."),
                # Seed for reproducibility (shared by WorkflowGenerator and NodeValidator)
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,  # 2**32 - 1 (4294967295) - maximum valid seed value
                    control_after_generate=True,
                    tooltip="Random seed.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="workflow_edges", tooltip="Initial workflow diagram (JSON)."),
                io.String.Output(display_name="workflow_edges (refined)", tooltip="Refined workflow diagram (JSON)."),
                io.String.Output(display_name="workflow_json", tooltip="Final ComfyUI workflow JSON."),
                io.String.Output(display_name="file_path", tooltip="Path to saved file."),
            ],
            is_output_node=True,  # Mark as output node since WorkflowBuilder saves files
        )

    @classmethod
    def validate_inputs(cls, instruction: str = "", **kwargs) -> bool | str:
        """
        Validate constant inputs before execution.

        Note: Only constant inputs (widget values) are available here.
        Connected inputs are not available and should be validated in execute().

        Returns:
            True if valid, or error message string if invalid

        Note: Validation moved to execute() to avoid ComfyUI's per-input error formatting.
        This prevents the "Custom validation failed for node: X -" prefix and duplicate errors
        that appear on every input field. The execute() method will raise a ValueError with
        a clear message if instruction is missing.
        """
        return True

    @classmethod
    def check_lazy_status(
        cls,
        refine_enabled: bool = True,
        use_llm_refinement: bool = False,
        refine_model_path: str = None,
        refine_context_size: int | None = None,
        refine_max_new_tokens: int | None = None,
        refine_temperature: float | None = None,
        refine_top_p: float | None = None,
        refine_auto_gpu_layers: bool | None = None,
        refine_n_gpu_layers: int | None = None,
        refine_dtype: str | None = None,
        refine_device_preference: str | None = None,
        refine_attn_implementation: str | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Determine which lazy inputs need to be evaluated.

        Returns:
            List of input names that need to be evaluated.
            Refine model inputs are only needed if refine_enabled=True AND use_llm_refinement=True.
        """
        if not (refine_enabled and use_llm_refinement):
            return []

        return check_llm_lazy_status(
            use_llm_refinement=True,
            refine_model_path=refine_model_path,
            refine_context_size=refine_context_size,
            refine_max_new_tokens=refine_max_new_tokens,
            refine_temperature=refine_temperature,
            refine_top_p=refine_top_p,
            refine_auto_gpu_layers=refine_auto_gpu_layers,
            refine_n_gpu_layers=refine_n_gpu_layers,
            refine_dtype=refine_dtype,
            refine_device_preference=refine_device_preference,
            refine_attn_implementation=refine_attn_implementation,
            **kwargs,
        )

    @classmethod
    def fingerprint_inputs(cls, instruction: str = "", seed: int = 0, **kwargs) -> Any:
        """
        Fingerprint inputs to determine if node should re-execute.

        Returns:
            Tuple of relevant inputs (no hashing needed - ComfyUI compares with !=)
        """
        # Pipeline contains WorkflowGenerator (stochastic) and potentially NodeValidator with LLM (stochastic)
        # Same instruction + same seed = same output
        return (instruction, seed)

    @classmethod
    def execute(
        cls,
        instruction: str,
        model_path: str = "workflow-generator-q8_0.gguf",
        max_new_tokens: int = 8192,
        top_p: float = 0.7,
        temperature: float = 0.95,
        allow_primitive_nodes: bool = False,
        refine_enabled: bool = True,
        use_llm_refinement: bool = False,
        refine_model_path: str | None = None,
        top_k: int = 5,
        refine_max_new_tokens: int | None = None,
        refine_temperature: float | None = None,
        refine_top_p: float | None = None,
        refine_auto_gpu_layers: bool | None = None,
        refine_n_gpu_layers: int | None = None,
        refine_dtype: str | None = None,
        refine_device_preference: str | None = None,
        refine_attn_implementation: str | None = None,
        refine_context_size: int | None = None,
        catalog_directory: str = "catalog",
        embedding_model_path: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device_preference: str = "auto",
        auto_gpu_layers: bool = True,
        n_gpu_layers: int = -1,
        dtype: str = "auto",
        attn_implementation: str = "auto",
        context_size: int = 4096,
        save_workflow: bool = True,
        filename_prefix: str = "generated_workflow",
        seed: int = 0,
    ) -> io.NodeOutput:
        """Execute complete ComfyUI-WorkflowGenerator pipeline."""
        try:
            logging.info("WorkflowGenerator Pipeline: Starting complete workflow generation")

            if not instruction or not instruction.strip():
                raise ValueError("instruction input is required - enter a description of the workflow you want to generate")

            max_seed = 2**32 - 1
            if seed < 0 or seed > max_seed:
                seed = seed % (max_seed + 1)

            random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)

            model_full_path = get_model_full_path(model_path)
            model_format = detect_model_format(model_path)

            embedding_model_full_path = get_model_full_path(embedding_model_path)

            workflow_generator_n_gpu_layers = n_gpu_layers
            if auto_gpu_layers:
                workflow_generator_n_gpu_layers = "auto"

            if refine_enabled and use_llm_refinement:
                if refine_model_path is None:
                    refine_model_path = "Qwen2.5-7B-Instruct-q8_0.gguf"
                if refine_context_size is None:
                    refine_context_size = 4096
                if refine_max_new_tokens is None:
                    refine_max_new_tokens = 4096
                if refine_temperature is None:
                    refine_temperature = 0.95
                if refine_top_p is None:
                    refine_top_p = 0.7
                if refine_auto_gpu_layers is None:
                    refine_auto_gpu_layers = False
                if refine_n_gpu_layers is None:
                    refine_n_gpu_layers = -1
                if refine_dtype is None:
                    refine_dtype = "auto"
                if refine_device_preference is None:
                    refine_device_preference = "auto"
                if refine_attn_implementation is None:
                    refine_attn_implementation = "auto"

                refine_model_full_path = get_model_full_path(refine_model_path)
                refine_model_format = detect_model_format(refine_model_path)

                refine_n_gpu_layers_value = refine_n_gpu_layers
                if refine_auto_gpu_layers:
                    refine_n_gpu_layers_value = "auto"
            else:
                refine_model_full_path = None
                refine_model_format = "gguf"
                refine_n_gpu_layers_value = -1
                refine_max_new_tokens = 4096
                refine_temperature = 0.95
                refine_top_p = 0.7
                refine_dtype = "auto"
                refine_device_preference = "auto"
                refine_attn_implementation = "auto"

            config_builder = ConfigBuilder()

            workflow_generator_cfg = config_builder.build_workflow_generator_config(
                model_format=model_format,
                model_path=model_full_path,
                tokenizer_path=None,  # Auto-detected in model loader
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                allow_primitive_nodes=allow_primitive_nodes,
            )

            node_base_cfg = config_builder.build_node_base_config(
                catalog_directory=catalog_directory, embedding_model_path=embedding_model_full_path
            )

            node_validator_cfg = config_builder.build_node_validator_config(
                local=use_llm_refinement if refine_enabled else False,
                model_format=refine_model_format if (refine_enabled and use_llm_refinement) else "gguf",
                model_path=refine_model_full_path if (refine_enabled and use_llm_refinement) else "",
                tokenizer_path=None,  # Auto-detected in model loader
                k=top_k,
                max_new_tokens=refine_max_new_tokens,
            )

            cfg = DictConfig(
                {
                    "workflow_generator": workflow_generator_cfg.workflow_generator,
                    "node_validator": node_validator_cfg.node_validator,
                    "node_base": node_base_cfg.node_base,
                }
            )

            logging.info("Pipeline Stage 1: WorkflowGenerator - Generating diagram")
            workflow_generator = WorkflowGenerator(
                cfg,
                n_gpu_layers=workflow_generator_n_gpu_layers,
                dtype=dtype,
                device_preference=device_preference,
                attn_implementation=attn_implementation,
                context_size=context_size,
            )

            try:
                diagram = workflow_generator.generate(instruction, seed=seed)
                diagram_json = json.dumps(diagram, indent=2)
            finally:
                workflow_generator.cleanup()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            refined_diagram = diagram
            refined_diagram_json = ""

            if refine_enabled:
                logging.info("Pipeline Stage 2: NodeValidator - Refining diagram")
                node_base = NodeBase(cfg)
                node_validator = NodeValidator(
                    node_base,
                    cfg,
                    n_gpu_layers=refine_n_gpu_layers_value,
                    dtype=refine_dtype,
                    device_preference=refine_device_preference,
                    attn_implementation=refine_attn_implementation,
                    context_size=refine_context_size,
                    temperature=refine_temperature,
                    top_p=refine_top_p,
                )

                try:
                    refine_seed = seed if use_llm_refinement else None
                    refine_prompt = instruction if use_llm_refinement else ""
                    refined_diagram, _ = node_validator.refine_diagram(diagram, prompt=refine_prompt, seed=refine_seed)
                    refined_diagram_json = json.dumps(refined_diagram, indent=2)
                finally:
                    node_validator.cleanup()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                refined_diagram_json = diagram_json

            refined_diagram_json_str = json.dumps(refined_diagram, sort_keys=True)
            cache_key_data = f"{refined_diagram_json_str}|{catalog_directory}"
            cache_key = hashlib.sha256(cache_key_data.encode("utf-8")).hexdigest()

            workflow_json = _pipeline_workflow_cache.get(cache_key)

            if workflow_json is not None:
                logging.info("Pipeline Stage 3: WorkflowBuilder - Using cached workflow JSON")
            else:
                logging.info("Pipeline Stage 3: WorkflowBuilder - Converting to workflow")
                node_base = NodeBase(cfg)
                workflow_builder = WorkflowBuilder(node_base)

                workflow = workflow_builder.parse_diagram_to_workflow(refined_diagram)
                workflow_json = json.dumps(workflow, indent=2)

                _pipeline_workflow_cache[cache_key] = workflow_json
                logging.info("Pipeline Stage 3: WorkflowBuilder - Workflow conversion complete (cached)")

            file_path = ""
            if save_workflow and COMFYUI_AVAILABLE:
                try:
                    output_dir = folder_paths.get_output_directory()
                    file_path = save_workflow_json(workflow_json, filename_prefix, output_dir)
                    logging.info(f"Pipeline: Workflow saved to {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to save workflow: {e}")

            logging.info("WorkflowGenerator Pipeline: Complete workflow generation finished")
            return io.NodeOutput(diagram_json, refined_diagram_json, workflow_json, file_path)

        except Exception as e:
            logging.error(f"WorkflowGenerator Pipeline error: {e}", exc_info=True)
            raise
