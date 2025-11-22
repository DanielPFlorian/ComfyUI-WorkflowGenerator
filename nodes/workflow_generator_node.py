"""
WorkflowGeneratorNode: Generates workflow diagrams from natural language descriptions.
Step 1 of the ComfyUI-WorkflowGenerator pipeline.
"""

import json
import logging
import random
from typing import Any

import numpy as np
import torch
from comfy_api.latest import io

from ..config.config import ConfigBuilder
from ..generators.workflow_generator import WorkflowGenerator
from ..utils.model_manager import (
    detect_model_format,
    get_model_full_path,
    get_workflow_generator_models,
    register_llm_folder_type,
)
from ..utils.node_utils import get_n_threads_options, parse_n_threads

# Generate options list at module import time (safe fallback if detection fails)
_N_THREADS_OPTIONS = get_n_threads_options()


class WorkflowGeneratorNode(io.ComfyNode):
    """
    WorkflowGenerator: Generates workflow diagrams from natural language descriptions.
    Uses a language model to convert user instructions into structured workflow diagrams.
    """

    @classmethod
    def define_schema(cls):
        # Ensure LLM folder is registered before getting model list
        register_llm_folder_type()
        model_options = get_workflow_generator_models()
        # If no models found, provide at least the default option
        if not model_options:
            model_options = ["workflow-generator-q8_0.gguf"]
        
        return io.Schema(
            node_id="WorkflowGenerator",
            category="WorkflowGenerator",
            display_name="1. WorkflowGenerator",
            description=(
                "Step 1 - Generates a workflow diagram from natural language instructions.\n"
                "This is the first stage of the pipeline, converting text to a structured plan."
            ),
            inputs=[
                io.String.Input("instruction", multiline=True, tooltip="Description of the desired ComfyUI workflow."),
                io.Combo.Input(
                    "model_path",
                    options=model_options,  # Populated after ensuring folder registration
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
                io.Boolean.Input("use_mmap", default=True, lazy=True, tooltip="Use memory-mapped loading (GGUF)."),
                io.Boolean.Input("use_mlock", default=False, lazy=True, tooltip="Lock memory to prevent swapping (GGUF)."),
                io.Int.Input("n_batch", default=512, min=32, max=2048, lazy=True, tooltip="Batch size for prompt processing (GGUF)."),
                io.Combo.Input(
                    "n_threads", options=_N_THREADS_OPTIONS, default="Auto", lazy=True, tooltip="CPU threads for inference (GGUF)."
                ),
                io.Float.Input("temperature", default=0.95, min=0.0, max=2.0, step=0.01, tooltip="Sampling temperature."),
                io.Float.Input("top_p", default=0.7, min=0.0, max=1.0, step=0.01, tooltip="Top-p sampling."),
                io.Boolean.Input(
                    "allow_primitive_nodes",
                    default=False,
                    tooltip="Include primitive nodes in diagram. False (recommended) removes them for cleaner workflows.",
                ),
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
                io.String.Output(display_name="workflow_edges", tooltip="Generated workflow diagram (JSON)."),
                io.String.Output(display_name="instruction", tooltip="Original instruction text (pass to Step 2)."),
            ],
        )

    @classmethod
    def validate_inputs(cls, instruction: str = "", **kwargs) -> bool | str:
        """
        Validate constant inputs before execution.

        Note: Only constant inputs (widget values) are available here.
        Connected inputs are not available and should be validated in execute().

        Returns:
            True if valid, or error message string if invalid
        """
        # Validation moved to execute() to avoid ComfyUI's per-input error formatting
        # This prevents the "Custom validation failed for node: X -" prefix and duplicate errors
        return True

    @classmethod
    def fingerprint_inputs(cls, instruction: str = "", seed: int = 0, **kwargs) -> Any:
        """
        Fingerprint inputs to determine if node should re-execute.

        Returns:
            Tuple of relevant inputs (no hashing needed - ComfyUI compares with !=)
        """
        # Return simple tuple - ComfyUI handles comparison
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
        device_preference: str = "auto",
        auto_gpu_layers: bool = True,
        n_gpu_layers: int = -1,
        dtype: str = "auto",
        attn_implementation: str = "auto",
        context_size: int = 4096,
        use_mmap: bool = True,
        use_mlock: bool = False,
        n_batch: int = 512,
        n_threads: str = "Auto",
        seed: int = 0,
    ) -> io.NodeOutput:
        """Execute WorkflowGenerator to generate workflow diagram."""
        if not instruction or not instruction.strip():
            raise ValueError("instruction input is required - enter a description of the workflow you want to generate")

        try:
            logging.info("WorkflowGenerator: Starting workflow diagram generation")

            model_full_path = get_model_full_path(model_path)
            model_format = detect_model_format(model_path)

            # Handle auto_gpu_layers: if True, calculate optimal layers; otherwise use manual value
            if auto_gpu_layers:
                # Auto-calculate will be done in model loader
                n_gpu_layers = "auto"  # Special value for model loader to trigger auto-calculation
            # Otherwise, n_gpu_layers is an integer: -1 (all), 0 (CPU only), or 1+ (specific count)
            # If value exceeds model layers, model loader will handle it gracefully

            # Build configuration (tokenizer path will be auto-detected)
            config_builder = ConfigBuilder()
            cfg = config_builder.build_workflow_generator_config(
                model_format=model_format,
                model_path=model_full_path,
                tokenizer_path=None,  # Auto-detected in model loader
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                allow_primitive_nodes=allow_primitive_nodes,
            )

            n_threads_parsed = parse_n_threads(n_threads)

            workflow_generator = WorkflowGenerator(
                cfg,
                n_gpu_layers=n_gpu_layers,
                dtype=dtype,
                device_preference=device_preference,
                attn_implementation=attn_implementation,
                context_size=context_size,
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                n_batch=n_batch,
                n_threads=n_threads_parsed,
            )

            try:
                max_seed = 2**32 - 1
                if seed < 0 or seed > max_seed:
                    seed = seed % (max_seed + 1)

                random.seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                torch.manual_seed(seed)

                diagram = workflow_generator.generate(instruction, seed=seed)
                diagram_json = json.dumps(diagram, indent=2)

                logging.info("WorkflowGenerator: Diagram generated successfully")
                return io.NodeOutput(diagram_json, instruction)

            finally:
                workflow_generator.cleanup()

        except Exception as e:
            logging.error(f"WorkflowGenerator error: {e}", exc_info=True)
            raise
