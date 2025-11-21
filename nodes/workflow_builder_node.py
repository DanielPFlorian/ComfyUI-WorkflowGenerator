"""
WorkflowBuilderNode: Converts workflow diagrams to ComfyUI workflow JSON.
Step 3 of the ComfyUI-WorkflowGenerator pipeline.
"""

import hashlib
import json
import logging
from typing import Any

from comfy_api.latest import io
from omegaconf import DictConfig

# Import utilities
from ..config.config import ConfigBuilder
from ..generators.node_validator import NodeBase

# Import generator classes
from ..generators.workflow_builder import WorkflowBuilder
from ..utils.file_utils import save_workflow_json

# Try to import ComfyUI utilities
try:
    import folder_paths

    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    logging.warning("ComfyUI folder_paths not available")

# Internal cache for built workflows (keyed by workflow_edges + catalog_directory hash)
_workflow_cache: dict[str, str] = {}


class WorkflowBuilderNode(io.ComfyNode):
    """
    WorkflowBuilder: Converts workflow diagrams to ComfyUI workflow JSON.
    Can save the workflow to a file in ComfyUI's output directory.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WorkflowBuilder",
            category="WorkflowGenerator",
            display_name="3. Build and Save Workflow",
            description=("Step 3 - Converts workflow diagram to ComfyUI JSON.\nCan save the workflow to a file for loading."),
            inputs=[
                io.String.Input("workflow_edges", force_input=True, tooltip="Input diagram from Step 1 or Step 2."),
                io.String.Input("catalog_directory", default="catalog", tooltip="Directory containing node catalog files."),
                io.Boolean.Input("save_workflow", default=True, tooltip="Save workflow to file."),
                io.String.Input("filename_prefix", default="generated_workflow", tooltip="Filename prefix or path for saved workflow."),
            ],
            outputs=[
                io.String.Output(display_name="workflow_json", tooltip="Complete ComfyUI workflow JSON."),
                io.String.Output(display_name="file_path", tooltip="Path to saved file."),
            ],
            is_output_node=True,  # Mark as output node since it can save files
        )

    @classmethod
    def validate_inputs(cls, catalog_directory: str = "", **kwargs) -> bool | str:
        """
        Validate constant inputs before execution.

        Note: Only constant inputs (widget values) are available here.
        Connected inputs (like workflow_edges) are not available and should be validated in execute().

        Returns:
            True if valid, or error message string if invalid
        """
        # workflow_edges is force_input=True (connection-only), so it's not available here
        # Validation of connected inputs happens in execute() when upstream nodes have run
        return True

    @classmethod
    def fingerprint_inputs(
        cls, workflow_edges: str = "", catalog_directory: str = "catalog", save_workflow: bool = False, filename_prefix: str = "", **kwargs
    ) -> Any:
        """
        Fingerprint inputs to determine if node should re-execute.

        Caching strategy:
        - Only includes inputs that affect the workflow build (workflow_edges, catalog_directory)
        - save_workflow and filename_prefix are excluded because they only affect saving, not building
        - Internal caching in execute() will reuse built workflow JSON when only save settings change
        - When rerunning with the same seed, workflow_edges will be identical (from cached upstream nodes),
          so the workflow build is cached and only the save step runs if needed

        Returns:
            Tuple of relevant inputs (no hashing needed - ComfyUI compares with !=)
        """
        # Only include inputs that affect the workflow build itself
        # save_workflow and filename_prefix don't affect the build, only the saving step
        # Internal cache in execute() handles reusing built workflow when only save settings change
        return (workflow_edges, catalog_directory)

    @classmethod
    def execute(
        cls,
        workflow_edges: str,
        catalog_directory: str = "catalog",
        save_workflow: bool = True,
        filename_prefix: str = "generated_workflow",
    ) -> io.NodeOutput:
        """
        Execute WorkflowBuilder to convert diagram to workflow.

        Uses internal caching to avoid rebuilding workflows when only save settings change.
        The workflow JSON is cached based on workflow_edges and catalog_directory.
        """
        try:
            # Create cache key from workflow_edges and catalog_directory
            cache_key_data = f"{workflow_edges}|{catalog_directory}"
            cache_key = hashlib.sha256(cache_key_data.encode("utf-8")).hexdigest()

            # Check if we have a cached workflow JSON
            workflow_json = _workflow_cache.get(cache_key)

            if workflow_json is not None:
                logging.info("WorkflowGenerator WorkflowBuilder: Using cached workflow JSON")
            else:
                logging.info("WorkflowGenerator WorkflowBuilder: Starting workflow conversion")

                # Validate workflow_edges is provided
                if not workflow_edges or not workflow_edges.strip():
                    raise ValueError(
                        "workflow_edges input is required - connect the 'workflow_edges' output from Step 1 (WorkflowGenerator) "
                        "or 'workflow_edges (refined)' output from Step 2 (NodeValidator). "
                        "If upstream nodes failed, check their error messages (e.g., missing llama-cpp-python)."
                    )

                # Parse workflow_edges
                try:
                    diagram_obj = json.loads(workflow_edges)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid workflow_edges JSON: {e}") from e

                # Build configuration
                # Note: catalog_directory is resolved by build_node_base_config using resolve_config_path
                # embedding_model_path is not needed for WorkflowBuilder (only Step 2 (NodeValidator) uses semantic search)
                config_builder = ConfigBuilder()
                node_base_cfg = config_builder.build_node_base_config(
                    catalog_directory=catalog_directory,
                    embedding_model_path=None,  # Not needed for WorkflowBuilder
                )

                # Merge configs
                cfg = DictConfig({"node_base": node_base_cfg.node_base})

                # Initialize NodeBase and WorkflowBuilder
                node_base = NodeBase(cfg)
                workflow_builder = WorkflowBuilder(node_base)

                # Convert diagram to workflow
                workflow = workflow_builder.parse_diagram_to_workflow(diagram_obj)

                # Convert to JSON string and cache it
                workflow_json = json.dumps(workflow, indent=2)
                _workflow_cache[cache_key] = workflow_json

                logging.info("WorkflowGenerator WorkflowBuilder: Workflow conversion complete (cached)")

            # Save workflow if requested (this step always runs, even with cached workflow)
            file_path = ""
            if save_workflow and COMFYUI_AVAILABLE:
                try:
                    output_dir = folder_paths.get_output_directory()
                    file_path = save_workflow_json(workflow_json, filename_prefix, output_dir)
                    logging.info(f"WorkflowGenerator WorkflowBuilder: Workflow saved to {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to save workflow: {e}")

            return io.NodeOutput(workflow_json, file_path)

        except Exception as e:
            logging.error(f"WorkflowGenerator WorkflowBuilder error: {e}", exc_info=True)
            raise
