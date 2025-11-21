"""
ComfyUI-WorkflowGenerator Custom Nodes for ComfyUI
Generates ComfyUI workflows from natural language descriptions using LLMs.
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .nodes.node_validator_node import NodeValidatorNode
from .nodes.pipeline_node import WGPipelineNode
from .nodes.update_catalog_node import UpdateNodeCatalogNode
from .nodes.workflow_builder_node import WorkflowBuilderNode

# Import node classes from individual files
from .nodes.workflow_generator_node import WorkflowGeneratorNode

# Import model manager for folder registration
from .utils.model_manager import register_llm_folder_type


class WGExtension(ComfyExtension):
    """ComfyUI WorkflowGenerator extension for ComfyUI V3 API."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """
        Return list of ComfyUI-WorkflowGenerator node classes.

        Returns:
            List of node classes to register
        """
        return [
            UpdateNodeCatalogNode,
            WorkflowGeneratorNode,
            NodeValidatorNode,
            WorkflowBuilderNode,
            WGPipelineNode,
        ]


def comfy_entrypoint():
    """Entry point for ComfyUI to load the extension."""
    # Register LLM folder type when extension loads
    register_llm_folder_type()
    return WGExtension()
