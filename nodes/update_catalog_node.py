"""
UpdateNodeCatalogNode: Scans ComfyUI installation and updates node catalog files.
Generates node_info.json and node_list.json for use by ComfyUI-WorkflowGenerator.
"""

import json
import logging
import os
from typing import Any

from comfy_api.latest import io

try:
    from comfy.utils import ProgressBar
    from comfy_execution.utils import get_executing_context

    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False
    logging.warning("ComfyUI progress utilities not available")

from ..cataloging.catalog import update_node_catalog
from ..config.config import get_custom_node_path, resolve_config_path


class UpdateNodeCatalogNode(io.ComfyNode):
    """
    UpdateNodeCatalog: Scans ComfyUI installation and updates node catalog files.
    Generates node_info.json and node_list.json for use by ComfyUI-WorkflowGenerator agents.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WG_UpdateNodeCatalog",
            category="WorkflowGenerator",
            display_name="UpdateNodeCatalog",
            description=(
                "Scans ComfyUI for nodes and updates catalog files (node_list.json, node_info.json).\n"
                "Run this after installing new custom nodes to make them available to the workflow validator."
            ),
            inputs=[
                io.String.Input("catalog_directory", default="catalog", tooltip="Directory to save catalog files."),
                io.Boolean.Input("force_update", default=False, tooltip="Force update even if files exist."),
            ],
            outputs=[
                io.String.Output(display_name="status", tooltip="Update status (JSON)."),
                io.Int.Output(display_name="node_count", tooltip="Number of nodes found."),
            ],
            is_output_node=True,  # Mark as output node so it can run standalone
        )

    @classmethod
    def validate_inputs(cls, catalog_directory: str = "", **kwargs) -> bool | str:
        """
        Validate inputs before execution.

        Returns:
            True if valid, or error message string if invalid
        """
        # Paths are validated during execution, no pre-validation needed
        return True

    @classmethod
    def fingerprint_inputs(cls, force_update: bool = False, **kwargs) -> Any:
        """
        Fingerprint inputs to determine if node should re-execute.

        Returns:
            Hashable value that changes when node should re-execute.
            For catalog updates, we want it to run when force_update changes.
        """
        # Return force_update value - node will re-execute when it changes
        # This allows manual triggering of catalog updates
        return force_update

    @classmethod
    def execute(
        cls,
        catalog_directory: str = "catalog",
        force_update: bool = False,
    ) -> io.NodeOutput:
        """Execute node catalog update."""
        try:
            logging.info("WorkflowGenerator UpdateNodeCatalog: Starting node catalog update")

            node_id = None
            if PROGRESS_AVAILABLE:
                try:
                    context = get_executing_context()
                    if context:
                        node_id = context.node_id
                except Exception:
                    pass

            custom_node_path = get_custom_node_path()
            catalog_dir = resolve_config_path(catalog_directory, custom_node_path)

            node_list_path = os.path.join(catalog_dir, "node_list.json")
            node_info_db_path = os.path.join(catalog_dir, "node_info.json")

            progress_bar = None
            if PROGRESS_AVAILABLE and node_id:
                progress_bar = ProgressBar(100, node_id=node_id)

            def progress_callback(current, total):
                if progress_bar:
                    scan_progress = int((current / total) * 80) if total > 0 else 0
                    progress_bar.update_absolute(scan_progress, total=100)
                    logging.debug(f"Progress: {current}/{total} nodes scanned ({scan_progress}%)")

            result = update_node_catalog(
                node_list_path=node_list_path,
                node_info_db_path=node_info_db_path,
                force_update=force_update,
                progress_callback=progress_callback,
            )

            if progress_bar:
                progress_bar.update_absolute(90, total=100)

            if progress_bar:
                progress_bar.update_absolute(100, total=100)

            status_json = json.dumps(result, indent=2)
            node_count = result.get("node_count", 0)

            logging.info(f"WorkflowGenerator UpdateNodeCatalog: Catalog update complete - {node_count} nodes")
            return io.NodeOutput(status_json, node_count)

        except Exception as e:
            logging.error(f"WorkflowGenerator UpdateNodeCatalog error: {e}", exc_info=True)
            error_result = {"status": "error", "message": str(e), "node_count": 0}
            status_json = json.dumps(error_result, indent=2)
            return io.NodeOutput(status_json, 0)
