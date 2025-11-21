"""
Node cataloging system for scanning and cataloging ComfyUI nodes.
Supports both V1 and V3 API nodes.
"""

import json
import logging
from pathlib import Path
from typing import Any

from tinydb import TinyDB

# Try to import ComfyUI nodes
try:
    import nodes

    COMFYUI_NODES_AVAILABLE = True
except ImportError:
    COMFYUI_NODES_AVAILABLE = False
    logging.warning("ComfyUI nodes module not available")

# Try to import V3 API
try:
    from comfy_api.latest import io  # noqa: F401

    V3_API_AVAILABLE = True
except ImportError:
    V3_API_AVAILABLE = False
    logging.warning("ComfyUI V3 API not available")


def extract_v1_node_info(node_name: str, node_class: Any) -> dict | None:
    """
    Extract information from a V1 API node.

    Args:
        node_name: Node class name
        node_class: Node class object

    Returns:
        Dictionary with node information or None if extraction fails
    """
    try:
        input_types = {}
        if hasattr(node_class, "INPUT_TYPES"):
            input_types = node_class.INPUT_TYPES()

        return_types = []
        if hasattr(node_class, "RETURN_TYPES"):
            return_types = node_class.RETURN_TYPES

        return_names = None
        if hasattr(node_class, "RETURN_NAMES"):
            return_names = node_class.RETURN_NAMES

        output_tooltips = None
        if hasattr(node_class, "OUTPUT_TOOLTIPS"):
            output_tooltips = node_class.OUTPUT_TOOLTIPS

        category = None
        if hasattr(node_class, "CATEGORY"):
            category = node_class.CATEGORY

        description = None
        if hasattr(node_class, "DESCRIPTION"):
            description = node_class.DESCRIPTION

        display_name = node_name
        if COMFYUI_NODES_AVAILABLE:
            try:
                if hasattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS") and node_name in nodes.NODE_DISPLAY_NAME_MAPPINGS:
                    display_name = nodes.NODE_DISPLAY_NAME_MAPPINGS[node_name]
            except Exception:
                pass

        is_output_node = False
        if hasattr(node_class, "OUTPUT_NODE") and node_class.OUTPUT_NODE:
            is_output_node = True

        is_deprecated = False
        if getattr(node_class, "DEPRECATED", False):
            is_deprecated = True

        is_experimental = False
        if getattr(node_class, "EXPERIMENTAL", False):
            is_experimental = True

        is_api_node = None
        if hasattr(node_class, "API_NODE"):
            is_api_node = node_class.API_NODE

        node_info = {
            "node_type": node_name,
            "display_name": display_name,
            "inputs": input_types,
            "outputs": return_types,
            "output_names": return_names,
            "output_tooltips": output_tooltips,
            "category": category,
            "description": description,
            "is_output_node": is_output_node,
            "is_deprecated": is_deprecated,
            "is_experimental": is_experimental,
            "is_api_node": is_api_node,
        }

        return node_info

    except Exception as e:
        logging.warning(f"Failed to extract V1 node info for '{node_name}': {e}")
        return None


def extract_v3_node_info(node_name: str, node_class: Any) -> dict | None:
    """
    Extract information from a V3 API node.

    Args:
        node_name: Node class name
        node_class: Node class object (io.ComfyNode)

    Returns:
        Dictionary with node information or None if extraction fails
    """
    try:
        schema = node_class.GET_SCHEMA()

        input_types = {}
        input_metadata = {}
        if hasattr(schema, "inputs") and schema.inputs is not None:
            inputs = schema.inputs
            required = {}
            optional = {}
            hidden = {}

            for input_def in inputs:
                input_name = getattr(input_def, "id", getattr(input_def, "name", None))
                if input_name is None:
                    continue
                input_type = input_def.io_type
                is_optional = getattr(input_def, "optional", False)
                is_required = not is_optional
                is_hidden = getattr(input_def, "hidden", False)

                input_data = [input_type]
                if hasattr(input_def, "default") and input_def.default is not None:
                    input_data.append({"default": input_def.default})

                input_metadata[input_name] = {
                    "display_name": getattr(input_def, "display_name", None),
                    "tooltip": getattr(input_def, "tooltip", None),
                }

                if is_hidden:
                    hidden[input_name] = input_data
                elif is_required:
                    required[input_name] = input_data
                else:
                    optional[input_name] = input_data

            if required:
                input_types["required"] = required
            if optional:
                input_types["optional"] = optional
            if hidden:
                input_types["hidden"] = hidden

        return_types = []
        return_names = []
        output_display_names = []
        output_tooltips = []
        if hasattr(schema, "outputs") and schema.outputs is not None:
            for output_def in schema.outputs:
                return_types.append(output_def.io_type)
                if hasattr(output_def, "name"):
                    return_names.append(output_def.name)
                else:
                    return_names.append(output_def.io_type)

                output_display_names.append(getattr(output_def, "display_name", None))
                output_tooltips.append(getattr(output_def, "tooltip", None))

        category = None
        if hasattr(schema, "category"):
            category = schema.category

        description = None
        if hasattr(schema, "description"):
            description = schema.description

        display_name = getattr(schema, "display_name", None)
        is_output_node = getattr(schema, "is_output_node", False)
        is_deprecated = getattr(schema, "is_deprecated", False)
        is_experimental = getattr(schema, "is_experimental", False)
        is_api_node = getattr(schema, "is_api_node", False)

        node_info = {
            "node_type": node_name,
            "display_name": display_name,
            "inputs": input_types,
            "input_metadata": input_metadata if input_metadata else None,
            "outputs": return_types,
            "output_names": return_names if return_names else None,
            "output_display_names": output_display_names if output_display_names else None,
            "output_tooltips": output_tooltips if output_tooltips else None,
            "category": category,
            "description": description,
            "is_output_node": is_output_node,
            "is_deprecated": is_deprecated,
            "is_experimental": is_experimental,
            "is_api_node": is_api_node,
        }

        return node_info

    except Exception as e:
        logging.warning(f"Failed to extract V3 node info for '{node_name}': {e}")
        return None


def scan_nodes(progress_callback=None) -> tuple[list[str], list[dict]]:
    """
    Scan all available ComfyUI nodes (V1 and V3).

    Args:
        progress_callback: Optional callback function(current, total) to report progress

    Returns:
        Tuple of (node_list, node_info_list)
        - node_list: List of node type names
        - node_info_list: List of node information dictionaries
    """
    if not COMFYUI_NODES_AVAILABLE:
        logging.error("ComfyUI nodes module not available")
        return [], []

    node_list = []
    node_info_list = []

    logging.info("Scanning ComfyUI nodes...")

    try:
        node_mappings = nodes.NODE_CLASS_MAPPINGS
        total_nodes = len(node_mappings)
        logging.info(f"Found {total_nodes} nodes in NODE_CLASS_MAPPINGS")

        processed = 0
        v1_count = 0
        v3_count = 0
        error_count = 0

        # Update progress every N nodes to avoid too many updates
        progress_update_interval = max(1, total_nodes // 100)

        for idx, (node_name, node_class) in enumerate(node_mappings.items(), 1):
            current_position = idx
            try:
                is_v3 = False
                if V3_API_AVAILABLE:
                    try:
                        if hasattr(node_class, "GET_SCHEMA"):
                            schema = node_class.GET_SCHEMA()
                            if hasattr(schema, "node_id"):
                                is_v3 = True
                    except Exception:
                        pass

                if is_v3:
                    node_info = extract_v3_node_info(node_name, node_class)
                    if node_info:
                        v3_count += 1
                    else:
                        node_info = extract_v1_node_info(node_name, node_class)
                        if node_info:
                            v1_count += 1
                else:
                    node_info = extract_v1_node_info(node_name, node_class)
                    if node_info:
                        v1_count += 1

                if node_info:
                    node_list.append(node_name)
                    node_info_list.append(node_info)
                    processed += 1
                else:
                    error_count += 1
                    logging.warning(f"Failed to extract info for node '{node_name}'")

                # Update progress periodically (every N nodes or on last node)
                if progress_callback and (current_position % progress_update_interval == 0 or current_position == total_nodes):
                    progress_callback(current_position, total_nodes)

            except Exception as e:
                error_count += 1
                logging.warning(f"Error processing node '{node_name}': {e}")
                if progress_callback and (current_position % progress_update_interval == 0 or current_position == total_nodes):
                    progress_callback(current_position, total_nodes)
                continue

        logging.info(f"Node scanning complete: {processed} nodes processed ({v1_count} V1, {v3_count} V3, {error_count} errors)")

        if progress_callback:
            progress_callback(total_nodes, total_nodes)

    except Exception as e:
        logging.error(f"Failed to scan nodes: {e}")
        return [], []

    return node_list, node_info_list


def update_node_catalog(node_list_path: str, node_info_db_path: str, force_update: bool = False, progress_callback=None) -> dict[str, Any]:
    """
    Update the node catalog by scanning ComfyUI nodes.

    Args:
        node_list_path: Path to save node list JSON
        node_info_db_path: Path to save node info database JSON
        force_update: If True, overwrite existing files

    Returns:
        Dictionary with update statistics
    """
    logging.info("Updating node catalog...")

    node_list_file = Path(node_list_path)
    node_info_file = Path(node_info_db_path)

    if not force_update and node_list_file.exists() and node_info_file.exists():
        logging.info("Node catalog files already exist, skipping update")
        return {"status": "skipped", "message": "Catalog files already exist"}

    node_list, node_info_list = scan_nodes(progress_callback=progress_callback)

    if not node_list:
        logging.error("No nodes found during scan")
        return {"status": "error", "message": "No nodes found during scan"}

    try:
        node_list_file.parent.mkdir(parents=True, exist_ok=True)
        with open(node_list_file, "w") as f:
            json.dump(node_list, f, indent=2)
        logging.info(f"Saved node list to {node_list_path} ({len(node_list)} nodes)")
    except Exception as e:
        logging.error(f"Failed to save node list: {e}")
        return {"status": "error", "message": f"Failed to save node list: {e}"}

    try:
        node_info_file.parent.mkdir(parents=True, exist_ok=True)

        db = TinyDB(node_info_file)
        db.truncate()
        db.insert_multiple(node_info_list)
        db.close()
        logging.info(f"Saved node info database to {node_info_db_path} ({len(node_info_list)} entries)")
    except Exception as e:
        logging.error(f"Failed to save node info database: {e}")
        return {"status": "error", "message": f"Failed to save node info database: {e}"}

    return {
        "status": "success",
        "node_count": len(node_list),
        "node_info_count": len(node_info_list),
        "node_list_path": str(node_list_path),
        "node_info_db_path": str(node_info_db_path),
    }
