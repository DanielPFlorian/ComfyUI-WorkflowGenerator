"""
Utility functions for processing ComfyUI-WorkflowGenerator workflows and diagrams.
"""

import json
import logging
from typing import Any


def format(node_type: str) -> str:
    """
    Format node type by removing the last underscore and everything after it.
    Also handles node names with parentheses (e.g., "Node Name (author)").

    Args:
        node_type: Node type string (e.g., "CheckpointLoaderSimple_1" or "SDXL Empty Latent Image (rgthree)")

    Returns:
        Formatted node type (e.g., "CheckpointLoaderSimple" or "SDXL Empty Latent Image (rgthree)")

    Note:
        - If node type contains underscore, removes suffix (e.g., "NodeName_0" -> "NodeName")
        - If node type contains parentheses, keeps as-is (e.g., "Node Name (author)" -> "Node Name (author)")
        - If node type has neither, returns as-is
    """
    last_index = node_type.rfind("_")
    if last_index != -1:
        suffix = node_type[last_index + 1 :]
        if suffix.isdigit() or (len(suffix) <= 2 and suffix.isalnum()):
            result = node_type[:last_index]
            return result
    return node_type


def del_digram_primitive(digram: list[list[str]]) -> list[list[str]]:
    """
    Remove primitive nodes from diagram.

    Args:
        digram: List of edges in the diagram, where each edge is [src_type, src_name, dst_type, dst_name]

    Returns:
        Filtered diagram without primitive nodes
    """
    result_digram = []
    for edge in digram:
        src_type, src_name, dst_type, dst_name = edge
        src_node = format(src_type)
        dst_node = format(dst_type)
        if src_node == "PrimitiveNode" or dst_node == "PrimitiveNode":
            continue
        result_digram.append(edge)
    return result_digram


def json_format(encode: str | Any) -> Any:
    """
    Extract and parse JSON from encoded string.

    Args:
        encode: String containing JSON (possibly wrapped in markdown code blocks) or already parsed JSON

    Returns:
        Parsed JSON object

    Raises:
        ValueError: If JSON format is invalid or cannot be parsed
    """
    if not isinstance(encode, str):
        return encode

    if "'''json" in encode:
        start = encode.find("'''json") + len("'''json")
        end = encode.find("'''", start)
        if start == -1 or end == -1:
            raise ValueError("Invalid JSON format: missing closing '''")
        encode = encode[start:end].strip().strip('"')

    if "```json" in encode:
        start = encode.find("```json") + len("```json")
        end = encode.find("```", start)
        if start == -1 or end == -1:
            raise ValueError("Invalid JSON format: missing closing ```")
        encode = encode[start:end].strip().strip('"')

    encode = encode.replace("'", '"')
    encode = encode.replace("\n", "")

    obj_start = encode.find("{")
    obj_end = encode.rfind("}")

    arr_start = encode.find("[")
    arr_end = encode.rfind("]")

    # Determine which format to use (prefer object if both exist)
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        count_open = encode.count("{")
        count_close = encode.count("}")
        if count_open != count_close:
            logging.debug(f"Invalid JSON format: object bracket mismatch. Content: {encode[:200]}")
            raise ValueError("Invalid JSON format: object bracket mismatch")
        encode = encode[obj_start : obj_end + 1]
    elif arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        count_open = encode.count("[")
        count_close = encode.count("]")
        if count_open != count_close:
            logging.debug(f"Invalid JSON format: array bracket mismatch. Content: {encode[:200]}")
            raise ValueError("Invalid JSON format: array bracket mismatch")
        encode = encode[arr_start : arr_end + 1]
    else:
        # Try to parse as-is (might already be valid JSON)
        pass

    try:
        result = json.loads(encode)
        return result
    except json.JSONDecodeError as e:
        logging.debug(f"JSON decode error: {e}")
        logging.debug(f"Content: {encode[:500]}")
        raise


def fetch_node_input_names(meta_info: dict[str, Any]) -> list[str]:
    """
    Extract input names from node meta info.

    Args:
        meta_info: Node metadata dictionary containing 'inputs' key

    Returns:
        List of input names
    """
    input_names = []
    if "inputs" not in meta_info or meta_info["inputs"] is None:
        return input_names

    inputs = meta_info["inputs"]
    for key, item in inputs.items():
        if key != "hidden" and isinstance(item, dict):
            for name in item:
                input_names.append(name)

    return input_names


def fetch_node_output_names(meta_info: dict[str, Any]) -> list[str]:
    """
    Extract output names from node meta info.

    Args:
        meta_info: Node metadata dictionary containing 'output_names' or 'outputs' key

    Returns:
        List of output names
    """
    if "output_names" in meta_info and meta_info["output_names"] is not None:
        return meta_info["output_names"]

    # Fallback: use output types as names
    if "outputs" in meta_info and meta_info["outputs"] is not None:
        return [str(output) for output in meta_info["outputs"]]

    return []
