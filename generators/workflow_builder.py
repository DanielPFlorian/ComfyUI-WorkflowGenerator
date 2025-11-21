"""
WorkflowBuilder: Converts workflow diagrams to ComfyUI workflow JSON.
Enhanced with error handling and ComfyUI integration.
"""

import logging
from typing import Any

from ..utils.process_utils import format

NO_LINK_TYPES = ["STRING", "FLOAT", "INT", "BOOLEAN"]


class NAME_EXCEPTION(Exception):
    """Exception raised when a node name is invalid."""

    pass


class REQUIR_MISS_EXCEPTION(Exception):
    """Exception raised when required inputs are missing."""

    pass


class WorkflowBuilder:
    """
    Builder responsible for converting workflow diagrams to ComfyUI workflow JSON.

    Takes a diagram (list of edges) and converts it into a complete ComfyUI
    workflow JSON structure with nodes, links, and metadata.
    """

    def __init__(self, node_base: Any):
        """
        Initialize WorkflowBuilder.

        Args:
            node_base: NodeBase instance for node information
        """
        logging.info("WorkflowGenerator WorkflowBuilder: Initializing...")
        self.node_base = node_base

    def parse_diagram_to_workflow(self, diagram: list, version: float = 0.4, type_to_pos: dict | None = None) -> dict:
        """
        Parse a workflow diagram into ComfyUI workflow JSON.

        Args:
            diagram: List of edges representing the workflow diagram
            version: Workflow version (default: 0.4)
            type_to_pos: Optional dictionary mapping node types to positions

        Returns:
            Complete ComfyUI workflow dictionary
        """
        logging.info("Parsing diagram to workflow...")
        logging.debug("Processing diagram edges...")

        count = 1
        type_to_id = {}
        links = []
        link_count = 1
        inputs_dict = {}
        outputs_dict = {}
        catch_candidate_inputs = {}
        processed_edges = 0

        for i, (src_node, src_name, dst_node, dst_name) in enumerate(diagram):
            processed_edges += 1

            if src_node not in type_to_id:
                type_to_id[src_node] = count
                count += 1
            if dst_node not in type_to_id:
                type_to_id[dst_node] = count
                count += 1

            src_id = type_to_id[src_node]
            dst_id = type_to_id[dst_node]
            src_type = format(src_node)
            dst_type = format(dst_node)

            if src_id not in outputs_dict:
                try:
                    outputs_dict[src_id] = self.fetch_node_outputs(src_type)
                except Exception as e:
                    logging.error(f"Failed to fetch outputs for node '{src_type}': {e}")
                    raise

            if dst_id not in inputs_dict:
                try:
                    inputs_dict[dst_id], catch_candidate_inputs[dst_type] = self.fetch_node_inputs(dst_type)
                except Exception as e:
                    logging.error(f"Failed to fetch inputs for node '{dst_type}': {e}")
                    raise

            outputs = outputs_dict[src_id]
            inputs = inputs_dict[dst_id]
            candidate_inputs = catch_candidate_inputs[dst_type]

            try:
                outputs, inputs, link, link_count = decode_edge(
                    outputs,
                    inputs,
                    candidate_inputs,
                    link_count,
                    src_id,
                    dst_id,
                    src_name,
                    dst_name,
                    dst_type,
                )

                # Only add link if decode_edge returned one (None means edge was skipped gracefully)
                if link is not None:
                    links.append(link)
                    link_count += 1
                else:
                    # Edge was skipped - don't increment link_count to avoid gaps
                    logging.debug(f"Edge {i} skipped (no valid connection found)")

            except NAME_EXCEPTION as e:
                logging.warning(f"Failed to decode edge {i}: {e}. Skipping edge.")
                logging.debug(f"  Source node: {src_type}, Destination node: {dst_type}")
                logging.debug(f"  Output name: {src_name}, Input name: {dst_name}")
                continue
            except Exception as e:
                logging.error(f"Failed to decode edge {i}: {e}")
                logging.error(f"  Source node: {src_type}, Destination node: {dst_type}")
                logging.error(f"  Output name: {src_name}, Input name: {dst_name}")
                raise

            outputs_dict[src_id] = outputs
            inputs_dict[dst_id] = inputs

        logging.debug(f"Processed {processed_edges} edges, created {len(type_to_id)} unique nodes")
        logging.debug("Building workflow structure...")

        for id, inputs in inputs_dict.items():
            tmp_input = []
            for input in inputs:
                if len(input) > 0:
                    tmp_input.append(input)
            inputs_dict[id] = tmp_input

        nodes = []
        for node_type, node_id in type_to_id.items():
            inputs = []
            if node_id in inputs_dict:
                inputs = inputs_dict[node_id]

            outputs = []
            if node_id in outputs_dict:
                outputs = outputs_dict[node_id]

            node = {
                "id": node_id,
                "type": format(node_type),
                "inputs": inputs,
                "outputs": outputs,
            }

            if type_to_pos is not None and node_type in type_to_pos:
                node["pos"] = type_to_pos[node_type]["pos"]
                node["size"] = type_to_pos[node_type]["size"]

            nodes.append(node)

        if type_to_pos is None:
            try:
                from ..utils.workflow_layout import WorkflowLayout

                logging.info("WorkflowGenerator WorkflowBuilder: Applying auto-layout to generated workflow")
                WorkflowLayout.process(nodes, links)
            except Exception as e:
                logging.warning(f"WorkflowGenerator WorkflowBuilder: Failed to apply auto-layout: {e}")

        decode_payload = {
            "last_link_id": link_count - 1,
            "nodes": nodes,
            "links": links,
            "version": version,
        }

        decode_payload["extra"] = {
            "extra": {
                "ds": {
                    "scale": 0.952486143572623,
                    "offset": [163.72358533046707, 22.018287636666063],
                }
            }
        }

        logging.info(f"Workflow structure created: {len(nodes)} nodes, {len(links)} links")

        return decode_payload

    def fetch_node_inputs(self, node_type: str) -> tuple[list, list]:
        """
        Fetch input information for a node type.

        Args:
            node_type: Node type name

        Returns:
            Tuple of (inputs list, candidate_inputs list)
        """
        try:
            meta_info = self.node_base.fetch_node_meta_info(node_type)
        except Exception as e:
            logging.error(f"Failed to fetch meta info for '{node_type}': {e}")
            raise

        if meta_info is None:
            logging.warning(f"Node type '{node_type}' not found in catalog, returning empty inputs")
            return [], []

        input_types = meta_info.get("inputs", {})
        if input_types is None:
            input_types = {}

        inputs = []
        else_inputs = []

        for key, item in input_types.items():
            if key != "hidden":
                for name, val in item.items():
                    val_type = val[0]

                    # Skip list types (COMBO)
                    if isinstance(val_type, list):
                        continue

                    # Non-linkable types (primitives)
                    if val_type not in NO_LINK_TYPES:
                        input = {"name": name, "type": val_type}
                        inputs.append(input)
                    elif len(val) > 1:
                        # Optional inputs with default values
                        val_info = val[1]
                        if "defaultInput" in val_info:
                            input = {"name": name, "type": val_type}
                            else_inputs.append(input)

        inputs.extend(else_inputs)
        candidate_inputs = get_node_candidate_inputs(input_types, inputs)

        return inputs, candidate_inputs

    def fetch_node_outputs(self, node_type: str) -> list[dict]:
        """
        Fetch output information for a node type.

        Args:
            node_type: Node type name

        Returns:
            List of output dictionaries
        """
        try:
            node_meta_info = self.node_base.fetch_node_meta_info(node_type)
        except Exception as e:
            logging.error(f"Failed to fetch meta info for '{node_type}': {e}")
            raise

        if node_meta_info is None:
            logging.warning(f"Node type '{node_type}' not found in catalog, returning empty outputs")
            return []

        output_types = node_meta_info.get("outputs", [])
        if output_types is None:
            output_types = []

        output_names = node_meta_info.get("output_names", None)
        outputs = []

        for i, output_type in enumerate(output_types):
            output = {"type": output_type, "slot_index": i, "links": []}

            if output_names is not None and i < len(output_names):
                output["name"] = output_names[i]
            else:
                output["name"] = output_type

            outputs.append(output)

        return outputs


def get_output_info(outputs: list[dict]) -> dict[str, int]:
    """
    Extract output name to index mapping from outputs list.

    Args:
        outputs: List of output dictionaries

    Returns:
        Dictionary mapping output names to indices
    """
    has_name = False
    error_tag = False
    type_dict = []
    output_names = {}

    for index, val in enumerate(outputs):
        if "name" in val:
            has_name = True
        if val["type"] in type_dict:
            error_tag = True
        type_dict.append(val["type"])
        output_names[val["name"]] = index

    if not has_name and error_tag:
        raise ValueError("Outputs format error: duplicate types without names")

    return output_names


def get_node_candidate_inputs(input_types: dict, inputs: list[dict]) -> list[dict]:
    """
    Get candidate inputs that can be added dynamically.

    Args:
        input_types: Dictionary of input type definitions
        inputs: List of existing inputs

    Returns:
        List of candidate input dictionaries
    """
    candidate_inputs = []
    inputs_names = [input["name"] for input in inputs]

    for key, item in input_types.items():
        if key != "hidden":
            for name, val in item.items():
                val_type = val[0]

                if name in inputs_names:
                    continue

                if val_type not in NO_LINK_TYPES and not isinstance(val, list):
                    continue

                input = {"name": name, "type": val_type}
                candidate_inputs.append(input)

    return candidate_inputs


def decode_edge(
    outputs: list[dict],
    inputs: list[dict],
    candidate_inputs: list[dict],
    link_count: int,
    src_id: int,
    dst_id: int,
    output_name: str,
    input_name: str,
    dst_node_type: str = None,
) -> tuple[list[dict], list[dict], list, int]:
    """
    Decode an edge and create a link between source and destination nodes.

    Args:
        outputs: List of output dictionaries from source node
        inputs: List of input dictionaries from destination node
        candidate_inputs: List of candidate inputs that can be added
        link_count: Current link count
        src_id: Source node ID
        dst_id: Destination node ID
        output_name: Name of the output port
        input_name: Name of the input port

    Returns:
        Tuple of (updated outputs, updated inputs, link, link_count)
    """
    output_names = get_output_info(outputs)

    if output_name not in output_names:
        raise NAME_EXCEPTION(f"Output name '{output_name}' not found")

    src_port = output_names[output_name]
    output = outputs[src_port]

    links = output.get("links", [])
    links.append(link_count)
    output["links"] = links
    outputs[src_port] = output

    input = None
    dst_port = None

    for i, _input in enumerate(inputs):
        if _input.get("name") == input_name:
            input = _input
            input["link"] = link_count
            dst_port = i
            break

    # If not found, try candidate inputs
    if input is None:
        for _input in candidate_inputs:
            if _input.get("name") == input_name:
                input = {
                    "name": input_name,
                    "type": _input["type"],
                    "link": link_count,
                    "widget": {"name": input_name},
                }
                dst_port = len(inputs)
                inputs.append(input)
                break

    # If still not found, try fuzzy matching on input names
    if input is None:
        all_input_names = [inp.get("name") for inp in inputs + candidate_inputs if inp.get("name")]

        best_match = None
        best_score = 0

        input_name_normalized = input_name.lower()
        for suffix in ["_list", "_array", "s", "_items", "_values"]:
            if input_name_normalized.endswith(suffix):
                input_name_base = input_name_normalized[: -len(suffix)]
                break
        else:
            input_name_base = input_name_normalized

        for available_name in all_input_names:
            available_normalized = available_name.lower()

            if available_normalized == input_name_normalized:
                best_match = available_name
                best_score = 1.0
                break

            if available_normalized == input_name_base or input_name_base == available_normalized:
                best_match = available_name
                best_score = 0.9
                continue

            if input_name_normalized in available_normalized or available_normalized in input_name_normalized:
                length_score = min(len(input_name_normalized), len(available_normalized)) / max(
                    len(input_name_normalized), len(available_normalized)
                )
                if available_normalized.startswith(input_name_base) or input_name_base.startswith(available_normalized):
                    length_score = max(length_score, 0.7)
                if length_score > best_score:
                    best_match = available_name
                    best_score = length_score

        if best_match and best_score > 0.5:
            logging.warning(
                f"Input name '{input_name}' not found for node, using similar name '{best_match}' (similarity: {best_score:.2f})"
            )
            for _input in inputs + candidate_inputs:
                if _input.get("name") == best_match:
                    input = {
                        "name": best_match,
                        "type": _input["type"],
                        "link": link_count,
                    }
                    if best_match not in [inp.get("name") for inp in inputs]:
                        input["widget"] = {"name": best_match}
                        dst_port = len(inputs)
                        inputs.append(input)
                    else:
                        for i, inp in enumerate(inputs):
                            if inp.get("name") == best_match:
                                inp["link"] = link_count
                                dst_port = i
                                input = inp
                                break
                    break
        else:
            type_matched_inputs = [inp for inp in inputs + candidate_inputs if inp.get("type") == output.get("type") and inp.get("name")]

            if type_matched_inputs:
                best_type_match = None
                for matched_input in type_matched_inputs:
                    matched_name = matched_input.get("name", "").lower()
                    if input_name_base in matched_name or matched_name.startswith(input_name_base):
                        best_type_match = matched_input
                        break

                if best_type_match is None:
                    best_type_match = type_matched_inputs[0]

                logging.warning(
                    f"Input name '{input_name}' not found for node '{dst_node_type}', "
                    f"using type-compatible input '{best_type_match.get('name')}' "
                    f"(type: {output.get('type')})"
                )

                matched_name = best_type_match.get("name")
                if matched_name not in [inp.get("name") for inp in inputs]:
                    input = {
                        "name": matched_name,
                        "type": best_type_match.get("type"),
                        "link": link_count,
                        "widget": {"name": matched_name},
                    }
                    dst_port = len(inputs)
                    inputs.append(input)
                else:
                    for i, inp in enumerate(inputs):
                        if inp.get("name") == matched_name:
                            inp["link"] = link_count
                            dst_port = i
                            input = inp
                            break
            else:
                suggestions = ", ".join(all_input_names[:5])
                node_info = f" for node '{dst_node_type}'" if dst_node_type else ""
                logging.warning(
                    f"Cannot connect edge: Input name '{input_name}' not found{node_info}. "
                    f"Output type: {output.get('type')}. "
                    f"Available inputs: {suggestions}" + ("..." if len(all_input_names) > 5 else "") + ". "
                    "Skipping this edge."
                )
                return outputs, inputs, None, link_count

    if input["type"] != output["type"]:
        logging.warning(f"Type mismatch for edge: output type '{output['type']}' != input type '{input['type']}'. Skipping this edge.")
        return outputs, inputs, None, link_count

    link = [link_count, src_id, src_port, dst_id, dst_port, input["type"]]

    return outputs, inputs, link, link_count
