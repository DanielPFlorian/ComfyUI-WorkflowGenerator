"""
NodeValidator: Validates and corrects node names in workflow diagrams.
Enhanced with GPU layer management, error handling, and ComfyUI integration.
"""

import json
import logging
import time
import traceback
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tinydb import Query, TinyDB

try:
    import nodes as comfy_nodes

    COMFY_NODES_AVAILABLE = True
except ImportError:
    COMFY_NODES_AVAILABLE = False
    logging.warning("ComfyUI nodes module not available, validation will use catalog only")

from ..utils.model_loaders.device_utils import load_model
from ..utils.process_utils import fetch_node_input_names, fetch_node_output_names, format, json_format


class NODE_MISS_EXCEPTION(Exception):
    """Exception raised when a node type is not found in the database."""

    pass


class NodeBase:
    """
    Base class for node information management and semantic search.

    Maintains a database of available ComfyUI nodes and provides
    semantic search capabilities for finding similar nodes.
    """

    def __init__(self, cfg: Any):
        """
        Initialize NodeBase with node database and optionally embedding model.

        Args:
            cfg: Configuration object with node_base settings
        """
        logging.info("Initializing NodeBase...")

        node_list_path = cfg.node_base.node_list_path
        logging.debug(f"Loading node list from: {node_list_path}")
        try:
            with open(node_list_path) as f:
                self.node_list = json.load(f)
            logging.info(f"Loaded {len(self.node_list)} node types")
        except Exception as e:
            logging.error(f"Failed to load node list: {e}")
            raise

        node_info_db_path = cfg.node_base.node_info_db_path
        logging.debug(f"Loading node info database from: {node_info_db_path}")
        try:
            self.node_info_db = TinyDB(node_info_db_path)
            node_info_count = len(self.node_info_db.all())
            logging.info(f"Loaded {node_info_count} node info entries")
        except (TypeError, ValueError) as e:
            # Handle case where file is in old format (plain JSON array instead of TinyDB format)
            logging.warning("Node info database appears to be in old format, attempting to convert...")
            try:
                with open(node_info_db_path, encoding="utf-8") as f:
                    old_data = json.load(f)

                if isinstance(old_data, list):
                    db = TinyDB(node_info_db_path)
                    db.truncate()
                    db.insert_multiple(old_data)
                    db.close()
                    logging.info(f"Converted {len(old_data)} entries from old format to TinyDB format")

                    self.node_info_db = TinyDB(node_info_db_path)
                    node_info_count = len(self.node_info_db.all())
                    logging.info(f"Loaded {node_info_count} node info entries")
                else:
                    raise ValueError(f"Unexpected data format in {node_info_db_path}")
            except Exception as conv_e:
                logging.error(f"Failed to convert node info database: {conv_e}")
                logging.error("Please run 'UpdateNodeCatalog' node to regenerate the catalog files")
                raise ValueError(
                    f"Node info database format is invalid. Please run 'UpdateNodeCatalog' node to regenerate it.\n"
                    f"Original error: {e}\n"
                    f"Conversion error: {conv_e}"
                ) from conv_e
        except Exception as e:
            logging.error(f"Failed to load node info database: {e}")
            raise

        self.cache_node_meta_info = {}

        # Load embedding model (optional - only needed for semantic search)
        embedding_model_path = getattr(cfg.node_base, "embedding_model_path", None)
        if embedding_model_path:
            logging.debug(f"Loading embedding model: {embedding_model_path}")
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logging.debug(f"Using {device} for embedding model")
                self.embedding_model = SentenceTransformer(embedding_model_path, device=device)
                logging.debug("Computing embeddings for semantic search...")
                self.list_embeddings = self.embedding_model.encode(self.node_list, batch_size=32, show_progress_bar=False)
                logging.info(f"Embeddings computed for {len(self.list_embeddings)} nodes")
            except Exception as e:
                logging.error(f"Failed to load embedding model: {e}")
                raise
        else:
            self.embedding_model = None
            self.list_embeddings = None
            logging.info("Embedding model not loaded (semantic search disabled)")

    def fetch_node_meta_info(self, node_type: str) -> dict | None:
        """
        Fetch metadata information for a node type.

        Args:
            node_type: Node type name

        Returns:
            Node metadata dictionary or None if not found

        Raises:
            NODE_MISS_EXCEPTION: If node type is not in the database
        """
        if node_type not in self.node_list:
            logging.warning(f"Node type '{node_type}' not in node list")
            return None

        node_meta_info = self.cache_node_meta_info.get(node_type, None)
        if node_meta_info is not None:
            return node_meta_info

        query = Query()
        node_meta_info_list = self.node_info_db.search(query.node_type == node_type)

        if len(node_meta_info_list) == 0:
            logging.warning(f"Node type '{node_type}' not found in database")
            raise NODE_MISS_EXCEPTION(node_type)

        node_meta_info = node_meta_info_list[0]
        self.cache_node_meta_info[node_type] = node_meta_info
        return node_meta_info

    def find_most_similar(self, node_type: str, k: int = 5) -> list[dict]:
        """
        Find the k most similar nodes to the given node type using semantic search.

        Args:
            node_type: Node type name to search for
            k: Number of similar nodes to return

        Returns:
            List of dictionaries containing node information

        Raises:
            ValueError: If embedding model is not loaded
        """
        if self.embedding_model is None or self.list_embeddings is None:
            raise ValueError("Embedding model not loaded. Semantic search requires embedding_model_path to be provided.")

        query_embedding = self.embedding_model.encode([node_type])
        similarities = cosine_similarity(query_embedding, self.list_embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        similar_node_list = [self.node_list[i] for i in top_k_indices]

        similar_node_infos = []
        for similar_node_type in similar_node_list:
            try:
                similar_node_infos.append(self.get_node_diagram_info(similar_node_type))
            except Exception as e:
                logging.warning(f"Failed to get info for node '{similar_node_type}': {e}")
                continue

        return similar_node_infos

    def not_in_sql(self, node_type: str) -> bool:
        """
        Check if a node type exists in the node list.

        Args:
            node_type: Node type name to check

        Returns:
            True if node exists, False otherwise
        """
        return node_type in self.node_list

    def is_valid_comfy_node(self, node_type: str) -> bool:
        """
        Check if a node type is a valid ComfyUI node class name.

        This checks against ComfyUI's actual NODE_CLASS_MAPPINGS, not just the catalog.
        The catalog may contain display names or aliases that aren't actual node class names.

        Args:
            node_type: Node type name to check

        Returns:
            True if node is a valid ComfyUI node class, False otherwise
        """
        if not COMFY_NODES_AVAILABLE:
            # Fallback to catalog check if ComfyUI nodes aren't available
            return node_type in self.node_list

        # Check against ComfyUI's actual node registry
        return node_type in comfy_nodes.NODE_CLASS_MAPPINGS

    def get_node_diagram_info(self, node_type: str) -> dict:
        """
        Get diagram information for a node type with enhanced metadata.

        Args:
            node_type: Node type name

        Returns:
            Dictionary with node_name, input_names, output_names, and additional metadata
        """
        meta_info = self.fetch_node_meta_info(node_type)
        input_names = fetch_node_input_names(meta_info)
        output_names = fetch_node_output_names(meta_info)

        info = {"node_name": node_type, "input_names": input_names, "output_names": output_names}

        if meta_info:
            if "category" in meta_info:
                info["category"] = meta_info["category"]
            if "display_name" in meta_info:
                info["display_name"] = meta_info["display_name"]
            if "description" in meta_info:
                info["description"] = meta_info["description"]
            if "is_deprecated" in meta_info:
                info["is_deprecated"] = meta_info["is_deprecated"]
            if "is_experimental" in meta_info:
                info["is_experimental"] = meta_info["is_experimental"]
            if "is_output_node" in meta_info:
                info["is_output_node"] = meta_info["is_output_node"]
            if "is_api_node" in meta_info:
                info["is_api_node"] = meta_info["is_api_node"]
            if "input_metadata" in meta_info:
                info["input_metadata"] = meta_info["input_metadata"]
            if "output_display_names" in meta_info:
                info["output_display_names"] = meta_info["output_display_names"]
            if "output_tooltips" in meta_info:
                info["output_tooltips"] = meta_info["output_tooltips"]

        return info


class NodeValidator:
    """
    Validator responsible for validating and correcting node names in workflow diagrams.

    Uses semantic search and optionally a language model to fix invalid node names
    by finding the most similar valid node names.
    """

    def __init__(
        self,
        node_base: NodeBase,
        cfg: Any,
        n_gpu_layers: int | str = "auto",
        dtype: str | torch.dtype = "auto",
        device_preference: str = "auto",
        attn_implementation: str = "auto",
        context_size: int = 4096,
        temperature: float = 0.95,
        top_p: float = 0.7,
        use_mmap: bool = True,
        use_mlock: bool = False,
        n_batch: int = 512,
        n_threads: int | None = None,
    ):
        """
        Initialize NodeValidator.

        Args:
            node_base: NodeBase instance for node information
            cfg: Configuration object with node_validator settings
            n_gpu_layers: For GGUF: "auto", -1 for all, 0 for CPU, or specific count
            dtype: For HuggingFace: "auto", "fp16", "bf16", "fp32", "fp8", or torch.dtype
            device_preference: "auto", "cuda", or "cpu"
        """
        logging.info("WorkflowGenerator NodeValidator: Initializing...")

        self.k = getattr(cfg.node_validator, "k", 5)
        self.node_base = node_base
        self.temperature = temperature
        self.top_p = top_p
        self.local = getattr(cfg.node_validator, "local", False)

        if self.local:
            self.max_new_tokens = getattr(cfg.node_validator, "max_new_tokens", 4096)
            model_format = getattr(cfg.node_validator, "model_format", "huggingface")
            model_path = cfg.node_validator.model_path
            tokenizer_path = getattr(cfg.node_validator, "tokenizer_path", None)

            logging.info(f"WorkflowGenerator NodeValidator: Loading local model: {model_format}, path: {model_path}")

            load_kwargs = {}

            if model_format.lower() == "huggingface":
                load_kwargs = {
                    "dtype": dtype,
                    "device_preference": device_preference,
                    "attn_implementation": attn_implementation,
                }
            else:
                # GGUF models use n_gpu_layers, context_size, and performance parameters
                load_kwargs = {
                    "n_gpu_layers": n_gpu_layers,
                    "device_preference": device_preference,
                    "context_size": context_size,
                    "use_mmap": use_mmap,
                    "use_mlock": use_mlock,
                    "n_batch": n_batch,
                    "n_threads": n_threads,
                }

            load_start = time.time()
            try:
                self.model_wrapper = load_model(
                    model_format=model_format, model_path=model_path, tokenizer_path=tokenizer_path, **load_kwargs
                )
                load_time = time.time() - load_start
                logging.info(f"WorkflowGenerator NodeValidator: Model loaded in {load_time:.2f}s")
            except Exception as e:
                logging.error(f"WorkflowGenerator NodeValidator: Failed to load model: {e}")
                raise

            self.model = self.model_wrapper.model
            self.tokenizer = self.model_wrapper.tokenizer

            try:
                if torch.cuda.is_available() and model_format.lower() == "huggingface":
                    vram_used = torch.cuda.memory_allocated() / 1024**3
                    logging.debug(f"GPU memory allocated: {vram_used:.2f} GB")
            except Exception:
                pass
        else:
            logging.info("WorkflowGenerator NodeValidator: Using non-local mode (semantic search only)")

    def refine_diagram(self, diagram: list, prompt: str, seed: int | None = None) -> tuple[list, dict]:
        """
        Refine a workflow diagram by validating and correcting node names.

        Args:
            diagram: List of edges representing the workflow diagram
            prompt: Original user prompt/description
            seed: Random seed for reproducible refinement (only used when local=True)

        Returns:
            Tuple of (refined_diagram, debug_info) where:
            - refined_diagram: Refined diagram with corrected node names
            - debug_info: Dictionary containing:
                - llm_prompts: List of prompts sent to LLM (one per invalid node)
                - candidate_nodes: Dictionary mapping invalid node names to their candidate nodes
        """
        if seed is not None:
            max_seed = 2**32 - 1
            if seed < 0 or seed > max_seed:
                seed = seed % (max_seed + 1)  # Wrap around if out of range

        logging.info("Refining workflow diagram...")

        invalid_nodes = []
        swap_map = {}
        debug_info = {"llm_prompts": [], "candidate_nodes": {}}

        logging.debug("Scanning diagram for invalid node names...")

        for edge in diagram:
            src_type, src_name, dst_type, dst_name = edge
            src_node = format(src_type)
            dst_node = format(dst_type)

            is_in_catalog = self.node_base.not_in_sql(src_node)
            is_valid_comfy = self.node_base.is_valid_comfy_node(src_node)

            if (not is_in_catalog or not is_valid_comfy) and src_node not in swap_map:
                invalid_nodes.append(src_node)
                if not is_in_catalog:
                    logging.debug(f"Source node '{src_node}' not found in node database")
                elif not is_valid_comfy:
                    logging.debug(f"Source node '{src_node}' is not a valid ComfyUI node class (may be display name/alias)")

                try:
                    candidate_nodes = self.node_base.find_most_similar(src_node, self.k)
                    debug_info["candidate_nodes"][src_node] = candidate_nodes
                except Exception as e:
                    logging.warning(f"Failed to get candidate nodes for '{src_node}': {e}")
                    debug_info["candidate_nodes"][src_node] = []

                if self.local:
                    swap_name, llm_prompt = self.fix_node_name_local(prompt, diagram, src_node, seed=seed, return_prompt=True)
                    if llm_prompt:
                        debug_info["llm_prompts"].append({"invalid_node": src_node, "prompt": llm_prompt})
                else:
                    swap_name = self.fix_node_name(prompt, diagram, src_node)
                if swap_name is not None:
                    swap_map[src_node] = swap_name
                    logging.info(f"Fixed node name: '{src_node}' -> '{swap_name}'")

            is_in_catalog = self.node_base.not_in_sql(dst_node)
            is_valid_comfy = self.node_base.is_valid_comfy_node(dst_node)

            if (not is_in_catalog or not is_valid_comfy) and dst_node not in swap_map:
                invalid_nodes.append(dst_node)
                if not is_in_catalog:
                    logging.debug(f"Destination node '{dst_node}' not found in node database")
                elif not is_valid_comfy:
                    logging.debug(f"Destination node '{dst_node}' is not a valid ComfyUI node class (may be display name/alias)")

                try:
                    candidate_nodes = self.node_base.find_most_similar(dst_node, self.k)
                    debug_info["candidate_nodes"][dst_node] = candidate_nodes
                except Exception as e:
                    logging.warning(f"Failed to get candidate nodes for '{dst_node}': {e}")
                    debug_info["candidate_nodes"][dst_node] = []

                if self.local:
                    swap_name, llm_prompt = self.fix_node_name_local(prompt, diagram, dst_node, seed=seed, return_prompt=True)
                    if llm_prompt:
                        debug_info["llm_prompts"].append({"invalid_node": dst_node, "prompt": llm_prompt})
                else:
                    swap_name = self.fix_node_name(prompt, diagram, dst_node)
                if swap_name is not None:
                    swap_map[dst_node] = swap_name
                    logging.info(f"Fixed node name: '{dst_node}' -> '{swap_name}'")

        logging.debug("Validating connections (input/output names)...")
        connection_issues = []

        for i, edge in enumerate(diagram):
            src_type, src_name, dst_type, dst_name = edge
            src_node = format(src_type)
            dst_node = format(dst_type)

            if src_node in swap_map:
                src_node = swap_map[src_node]
                edge[0] = src_node
            if dst_node in swap_map:
                dst_node = swap_map[dst_node]
                edge[2] = dst_node

            try:
                src_meta = self.node_base.fetch_node_meta_info(src_node)
                if src_meta:
                    src_output_names = fetch_node_output_names(src_meta)
                    if src_name not in src_output_names:
                        connection_issues.append(
                            {"edge_index": i, "node": src_node, "type": "output", "name": src_name, "available": src_output_names}
                        )
                        logging.debug(f"Source node '{src_node}' has no output '{src_name}'. Available: {src_output_names[:5]}...")
            except Exception as e:
                logging.warning(f"Failed to validate source node '{src_node}' output: {e}")

            try:
                dst_meta = self.node_base.fetch_node_meta_info(dst_node)
                if dst_meta:
                    dst_input_names = fetch_node_input_names(dst_meta)
                    if dst_name not in dst_input_names:
                        connection_issues.append(
                            {"edge_index": i, "node": dst_node, "type": "input", "name": dst_name, "available": dst_input_names}
                        )
                        logging.debug(f"Destination node '{dst_node}' has no input '{dst_name}'. Available: {dst_input_names[:5]}...")

                        if dst_node not in swap_map:
                            try:
                                candidate_nodes = self.node_base.find_most_similar(dst_node, self.k)
                                debug_info["candidate_nodes"][dst_node] = candidate_nodes

                                for candidate in candidate_nodes:
                                    candidate_meta = self.node_base.fetch_node_meta_info(candidate["node_name"])
                                    if candidate_meta:
                                        candidate_inputs = fetch_node_input_names(candidate_meta)
                                        if dst_name in candidate_inputs:
                                            logging.debug(f"Candidate node '{candidate['node_name']}' has input '{dst_name}'")
                                            if self.local:
                                                connection_issue_info = {"type": "input", "name": dst_name, "available": dst_input_names}
                                                swap_name, llm_prompt = self.fix_node_name_local(
                                                    prompt,
                                                    diagram,
                                                    dst_node,
                                                    seed=seed,
                                                    return_prompt=True,
                                                    preferred_candidate=candidate["node_name"],
                                                    connection_issue=connection_issue_info,
                                                )
                                                if llm_prompt:
                                                    debug_info["llm_prompts"].append(
                                                        {
                                                            "invalid_node": dst_node,
                                                            "invalid_connection": f"input '{dst_name}'",
                                                            "prompt": llm_prompt,
                                                        }
                                                    )
                                                if swap_name and swap_name != dst_node:
                                                    swap_map[dst_node] = swap_name
                                                    logging.info(
                                                        f"Fixed node name: '{dst_node}' -> '{swap_name}' (has required input '{dst_name}')"
                                                    )
                                                    break
                                            else:
                                                swap_map[dst_node] = candidate["node_name"]
                                                logging.info(
                                                    f"Fixed node name: '{dst_node}' -> '{candidate['node_name']}' (has required input '{dst_name}')"
                                                )
                                                break
                            except Exception as e:
                                logging.warning(f"Failed to find candidate nodes for '{dst_node}': {e}")
            except Exception as e:
                logging.warning(f"Failed to validate destination node '{dst_node}' input: {e}")

        if len(swap_map) == 0 and len(connection_issues) == 0:
            logging.info("Diagram refinement complete: no corrections needed")
        else:
            if len(swap_map) > 0:
                logging.debug(f"Applying {len(swap_map)} node name corrections...")
                for i, edge in enumerate(diagram):
                    src_type, src_name, dst_type, dst_name = edge
                    src_node = format(src_type)
                    dst_node = format(dst_type)

                    if src_node in swap_map:
                        swap_name = swap_map[src_node]
                        edge[0] = edge[0].replace(src_node, swap_name)

                    if dst_node in swap_map:
                        swap_name = swap_map[dst_node]
                        edge[2] = edge[2].replace(dst_node, swap_name)

                    diagram[i] = edge

            if len(connection_issues) > 0:
                logging.warning(f"Found {len(connection_issues)} connection issues")

            logging.info(f"Diagram refinement complete: {len(swap_map)} node corrections, {len(connection_issues)} connection issues")

        return diagram, debug_info

    def fix_node_name(self, desc: str, diagram: list, error_name: str) -> str | None:
        """
        Fix node name using semantic search only (non-local mode).

        Args:
            desc: User description
            diagram: Workflow diagram
            error_name: Invalid node name to fix

        Returns:
            Corrected node name or None if no suitable match found
        """
        logging.info(f"Fixing node name '{error_name}' using semantic search")

        try:
            candidate_nodes = self.node_base.find_most_similar(error_name, self.k)
            if candidate_nodes:
                return candidate_nodes[0]["node_name"]
        except Exception as e:
            logging.warning(f"Failed to find similar node for '{error_name}': {e}")

        return None

    def fix_node_name_local(
        self,
        desc: str,
        diagram: list,
        error_name: str,
        seed: int | None = None,
        return_prompt: bool = False,
        preferred_candidate: str | None = None,
        connection_issue: dict | None = None,
    ) -> str | None | tuple[str | None, str | None]:
        """
        Fix node name using local language model.

        Args:
            desc: User description
            diagram: Workflow diagram
            error_name: Invalid node name to fix
            seed: Random seed for reproducible generation (optional)
            return_prompt: If True, return tuple of (result, full_prompt_text), else just result
            preferred_candidate: Specific candidate to prioritize (e.g., one that has required input/output)
            connection_issue: Dictionary describing the connection problem, e.g.:
                {"type": "input", "name": "text_list", "available": ["text_a", "text_b", ...]}

        Returns:
            If return_prompt=False: Corrected node name or None if no suitable match found
            If return_prompt=True: Tuple of (corrected_node_name, full_prompt_text) where full_prompt_text is the complete prompt sent to LLM
        """
        if not self.local:
            result = self.fix_node_name(desc, diagram, error_name)
            if return_prompt:
                return result, None
            return result

        logging.info(f"Fixing node name '{error_name}' using local model")

        try:
            candidate_nodes = self.node_base.find_most_similar(error_name, self.k)
            candidate_node_names = [e["node_name"] for e in candidate_nodes]

            if preferred_candidate and preferred_candidate in candidate_node_names:
                preferred_nodes = []
                other_nodes = []
                for c in candidate_nodes:
                    if c["node_name"] == preferred_candidate:
                        c_copy = c.copy()
                        c_copy["_recommended"] = True
                        if connection_issue:
                            c_copy["_reason"] = f"Has required {connection_issue['type']}: '{connection_issue['name']}'"
                        preferred_nodes.append(c_copy)
                    else:
                        other_nodes.append(c)
                candidate_nodes = preferred_nodes + other_nodes
                candidate_node_names = [e["node_name"] for e in candidate_nodes]

            if connection_issue:
                issue_type = connection_issue.get("type", "input")
                missing_name = connection_issue.get("name", "unknown")
                available_names = connection_issue.get("available", [])
                available_str = ", ".join([f"'{n}'" for n in available_names[:10]])
                if len(available_names) > 10:
                    available_str += ", ..."

                prompt = (
                    f"I would like you to act as an expert in ComfyUI platform. "
                    f"I will provide a workflow description and a logical diagram in JSON format representing the ComfyUI workflow. "
                    f"The logical diagram is a list of edges [edge_1, edge_2, ...], where each edge is "
                    f"[source_node_name, output_name, destination_node_name, input_name], representing a connection from source to destination.\n\n"
                    f"**Workflow Description:** {desc if desc else 'Not provided'}\n\n"
                    f"**Logical Diagram:** {str(diagram)}\n\n"
                    f"**PROBLEM DETECTED:**\n"
                    f"The workflow uses node '{error_name}', but there is a CONNECTION ERROR:\n"
                    f"- The workflow tries to connect to {issue_type} '{missing_name}'\n"
                    f"- However, node '{error_name}' does NOT have this {issue_type}\n"
                    f"- Available {issue_type}s on '{error_name}': {available_str}\n\n"
                    f"**YOUR TASK:**\n"
                    f"Select a replacement node from the candidates below that:\n"
                    f"1. Has the required {issue_type} '{missing_name}' (CRITICAL)\n"
                    f"2. Is semantically similar to '{error_name}'\n"
                    f"3. Fits the workflow's purpose\n\n"
                    f"**Candidate Nodes:** {str(candidate_nodes)}\n\n"
                    f"**IMPORTANT NOTES:**\n"
                    f"- Look for candidates with '_recommended': True - these have the required {issue_type}\n"
                    f"- Check each candidate's '{issue_type}_names' list carefully\n"
                    f"- The replacement MUST have '{missing_name}' in its {issue_type}s\n\n"
                    f"**Response Format:**\n"
                    f"Return ONLY the chosen node name in pure JSON format:\n"
                    f"'''json{{\"candidate_node_name\":\"...\"}}'''"
                )
            else:
                prompt = (
                    f"I would like you to act as an expert in ComfyUI platform. "
                    f"I will provide a workflow description and a logical diagram in JSON format representing the ComfyUI workflow. "
                    f"The logical diagram is a list of edges [edge_1, edge_2, ...], where each edge is "
                    f"[source_node_name, output_name, destination_node_name, input_name], representing a connection from source to destination.\n\n"
                    f"**Workflow Description:** {desc if desc else 'Not provided'}\n\n"
                    f"**Logical Diagram:** {str(diagram)}\n\n"
                    f"**PROBLEM DETECTED:**\n"
                    f"The node name '{error_name}' is not valid in ComfyUI.\n\n"
                    f"**YOUR TASK:**\n"
                    f"Select the most suitable replacement node from the candidates below that:\n"
                    f"1. Is semantically similar to '{error_name}'\n"
                    f"2. Fits the workflow's purpose based on the description and diagram\n"
                    f"3. Has compatible inputs/outputs for the connections in the diagram\n\n"
                    f"**Candidate Nodes:** {str(candidate_nodes)}\n\n"
                    f"**Response Format:**\n"
                    f"Return ONLY the chosen node name in pure JSON format:\n"
                    f"'''json{{\"candidate_node_name\":\"...\"}}'''"
                )

            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            # Use HuggingFace tokenizer even for GGUF models (embedded tokenizer doesn't expose apply_chat_template)
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            full_prompt_text = text if return_prompt else None

            model_inputs = self.tokenizer([text], return_tensors="pt")
            if hasattr(self.model_wrapper, "device") and isinstance(self.model_wrapper.device, torch.device):
                model_inputs = model_inputs.to(self.model_wrapper.device)

            generate_kwargs = {"max_new_tokens": self.max_new_tokens, "temperature": self.temperature, "top_p": self.top_p}
            if seed is not None:
                max_seed = 2**32 - 1
                if seed < 0 or seed > max_seed:
                    seed = seed % (max_seed + 1)  # Wrap around if out of range
                generate_kwargs["seed"] = seed

            generated_ids = self.model_wrapper.generate(model_inputs.input_ids, **generate_kwargs)

            generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

            result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            try:
                result = json_format(result)
                if isinstance(result, dict):
                    result = result.get("candidate_node_name", None)
            except Exception as e:
                logging.warning(f"Failed to parse JSON response: {e}")
                traceback.print_exc()
                if return_prompt:
                    return None, full_prompt_text
                return None

            if result not in candidate_node_names:
                logging.warning(f"Model returned '{result}' which is not in candidate list")
                if return_prompt:
                    return None, full_prompt_text
                return None

            if return_prompt:
                return result, full_prompt_text
            return result

        except Exception as e:
            logging.error(f"Failed to fix node name using local model: {e}")
            traceback.print_exc()
            if return_prompt:
                return None, None
            return None

    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, "model_wrapper"):
            del self.model_wrapper
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info("WorkflowGenerator NodeValidator: Resources cleaned up")
