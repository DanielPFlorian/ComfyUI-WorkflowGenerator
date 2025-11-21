# Node Reference

Detailed documentation for each node in ComfyUI-WorkflowGenerator.

## Workflow Generator Pipeline

**Category**: `WorkflowGenerator`

**Purpose**: One-click solution for complete workflow generation. Processes your instruction through all three stages sequentially: generates the workflow diagram, validates and corrects node names, then builds the final ComfyUI workflow JSON.

**Inputs**:
- `instruction` (string, required): Natural language description of the workflow
- `workflow_generator_model_path` (string): Path to WorkflowGenerator model (GGUF or HuggingFace, default: `workflow-generator-q8_0.gguf`)
- `workflow_generator_tokenizer_path` (string, optional): Path to tokenizer (auto-detected if not provided)
- `model_format` (string): "gguf" or "huggingface" (default: "gguf")
- `auto_gpu_layers` (bool): Auto-calculate GPU layers based on VRAM (default: True)
- `n_gpu_layers` (int/string): GPU layers for GGUF ("auto", "all", "none", or 0-50)
- `device_preference` (string): "auto", "cuda", or "cpu"
- `use_llm_refinement` (bool): Enable LLM refinement in NodeValidator (default: False)
- `node_validator_model_path` (string, optional): Path to NodeValidator model (required if use_llm_refinement=True)
- `catalog_directory` (string): Directory containing node_list.json and node_info.json
- `filename_prefix` (string): Prefix for saved workflow files
- `save_workflow_json` (bool): Whether to save workflow to file

**Outputs**:
- `workflow_json` (string): Complete executable ComfyUI workflow JSON
- `instruction` (string): Pass-through of input instruction

**Usage**: All-in-one solution. No other nodes required.

**Best for**: Quick workflow creation without intermediate steps

## WorkflowGenerator Node

**Category**: `WorkflowGenerator`

**Purpose**: Generates workflow diagrams from natural language instructions using a fine-tuned LLM.

**Usage**: **Step 1** in the pipeline. Connects to NodeValidator (optional) or directly to WorkflowBuilder.

**Inputs**:
- `instruction` (string, required): Natural language description of the workflow
- `model_path` (string): Path to WorkflowGenerator model (default: `workflow-generator-q8_0.gguf`)
- `tokenizer_path` (string, optional): Path to tokenizer (auto-detected if not provided)
- `model_format` (string): "gguf" or "huggingface"
- `auto_gpu_layers` (bool): Auto-calculate GPU layers (default: True)
- `n_gpu_layers` (int/string): GPU layers for GGUF
- `device_preference` (string): "auto", "cuda", or "cpu"
- `context_size` (int): Context window size (default: 4096)
- `top_p` (float): Top-p sampling parameter (default: 0.7)
- `max_new_tokens` (int): Maximum tokens to generate (default: 8192)
- `temperature` (float): Sampling temperature (default: 0.95)
- `allow_primitive_nodes` (bool): Whether to include primitive nodes in the diagram (default: False)

**Outputs**:
- `workflow_edges` (string): Workflow diagram in JSON format (list of edges)
- `instruction` (string): Pass-through of input instruction

### Understanding `allow_primitive_nodes`

The `allow_primitive_nodes` parameter controls whether primitive nodes (like integers, strings, floats used as inputs) are included in the generated workflow diagram.

- **False (Default & Recommended)**: Removes primitive nodes from the diagram. This results in a cleaner workflow where values are set directly in the node widgets rather than via external primitive nodes. This is usually preferred for readability.
- **True**: Includes primitive nodes. This can be useful if you prefer to have all inputs explicitly visible as separate nodes, but it can clutter the workflow diagram.

## NodeValidator Node

**Category**: `WorkflowGenerator`

**Purpose**: Validates and corrects invalid node names in workflow diagrams using semantic search and optional LLM refinement.

**Usage**: **Step 2** (Optional). Connects between WorkflowGenerator and WorkflowBuilder. Can be skipped if validation is not needed.

**Inputs**:
- `workflow_edges` (string, required): Workflow diagram from WorkflowGenerator
- `instruction` (string, optional): Context instruction for LLM refinement
- `use_llm_refinement` (bool): Enable LLM refinement (default: False)
- `node_validator_model_path` (string, optional): Path to NodeValidator model (required if use_llm_refinement=True)
- `node_validator_tokenizer_path` (string, optional): Path to tokenizer
- `embedding_model_path` (string): Path to embedding model for semantic search
- `catalog_directory` (string): Directory containing node_list.json and node_info.json
- `top_k` (int): Number of candidate nodes to consider (default: 5)
- `similarity_threshold` (float): Minimum similarity score (default: 0.5)
- `seed` (int, lazy): Random seed (only if LLM enabled)

**Outputs**:
- `workflow_edges (refined)` (string): Corrected workflow diagram with valid node names

**Two modes**:
- **Semantic search only** (faster, deterministic): `use_llm_refinement=False`
- **LLM refinement** (slower, more accurate): `use_llm_refinement=True`

See [Instruction Prompt Usage](Instruction-Prompt-Usage) for details on how the instruction is used in this node.

## WorkflowBuilder Node

**Category**: `WorkflowGenerator`

**Purpose**: Converts workflow diagrams into executable ComfyUI workflow JSON with proper node configurations and connections.

**Usage**: **Step 3** (Final). Connects to NodeValidator or WorkflowGenerator.

**Inputs**:
- `workflow_edges` (string, required): Refined workflow diagram (connection-only input)
- `catalog_directory` (string): Directory containing node_list.json and node_info.json
- `filename_prefix` (string): Prefix for saved workflow files
- `save_workflow_json` (bool): Whether to save workflow to file

**Outputs**:
- `workflow_json` (string): Executable ComfyUI workflow JSON

## UpdateNodeCatalog Node

**Category**: `WorkflowGenerator`

**Purpose**: Scans and catalogs all available ComfyUI nodes (native and custom) for use in validation and workflow building.

**Inputs**:
- `catalog_directory` (string): Directory to save node_list.json and node_info.json files (default: "catalog")
- `force_update` (bool): Force update even if catalog files already exist

**Outputs**:
- `status` (string): Status message ("success" or error)
- `node_count` (int): Number of nodes cataloged

**When to run**: After installing new custom nodes or updating ComfyUI

---

[← Back to Home](Home) | [← Installation](Installation) | [Next: Configuration →](Configuration)

