# ComfyUI Workflow Generator

Generate ComfyUI workflows from natural language descriptions using Large Language Models (LLMs).

This custom node enables users to describe a workflow in natural language (e.g., *"Create a text-to-image workflow using SDXL"*) and automatically constructs the corresponding node graph. It leverages Large Language Models (LLMs) to bridge the gap between intent and execution.

## Implementation Note

This project is an independent implementation of the [ComfyGPT research](https://arxiv.org/abs/2503.17671), designed to bring that architecture directly into ComfyUI as a native node suite.

My goal was to preserve the core functionality of the original research—specifically the multi-stage pipeline of generation, validation, and construction, while optimizing it for the local ComfyUI environment. This implementation focuses on modularity, allowing users to inspect and intervene at each stage of the generation process.

**Current Limitations:**
The system's knowledge is bounded by its training data. While it excels at standard patterns and known nodes, it cannot inherently "know" about custom nodes released after its training cutoff without additional context. It acts as a powerful accelerator for workflow creation, but supervision is recommended.

## Based on ComfyGPT Research

**Title:** "ComfyGPT: A Self-Optimizing Multi-Agent System for Comprehensive ComfyUI Workflow Generation"
- **Original Repository:** https://github.com/comfygpt/comfygpt
- **Project Website:** https://comfygpt.github.io/

### How It Works

This implementation uses a specialized LLM fine-tuned on workflow data to execute a three-stage pipeline:
1.  **Generator:** Interprets the natural language prompt to generate a logical graph structure (JSON).
2.  **Validator:** Verifies node names against the local installation or semantic embeddings to ensure compatibility.
3.  **Builder:** Compiles the validated structure into an executable ComfyUI workflow format.

---

### Model Sources

The models used in this implementation are based on the original ComfyGPT research:

- **WorkflowGenerator Model**: Original fine-tuned model from [xiatianzs/resources](https://huggingface.co/xiatianzs/resources/tree/main) (ComfyGPT research team)
  - Base model: Qwen2.5-14B, fine-tuned for workflow generation
  - Training context window: 8,192 tokens (cutoff length used during training - see [training config](https://github.com/comfygpt/comfygpt/blob/main/train/sft/qwen2.5_full_sft.yaml))
  - Model architecture context window: 131,072 tokens (128K) - Maximum supported by model architecture (see [model config](https://huggingface.co/xiatianzs/resources/blob/main/model/flow_agent/config.json))
  - Quantized to GGUF format (q8_0) for efficient inference

- **Embedding Model**: [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (original SentenceTransformer model)
  - Used for semantic search in NodeValidator

- **NodeValidator Model**: Base [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model (not fine-tuned)
  - Context window: 32,768 tokens
  - Used for LLM refinement (optional)
  - Quantized to GGUF format (q8_0) for efficient inference

**Model Repositories:**
- **Original Models**: [xiatianzs/resources](https://huggingface.co/xiatianzs/resources/tree/main) - Original fine-tuned models from ComfyGPT research team (HuggingFace format)
- **Pre-quantized GGUF Models**: [DanielPFlorian/comfyui-workflowgenerator-models](https://huggingface.co/DanielPFlorian/comfyui-workflowgenerator-models) - Ready-to-use quantized GGUF models for this implementation


## Installation

### Prerequisites

- ComfyUI installed and running
- Python 3.10 or higher (see note below)
- CUDA-capable GPU (recommended) or CPU
- Git installed (for cloning the repository)

**Python Requirements:**
- **Standard ComfyUI Installation:** Python 3.10 or higher must be installed separately. ComfyUI requires Python to run, so if ComfyUI is working, Python is already installed.
- **Portable ComfyUI Installation:** Python is included (embedded in the `python_embeded` folder), so no separate Python installation is needed. The custom node will use ComfyUI's embedded Python.

**Note:** If Git is not installed, download it from [git-scm.com](https://git-scm.com/). You can verify Git installation by running `git --version` in your terminal.

### Model Recommendation

**Model Format Comparison:**

- **GGUF models**: Use significantly less VRAM with similar generation quality compared to HuggingFace models. Quantization (q8_0, q4_0) provides a good balance between quality and memory usage.

- **HuggingFace models**: Use more VRAM but offer full precision. Both formats are fully supported and produce similar quality results.

### Step 1: Clone the Repository

Navigate to your ComfyUI `custom_nodes` directory and clone this repository:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/danielpflorian/ComfyUI-WorkflowGenerator.git
```

### Step 2: Install Dependencies

Install the required Python dependencies by opening your terminal inside the ComfyUI-WorkflowGenerator folder:

```bash
cd ComfyUI/custom_nodes/ComfyUI-WorkflowGenerator
pip install -r requirements.txt
```

**For Portable ComfyUI Installations:**

For portable installations, use the embedded Python from the portable ComfyUI root directory:

```bash
cd <portable_comfyui_root>
python_embeded\python.exe -s -m pip install -r ComfyUI\custom_nodes\ComfyUI-WorkflowGenerator\requirements.txt
```

### Step 3: Install llama-cpp-python (Required for GGUF Models)

**Important:** `llama-cpp-python` is **required** for GGUF models. It must be installed separately based on your system configuration.

**Note:** The quick install commands below may not work for all Windows/CUDA configurations. If they fail or if CUDA support isn't detected, **local compilation is often necessary**. See the [Wiki - llama-cpp-python Installation](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/wiki/Installation#llama-cpp-python-installation) for detailed instructions on compiling from source.

**Quick install (try this first):**
- **CPU only**: `pip install llama-cpp-python`
- **CUDA (NVIDIA)**: `pip install llama-cpp-python[cuda]`
- **Metal (macOS)**: `pip install llama-cpp-python[metal]`

**For Portable ComfyUI Installations:**

For portable installations, use the embedded Python from the portable ComfyUI root directory:

```bash
cd <portable_comfyui_root>
python_embeded\python.exe -s -m pip install llama-cpp-python[cuda]
```

**Note:** If you plan to use HuggingFace models instead, you can skip this step.

### Step 4: Copy Models

Copy your GGUF models and tokenizers to `ComfyUI/models/LLM/`:

```
ComfyUI/models/LLM/
├── workflow-generator-q8_0.gguf    # WorkflowGenerator model
├── workflow-generator/             # WorkflowGenerator tokenizer
├── Qwen2.5-7B-Instruct-q8_0.gguf   # NodeValidator model (optional)
├── Qwen2.5-7B-Instruct/            # NodeValidator tokenizer (optional)
└── paraphrase-multilingual-MiniLM-L12-v2/  # Embedding model
```

### Step 5: Restart ComfyUI

Restart ComfyUI to load the custom nodes. The nodes will appear in the `WorkflowGenerator` category.

## Quick Start

The easiest way to get started is using the **Workflow Generator Pipeline** node, which processes your instruction through all three stages sequentially:

1. **Install the custom node** (see Installation above)
2. **Install llama-cpp-python** (required for GGUF models - see [Wiki - llama-cpp-python Installation](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/wiki/Installation#llama-cpp-python-installation))
3. **Run the "Update Node Catalog" node first** - This scans and catalogs all available ComfyUI nodes (native and custom) and is required before generating workflows
4. **Add the "Workflow Generator Pipeline" node** to your ComfyUI workflow
5. **Enter your instruction**, for example: `"Create a text-to-image workflow"`
6. **Configure model path** (see Model Format Comparison above for guidance)
7. **Execute the node** to generate a complete workflow
8. **Use the generated workflow** - it will appear as a workflow JSON that you can load or save

**Expected Results:**
- The Pipeline node automatically generates the workflow diagram, validates node names, and converts it to executable ComfyUI workflow JSON
- You can save the workflow to a file for later use. By default it will be saved in comfyUI/output.

**Note:** For detailed documentation on individual nodes and advanced usage, see the [Wiki](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/wiki).

## Available Nodes

This section provides a high-level overview of each node.

#### [Workflow Generator Pipeline](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/wiki/Node-Reference#workflow-generator-pipeline)

- **Purpose**: One-click solution for complete workflow generation
- **What it does**: Processes your instruction through all three stages sequentially: generates the workflow diagram, validates and corrects node names, then builds the final ComfyUI workflow JSON
- **Best for**: Quick workflow creation without intermediate steps
- **Key inputs**: Instruction, model path, configuration options
- **Key outputs**: Complete workflow JSON, optional file save

For more control, you can use individual nodes to inspect and modify intermediate results:

**Important:** Before using individual nodes or the pipeline, **run the "Update Node Catalog" node first** to scan and catalog all available ComfyUI nodes. This is required for proper node validation and workflow building.

**Note:** WorkflowGenerator, NodeValidator, and WorkflowBuilder are designed to be used in sequence. NodeValidator is optional—you can connect WorkflowGenerator directly to WorkflowBuilder if you want to skip validation.

#### 1. [WorkflowGenerator](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/wiki/Node-Reference#workflowgenerator-node)

- **Purpose**: Generates workflow diagrams from natural language
- **Key inputs**: Instruction, model selection
- **Key outputs**: Workflow diagram (JSON string)
- **Usage**: First step in the pipeline

#### 2. [NodeValidator](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/wiki/Node-Reference#nodevalidator-node)

- **Purpose**: Validates and corrects node names in workflow diagrams
- **Key inputs**: Workflow diagram, optional instruction for context
- **Key outputs**: Refined diagram with corrected node names
- **Usage**: Optional second step (can be skipped)
- **Two modes**: 
  - Semantic search only (faster, deterministic)
  - LLM refinement (slower, more accurate)

#### 3. [WorkflowBuilder](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/wiki/Node-Reference#workflowbuilder-node)

- **Purpose**: Converts workflow diagrams into executable ComfyUI workflow JSON
- **Key inputs**: Refined diagram (from NodeValidator) or raw diagram (from WorkflowGenerator)
- **Key outputs**: Workflow JSON, optional file save
- **Usage**: Final step in the pipeline

#### [UpdateNodeCatalog](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/wiki/Node-Reference#updatenodecatalog-node)

- **Purpose**: Scans and catalogs all available ComfyUI nodes (native and custom)
- **When to run**: After installing new custom nodes or updating ComfyUI
- **Key inputs**: Catalog directory path (optional)
- **Key outputs**: Updated catalog files

## Troubleshooting

**Quick fixes:**
- **Models not found**: Verify models are in `ComfyUI/models/LLM/`
- **OOM errors**: `auto_gpu_layers` is enabled by default to prevent this. If issues persist, verify your VRAM is sufficient for the model size.
- **Slow performance**: Use GGUF models (lower VRAM usage), disable LLM refinement
- **Invalid nodes**: Run **Update Node Catalog** node

## Architectural Insights & Future Vision

While this implementation successfully brings the ComfyGPT architecture to ComfyUI, the rapid evolution of the generative AI landscape suggests that the next generation of workflow generation tools will need to evolve beyond static fine-tuning.

### The Problem with Static Models
The ComfyUI custom node ecosystem changes daily. New nodes and unforeseen architectures are released constantly. A model fine-tuned on a dataset from last month (like the current WorkflowGenerator) is inherently "frozen" in time. It cannot hallucinate the correct connections for a node it has never seen during training.

### Ideas on Future Workflow Generators
To achieve true state-of-the-art performance that keeps pace with the community, future architectures should likely move toward:

1.  **Retrieval-Augmented Generation (RAG) for Nodes**: Instead of baking node knowledge into the model weights, an agent could query a dynamic, vector-embedded database that includes both the user's *currently installed* nodes and an updated database of repositories. This allows the system to discover and use completely new nodes or updated versions directly from the internet, enabling it to "read" the documentation of a node released today and use it immediately.

2.  **Input/Output Type Awareness**: The current approach uses semantic search to correct node names (e.g., matching "Load Image" to "Image Loader"). However, a robust system needs to understand **I/O Schema**.
    *   *Current*: "Does this node name look right?"
    *   *Future*: "Does the `LATENT` output of Node A actually fit into the `LATENT` input of Node B?"
    *   An agent needs to reason about data types (Float, Image, Conditioning, Model) to ensure generated workflows are not just linguistically plausible, but executably valid.

3.  **Small Graph-Reasoning Models (SLMs)**: We may not need massive 70B+ parameter models for this task. Specialized, smaller models trained specifically on Graph Theory and Directed Acyclic Graphs (DAGs) could offer superior reasoning capabilities for connecting logical blocks, while relying on the RAG system for the specific node vocabulary.

4.  **Case-Based Reasoning & Workflow Retrieval**: Thousands of high-quality workflows exist online, often accompanied by images and descriptions. A future system should not just generate from scratch but actively **retrieve and adapt existing workflows**. By indexing these community-created workflows as "knowledge patterns," the agent can identify how experts typically solve specific sub-problems (e.g. "How is ControlNet usually connected to Wan Video Generation Nodes?") and apply those proven sub-graphs to the current task.

This project serves as a foundational step, implementing the current research (ComfyGPT). However, the future lies in agents that can actively explore, reason about, and validate the live ComfyUI environment they inhabit.

## License

GNU General Public License v3

## Acknowledgments

- **ComfyUI**: Built on top of [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- **ComfyGPT Research Team**: This project is based on the [ComfyGPT research paper](https://arxiv.org/abs/2503.17671) and [original repository](https://github.com/comfygpt/comfygpt) by Oucheng Huang, Yuhang Ma, Zeng Zhao, Mingrui Wu, Jiayi Ji, Rongsheng Zhang, Zhipeng Hu, Xiaoshuai Sun, and Rongrong Ji
- **llama-cpp-python**: Uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for GGUF model support

