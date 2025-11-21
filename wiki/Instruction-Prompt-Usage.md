# Instruction Prompt Usage Across Steps

This page explains how the instruction prompt is utilized across the different stages of the workflow generation pipeline.

## Step 1 (WorkflowGenerator)

The instruction prompt is used to generate the initial workflow diagram. This is the primary input that determines the overall workflow structure.

## Step 2 (NodeValidator)

The instruction prompt serves as **context** for the LLM when correcting invalid node names. It helps the LLM understand the workflow's intent and choose the most appropriate node from candidate matches.

### How the Instruction is Used in Step 2

The instruction does **NOT** override the refinement prompt. Instead, it is **inserted as context** within the prompt template.

When LLM refinement is enabled (`use_llm_refinement=True`), the instruction you provide is inserted into a structured prompt that asks the LLM to select the best node from candidate matches. The full prompt structure is:

```
"I would like you to act as an expert in ComfyUI platform. 
I will provide a example, including a description about ComfyUI workflow 
and a logical diagram in json format represents the comfyui workflow. 
The logical diagram is a edges list [edge_1, edge_2, edge_3, ... , edge_n], 
each edge is consist of [output_node_name,output_name,input_node_name,input_name], 
represents a line between output node and input node. 
Example: description: {YOUR_INSTRUCTION_HERE}. logical diagram: {diagram}. 
Now, This logical diagram has one error node name. error name: {error_name}. 
I will give you some candidate nodes. Please combine the above information 
to select the most suitable candidate node. Candidate nodes: {candidate_nodes}. 
You just need to return you choose node name. 
Please return result in pure JSON format, including: 
'''json{"candidate_node_name":"..."}'''"
```

**Key Points:**
- Your instruction replaces `{YOUR_INSTRUCTION_HERE}` in the prompt
- The instruction provides **context** about the workflow's intent
- The LLM uses this context to make better decisions when selecting from candidate nodes
- The prompt structure itself is fixed and cannot be overridden

### How Different Instructions Affect Refinement

When using **LLM refinement** (`use_llm_refinement=True`), the instruction prompt in Step 2 directly influences node selection:

| Scenario | Effect on Refinement |
|----------|---------------------|
| **Same instruction as Step 1** | ✅ **Recommended** - Provides consistent context, helps LLM maintain original intent |
| **More specific/detailed instruction** | ✅ **Beneficial** - Can guide LLM toward more precise node choices (e.g., "use SDXL nodes" vs "use SD 1.5 nodes") |
| **Different focus/emphasis** | ⚠️ **Can be beneficial or problematic** - May bias workflow toward different goals (e.g., "optimize for speed" vs "optimize for quality") |
| **Contradictory instruction** | ❌ **Not recommended** - May cause confusion and incorrect node selections |
| **Empty instruction** | ⚠️ **Works but less effective** - LLM still works but has less context to make decisions |

### Example Use Cases for Different Instructions

1. **Refining for Specific Model Type:**
   - Step 1: `"Create a text-to-image workflow"`
   - Step 2: `"Create a text-to-image workflow using SDXL models"` → Guides LLM to select SDXL-specific nodes

2. **Refining for Performance:**
   - Step 1: `"Create an image upscaling workflow"`
   - Step 2: `"Create a fast image upscaling workflow using lightweight models"` → May prefer faster upscaling nodes

3. **Refining for Quality:**
   - Step 1: `"Create an image generation workflow"`
   - Step 2: `"Create a high-quality image generation workflow with advanced sampling"` → May prefer higher-quality sampling nodes

### Best Practices

- **Default**: Connect the `instruction` output from Step 1 to Step 2 for consistent context
- **Advanced**: Use a modified instruction in Step 2 only when you want to guide refinement in a specific direction
- **When LLM refinement is disabled** (`use_llm_refinement=False`): The instruction has no effect (only semantic search is used)

---

[← Back to Home](Home)

