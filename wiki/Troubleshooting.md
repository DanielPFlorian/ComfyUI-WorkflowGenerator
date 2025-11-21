# Troubleshooting

Common issues, solutions, and frequently asked questions.

## Common Issues

### Models Not Found

**Symptoms**: Error message about model file not found

**Solutions**:
- Verify models are in `ComfyUI/models/LLM/` directory
- Check model paths are correct (use absolute paths if needed)
- Ensure tokenizer directories exist and contain required files
- Restart ComfyUI after moving files

### Out of Memory (OOM) Errors

**Symptoms**: CUDA out of memory errors, crashes during model loading

**Solutions**:
- Reduce `n_gpu_layers` (try "auto" or lower number)
- Use smaller GGUF models (q8_0 or q4_0 quantization)
- Set `device_preference` to "cpu" for some processing stages (e.g., NodeValidator semantic search)
- Close other applications using GPU memory
- Use CPU-only mode if GPU memory is insufficient

### Slow Performance

**Symptoms**: Workflow generation takes a very long time

**Solutions**:
- Use GGUF models (lower VRAM usage)
- Set `use_llm_refinement=False` (semantic search only)
- Increase `n_gpu_layers` if VRAM available
- Use smaller models if accuracy allows
- Ensure CUDA/Metal acceleration is properly installed

### Invalid Node Names

**Symptoms**: Errors about node types not found, incorrect node names in workflow

**Solutions**:
- Run **Update Node Catalog** node to refresh database
- Enable `use_llm_refinement=True` for better accuracy
- Check that custom nodes are properly installed
- Verify catalog files are up to date

### Import Errors

**Symptoms**: Python import errors, missing modules

**Solutions**:
- Verify dependencies: `pip install -r requirements.txt`
- Ensure `llama-cpp-python` is installed (see [Installation](Installation))
- Check Python version (3.10+ required)
- Restart ComfyUI after installing dependencies
- Verify you're using the correct Python environment

### Tokenizer Not Found

**Symptoms**: Errors about missing tokenizer files

**Solutions**:
- Ensure tokenizer directory exists with the correct name
- Verify tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`) are present
- Check directory structure matches expected format
- Tokenizer directory name must match model name (without `.gguf` and quantization suffix)

## Limitations & Prompting Tips

**"Why isn't it generating the perfect workflow?"**

It's important to understand that the system is not magic. It relies on a fine-tuned language model that has seen a specific set of training data (around 13,000 workflows).

### Limitations
- **Training Data Bias**: The model is biased towards the workflows it was trained on (mostly SD 1.5, SDXL, SVD, etc.). It may struggle with very new nodes or niche custom node packs released after the dataset was created.
- **Hallucination**: Like all LLMs, it can hallucinate connections that don't exist or propose nodes that are similar to but not exactly what you have installed.
- **Complexity**: Extremely complex workflows with circular dependencies or very specific control flows may be simplified or incorrect.

### Tips for Better Results
- **Be Specific**: Instead of "make an image", try "create a text to image workflow using SDXL with a refiner and simple styles".
- **Re-roll**: If the output looks wrong, try changing the seed or slightly rewording the prompt.
- **Step-by-Step**: Use the individual nodes. Generate the diagram first, inspect it, then validate.
- **Check Node Names**: If it's picking the wrong nodes, run the **Update Node Catalog** node to ensure it knows about your currently installed nodes.

## FAQ

**Q: Can I use HuggingFace models instead of GGUF?**
A: Yes, but they use more VRAM and may be slower. GGUF models are recommended for better performance.

**Q: Do I need the NodeValidator model?**
A: Only if you want to use LLM refinement. Semantic search works without it but is less accurate.

**Q: How often should I update the node catalog?**
A: After installing new custom nodes or updating ComfyUI. The catalog is cached for performance.

**Q: Can I modify the generated workflow?**
A: Yes, the workflow JSON can be edited manually or loaded into ComfyUI for further modification.

**Q: Why is my workflow generation slow?**
A: Check that you're using GGUF models, have GPU acceleration enabled, and LLM refinement is disabled if speed is priority.

**Q: The quick install for llama-cpp-python didn't work. What should I do?**
A: Local compilation is often necessary, especially on Windows. See [Installation](Installation) for detailed compilation instructions.

---

[← Back to Home](Home) | [← Configuration](Configuration)

