# Configuration Guide

Complete configuration options, performance tuning, and model-specific settings.

## Model Paths

- **Relative paths**: `models/LLM/workflow-generator-q8_0.gguf` (relative to ComfyUI root)
- **Absolute paths**: `/path/to/model.gguf`
- System automatically resolves paths relative to ComfyUI's model directories

## GPU Settings

- **GPU Layers**: `"auto"` (default & recommended), `"all"`, `"none"`, or number (0-50)
  - `"auto"`: Automatically determines optimal GPU layer allocation based on available VRAM to prevent OOM.
  - `"all"`: Use all available GPU layers (fastest, but may cause OOM on lower VRAM).
  - `"none"`: CPU only.
  - Number: Specific count of GPU layers to use.
- **Device Preference**: `"auto"` (detects best), `"cuda"`, or `"cpu"`

## Model-Specific Settings

### GGUF Models

- Handle quantization internally (no dtype settings needed)
- Use `n_gpu_layers` to control GPU/CPU split
- Use `context_size` to set context window (default: 4096)
- Performance parameters:
  - `use_mmap` (bool): Memory-mapped file access (default: True)
  - `use_mlock` (bool): Lock memory pages (default: False)
  - `n_batch` (int): Batch size for processing (default: 512)
  - `n_threads` (int, optional): CPU thread count (auto-detected if not set)

### HuggingFace Models

- Support dtype options: `"auto"`, `"fp16"`, `"bf16"`, `"fp32"`, `"fp8"`, or `torch.dtype`
- Attention implementations:
  - `"auto"`: Automatically selects best available
  - `"flash_attention_2"`: Flash Attention 2 (requires flash-attn package)
  - `"sdpa"`: Scaled Dot Product Attention (built into PyTorch 2.0+)
- Device map: Automatically handles device placement

## Performance Tuning

### For Lower VRAM Usage

- Use GGUF models (typically 4-8GB for q8_0 quantization vs 12-16GB+ for HuggingFace)
- Set `n_gpu_layers="all"` if VRAM allows
- Disable LLM refinement (`use_llm_refinement=False`)
- Use smaller context windows if possible

### For Lower VRAM Usage

- Reduce `n_gpu_layers` (try "auto" or lower number)
- Use smaller quantization (q4_0 instead of q8_0)
- Set `device_preference="cpu"` for some operations
- Use HuggingFace models with CPU offloading

### For Better Accuracy

- Enable LLM refinement (`use_llm_refinement=True`)
- Use larger models (q8_0 instead of q4_0)
- Increase `top_k` in NodeValidator for more candidate nodes

---

[← Back to Home](Home) | [← Node Reference](Node-Reference) | [Next: Troubleshooting →](Troubleshooting)

