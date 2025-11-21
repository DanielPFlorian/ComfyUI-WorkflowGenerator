# Installation Guide

This guide covers detailed installation instructions, including the `llama-cpp-python` setup required for GGUF models.

## Prerequisites

- ComfyUI installed and running
- Python 3.10 or higher
- CUDA-capable GPU (recommended) or CPU
- Git installed

## Basic Installation

Follow the main [README](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator) for basic installation steps (cloning, dependencies, etc.).

## llama-cpp-python Installation

**Important:** `llama-cpp-python` is **required** for GGUF models (recommended). It must be installed separately based on your system configuration. It is not included in `requirements.txt` because it requires different installation methods for CPU, CUDA, and Metal.

For detailed `llama-cpp-python` installation instructions, see [INSTALL_LLAMACPP.md](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/blob/main/INSTALL_LLAMACPP.md) in the main repository.

### Quick Install Options

#### CPU Only (No GPU acceleration)
```bash
pip install llama-cpp-python
```

#### CUDA Support (NVIDIA GPUs) - Recommended

**IMPORTANT NOTE FOR WINDOWS USERS:**
Installing `llama-cpp-python` with CUDA support on Windows is notoriously difficult because pre-built binaries (wheels) often don't match your specific combination of Python version, CUDA version, and OS.

**If the quick install command below fails or falls back to CPU mode, you MUST compile from source.** This is normal and expected.

**Try this first (Quick Install):**
```bash
pip install llama-cpp-python[cuda]
```

**If that fails (Compile from Source):**
This is the most reliable method for Windows. It takes about 20-30 minutes to compile.

1. **Install Prerequisites:**
   - [Visual Studio 2022 Community](https://visualstudio.microsoft.com/vs/community/) (Select "Desktop development with C++")
   - [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) (Match your driver version)
   - [CMake](https://cmake.org/download/) (Add to PATH during installation)

2. **Compile and Install:**
   ```powershell
   # PowerShell
   $env:CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native"
   $env:FORCE_CMAKE=1
   pip install llama-cpp-python --no-cache-dir --verbose --force-reinstall
   ```

   ```batch
   :: Command Prompt (cmd)
   set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native
   set FORCE_CMAKE=1
   pip install llama-cpp-python --no-cache-dir --verbose --force-reinstall
   ```

For a complete step-by-step guide with troubleshooting, please see the [Detailed Installation Guide (INSTALL_LLAMACPP.md)](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/blob/main/INSTALL_LLAMACPP.md).

#### Metal Support (macOS with Apple Silicon)
```bash
pip install llama-cpp-python[metal]
```

### CUDA/PyTorch Compatibility

`llama-cpp-python` with CUDA support is compatible with:
- **PyTorch versions:** 2.0+ (tested with 2.1.0, 2.2.0, 2.3.0, 2.4.0+)
- **CUDA versions:** 11.8, 12.1, 12.4+ (pre-built wheels available for common versions)
- **Python versions:** 3.10, 3.11, 3.12

**Note:** Pre-built wheels are available for common CUDA/Python combinations. If your specific combination isn't available, you'll need to compile from source.

### Quick Check: Will I Need to Compile from Source?

To check if a pre-built wheel is available for your setup:

```bash
# Check your Python, PyTorch, and CUDA versions
python -c "import sys, torch; print(f'Python: {sys.version_info.major}.{sys.version_info.minor}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Try installing (this will show if it's downloading a wheel or compiling)
pip install llama-cpp-python[cuda] --verbose
```

**Signs a pre-built wheel is available:**
- Installation completes in 1-2 minutes
- You see "Downloading" messages for `.whl` files
- No compilation messages (no "Building wheel", "CMake", etc.)

**Signs you'll need to compile from source:**
- Installation takes 20-30 minutes
- You see "Building wheel" or "CMake" messages
- Requires Visual Studio Build Tools (Windows) or build-essential (Linux)

### Troubleshooting CUDA Installation

**Problem: "No module named 'llama_cpp'" after installation**
- Solution: Make sure you installed the CUDA version, not the CPU version
- Reinstall: `pip uninstall llama-cpp-python && pip install llama-cpp-python[cuda]`

**Problem: Installation takes a very long time (20-30 minutes)**
- This means it's compiling from source (no pre-built wheel available)
- Ensure you have Visual Studio Build Tools (Windows) or build-essential (Linux) installed

**Problem: "CUDA not found" or "CUDA version mismatch"**
- Check your CUDA version: `python -c "import torch; print(torch.version.cuda)"`
- Ensure CUDA is properly installed and accessible
- See [INSTALL_LLAMACPP.md](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/blob/main/INSTALL_LLAMACPP.md) for detailed troubleshooting

**Problem: "CMake not found"**
- Install CMake: `pip install cmake` or download from https://cmake.org/
- On Windows, you may also need Visual Studio Build Tools

### Portable ComfyUI Installation

For portable ComfyUI installations (Windows), use the embedded Python:

```bash
cd <portable_comfyui_root>
.\python_embeded\python.exe -s -m pip install llama-cpp-python[cuda]
```

Or if compiling from source:
```batch
.\python_embeded\python.exe -s -m pip install cmake
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
set FORCE_CMAKE=1
.\python_embeded\python.exe -s -m pip install llama-cpp-python --no-cache-dir
```

---

[← Back to Home](Home) | [Next: Node Reference →](Node-Reference)

