# Installation Guide

This guide covers installation of ComfyUI-WorkflowGenerator and the required `llama-cpp-python` library for GGUF models.

---

## Prerequisites

- ComfyUI installed and running
- Python 3.10, 3.11, or 3.12
- Git installed
- **For GPU acceleration:** NVIDIA GPU with CUDA support (recommended) or Apple Silicon Mac

---

## Step 1: Install ComfyUI-WorkflowGenerator

Follow the main [README](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator) for basic installation steps (cloning, dependencies, etc.).

---

## Step 2: Install llama-cpp-python

**Why separate?** `llama-cpp-python` is required for GGUF models but needs different installation methods for CPU/CUDA/Metal, so it's not included in `requirements.txt`.

### Choose Your Installation Path

<table>
<tr>
<td><strong>Standard Python Installation</strong></td>
<td><strong>Portable ComfyUI (Windows)</strong></td>
</tr>
<tr>
<td>Use regular <code>pip install</code> commands</td>
<td>Use <code>python_embeded\python.exe -s -m pip</code></td>
</tr>
<tr>
<td>→ See <a href="#standard-installation">Standard Installation</a></td>
<td>→ See <a href="#portable-comfyui-windows">Portable Installation</a></td>
</tr>
</table>

---

## Standard Installation

### Option 1: CPU Only (No GPU)

**All Platforms:**
```bash
pip install llama-cpp-python
```

**Who should use this:** Systems without NVIDIA GPU or Apple Silicon, or for testing purposes.

---

### Option 2: CUDA Support (NVIDIA GPUs) - Recommended

**Compatibility:**
- Python 3.10, 3.11, 3.12
- CUDA 11.8, 12.1, 12.4+
- PyTorch 2.0+

#### Try Quick Install First

**All Platforms:**
```bash
pip install llama-cpp-python[cuda]
```

**This works if:**
- Installation completes in 1-2 minutes
- You see "Downloading .whl" messages
- ✅ You're done! Skip to [testing](#testing-your-installation)

**If it fails or takes 20+ minutes, you need to compile from source →**

---

#### Compile from Source (Windows)

**When needed:** Pre-built wheel not available for your Python/CUDA combination (common on Windows)

**Prerequisites (install these first):**
1. [Visual Studio 2022 Community](https://visualstudio.microsoft.com/vs/community/) - Select "Desktop development with C++"
2. [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) - Match your driver version
3. [CMake](https://cmake.org/download/) - Add to PATH during installation
   - **Important:** After installing CMake, close and reopen your terminal (no restart needed)

**PowerShell:**
```powershell
# Install build dependencies
pip install cmake scikit-build-core

# Set build flags and compile (takes 20-30 minutes)
$env:CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native"
$env:FORCE_CMAKE=1
pip install llama-cpp-python --no-cache-dir --verbose --force-reinstall
```

**Command Prompt (cmd):**
```cmd
REM Install build dependencies
pip install cmake scikit-build-core

REM Set build flags and compile (takes 20-30 minutes)
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native
set FORCE_CMAKE=1
pip install llama-cpp-python --no-cache-dir --verbose --force-reinstall
```

---

#### Compile from Source (Linux/macOS)

**Prerequisites:**
- GCC/Clang compiler
- CUDA Toolkit installed
- CMake installed

**Linux/macOS:**
```bash
# Install build dependencies
pip install cmake scikit-build-core

# Set build flags and compile
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --verbose --force-reinstall
```

---

### Option 3: Metal Support (macOS Apple Silicon)

**macOS Only:**
```bash
pip install llama-cpp-python[metal]
```

**Who should use this:** Mac users with M1/M2/M3/M4 chips for GPU acceleration.

---

## Portable ComfyUI (Windows)

For Windows portable ComfyUI installations, all commands use the embedded Python interpreter.

### Quick Install (Try First)

**PowerShell:**
```powershell
cd <your_portable_comfyui_folder>
python_embeded\python.exe -s -m pip install llama-cpp-python[cuda]
```

**Command Prompt:**
```cmd
cd <your_portable_comfyui_folder>
python_embeded\python.exe -s -m pip install llama-cpp-python[cuda]
```

---

### Compile from Source (Portable)

**Prerequisites:** Same as standard Windows installation (Visual Studio, CUDA Toolkit, CMake)

**PowerShell:**
```powershell
# Step 1: Install build dependencies
python_embeded\python.exe -s -m pip install cmake scikit-build-core

# Step 2: Set environment and compile
$env:CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native"
$env:FORCE_CMAKE=1
python_embeded\python.exe -s -m pip install llama-cpp-python --no-cache-dir --verbose --force-reinstall
```

**Command Prompt:**
```cmd
REM Step 1: Install build dependencies
python_embeded\python.exe -s -m pip install cmake scikit-build-core

REM Step 2: Set environment and compile
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native
set FORCE_CMAKE=1
python_embeded\python.exe -s -m pip install llama-cpp-python --no-cache-dir --verbose --force-reinstall
```

**Note:** The `-s` flag prevents Python from adding user site-packages to sys.path.

---

## Testing Your Installation

**Check if CUDA is working:**
```python
python -c "from llama_cpp import Llama; print('✓ llama-cpp-python installed successfully')"
```

**Check your versions:**
```python
python -c "import sys, torch; print(f'Python: {sys.version_info.major}.{sys.version_info.minor}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

---

## Troubleshooting

### "No module named 'llama_cpp'" after installation

**Cause:** CPU version installed instead of CUDA version

**Solution - Reinstall with CUDA support:**

**PowerShell:**
```powershell
pip uninstall llama-cpp-python; pip install llama-cpp-python[cuda]
```

**Command Prompt / Linux / macOS:**
```bash
pip uninstall llama-cpp-python && pip install llama-cpp-python[cuda]
```

**Portable ComfyUI:**
```cmd
python_embeded\python.exe -s -m pip uninstall llama-cpp-python && python_embeded\python.exe -s -m pip install llama-cpp-python[cuda]
```

---

### "Cannot import 'scikit_build_core.build'"

**Cause:** Missing build dependencies

**Solution:**
```bash
pip install cmake scikit-build-core
```

**Portable ComfyUI:**
```cmd
python_embeded\python.exe -s -m pip install cmake scikit-build-core
```

Then retry the installation with CMAKE_ARGS set.

---

### "CMake not found"

**Solution:**
1. Install CMake: `pip install cmake` or download from https://cmake.org/
2. **Windows users:** Close and reopen your terminal for PATH changes (no restart needed)
3. If still not working, you may need Visual Studio Build Tools

---

### "CUDA not found" or "CUDA version mismatch"

**Check your CUDA version:**
```python
python -c "import torch; print(torch.version.cuda)"
```

**Solutions:**
- Ensure CUDA Toolkit is installed and matches your PyTorch CUDA version
- Check that your GPU drivers are up to date
- Verify CUDA is in your system PATH

For detailed troubleshooting, see [INSTALL_LLAMACPP.md](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator/blob/main/INSTALL_LLAMACPP.md)

---

### Installation takes 20-30 minutes

**This is normal!** It means you're compiling from source because no pre-built wheel is available for your system.

**Requirements:**
- **Windows:** Visual Studio Build Tools with C++ support
- **Linux:** build-essential package (`sudo apt install build-essential`)
- Be patient - the compilation will complete

---
[← Back to Home](Home) | [Next: Node Reference →](Node-Reference)
