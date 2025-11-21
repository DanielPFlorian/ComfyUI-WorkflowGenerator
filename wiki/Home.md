# ComfyUI-WorkflowGenerator Wiki

Welcome to the ComfyUI-WorkflowGenerator documentation wiki. This wiki contains detailed guides, references, and troubleshooting information.

## Quick Links

- [Installation Guide](Installation) - Complete setup instructions including llama-cpp-python installation
- [Node Reference](Node-Reference) - Detailed documentation for each node
- [Configuration Guide](Configuration) - Model paths, GPU settings, and performance tuning
- [Troubleshooting](Troubleshooting) - Common issues and solutions
- [Instruction Prompt Usage](Instruction-Prompt-Usage) - How instructions are used across pipeline steps

## Overview

ComfyUI-WorkflowGenerator is a custom node suite that generates ComfyUI workflows from natural language descriptions. It uses a three-stage pipeline:

1. **WorkflowGenerator** - Converts natural language to workflow diagrams
2. **NodeValidator** (optional) - Validates and corrects node names
3. **WorkflowBuilder** - Converts diagrams to executable ComfyUI workflow JSON

## Getting Started

1. Follow the [Installation Guide](Installation) to set up the custom node
2. Read the [Node Reference](Node-Reference) to understand each component
3. Check [Troubleshooting](Troubleshooting) if you encounter issues

## Main Repository

For the main project repository, installation instructions, and quick start guide, see the [GitHub README](https://github.com/danielpflorian/ComfyUI-WorkflowGenerator).

