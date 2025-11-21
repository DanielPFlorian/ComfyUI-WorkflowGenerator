"""
WorkflowGenerator: Generates workflow diagrams from natural language descriptions.
Enhanced with GPU layer management, error handling, and ComfyUI integration.
"""

import json
import logging
import time
from typing import Any

import torch

from ..utils.model_loaders.device_utils import load_model
from ..utils.process_utils import del_digram_primitive


class WorkflowGenerator:
    """
    Generator responsible for creating workflow diagrams from natural language.

    Uses a language model to convert user descriptions into structured workflow
    diagrams represented as JSON edge lists.
    """

    def __init__(
        self,
        cfg: Any,
        n_gpu_layers: int | str = "auto",
        dtype: str | torch.dtype = "auto",
        device_preference: str = "auto",
        attn_implementation: str = "auto",
        context_size: int = 4096,
        use_mmap: bool = True,
        use_mlock: bool = False,
        n_batch: int = 512,
        n_threads: int | None = None,
    ):
        """
        Initialize WorkflowGenerator with model configuration.

        Args:
            cfg: Configuration object with workflow_generator settings
            n_gpu_layers: For GGUF: "auto", -1 for all, 0 for CPU, or specific count
            dtype: For HuggingFace: "auto", "fp16", "bf16", "fp32", "fp8", or torch.dtype
            device_preference: "auto", "cuda", or "cpu"
            attn_implementation: For HuggingFace: "auto", "flash_attention_2", or "sdpa"
        """
        logging.info("WorkflowGenerator: Initializing...")

        # Get model format from config (default to huggingface for backward compatibility)
        model_format = getattr(cfg.workflow_generator, "model_format", "huggingface")
        model_path = cfg.workflow_generator.model_path
        tokenizer_path = getattr(cfg.workflow_generator, "tokenizer_path", None)

        logging.info(f"Model format: {model_format}, path: {model_path}")

        load_kwargs = {}

        if model_format.lower() == "huggingface":
            # HuggingFace models use dtype, device_map, and attention implementation
            load_kwargs = {
                "dtype": dtype,
                "device_preference": device_preference,
                "attn_implementation": attn_implementation,
            }
            logging.debug("Model format: HuggingFace")
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
            logging.debug("Model format: GGUF (already quantized)")

        load_start = time.time()
        try:
            self.model_wrapper = load_model(model_format=model_format, model_path=model_path, tokenizer_path=tokenizer_path, **load_kwargs)
            load_time = time.time() - load_start
            logging.info(f"Model loaded in {load_time:.2f}s")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

        self.model = self.model_wrapper.model
        self.tokenizer = self.model_wrapper.tokenizer

        try:
            if torch.cuda.is_available() and model_format.lower() == "huggingface":
                vram_used = torch.cuda.memory_allocated() / 1024**3
                logging.debug(f"GPU memory allocated: {vram_used:.2f} GB")
        except Exception:
            pass

        self.top_p = getattr(cfg.workflow_generator, "top_p", 0.7)
        self.max_new_tokens = getattr(cfg.workflow_generator, "max_new_tokens", 8192)
        self.temperature = getattr(cfg.workflow_generator, "temperature", 0.95)
        self.allow_primitive_nodes = getattr(cfg.workflow_generator, "allow_primitive_nodes", False)
        # Template matches FlowDataset training format: instruction + "description: [description]"
        # Model (Qwen2.5-14B, not instruct) was fine-tuned on this exact format
        self.template = "Based on the description I provided, generate a JSON example of the required ComfyUi workflow. description:"

        logging.info("WorkflowGenerator: Initialized successfully")

    def generate(self, prompt: str, seed: int | None = None) -> list:
        """
        Generate workflow diagram from natural language prompt.

        Args:
            prompt: Natural language description of the desired workflow
            seed: Random seed for reproducible generation (optional)

        Returns:
            List of edges representing the workflow diagram
        """
        # Format prompt to match FlowDataset training format: Model (Qwen2.5-14B, not instruct) was fine-tuned on this exact format
        connection_guidance = (
            " Ensure all connections are valid: every input_name must exist on the destination node, "
            "and every output_name must exist on the source node."
        )

        if "Based on the description I provided" not in prompt:
            formatted_prompt = self.template + " " + prompt + connection_guidance
        else:
            if "description:" in prompt:
                desc_idx = prompt.find("description:")
                if desc_idx != -1:
                    desc_prefix = prompt[: desc_idx + len("description:")]
                    desc_content = prompt[desc_idx + len("description:") :].strip()
                    formatted_prompt = desc_prefix + " " + desc_content + connection_guidance
                else:
                    formatted_prompt = prompt + connection_guidance
            else:
                formatted_prompt = prompt + " description: " + connection_guidance

        prompt = formatted_prompt

        messages = [{"role": "user", "content": prompt}]

        try:
            # Use HuggingFace tokenizer even for GGUF models (embedded tokenizer doesn't expose apply_chat_template)
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            logging.error(f"Failed to apply chat template: {e}")
            text = prompt

        logging.debug(f"Tokenizing input (max_new_tokens={self.max_new_tokens}, temperature={self.temperature}, top_p={self.top_p})...")

        try:
            model_inputs = self.tokenizer([text], return_tensors="pt")
            if hasattr(self.model_wrapper, "device") and isinstance(self.model_wrapper.device, torch.device):
                model_inputs = model_inputs.to(self.model_wrapper.device)
            input_tokens = model_inputs.input_ids.shape[1]
            logging.debug(f"Input tokens: {input_tokens}")
        except Exception as e:
            logging.error(f"Failed to tokenize input: {e}")
            raise

        logging.info("Generating workflow diagram...")

        try:
            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": True,
                "top_p": self.top_p,
            }
            if seed is not None:
                max_seed = 2**32 - 1
                if seed < 0 or seed > max_seed:
                    seed = seed % (max_seed + 1)
                generate_kwargs["seed"] = seed

            generated_ids = self.model_wrapper.generate(model_inputs.input_ids, **generate_kwargs)

            generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            output_tokens = len(generated_ids[0])
            logging.debug(f"Generated {output_tokens} tokens")
        except Exception as e:
            logging.error(f"Failed to generate tokens: {e}")
            raise

        try:
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            logging.error(f"Failed to decode tokens: {e}")
            raise

        logging.debug("Post-processing diagram...")
        try:
            diagram = self.post_process(response)
            logging.info(f"Workflow diagram generated successfully: {len(diagram)} edges")
            return diagram
        except Exception as e:
            logging.error(f"Failed to post-process diagram: {e}")
            raise

    def post_process(self, response: str | list) -> list:
        """
        Post-process the model response into a valid diagram format.

        Args:
            response: Raw model response (string or list)

        Returns:
            List of edges representing the workflow diagram
        """
        diagram = response

        if isinstance(diagram, str):
            try:
                diagram = json.loads(diagram)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response: {e}")
                logging.error(f"Response content: {response[:500]}")
                raise

        if not self.allow_primitive_nodes:
            try:
                diagram = del_digram_primitive(diagram)
            except Exception as e:
                logging.warning(f"Failed to remove primitive nodes: {e}")

        if not isinstance(diagram, list):
            raise ValueError(f"Invalid diagram format: expected list, got {type(diagram)}")

        return diagram

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

        logging.info("WorkflowGenerator: Resources cleaned up")
