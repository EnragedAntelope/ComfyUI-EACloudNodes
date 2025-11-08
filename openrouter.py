"""
OpenRouter Chat Node for ComfyUI v3
Supports text and vision-language models through OpenRouter's API.
"""

import base64
import json
import requests
import time
from PIL import Image
import io as python_io
import torch
from torchvision.transforms import ToPILImage
import random

from comfy_api.latest import ComfyExtension, io


class OpenRouterModelEnum(io.ComboInput):
    """Enum for OpenRouter model selection - Free models"""
    OPTIONS = [
        # Google Models
        "google/gemini-2.5-pro-exp:free",
        "google/gemini-2.0-flash-exp:free",
        "google/gemini-2.0-flash-thinking-exp:free",
        "google/gemma-2-9b-it:free",
        # Meta Llama Models
        "meta-llama/llama-3.3-70b-instruct:free",
        "meta-llama/llama-3.1-70b-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "meta-llama/llama-3.2-90b-vision-instruct:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "meta-llama/llama-4-maverick-17b-128e-instruct:free",
        "meta-llama/llama-4-scout-17b-16e-instruct:free",
        # Mistral Models
        "mistralai/mistral-small-3.1:free",
        "mistralai/mistral-small-3:free",
        "mistralai/mistral-saba-24b:free",
        "mistralai/mistral-7b-instruct:free",
        # Microsoft Models
        "microsoft/phi-3-medium-128k-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        # Qwen Models
        "qwen/qwen-2.5-7b-instruct:free",
        "qwen/qwen-2-7b-instruct:free",
        # DeepSeek Models
        "deepseek/deepseek-r1-distill-llama-70b:free",
        "deepseek/deepseek-r1-distill-qwen-32b:free",
        "deepseek/deephermes-3:free",
        # Other Models
        "openchat/openchat-7b:free",
        "sophosympatheia/rogue-rose-103b-v0.2:free",
        # Manual input option
        "Manual Input"
    ]


class SendSystemEnum(io.ComboInput):
    """Enum for system prompt toggle"""
    OPTIONS = ["yes", "no"]


class ResponseFormatEnum(io.ComboInput):
    """Enum for response format"""
    OPTIONS = ["text", "json_object"]


class SeedModeEnum(io.ComboInput):
    """Enum for seed mode"""
    OPTIONS = ["fixed", "random", "increment", "decrement"]


class DebugModeEnum(io.ComboInput):
    """Enum for debug mode"""
    OPTIONS = ["off", "on"]


class OpenrouterNode(io.ComfyNode):
    """
    A node for interacting with OpenRouter API.
    Supports text and vision-language models through OpenRouter's API.
    """

    # JavaScript safe integer limit (2^53 - 1)
    MAX_SAFE_INTEGER = 9007199254740991

    # Models that support vision capabilities (based on model name)
    VISION_MODELS = [
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "meta-llama/llama-3.2-90b-vision-instruct:free",
        "meta-llama/llama-4-maverick-17b-128e-instruct:free",
        "meta-llama/llama-4-scout-17b-16e-instruct:free"
    ]

    # Class-level storage for seed tracking (since nodes are stateless)
    _last_seed = {}

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OpenrouterNode",
            display_name="OpenRouter Chat",
            category="OpenRouter",
            description="Access OpenRouter's multi-provider API for various AI models. Supports free models, vision analysis, JSON output, and comprehensive error handling.",
            inputs=[
                io.String.Input(
                    "api_key",
                    default="",
                    multiline=False,
                    tooltip="⚠️ Your OpenRouter API key from https://openrouter.ai/keys (Note: key will be visible - take care when sharing workflows)"
                ),
                OpenRouterModelEnum.Input(
                    "model",
                    default="meta-llama/llama-3.3-70b-instruct:free",
                    tooltip="Select a free OpenRouter model or choose 'Manual Input' for custom models. Models with 'vision' support image inputs."
                ),
                io.String.Input(
                    "manual_model",
                    default="",
                    multiline=False,
                    tooltip="Enter a custom model identifier (only used when 'Manual Input' is selected). Format: provider/model-name[:free]. Leave empty if using dropdown."
                ),
                io.String.Input(
                    "base_url",
                    default="https://openrouter.ai/api/v1/chat/completions",
                    multiline=False,
                    tooltip="OpenRouter API endpoint URL. Leave as default unless using a proxy or alternate endpoint."
                ),
                io.String.Input(
                    "system_prompt",
                    default="You are a helpful AI assistant. Please provide clear, accurate, and ethical responses.",
                    multiline=True,
                    tooltip="Optional system prompt to set the AI's behavior and context. Defines the assistant's role, personality, and guidelines."
                ),
                io.String.Input(
                    "user_prompt",
                    default="",
                    multiline=True,
                    tooltip="Main prompt or question for the model. For vision models, describe what you want to know about the image. Required field."
                ),
                SendSystemEnum.Input(
                    "send_system",
                    default="yes",
                    tooltip="Toggle system prompt sending. Set to 'no' if the model doesn't support system prompts or you want to skip it."
                ),
                io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Controls response randomness and creativity. Lower values (0.0-0.3) = more focused and deterministic. Higher values (0.7-2.0) = more creative and varied."
                ),
                io.Float.Input(
                    "top_p",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Nucleus sampling threshold. Controls diversity of word choices. Lower values (0.0-0.3) = more focused vocabulary. Higher values (0.7-1.0) = more diverse word selection."
                ),
                io.Int.Input(
                    "top_k",
                    default=50,
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="Limits vocabulary to top K most likely tokens. Lower values = more focused. Higher values = more diverse. 50 is a balanced default. Range: 1-1000."
                ),
                io.Int.Input(
                    "max_tokens",
                    default=1000,
                    min=1,
                    max=32768,
                    step=1,
                    tooltip="Maximum number of tokens to generate in the response. Note: actual limit varies by model. Higher values allow longer responses. Range: 1-32,768."
                ),
                io.Float.Input(
                    "frequency_penalty",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Penalizes tokens based on their frequency in the output. Positive values reduce word repetition. Range: -2.0 to 2.0. 0.0 = no penalty."
                ),
                io.Float.Input(
                    "presence_penalty",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Penalizes tokens that have already appeared in the output. Positive values encourage topic diversity. Range: -2.0 to 2.0. 0.0 = no penalty."
                ),
                io.Float.Input(
                    "repetition_penalty",
                    default=1.1,
                    min=1.0,
                    max=2.0,
                    step=0.01,
                    tooltip="OpenRouter-specific repetition penalty. Values > 1.0 reduce repetition. 1.0 = off. Higher values = stronger penalty. Range: 1.0-2.0."
                ),
                ResponseFormatEnum.Input(
                    "response_format",
                    default="text",
                    tooltip="Response format: 'text' for natural language, 'json_object' for structured JSON output. When using JSON, instruct the model in your prompt to output JSON."
                ),
                SeedModeEnum.Input(
                    "seed_mode",
                    default="random",
                    tooltip="Seed behavior control: 'fixed' uses the seed_value below, 'random' generates new seed each time, 'increment' increases by 1, 'decrement' decreases by 1."
                ),
                io.Int.Input(
                    "seed_value",
                    default=0,
                    min=0,
                    max=9007199254740991,
                    step=1,
                    tooltip="Seed value for reproducibility when seed_mode is 'fixed'. Use same seed + parameters for similar outputs. Valid range: 0-9007199254740991 (JavaScript safe integer limit)."
                ),
                io.Int.Input(
                    "max_retries",
                    default=3,
                    min=0,
                    max=5,
                    step=1,
                    tooltip="Maximum number of automatic retry attempts for recoverable errors (rate limits, temporary server issues). 0 disables retries. Range: 0-5."
                ),
                DebugModeEnum.Input(
                    "debug_mode",
                    default="off",
                    tooltip="Enable detailed error messages and request debugging information. Useful for troubleshooting API issues or parameter problems."
                ),
                io.Image.Input(
                    "image_input",
                    optional=True,
                    tooltip="Optional image input for vision-capable models. Supported models: llama-3.2-*-vision, llama-4-maverick, llama-4-scout. Maximum size: 2048x2048."
                ),
                io.String.Input(
                    "additional_params",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Additional OpenRouter API parameters in JSON format. Example: {\"min_p\": 0.1, \"top_a\": 0.5}. Use for advanced model-specific parameters not exposed in the UI."
                )
            ],
            outputs=[
                io.String.Output(
                    "response",
                    tooltip="The model's generated text or JSON response"
                ),
                io.String.Output(
                    "status",
                    tooltip="Detailed information about the request including model used, seed value, and token counts"
                ),
                io.String.Output(
                    "help",
                    tooltip="Static help text with usage information and repository URL"
                )
            ],
            is_output_node=True
        )

    @classmethod
    def validate_inputs(cls, api_key, model, manual_model, user_prompt, base_url, **kwargs):
        """Validate inputs before execution"""
        # Validate API key
        if not api_key or not api_key.strip():
            return "OpenRouter API key is required. Get one at https://openrouter.ai/keys"

        # Validate model selection
        actual_model = manual_model if model == "Manual Input" else model
        if model == "Manual Input" and (not manual_model or not manual_model.strip()):
            return "Manual model identifier is required when 'Manual Input' is selected"

        # Validate user prompt
        if not user_prompt or not user_prompt.strip():
            return "User prompt is required"

        # Validate base URL
        if not base_url or not base_url.strip():
            return "OpenRouter API endpoint URL is required"

        if not base_url.startswith(("http://", "https://")):
            return "Invalid API endpoint URL format (must start with http:// or https://)"

        # Validate additional_params if provided
        additional_params = kwargs.get("additional_params", "")
        if additional_params and additional_params.strip():
            try:
                json.loads(additional_params)
            except json.JSONDecodeError:
                return "Invalid JSON in additional parameters. Example format: {\"top_a\": 0.5}"

        return True

    @classmethod
    def execute(
        cls,
        api_key: str,
        model: str,
        manual_model: str,
        base_url: str,
        system_prompt: str,
        user_prompt: str,
        send_system: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        frequency_penalty: float,
        presence_penalty: float,
        repetition_penalty: float,
        response_format: str,
        seed_mode: str,
        seed_value: int,
        max_retries: int,
        debug_mode: str,
        image_input=None,
        additional_params: str = ""
    ) -> io.NodeOutput:
        """
        Execute chat completion request to OpenRouter API
        """

        help_text = """ComfyUI-EACloudNodes - OpenRouter Chat (v3)
Repository: https://github.com/EnragedAntelope/ComfyUI-EACloudNodes

Key Settings:
- API Key: Get from https://openrouter.ai/keys
- Model: Choose from free models or use Manual Input
  * Default: meta-llama/llama-3.3-70b-instruct:free
  * Vision: llama-3.2-*-vision, llama-4-maverick/scout
  * Google: gemini-2.5-pro, gemini-2.0-flash variants
  * Other: Mistral, Qwen, DeepSeek, Phi, and more
- Manual Model: Custom model ID (provider/model-name[:free])
- Base URL: API endpoint (usually leave as default)
- System Prompt: Set AI behavior/context
- User Prompt: Main input for the model (required)
- Send System: Toggle system prompt on/off
- Temperature: 0.0 (focused) to 2.0 (creative)
- Top-p: Nucleus sampling threshold (0.0-1.0)
- Top-k: Vocabulary limit (1-1000)
- Max Tokens: Response length limit (1-32,768)
- Frequency Penalty: Reduce token frequency (-2.0 to 2.0)
- Presence Penalty: Encourage topic diversity (-2.0 to 2.0)
- Repetition Penalty: Reduce repetition (1.0-2.0, 1.0=off)
- Response Format: Text or JSON object output
- Seed Mode: Fixed/random/increment/decrement for reproducibility
- Seed Value: Seed for 'fixed' mode (0-9007199254740991)
- Max Retries: Auto-retry on errors (0-5)
- Debug Mode: Enable for detailed error messages

Optional:
- Image Input: For vision-capable models only
  * llama-3.2-11b-vision-instruct:free
  * llama-3.2-90b-vision-instruct:free
  * llama-4-maverick-17b-128e-instruct:free
  * llama-4-scout-17b-16e-instruct:free
  * Max size: 2048x2048
- Additional Params: Extra model parameters in JSON

Vision Models:
1. Select a vision-capable model (contains 'vision' in name)
2. Connect an image to image_input
3. Describe what you want to know about the image in user_prompt

Free Models:
All models in the dropdown include ':free' suffix for free-tier access.
OpenRouter provides free access to many models with rate limits.

For full documentation and examples, visit:
https://github.com/EnragedAntelope/ComfyUI-EACloudNodes"""

        try:
            # Sanitize and validate numeric inputs
            try:
                temperature = max(0.0, min(2.0, float(temperature)))
                top_p = max(0.0, min(1.0, float(top_p)))
                top_k = max(1, min(1000, int(top_k)))
                max_tokens = max(1, min(32768, int(max_tokens)))
                frequency_penalty = max(-2.0, min(2.0, float(frequency_penalty)))
                presence_penalty = max(-2.0, min(2.0, float(presence_penalty)))
                repetition_penalty = max(1.0, min(2.0, float(repetition_penalty)))
                max_retries = max(0, min(5, int(max_retries)))
                seed_value = max(0, min(cls.MAX_SAFE_INTEGER, int(seed_value)))
            except (ValueError, TypeError) as e:
                return io.NodeOutput("", f"Error: Invalid parameter value - {str(e)}", help_text)

            # Use manual_model if "Manual Input" is selected
            actual_model = manual_model.strip() if model == "Manual Input" else model

            # Handle seed based on mode
            node_id = id(cls)  # Use class id as a simple identifier
            if seed_mode == "random":
                seed = random.randint(0, cls.MAX_SAFE_INTEGER)
            elif seed_mode == "increment":
                last_seed = cls._last_seed.get(node_id, 0)
                seed = (last_seed + 1) % cls.MAX_SAFE_INTEGER
            elif seed_mode == "decrement":
                last_seed = cls._last_seed.get(node_id, 0)
                seed = (last_seed - 1) if last_seed > 0 else cls.MAX_SAFE_INTEGER
            else:  # "fixed"
                seed = seed_value

            # Store the seed we're using
            cls._last_seed[node_id] = seed

            # Check if model supports vision capabilities
            is_vision_model = "vision" in actual_model.lower() or actual_model in cls.VISION_MODELS

            # Vision model validation
            if image_input is not None and not is_vision_model:
                return io.NodeOutput(
                    "",
                    f"Warning: Model '{actual_model}' may not support vision inputs. Consider using a model with 'vision' in its name. Vision-capable models: {', '.join(cls.VISION_MODELS)}",
                    help_text
                )

            # Prepare headers
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Initialize messages list
            messages = []

            # Add system prompt if provided and enabled
            if system_prompt and system_prompt.strip() and send_system == "yes":
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Handle image input if provided
            if image_input is not None:
                try:
                    # Process image for vision models
                    if isinstance(image_input, torch.Tensor):
                        if image_input.dim() == 4:
                            image_input = image_input.squeeze(0)
                        if image_input.dim() != 3:
                            return io.NodeOutput("", "Error: Image tensor must be 3D after squeezing", help_text)

                        if image_input.shape[-1] in [1, 3, 4]:
                            image_input = image_input.permute(2, 0, 1)

                        pil_image = ToPILImage()(image_input)
                    elif isinstance(image_input, Image.Image):
                        pil_image = image_input
                    else:
                        return io.NodeOutput("", "Error: Unsupported image input type", help_text)

                    # Add size validation
                    if pil_image.size[0] * pil_image.size[1] > 2048 * 2048:
                        return io.NodeOutput(
                            "",
                            "Error: Image too large. Maximum size is 2048x2048. Please resize your image.",
                            help_text
                        )

                    # Convert image to base64
                    buffered = python_io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    # Add user message with image for vision models
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                        ]
                    })
                except Exception as img_err:
                    return io.NodeOutput("", f"Image Processing Error: {str(img_err)}", help_text)
            else:
                # Add text-only user message
                messages.append({
                    "role": "user",
                    "content": user_prompt
                })

            # Prepare request body
            body = {
                "model": actual_model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }

            # Add optional parameters
            if top_k != 50:
                body["top_k"] = top_k

            if frequency_penalty != 0:
                body["frequency_penalty"] = frequency_penalty

            if presence_penalty != 0:
                body["presence_penalty"] = presence_penalty

            if repetition_penalty != 1.0:
                body["repetition_penalty"] = repetition_penalty

            if seed is not None:
                body["seed"] = seed

            # Add response format if json_object is selected
            if response_format == "json_object":
                body["response_format"] = {"type": "json_object"}

            # Parse and add additional parameters if provided
            if additional_params and additional_params.strip():
                try:
                    extra_params = json.loads(additional_params)
                    body.update(extra_params)
                except json.JSONDecodeError:
                    return io.NodeOutput("", "Error: Invalid JSON in additional parameters. Example format: {\"top_a\": 0.5}", help_text)

            # Make API request with retry logic
            retries = 0
            while True:
                try:
                    response = requests.post(base_url, headers=headers, json=body, timeout=120)

                    # Define retryable status codes
                    retryable_codes = {429, 500, 502, 503, 504}

                    if response.status_code in retryable_codes and retries < max_retries:
                        retries += 1
                        time.sleep(2 ** retries)  # Exponential backoff: 2, 4, 8, 16... seconds
                        continue

                    # Handle 400 errors with detailed information
                    if response.status_code == 400:
                        try:
                            error_json = response.json()
                            error_message = error_json.get("error", {}).get("message", "Unknown error")

                            if debug_mode == "on":
                                return io.NodeOutput(
                                    "",
                                    f"Error 400: {error_message}\n\nRequest body:\n{json.dumps(body, indent=2)}",
                                    help_text
                                )
                            else:
                                return io.NodeOutput("", f"Error 400: {error_message}", help_text)
                        except Exception:
                            return io.NodeOutput(
                                "",
                                "Error: Bad request - check model name and parameters (enable debug mode for details)",
                                help_text
                            )

                    # Handle other response codes
                    if response.status_code == 401:
                        return io.NodeOutput("", "Error: Invalid API key or unauthorized access", help_text)
                    elif response.status_code == 413:
                        return io.NodeOutput("", "Error: Payload too large - try reducing prompt or image size", help_text)
                    elif response.status_code == 429:
                        return io.NodeOutput("", f"Error: Rate limit exceeded. Tried {retries} times", help_text)
                    elif response.status_code in {500, 502, 503, 504}:
                        return io.NodeOutput("", f"Error: OpenRouter service error (status {response.status_code}). Tried {retries} times", help_text)
                    elif response.status_code != 200:
                        return io.NodeOutput("", f"Error: API returned status {response.status_code}. Tried {retries} times", help_text)

                    response_json = response.json()

                    # Extract information for status
                    model_used = response_json.get("model", "unknown")
                    tokens = response_json.get("usage", {})
                    prompt_tokens = tokens.get("prompt_tokens", 0)
                    completion_tokens = tokens.get("completion_tokens", 0)
                    total_tokens = prompt_tokens + completion_tokens

                    status_msg = f"Success: Model={model_used} | Seed={seed} | Tokens: {prompt_tokens}+{completion_tokens}={total_tokens}"

                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        content = response_json["choices"][0].get("message", {}).get("content", "")
                        return io.NodeOutput(content, status_msg, help_text)
                    else:
                        return io.NodeOutput("", "Error: No response content from the model", help_text)

                except requests.exceptions.Timeout:
                    if retries < max_retries:
                        retries += 1
                        time.sleep(2 ** retries)
                        continue
                    return io.NodeOutput("", f"Error: Request timed out after {retries} tries. Please try again", help_text)
                except requests.exceptions.RequestException as req_err:
                    # Retry network-related errors
                    if retries < max_retries:
                        retries += 1
                        time.sleep(2 ** retries)
                        continue
                    return io.NodeOutput("", f"Network Error: {str(req_err)}. Tried {retries} times.", help_text)
                except json.JSONDecodeError:
                    return io.NodeOutput("", "Error: Invalid JSON response from OpenRouter", help_text)

                # Break out of retry loop
                break

        except Exception as e:
            return io.NodeOutput("", f"Unexpected Error: {str(e)}", help_text)


class OpenRouterExtension(ComfyExtension):
    """Extension class for OpenRouter nodes"""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [OpenrouterNode]


async def comfy_entrypoint() -> ComfyExtension:
    """Entry point for ComfyUI v3"""
    return OpenRouterExtension()


# Legacy v1 compatibility (for nodes that still use old API)
NODE_CLASS_MAPPINGS = {
    "OpenrouterNode": OpenrouterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenrouterNode": "OpenRouter Chat"
}
