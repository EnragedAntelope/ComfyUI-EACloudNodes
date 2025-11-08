"""
Groq Chat Node for ComfyUI v3
Supports text and vision-language models through Groq's API.
"""

import json
import requests
import base64
from PIL import Image
import io as python_io
import torch
from torchvision.transforms import ToPILImage
import random

from comfy_api.latest import ComfyExtension, io


class GroqModelEnum(io.ComboInput):
    """Enum for Groq model selection"""
    OPTIONS = [
        # Production Models
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-guard-4-12b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "whisper-large-v3",
        "whisper-large-v3-turbo",
        # Production Systems
        "groq/compound",
        "groq/compound-mini",
        # Preview Models
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m",
        "moonshotai/kimi-k2-instruct-0905",
        "openai/gpt-oss-safeguard-20b",
        "playai-tts",
        "playai-tts-arabic",
        "qwen/qwen3-32b",
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


class GroqNode(io.ComfyNode):
    """
    A node for interacting with Groq's API.
    Supports text and vision-language models through Groq's API.
    """

    # JavaScript safe integer limit (2^53 - 1)
    MAX_SAFE_INTEGER = 9007199254740991

    # Models that support vision capabilities
    VISION_MODELS = [
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]

    # Class-level storage for seed tracking (since nodes are stateless)
    _last_seed = {}

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GroqNode",
            display_name="Groq Chat",
            category="Groq",
            description="Interact with Groq's API for ultra-fast inference with various LLM models. Supports text generation, JSON output, and vision analysis with compatible models.",
            inputs=[
                io.String.Input(
                    "api_key",
                    default="",
                    multiline=False,
                    tooltip="⚠️ Your Groq API key from https://console.groq.com/keys (Note: key will be visible - take care when sharing workflows)"
                ),
                GroqModelEnum.Input(
                    "model",
                    default="llama-3.3-70b-versatile",
                    tooltip="Select a Groq model or choose 'Manual Input' to specify a custom model. Production models are stable; Preview models are for evaluation only."
                ),
                io.String.Input(
                    "manual_model",
                    default="",
                    multiline=False,
                    tooltip="Enter a custom model identifier (only used when 'Manual Input' is selected above). Leave empty if using dropdown selection."
                ),
                io.String.Input(
                    "system_prompt",
                    default="You are a helpful AI assistant. Please provide clear, accurate, and ethical responses.",
                    multiline=True,
                    tooltip="Optional system prompt to set the AI's behavior and context. Note: Vision models may not support system prompts - toggle 'send_system' to 'no' if needed."
                ),
                io.String.Input(
                    "user_prompt",
                    default="",
                    multiline=True,
                    tooltip="Main prompt or question for the model. For vision tasks, describe what you want to know about the image."
                ),
                SendSystemEnum.Input(
                    "send_system",
                    default="yes",
                    tooltip="Toggle system prompt sending. Set to 'no' for vision models that don't accept system prompts (e.g., Llama-4 vision models)."
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
                    "max_completion_tokens",
                    default=1000,
                    min=1,
                    max=131072,
                    step=1,
                    tooltip="Maximum number of tokens to generate in the response. Note: actual limit varies by model (check model documentation). Range: 1-131,072."
                ),
                io.Float.Input(
                    "frequency_penalty",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Penalizes tokens based on their frequency in the output. Positive values reduce repetition. Range: -2.0 to 2.0. Note: not all models support this parameter."
                ),
                io.Float.Input(
                    "presence_penalty",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Penalizes tokens that have already appeared in the output. Positive values encourage topic diversity. Range: -2.0 to 2.0. Note: not all models support this parameter."
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
                    tooltip="Seed value for reproducibility when seed_mode is 'fixed'. Use same seed + parameters for identical outputs. Valid range: 0-9007199254740991 (JavaScript safe integer limit)."
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
                    tooltip="Optional image input for vision-capable models. Currently supported models: meta-llama/llama-4-maverick-17b-128e-instruct and meta-llama/llama-4-scout-17b-16e-instruct."
                ),
                io.String.Input(
                    "additional_params",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Additional Groq API parameters in JSON format. Example: {\"stop\": [\"\\n\"], \"min_p\": 0.1}. Use for advanced model-specific parameters not exposed in the UI."
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
    def validate_inputs(cls, api_key, model, manual_model, user_prompt, **kwargs):
        """Validate inputs before execution"""
        # Validate API key
        if not api_key or not api_key.strip():
            return "Groq API key is required. Get one at https://console.groq.com/keys"

        # Validate model selection
        actual_model = manual_model if model == "Manual Input" else model
        if model == "Manual Input" and (not manual_model or not manual_model.strip()):
            return "Manual model identifier is required when 'Manual Input' is selected"

        # Validate user prompt
        if not user_prompt or not user_prompt.strip():
            return "User prompt is required"

        # Validate additional_params if provided
        additional_params = kwargs.get("additional_params", "")
        if additional_params and additional_params.strip():
            try:
                json.loads(additional_params)
            except json.JSONDecodeError:
                return "Invalid JSON in additional parameters. Example format: {\"stop\": [\"\\n\"]}"

        return True

    @classmethod
    def execute(
        cls,
        api_key: str,
        model: str,
        manual_model: str,
        system_prompt: str,
        user_prompt: str,
        send_system: str,
        temperature: float,
        top_p: float,
        max_completion_tokens: int,
        frequency_penalty: float,
        presence_penalty: float,
        response_format: str,
        seed_mode: str,
        seed_value: int,
        max_retries: int,
        debug_mode: str,
        image_input=None,
        additional_params: str = ""
    ) -> io.NodeOutput:
        """
        Execute chat completion request to Groq API
        """

        help_text = """ComfyUI-EACloudNodes - Groq Chat (v3)
Repository: https://github.com/EnragedAntelope/ComfyUI-EACloudNodes

Key Settings:
- API Key: Get from https://console.groq.com/keys
- Model: Choose from dropdown or use Manual Input
  * Production: llama-3.3-70b-versatile (default), llama-3.1-8b-instant, etc.
  * Preview: llama-4-maverick/scout (vision), qwen3-32b, etc.
- System Prompt: Set AI behavior/context (disable for vision models)
- User Prompt: Main input for the model
- Send System: Toggle system prompt (off for vision models)
- Temperature: 0.0 (focused) to 2.0 (creative)
- Top-p: Nucleus sampling threshold (0.0-1.0)
- Max Tokens: Response length limit (varies by model)
- Frequency Penalty: Reduce token frequency (-2.0 to 2.0)
- Presence Penalty: Encourage topic diversity (-2.0 to 2.0)
- Response Format: Text or JSON object output
- Seed Mode: Fixed/random/increment/decrement for reproducibility
- Seed Value: Seed for 'fixed' mode (0-9007199254740991)
- Max Retries: Auto-retry on errors (0-5)
- Debug Mode: Enable for detailed error messages

Optional:
- Image Input: For Llama-4 vision models only
  * meta-llama/llama-4-maverick-17b-128e-instruct
  * meta-llama/llama-4-scout-17b-16e-instruct
- Additional Params: Extra model parameters in JSON

Vision Models:
1. Connect an image to image_input
2. Select a vision-capable model (Llama-4 Maverick or Scout)
3. Set 'send_system' to 'no'
4. Describe what you want to know about the image in user_prompt

Production vs Preview Models:
- Production: Stable, reliable, recommended for production use
- Preview: Experimental, may be deprecated, for evaluation only

For full documentation and examples, visit:
https://github.com/EnragedAntelope/ComfyUI-EACloudNodes"""

        try:
            # Sanitize and validate numeric inputs
            try:
                temperature = max(0.0, min(2.0, float(temperature)))
                top_p = max(0.0, min(1.0, float(top_p)))
                max_completion_tokens = max(1, min(131072, int(max_completion_tokens)))
                frequency_penalty = max(-2.0, min(2.0, float(frequency_penalty)))
                presence_penalty = max(-2.0, min(2.0, float(presence_penalty)))
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
            is_vision_model = actual_model in cls.VISION_MODELS

            # Vision model validation
            if image_input is not None and not is_vision_model:
                return io.NodeOutput(
                    "",
                    f"Error: Model '{actual_model}' does not support vision inputs. Only the following Groq models support vision: {', '.join(cls.VISION_MODELS)}",
                    help_text
                )

            # Initialize messages list
            messages = []

            # Add system prompt if provided and enabled
            if system_prompt and system_prompt.strip() and send_system == "yes":
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Handle different message formats based on whether it's a vision model with image
            if image_input is not None and is_vision_model:
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

            # Prepare request body with only supported parameters
            body = {
                "model": actual_model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_completion_tokens
            }

            # Add seed
            if seed is not None:
                body["seed"] = seed

            # Only add penalty parameters if non-zero (not all models support them)
            if frequency_penalty != 0:
                body["frequency_penalty"] = frequency_penalty

            if presence_penalty != 0:
                body["presence_penalty"] = presence_penalty

            # Add response format if json_object is selected
            if response_format == "json_object":
                body["response_format"] = {"type": "json_object"}

            # Parse and add additional parameters if provided
            if additional_params and additional_params.strip():
                try:
                    extra_params = json.loads(additional_params)
                    body.update(extra_params)
                except json.JSONDecodeError:
                    return io.NodeOutput("", "Error: Invalid JSON in additional parameters. Example format: {\"stop\": [\"\\n\"]}", help_text)

            # Make API request with retry logic
            retries = 0
            while True:
                try:
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json=body,
                        timeout=120
                    )

                    # Define retryable status codes
                    retryable_codes = {429, 500, 502, 503, 504}

                    if response.status_code in retryable_codes and retries < max_retries:
                        retries += 1
                        import time
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
                        return io.NodeOutput("", "Error: Invalid API key", help_text)
                    elif response.status_code == 429:
                        return io.NodeOutput("", f"Error: Rate limit exceeded. Tried {retries} times", help_text)
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
                        return io.NodeOutput("", "Error: No response content from model", help_text)

                except requests.exceptions.RequestException as req_err:
                    # Retry network-related errors
                    if retries < max_retries:
                        retries += 1
                        import time
                        time.sleep(2 ** retries)
                        continue
                    return io.NodeOutput("", f"Network Error: {str(req_err)}. Tried {retries} times.", help_text)

                # Break out of retry loop
                break

        except Exception as e:
            return io.NodeOutput("", f"Unexpected Error: {str(e)}", help_text)


class GroqExtension(ComfyExtension):
    """Extension class for Groq nodes"""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [GroqNode]


async def comfy_entrypoint() -> ComfyExtension:
    """Entry point for ComfyUI v3"""
    return GroqExtension()


# Legacy v1 compatibility (for nodes that still use old API)
NODE_CLASS_MAPPINGS = {
    "GroqNode": GroqNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroqNode": "Groq Chat"
}
