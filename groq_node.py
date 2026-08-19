"""
Groq Chat Node for ComfyUI v3
Supports text and vision-language models through Groq's API.
"""

import json
import requests
import base64
import time
from PIL import Image
import io as python_io
import torch
from torchvision.transforms import ToPILImage
import random

from comfy_api.latest import ComfyExtension, io

# ============================================================================
# MODULE-LEVEL CONSTANTS (Dynamic Model Fetching)
# ============================================================================

# Module-level cache for dynamically fetched models (5-minute TTL)
_groq_model_cache = {
    "models": None,
    "vision_models": None,
    "last_fetch": 0,
    "cache_ttl": 300,  # 5 minutes
    "last_failure": 0,
    "failure_backoff": 60  # don't re-try a failing fetch on every execution
}

# Model categorization mapping (hybrid approach - applied to fetched models)
MODEL_CATEGORIES = {
    "Featured": ["groq/compound", "openai/gpt-oss-120b"],
    "Production: Chat": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"],
    "Production: Systems": ["groq/compound-mini"],
    "Preview: Chat": [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-safeguard-20b",
        "qwen/qwen3-32b",
    ],
    "Preview: Safety": [
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m",
    ],
}

# Speech-to-text and text-to-speech models are served by /audio/transcriptions and
# /audio/speech, not /chat/completions, so they can never work in this node.
AUDIO_MODEL_PATTERNS = ["whisper", "orpheus", "playai-tts", "tts-"]

# Prefix used for the non-selectable category separators in the model dropdown.
CATEGORY_SEPARATOR_PREFIX = "---"

# Known vision models (hybrid detection: hardcoded list + pattern matching)
KNOWN_VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
]
VISION_PATTERNS = ["vision", "vl", "-4-"]  # Patterns for detecting unknown vision models

# Static fallback list (used when API unavailable)
STATIC_FALLBACK_MODELS = [
    "--- Featured ---",
    "groq/compound",
    "openai/gpt-oss-120b",
    "--- Production: Chat ---",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "--- Production: Systems ---",
    "groq/compound-mini",
    "--- Preview: Chat ---",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3-32b",
    "--- Preview: Safety ---",
    "meta-llama/llama-prompt-guard-2-22m",
    "meta-llama/llama-prompt-guard-2-86m",
    "Manual Input",
]


# ============================================================================
# MODULE-LEVEL FUNCTIONS (Dynamic Model Fetching)
# ============================================================================

def _is_audio_model(model_id: str) -> bool:
    """True for speech-to-text / text-to-speech models, which this node cannot call."""
    lowered = model_id.lower()
    return any(pattern in lowered for pattern in AUDIO_MODEL_PATTERNS)


def _is_category_separator(model_id: str) -> bool:
    """True for the '--- Category ---' rows used to group the dropdown."""
    return model_id.strip().startswith(CATEGORY_SEPARATOR_PREFIX)


def _get_static_fallback_models() -> tuple[list[str], list[str]]:
    """Return comprehensive static fallback list."""
    return STATIC_FALLBACK_MODELS.copy(), KNOWN_VISION_MODELS.copy()


def _categorize_groq_models(api_models: list[dict]) -> list[str]:
    """
    Apply hardcoded categorization to fetched models.
    Models not in mapping go to 'Other' category.
    """
    # Build reverse mapping: model_id -> category
    model_to_category = {}
    for category, model_list in MODEL_CATEGORIES.items():
        for model_id in model_list:
            model_to_category[model_id] = category
    
    # Group fetched models by category
    categorized = {cat: [] for cat in MODEL_CATEGORIES.keys()}
    categorized["Other"] = []
    
    for model in api_models:
        model_id = model.get("id", "")
        if not model_id or not model.get("active", True):
            continue

        # Audio models use different endpoints and cannot be called from this node
        if _is_audio_model(model_id):
            continue

        # Find matching category
        if model_id in model_to_category:
            categorized[model_to_category[model_id]].append(model_id)
        else:
            categorized["Other"].append(model_id)
    
    # Build final list with category headers
    result = []
    for category in MODEL_CATEGORIES.keys():
        if categorized[category]:
            result.append(f"--- {category} ---")
            result.extend(sorted(categorized[category]))
    
    if categorized["Other"]:
        result.append("--- Other ---")
        result.extend(sorted(categorized["Other"]))
    
    result.append("Manual Input")
    return result


def _detect_vision_models(api_models: list[dict]) -> list[str]:
    """
    Detect vision-capable models using hybrid approach:
    1. Include all KNOWN_VISION_MODELS that exist in API response
    2. Pattern-match model IDs for vision indicators
    """
    vision_models = []
    
    for model in api_models:
        model_id = model.get("id", "")
        if not model_id or not model.get("active", True) or _is_audio_model(model_id):
            continue

        # Check hardcoded list
        if model_id in KNOWN_VISION_MODELS:
            vision_models.append(model_id)
            continue
        
        # Pattern matching
        if any(pattern in model_id.lower() for pattern in VISION_PATTERNS):
            vision_models.append(model_id)
    
    return vision_models


def _fetch_groq_models(api_key: str = None) -> tuple[list[str], list[str]]:
    """
    Fetch available models from Groq API with 5-minute caching.
    
    Args:
        api_key: Optional Groq API key. If not provided, returns static fallback.
    
    Returns:
        tuple: (categorized_model_list, vision_model_list)
               Returns static fallback if API call fails or no key provided.
    """
    now = time.time()

    # Return cached results if still fresh
    if (_groq_model_cache["models"] is not None and
            now - _groq_model_cache["last_fetch"] < _groq_model_cache["cache_ttl"]):
        return _groq_model_cache["models"], _groq_model_cache["vision_models"]

    # If no API key, return static fallback
    if not api_key or not api_key.strip():
        return _get_static_fallback_models()

    # Back off after a failure too, so a broken key or an offline host does not
    # add a request (and its timeout) to every single node execution.
    if now - _groq_model_cache["last_failure"] < _groq_model_cache["failure_backoff"]:
        if _groq_model_cache["models"] is not None:
            return _groq_model_cache["models"], _groq_model_cache["vision_models"]
        return _get_static_fallback_models()

    try:
        # Fetch from Groq API
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5
        )
        
        if response.status_code != 200:
            raise Exception(f"API returned status {response.status_code}")
        
        data = response.json().get("data", [])
        
        # Build categorized model list
        categorized_models = _categorize_groq_models(data)
        
        # Detect vision-capable models
        vision_models = _detect_vision_models(data)
        
        # Update cache
        _groq_model_cache["models"] = categorized_models
        _groq_model_cache["vision_models"] = vision_models
        _groq_model_cache["last_fetch"] = now
        
        return categorized_models, vision_models
        
    except Exception:
        _groq_model_cache["last_failure"] = now
        # Return previously cached results if available
        if _groq_model_cache["models"] is not None:
            return _groq_model_cache["models"], _groq_model_cache["vision_models"]
        # Return static fallback
        return _get_static_fallback_models()


# ============================================================================
# GROQ NODE CLASS
# ============================================================================

class GroqNode(io.ComfyNode):
    """
    A node for interacting with Groq's API.
    Supports text and vision-language models through Groq's API.
    """

    # JavaScript safe integer limit (2^53 - 1)
    MAX_SAFE_INTEGER = 9007199254740991

    # Class-level storage for seed counters, keyed by (model, starting seed)
    _last_seed = {}

    # Upper bound on tracked seed counters, so the dict cannot grow without bound
    MAX_TRACKED_SEEDS = 1024

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GroqNode",
            display_name="Groq Chat",
            category="Groq",
            description="Interact with Groq's API for ultra-fast inference. Model list dynamically fetched from Groq API (5-min cache). Supports text generation, JSON output, and vision analysis with compatible models.",
            inputs=[
                io.String.Input(
                    "api_key",
                    default="",
                    multiline=False,
                    tooltip="⚠️ Your Groq API key from https://console.groq.com/keys (Note: key will be visible - take care when sharing workflows)"
                ),
                io.Combo.Input(
                    "model",
                options=_fetch_groq_models(api_key=None)[0],
                    default="llama-3.3-70b-versatile",
                tooltip="Select a Groq model or choose 'Manual Input'. Categories: Featured, Production (stable), Preview (evaluation). Use ComfyUI Refresh to update model list from Groq API."
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
                io.Combo.Input(
                    "send_system",
                    options=["yes", "no"],
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
                io.Combo.Input(
                    "response_format",
                    options=["text", "json_object"],
                    default="text",
                    tooltip="Response format: 'text' for natural language, 'json_object' for structured JSON output. When using JSON, instruct the model in your prompt to output JSON."
                ),
                io.Combo.Input(
                    "seed_mode",
                    options=["fixed", "random", "increment", "decrement"],
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
                io.Combo.Input(
                    "debug_mode",
                    options=["off", "on"],
                    default="off",
                    tooltip="Enable detailed error messages and request debugging information. Useful for troubleshooting API issues or parameter problems."
                ),
                io.Image.Input(
                    "image_input",
                    optional=True,
                    tooltip="Optional image input for vision-capable models. Currently supported: meta-llama/llama-4-scout-17b-16e-instruct. Maximum size: 2048x2048."
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
                    display_name="response"
                ),
                io.String.Output(
                    display_name="status"
                ),
                io.String.Output(
                    display_name="help"
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
        if model == "Manual Input" and (not manual_model or not manual_model.strip()):
            return "Manual model identifier is required when 'Manual Input' is selected"

        # Category headers are group labels, not selectable models
        if _is_category_separator(model):
            return f"'{model}' is a category label, not a model. Please select a model from the dropdown."

        # Audio models are served by different Groq endpoints and cannot be used here
        actual_model = manual_model if model == "Manual Input" else model
        if actual_model and _is_audio_model(actual_model):
            return (
                f"'{actual_model}' is a speech model served by Groq's audio endpoints "
                "and cannot be used for chat completions. Please select a chat model."
            )

        # Validate additional_params if provided
        additional_params = kwargs.get("additional_params", "")
        if additional_params and additional_params.strip():
            try:
                parsed = json.loads(additional_params)
            except json.JSONDecodeError:
                return "Invalid JSON in additional parameters. Example format: {\"stop\": [\"\\n\"]}"
            if not isinstance(parsed, dict):
                return "Additional parameters must be a JSON object. Example format: {\"stop\": [\"\\n\"]}"

        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        """
        Equivalent of V1's IS_CHANGED.

        The seed is derived inside execute(), so with unchanged widgets ComfyUI would
        serve a cached response and the non-fixed seed modes would never take effect.
        Returning NaN marks the node dirty whenever the seed is meant to move.
        """
        if kwargs.get("seed_mode", "fixed") != "fixed":
            return float("nan")
        return kwargs.get("seed_value", 0)

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
  * Also used to refresh the model list from the Groq API
- Model: Pick from the dropdown or choose 'Manual Input'
  * Featured: groq/compound, openai/gpt-oss-120b
  * Production: stable models (llama-3.3-70b-versatile is the default)
  * Preview: experimental models, may be deprecated without notice
  * Rows shown as '--- Category ---' are labels, not selectable models
  * Speech models (Whisper, Orpheus) are excluded: they use Groq's audio
    endpoints, not chat completions
- Manual Model: custom model id, used only when 'Manual Input' is selected
- System Prompt: sets AI behavior/context
- User Prompt: main input for the model (required)
- Send System: set to 'no' for vision models that reject system prompts
- Temperature: 0.0 (focused) to 2.0 (creative)
- Top-p: nucleus sampling threshold (0.0-1.0)
- Max Completion Tokens: response length limit (varies by model)
- Frequency Penalty: reduce token frequency (-2.0 to 2.0), sent only when non-zero
- Presence Penalty: encourage topic diversity (-2.0 to 2.0), sent only when non-zero
- Response Format: text or JSON object output
- Seed Mode: fixed / random / increment / decrement
- Seed Value: seed used by 'fixed' mode (0-9007199254740991)
- Max Retries: auto-retry on rate limits and 5xx errors (0-5)
- Debug Mode: include the request body in 400-error messages

Optional:
- Image Input: for vision-capable models
  * Known: meta-llama/llama-4-scout-17b-16e-instruct
  * Also detected from model ids containing 'vision', 'vl', or '-4-'
  * Max size: 2048x2048 per dimension; only the first image of a batch is sent
- Additional Params: extra Groq parameters as a JSON object, merged into the
  request body (it overrides the widgets above on key collisions)

Vision Models:
1. Connect an image to image_input
2. Select a vision-capable model
3. Set 'send_system' to 'no' (vision models often reject system prompts)
4. Describe what you want to know about the image in user_prompt

Model List:
- Ships with a static list of known chat models
- Run the node once with a valid API key, then use ComfyUI's Refresh button to
  replace the dropdown with the live list from the Groq API (cached 5 minutes)
- Falls back to the static list whenever the API is unreachable

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

            # Validate user prompt (delayed until execute to handle connected inputs)
            if not user_prompt or not user_prompt.strip():
                return io.NodeOutput("", "User prompt is required", help_text)


            # Use manual_model if "Manual Input" is selected
            actual_model = manual_model.strip() if model == "Manual Input" else model

            if _is_category_separator(actual_model):
                return io.NodeOutput(
                    "",
                    f"Error: '{actual_model}' is a category label, not a model. Please select a model from the dropdown.",
                    help_text
                )

            if _is_audio_model(actual_model):
                return io.NodeOutput(
                    "",
                    f"Error: '{actual_model}' is a speech model served by Groq's audio endpoints "
                    "and cannot be used for chat completions. Please select a chat model.",
                    help_text
                )

            # Handle seed based on mode.
            # Counters are keyed by (model, starting seed) and capped so long-running
            # sessions cannot grow this dict without bound.
            node_key = (actual_model, seed_value)
            if seed_mode == "random":
                seed = random.randint(0, cls.MAX_SAFE_INTEGER)
            elif seed_mode == "increment":
                last_seed = cls._last_seed.get(node_key, seed_value)
                seed = (last_seed + 1) % cls.MAX_SAFE_INTEGER
            elif seed_mode == "decrement":
                last_seed = cls._last_seed.get(node_key, seed_value)
                seed = (last_seed - 1) if last_seed > 0 else cls.MAX_SAFE_INTEGER
            else:  # "fixed"
                seed = seed_value

            # Store the seed we're using
            if len(cls._last_seed) >= cls.MAX_TRACKED_SEEDS:
                cls._last_seed.clear()
            cls._last_seed[node_key] = seed

            # Check if model supports vision capabilities.
            # Passing the key here is what warms the module cache, so a later ComfyUI
            # Refresh can rebuild the dropdown from the live Groq model list.
            _, vision_models = _fetch_groq_models(api_key=api_key)
            if vision_models is None:
                vision_models = KNOWN_VISION_MODELS
            is_vision_model = (
                actual_model in vision_models or
                any(pattern in actual_model.lower() for pattern in VISION_PATTERNS)
            )

            # Vision model validation
            if image_input is not None and not is_vision_model:
                return io.NodeOutput(
                    "",
                    f"Error: Model '{actual_model}' does not support vision inputs. Vision-capable models are auto-detected from Groq API. Currently known: {', '.join(vision_models)}",
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
                        # ComfyUI IMAGE tensors are [batch, height, width, channels];
                        # take the first frame rather than failing on batches > 1.
                        if image_input.dim() == 4:
                            image_input = image_input[0]
                        if image_input.dim() != 3:
                            return io.NodeOutput(
                                "",
                                f"Error: Expected a 3D or 4D image tensor, got {image_input.dim()}D",
                                help_text
                            )

                        if image_input.shape[-1] in [1, 3, 4]:
                            image_input = image_input.permute(2, 0, 1)

                        image_input = image_input.cpu()
                        # ComfyUI IMAGE tensors are floats in 0..1; clamping keeps an
                        # out-of-range upstream result from wrapping around on convert.
                        # Integer tensors are already in 0..255 and must not be clamped.
                        if image_input.is_floating_point():
                            image_input = image_input.clamp(0, 1)
                        pil_image = ToPILImage()(image_input)
                    elif isinstance(image_input, Image.Image):
                        pil_image = image_input
                    else:
                        return io.NodeOutput("", "Error: Unsupported image input type", help_text)

                    # Validate image dimensions (max 2048 in either dimension)
                    if pil_image.size[0] > 2048 or pil_image.size[1] > 2048:
                        return io.NodeOutput(
                            "",
                            f"Error: Image too large ({pil_image.size[0]}x{pil_image.size[1]}). Maximum is 2048 pixels in either dimension. Please resize your image.",
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

            # Prepare request body with only supported parameters
            body = {
                "model": actual_model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                # Groq deprecated "max_tokens" in favour of "max_completion_tokens"
                "max_completion_tokens": max_completion_tokens
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
                except json.JSONDecodeError:
                    return io.NodeOutput("", "Error: Invalid JSON in additional parameters. Example format: {\"stop\": [\"\\n\"]}", help_text)
                if not isinstance(extra_params, dict):
                    return io.NodeOutput("", "Error: Additional parameters must be a JSON object. Example format: {\"stop\": [\"\\n\"]}", help_text)
                body.update(extra_params)

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

                except requests.exceptions.JSONDecodeError:
                    # A 200 with a malformed body is not worth retrying
                    return io.NodeOutput("", "Error: Invalid JSON response from Groq", help_text)
                except requests.exceptions.RequestException as req_err:
                    # Retry network-related errors
                    if retries < max_retries:
                        retries += 1
                        time.sleep(2 ** retries)
                        continue
                    return io.NodeOutput("", f"Network Error: {str(req_err)}. Tried {retries} times.", help_text)

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
