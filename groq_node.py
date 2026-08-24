"""
Groq Chat Node for ComfyUI v3
Supports text and vision-language models through Groq's API.
"""

import json
import requests
import time
from PIL import Image
import torch

# Relative: ComfyUI loads this folder as a package and never puts it on
# sys.path, so an absolute `import chat_common` fails at registration time.
from . import chat_common
from comfy_api.latest import io

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
# Curated chat-capable models, mirroring Groq's own Production/Preview grouping.
# Audio models are excluded because this node cannot call them at all; the
# prompt-guard classifiers are excluded because a 512-token safety classifier is
# not useful as a chat model here (they remain reachable via 'Manual Input').
MODEL_CATEGORIES = {
    "Featured": ["groq/compound", "openai/gpt-oss-120b"],
    "Production: Chat": ["openai/gpt-oss-20b"],
    "Production: Systems": ["groq/compound-mini"],
    "Preview: Chat": [
        "minimaxai/minimax-m2.7",
        "openai/gpt-oss-safeguard-20b",
        "qwen/qwen3.6-27b",
    ],
}

# Speech-to-text and text-to-speech models are served by /audio/transcriptions and
# /audio/speech, not /chat/completions, so they can never work in this node.
AUDIO_MODEL_PATTERNS = ["whisper", "orpheus", "playai-tts", "tts-"]

# Prefix used for the non-selectable category separators in the model dropdown.
CATEGORY_SEPARATOR_PREFIX = "---"

# Vision detection is advisory only - it shapes hints, never blocks a request.
# Order of authority: modality metadata from the live API, then the naming
# conventions below, then this seed list. Everything here may go stale without
# breaking anything, because an image is always sent and Groq itself decides.
KNOWN_VISION_MODELS = [
    "qwen/qwen3.6-27b",
]

# Field names Groq has used, or may use, to describe a model's accepted inputs.
MODALITY_FIELDS = ["input_modalities", "modalities", "supported_modalities"]

VISION_PATTERNS = ["vision", "vl", "multimodal", "omni"]

# Preferred defaults, most wanted first. The first one present in the resolved
# model list wins; if none survive a Groq reshuffle, the first real model does.
# This is what keeps a retired default from breaking the node out of the box.
PREFERRED_DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "groq/compound",
]

# Static fallback list (used when API unavailable)
STATIC_FALLBACK_MODELS = [
    "--- Featured ---",
    "groq/compound",
    "openai/gpt-oss-120b",
    "--- Production: Chat ---",
    "openai/gpt-oss-20b",
    "--- Production: Systems ---",
    "groq/compound-mini",
    "--- Preview: Chat ---",
    "minimaxai/minimax-m2.7",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3.6-27b",
    "Manual Input",
]


# ============================================================================
# MODULE-LEVEL FUNCTIONS (Dynamic Model Fetching)
# ============================================================================

def _is_audio_model(model_id: str, model: dict = None) -> bool:
    """
    True for speech-to-text / text-to-speech models, which this node cannot call.

    A model entry that positively reports text output is taken at its word, so a
    future chat model whose name happens to trip a pattern is never excluded.
    Naming conventions are only the fallback for entries carrying no metadata.
    """
    if model:
        for field in ("output_modalities", "output_modality"):
            value = model.get(field)
            if isinstance(value, list) and value:
                return not any(str(item).lower() == "text" for item in value)
            if isinstance(value, str) and value:
                return "text" not in value.lower()

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
        if _is_audio_model(model_id, model):
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


def _model_reports_image_input(model: dict):
    """
    Read image capability out of a Groq model entry's own metadata.

    Groq has not committed to a modality field, so several plausible names are
    checked. Returns None when the entry says nothing, which the caller treats
    as "unknown" rather than "no" - that is what stops this going stale.
    """
    for field in MODALITY_FIELDS:
        value = model.get(field)
        if isinstance(value, list) and value:
            return any(str(item).lower() == "image" for item in value)
        if isinstance(value, str) and value:
            # e.g. "text+image->text"; only the input side counts
            return "image" in value.split("->")[0].lower()
    return None


def _detect_vision_models(api_models: list[dict]) -> list[str]:
    """
    Collect the models that look vision-capable, most authoritative signal first:
    the entry's own modality metadata, then naming conventions, then the seed list.

    This only feeds hints, so a miss costs a less specific error message rather
    than a blocked request.
    """
    vision_models = []

    for model in api_models:
        model_id = model.get("id", "")
        if not model_id or not model.get("active", True) or _is_audio_model(model_id, model):
            continue

        reported = _model_reports_image_input(model)
        if reported is not None:
            if reported:
                vision_models.append(model_id)
            continue

        if (model_id in KNOWN_VISION_MODELS or
                any(pattern in model_id.lower() for pattern in VISION_PATTERNS)):
            vision_models.append(model_id)

    return vision_models


def _pick_default_model(models: list[str]) -> str:
    """
    Choose a default that exists in the given list.

    Groq retires models regularly - a hardcoded default that disappears makes the
    node fail on a fresh drop-in, so fall back through preferences and then to
    whatever the list actually offers.
    """
    for preferred in PREFERRED_DEFAULT_MODELS:
        if preferred in models:
            return preferred
    for model_id in models:
        if not _is_category_separator(model_id) and model_id != "Manual Input":
            return model_id
    return "Manual Input"


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

    with chat_common.LOCK:
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
        # Resolved together so the default is always one of the offered options,
        # whatever Groq's catalogue looks like on the day.
        model_options = _fetch_groq_models(api_key=None)[0]
        default_model = _pick_default_model(model_options)

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
                    options=model_options,
                    default=default_model,
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
                    tooltip="Toggle system prompt sending. Set to 'no' for models that reject system prompts, which is common for vision models."
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
                    tooltip="Optional image input for vision-capable models (qwen/qwen3.6-27b at time of writing). The image is always sent and Groq decides whether the model accepts it, so newly released vision models work without updating this node. Maximum size: 2048x2048 (only the first image of a batch is sent)."
                ),
                io.Combo.Input(
                    "image_format",
                    options=["png", "jpeg"],
                    default="png",
                    tooltip="Encoding for the attached image. PNG is lossless (best for screenshots and text); JPEG produces a much smaller request for photographic content, cutting latency and token overhead."
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
        image_format: str = "png",
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
  * Featured: groq/compound, openai/gpt-oss-120b (default)
  * Production: stable models (openai/gpt-oss-120b is the default)
  * Preview: experimental models, may be deprecated without notice
  * Rows shown as '--- Category ---' are labels, not selectable models
  * Speech models (Whisper, Orpheus) are excluded and rejected if entered:
    they use Groq's audio endpoints, not chat completions
  * The prompt-guard safety classifiers are left out of the curated list too
    (512-token classifiers); reach them with 'Manual Input' if you need them
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
- Debug Mode: include the request body in 400-error messages (image data
  is summarized rather than dumped)

Optional:
- Image Input: for vision-capable models (qwen/qwen3.6-27b at time of writing)
  * An attached image is ALWAYS sent. This node does not second-guess which
    models accept images - Groq decides, so a vision model released after this
    node shipped works immediately, with no update here
  * If Groq refuses it, the error carries a hint about what did look capable
  * Max size: 2048x2048 per dimension; only the first image of a batch is sent
- Image Format: PNG (lossless) or JPEG (smaller payload for photos)
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
            with chat_common.LOCK:
                seed = chat_common.derive_seed(
                    cls._last_seed, node_key, seed_mode, seed_value, cls.MAX_SAFE_INTEGER)
                chat_common.store_seed(
                    cls._last_seed, node_key, seed, cls.MAX_TRACKED_SEEDS)

            # Warm the module cache so a later ComfyUI Refresh can rebuild the
            # dropdown from the live Groq model list.
            _, vision_models = _fetch_groq_models(api_key=api_key)
            if vision_models is None:
                vision_models = KNOWN_VISION_MODELS

            # Deliberately NOT a gate. Refusing an image on the strength of a
            # capability list means every Groq model reshuffle silently blocks a
            # model that actually works. The image is always sent; Groq rejects it
            # if the model cannot take it, and that 400 is annotated below with
            # whichever models did look capable at the time.
            is_vision_model = (
                actual_model in vision_models or
                any(pattern in actual_model.lower() for pattern in VISION_PATTERNS)
            )

            # Appended to a 400 when an image was attached to a model that did not
            # look vision-capable, turning Groq's generic complaint into a lead.
            vision_hint = ""
            if image_input is not None and not is_vision_model:
                capable = [m for m in vision_models if not _is_category_separator(m)]
                vision_hint = (
                    f"\n\nHint: an image was attached and '{actual_model}' did not look "
                    "vision-capable. "
                    + (f"Models that currently do: {', '.join(capable)}."
                       if capable else
                       "No model in the current list advertises image input.")
                )

            # Initialize messages list
            messages = []

            # Add system prompt if provided and enabled
            if system_prompt and system_prompt.strip() and send_system == "yes":
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # An attached image is always forwarded. Choosing the text-only shape
            # here on a capability guess would silently drop the user's image.
            if image_input is not None:
                try:
                    if isinstance(image_input, torch.Tensor):
                        pil_image = chat_common.tensor_to_pil(image_input)
                    elif isinstance(image_input, Image.Image):
                        pil_image = image_input
                    else:
                        return io.NodeOutput("", "Error: Unsupported image input type", help_text)

                    messages.append(chat_common.encode_image_message(
                        pil_image, user_prompt, image_format))
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
                "max_completion_tokens": max_completion_tokens,
                "seed": seed
            }

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

            response, transport_error = chat_common.post_with_retries(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                body=body,
                max_retries=max_retries,
            )

            if transport_error is not None:
                return io.NodeOutput("", transport_error, help_text)

            # Handle 400 errors with detailed information
            if response.status_code == 400:
                try:
                    error_json = response.json()
                    error_message = error_json.get("error", {}).get("message", "Unknown error")

                    if debug_mode == "on":
                        return io.NodeOutput(
                            "",
                            f"Error 400: {error_message}{vision_hint}"
                            f"\n\nRequest body:\n{json.dumps(chat_common.redact_body(body), indent=2)}",
                            help_text
                        )
                    else:
                        return io.NodeOutput("", f"Error 400: {error_message}{vision_hint}", help_text)
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
                return io.NodeOutput(
                    "", f"Error: Rate limit exceeded even after {max_retries + 1} attempt(s)", help_text)
            elif response.status_code != 200:
                return io.NodeOutput(
                    "", f"Error: API returned status {response.status_code}", help_text)

            try:
                response_json = response.json()
            except requests.exceptions.JSONDecodeError:
                # A 200 with a malformed body is not worth retrying
                return io.NodeOutput("", "Error: Invalid JSON response from Groq", help_text)

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

        except Exception as e:
            # ComfyUI's cancel signal must propagate, not become a chat error.
            if type(e).__name__ == "InterruptProcessingException":
                raise
            return io.NodeOutput("", f"Unexpected Error: {str(e)}", help_text)


# Legacy v1 compatibility (for nodes that still use old API)
NODE_CLASS_MAPPINGS = {
    "GroqNode": GroqNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroqNode": "Groq Chat"
}
