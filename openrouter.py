"""
OpenRouter Chat Node for ComfyUI v3
Supports text and vision-language models through OpenRouter's API.
"""

import json
import requests
import time
from PIL import Image
import torch
from urllib.parse import urlsplit

# Relative: ComfyUI loads this folder as a package and never puts it on
# sys.path, so an absolute `import chat_common` fails at registration time.
from . import chat_common
from comfy_api.latest import io


# Module-level cache for dynamically fetched models
_openrouter_model_cache = {
    "models": None,
    "vision_models": None,
    "known_models": None,
    "last_fetch": 0,
    "cache_ttl": 300,  # 5 minutes
    "last_failure": 0,
    "failure_backoff": 60  # don't re-try a failing fetch on every execution
}

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Hosts a plain-http request is allowed to target: local proxies only. The
# Authorization header would otherwise cross the network in cleartext.
LOCAL_HTTP_HOSTS = {"localhost", "127.0.0.1", "::1"}

# Preferred defaults, most wanted first; mirrors the Groq node's strategy.
# The first entry present in the fetched free list wins, falling back to the
# first real model, so a retired default cannot break the node out of the box.
PREFERRED_DEFAULT_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
]


def _endpoint_error(base_url: str):
    """
    Reject an endpoint the API key must not be sent to; None means it is allowed.

    Enforced in execute() as well as validate_inputs(), because validate_inputs
    only sees a literal widget value - once base_url is converted to an input
    socket its value is not known until the graph runs, and execute() is the
    point where the Authorization header actually goes on the wire.
    """
    if not base_url or not base_url.strip():
        return "OpenRouter API endpoint URL is required"

    try:
        parts = urlsplit(base_url.strip())
    except ValueError:
        return "Invalid API endpoint URL format"

    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()

    if scheme not in ("http", "https"):
        return "Invalid API endpoint URL format (must start with http:// or https://)"

    # Workflows are shared as JSON files, and this header carries the user's
    # key. Plain http would put it on the wire in cleartext; only a local
    # proxy has a reason to do that.
    if scheme == "http" and host not in LOCAL_HTTP_HOSTS:
        return ("Refusing to send your API key over plain http://. Use an "
                "https:// endpoint, or point base_url at a localhost proxy.")

    return None


# Embedding, reranking and speech models are priced at $0 and so pass a
# pricing-only "free" test, but none of them can answer a chat completion.
NON_CHAT_NAME_PATTERNS = ["embed", "rerank", "-tts", "tts-", "/tts", "-stt", "whisper"]


def _model_supports_chat(model: dict) -> bool:
    """
    True when a catalogue entry looks like a chat model.

    Judged on output modality first (a chat model emits text; a TTS model emits
    audio), then on the naming conventions used for embedding and reranking
    models. Entries carrying no modality metadata are kept rather than dropped,
    so an unfamiliar shape never silently hides a usable model.
    """
    arch = model.get("architecture") or {}

    output_modalities = arch.get("output_modalities")
    if isinstance(output_modalities, list) and output_modalities:
        # Metadata is authoritative in both directions: a model that says it
        # emits text is a chat model even if its name looks like something else.
        return any(str(m).lower() == "text" for m in output_modalities)

    modality = str(arch.get("modality") or "")
    if "->" in modality:
        outputs = modality.split("->", 1)[1].lower()
        if outputs:
            return "text" in outputs

    # No modality metadata: fall back on naming conventions.
    haystack = f"{model.get('id', '')} {model.get('name', '')}".lower()
    return not any(pattern in haystack for pattern in NON_CHAT_NAME_PATTERNS)


def _model_accepts_images(model: dict) -> bool:
    """
    Decide whether an OpenRouter model entry accepts image input.

    Prefers the architecture metadata (``input_modalities`` on current API
    responses, ``modality`` on older ones) and falls back to the naming
    convention used by most vision models.
    """
    arch = model.get("architecture") or {}

    input_modalities = arch.get("input_modalities") or []
    if isinstance(input_modalities, list):
        if any(str(m).lower() == "image" for m in input_modalities):
            return True

    modality = str(arch.get("modality") or "").lower()
    # e.g. "text+image->text"; only the input side (before "->") counts
    if "image" in modality.split("->")[0]:
        return True

    haystack = f"{model.get('id', '')} {model.get('name', '')}".lower()
    return "vision" in haystack or "-vl" in haystack or "vl-" in haystack


def _fetch_openrouter_free_models():
    """
    Fetch free models from OpenRouter's public API with caching.
    Returns (model_list, vision_model_list) or (None, None) on failure.
    The model list includes 'Manual Input' as the last entry.
    """
    now = time.time()

    with chat_common.LOCK:
        # Return cached results if still fresh
        if (_openrouter_model_cache["models"] is not None and
                now - _openrouter_model_cache["last_fetch"] < _openrouter_model_cache["cache_ttl"]):
            return _openrouter_model_cache["models"], _openrouter_model_cache["vision_models"]

        # Back off after a failure too, so an offline host does not add a request
        # (and its timeout) to every node execution and every /object_info refresh.
        if now - _openrouter_model_cache["last_failure"] < _openrouter_model_cache["failure_backoff"]:
            return _openrouter_model_cache["models"], _openrouter_model_cache["vision_models"]

        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                timeout=5
            )
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}")

            data = response.json().get("data", [])

            free_models = []
            vision_models = []
            known_models = []

            for model in data:
                model_id = model.get("id", "")
                if not model_id:
                    continue

                # Embeddings, rerankers and TTS models are $0 and would otherwise pass
                # the pricing-only "free" test straight into a chat dropdown.
                if not _model_supports_chat(model):
                    continue

                known_models.append(model_id)

                # Vision capability is tracked for every model, not just the free ones,
                # so models entered via 'Manual Input' can be checked too.
                if _model_accepts_images(model):
                    vision_models.append(model_id)

                pricing = model.get("pricing") or {}
                try:
                    is_free = (
                        float(pricing.get("prompt", "1")) == 0 and
                        float(pricing.get("completion", "1")) == 0
                    )
                except (ValueError, TypeError):
                    continue

                if is_free:
                    free_models.append(model_id)

            free_models.sort()
            free_models.append("Manual Input")

            _openrouter_model_cache["models"] = free_models
            _openrouter_model_cache["vision_models"] = vision_models
            _openrouter_model_cache["known_models"] = known_models
            _openrouter_model_cache["last_fetch"] = now

            return free_models, vision_models

        except Exception:
            _openrouter_model_cache["last_failure"] = now
            # Return previously cached results if available, otherwise None
            if _openrouter_model_cache["models"] is not None:
                return _openrouter_model_cache["models"], _openrouter_model_cache["vision_models"]
            return None, None


def _pick_default_model(models: list[str]) -> str:
    """Choose a default that exists in the given list, mirroring the Groq node."""
    for preferred in PREFERRED_DEFAULT_MODELS:
        if preferred in models:
            return preferred
    for model_id in models:
        if model_id != "Manual Input":
            return model_id
    return "Manual Input"


class OpenrouterNode(io.ComfyNode):
    """
    A node for interacting with OpenRouter API.
    Supports text and vision-language models through OpenRouter's API.
    """

    # JavaScript safe integer limit (2^53 - 1)
    MAX_SAFE_INTEGER = 9007199254740991

    # Class-level storage for seed counters, keyed by (model, starting seed)
    _last_seed = {}

    # Upper bound on tracked seed counters, so the dict cannot grow without bound
    MAX_TRACKED_SEEDS = 1024

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Dynamically fetch free models from OpenRouter's public API
        models, _ = _fetch_openrouter_free_models()
        if models is None:
            # API unreachable - provide Manual Input as the only option
            models = ["Manual Input"]

        # Pick a sensible default from the fetched list
        default_model = _pick_default_model(models)

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
                io.Combo.Input(
                    "model",
                    options=models,
                    default=default_model,
                    tooltip="Select a free OpenRouter chat model, or choose 'Manual Input' for a paid or custom one. The list is fetched live from OpenRouter and excludes embedding, reranking and speech models, which cannot answer a chat request. Use ComfyUI's Refresh to update it."
                ),
                io.String.Input(
                    "manual_model",
                    default="",
                    multiline=False,
                    tooltip="Enter a custom model identifier (only used when 'Manual Input' is selected). Format: provider/model-name[:free]. Leave empty if using dropdown."
                ),
                io.String.Input(
                    "base_url",
                    default=DEFAULT_BASE_URL,
                    multiline=False,
                    tooltip="OpenRouter API endpoint URL. Leave as default unless using a proxy or alternate endpoint. Must be https:// unless it points at localhost; non-OpenRouter endpoints get a visible warning on every run, because your key is sent there."
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
                io.Combo.Input(
                    "send_system",
                    options=["yes", "no"],
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
                io.Combo.Input(
                    "debug_mode",
                    options=["off", "on"],
                    default="off",
                    tooltip="Enable detailed error messages and request debugging information. Useful for troubleshooting API issues or parameter problems."
                ),
                io.Image.Input(
                    "image_input",
                    optional=True,
                    tooltip="Optional image input. Capability is checked against OpenRouter's model catalogue; models it lists as text-only are rejected. Maximum size: 2048x2048 (only the first image of a batch is sent)."
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
                    tooltip="Additional OpenRouter API parameters in JSON format. Example: {\"min_p\": 0.1, \"top_a\": 0.5}. Use for advanced model-specific parameters not exposed in the UI."
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
    def validate_inputs(cls, api_key, model, manual_model, user_prompt, base_url, **kwargs):
        """Validate inputs before execution"""
        # Validate API key
        if not api_key or not api_key.strip():
            return "OpenRouter API key is required. Get one at https://openrouter.ai/keys"

        # Validate model selection
        if model == "Manual Input" and (not manual_model or not manual_model.strip()):
            return "Manual model identifier is required when 'Manual Input' is selected"

        # Validate base URL
        endpoint_error = _endpoint_error(base_url)
        if endpoint_error is not None:
            return endpoint_error

        # Validate additional_params if provided
        additional_params = kwargs.get("additional_params", "")
        if additional_params and additional_params.strip():
            try:
                parsed = json.loads(additional_params)
            except json.JSONDecodeError:
                return "Invalid JSON in additional parameters. Example format: {\"top_a\": 0.5}"
            if not isinstance(parsed, dict):
                return "Additional parameters must be a JSON object. Example format: {\"top_a\": 0.5}"

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
        image_format: str = "png",
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
- Model: Dropdown auto-populates with free models from OpenRouter's API.
  Use ComfyUI's Refresh to update the list. Choose 'Manual Input' for custom models.
- Manual Model: Custom model ID (provider/model-name[:free])
- Base URL: API endpoint (usually leave as default). https:// required unless
  the endpoint is localhost; a non-OpenRouter endpoint warns on every run,
  because your API key is sent wherever base_url points.
- System Prompt: Set AI behavior/context
- User Prompt: Main input for the model (required)
- Send System: Toggle system prompt on/off
- Temperature: 0.0 (focused) to 2.0 (creative)
- Top-p: Nucleus sampling threshold (0.0-1.0)
- Top-k: Vocabulary limit (1-1000); only sent when changed from the default of 50
- Max Tokens: Response length limit (1-32,768)
- Frequency Penalty: Reduce token frequency (-2.0 to 2.0)
- Presence Penalty: Encourage topic diversity (-2.0 to 2.0)
- Repetition Penalty: Reduce repetition (1.0-2.0, 1.0=off)
- Response Format: Text or JSON object output
- Seed Mode: Fixed/random/increment/decrement for reproducibility
- Seed Value: Seed for 'fixed' mode (0-9007199254740991)
- Max Retries: Auto-retry on errors (0-5)
- Debug Mode: Enable for detailed error messages (image data is summarized
  rather than dumped)

Optional:
- Image Input: For vision-capable models
  * Capability is read from OpenRouter's model catalogue; a model the catalogue
    lists as text-only is rejected, and ids it does not know are passed through
  * Max size: 2048x2048 per dimension; only the first image of a batch is sent
- Image Format: PNG (lossless) or JPEG (smaller payload for photos)
- Additional Params: Extra model parameters as a JSON object, merged into the
  request body (it overrides the widgets above on key collisions)

Vision Models:
1. Select a vision-capable model (dropdown, or 'Manual Input' for paid models)
2. Connect an image to image_input
3. Describe what you want to know about the image in user_prompt

Note on the model dropdown:
- It lists OpenRouter's *free* models only, refreshed from the public catalogue
- Paid models are reached with 'Manual Input' plus the provider/model id

For full documentation and examples, visit:
https://github.com/EnragedAntelope/ComfyUI-EACloudNodes"""

        # Workflows travel as JSON files, and whatever sits in base_url receives
        # the Authorization header. Make a redirected request impossible to miss.
        try:
            endpoint_host = (urlsplit((base_url or "").strip()).hostname or "").lower()
        except ValueError:
            endpoint_host = ""
        custom_endpoint = bool(endpoint_host) and endpoint_host != "openrouter.ai"

        # Only claimed once the request has actually been issued: the same status
        # helper serves the early validation returns, and telling a user their key
        # was sent when nothing left the machine is its own kind of alarming.
        request_issued = []

        def out(response_text: str, status: str) -> io.NodeOutput:
            warning = ""
            if custom_endpoint and request_issued:
                warning = (f"\n⚠️ Custom endpoint: your API key and prompt were sent to "
                           f"'{endpoint_host}', not openrouter.ai.")
            elif custom_endpoint:
                warning = (f"\n⚠️ Custom endpoint: base_url points at '{endpoint_host}', "
                           "not openrouter.ai. Your API key is sent wherever it points.")
            return io.NodeOutput(response_text, status + warning, help_text)

        try:
            # Re-checked here, not just in validate_inputs: once base_url is
            # converted to an input socket its value is unknown until the graph
            # runs, and this is where the Authorization header goes on the wire.
            endpoint_error = _endpoint_error(base_url)
            if endpoint_error is not None:
                return out("", f"Error: {endpoint_error}")

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
                return out("", f"Error: Invalid parameter value - {str(e)}")

            # Validate user prompt (delayed until execute to handle connected inputs)
            if not user_prompt or not user_prompt.strip():
                return out("", "User prompt is required")

            # Use manual_model if "Manual Input" is selected
            actual_model = manual_model.strip() if model == "Manual Input" else model

            # Handle seed based on mode.
            # Counters are keyed by (model, starting seed) and capped so long-running
            # sessions cannot grow this dict without bound.
            node_key = (actual_model, seed_value)
            with chat_common.LOCK:
                seed = chat_common.derive_seed(
                    cls._last_seed, node_key, seed_mode, seed_value, cls.MAX_SAFE_INTEGER)
                chat_common.store_seed(
                    cls._last_seed, node_key, seed, cls.MAX_TRACKED_SEEDS)

            # Vision gating. Only refuse when OpenRouter's own catalogue tells us the
            # selected model does not accept images; unknown ids (custom endpoints,
            # brand-new models, or an unreachable catalogue) are passed through so the
            # API can answer for itself instead of us blocking a valid request.
            if image_input is not None:
                _fetch_openrouter_free_models()
                vision_models = _openrouter_model_cache.get("vision_models") or []
                known_models = _openrouter_model_cache.get("known_models") or []
                if actual_model in known_models and actual_model not in vision_models:
                    return out(
                        "",
                        f"Error: Model '{actual_model}' does not accept image input according to "
                        "OpenRouter's model catalogue. Choose a vision-capable model, or disconnect "
                        "the image input."
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
                    if isinstance(image_input, torch.Tensor):
                        pil_image = chat_common.tensor_to_pil(image_input)
                    elif isinstance(image_input, Image.Image):
                        pil_image = image_input
                    else:
                        return out("", "Error: Unsupported image input type")

                    messages.append(chat_common.encode_image_message(
                        pil_image, user_prompt, image_format))
                except Exception as img_err:
                    return out("", f"Image Processing Error: {str(img_err)}")
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
                "max_tokens": max_tokens,
                "seed": seed
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

            # Add response format if json_object is selected
            if response_format == "json_object":
                body["response_format"] = {"type": "json_object"}

            # Parse and add additional parameters if provided
            if additional_params and additional_params.strip():
                try:
                    extra_params = json.loads(additional_params)
                except json.JSONDecodeError:
                    return out("", "Error: Invalid JSON in additional parameters. Example format: {\"top_a\": 0.5}")
                if not isinstance(extra_params, dict):
                    return out("", "Error: Additional parameters must be a JSON object. Example format: {\"top_a\": 0.5}")
                body.update(extra_params)

            request_issued.append(True)
            response, transport_error = chat_common.post_with_retries(
                base_url, headers=headers, body=body, max_retries=max_retries)

            if transport_error is not None:
                return out("", transport_error)

            # Handle 400 errors with detailed information
            if response.status_code == 400:
                try:
                    error_json = response.json()
                    error_message = error_json.get("error", {}).get("message", "Unknown error")

                    if debug_mode == "on":
                        return out(
                            "",
                            f"Error 400: {error_message}"
                            f"\n\nRequest body:\n{json.dumps(chat_common.redact_body(body), indent=2)}"
                        )
                    else:
                        return out("", f"Error 400: {error_message}")
                except Exception:
                    return out(
                        "",
                        "Error: Bad request - check model name and parameters (enable debug mode for details)"
                    )

            # Handle other response codes
            if response.status_code == 401:
                return out("", "Error: Invalid API key or unauthorized access")
            elif response.status_code == 413:
                return out("", "Error: Payload too large - try reducing prompt or image size")
            elif response.status_code == 429:
                return out(
                    "", f"Error: Rate limit exceeded even after {max_retries + 1} attempt(s)")
            elif response.status_code in {500, 502, 503, 504}:
                return out(
                    "", f"Error: OpenRouter service error (status {response.status_code})")
            elif response.status_code != 200:
                return out("", f"Error: API returned status {response.status_code}")

            try:
                response_json = response.json()
            except requests.exceptions.JSONDecodeError:
                # A 200 with a malformed body is not worth retrying
                return out("", "Error: Invalid JSON response from OpenRouter")

            # Extract information for status
            model_used = response_json.get("model", "unknown")
            tokens = response_json.get("usage", {})
            prompt_tokens = tokens.get("prompt_tokens", 0)
            completion_tokens = tokens.get("completion_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            status_msg = f"Success: Model={model_used} | Seed={seed} | Tokens: {prompt_tokens}+{completion_tokens}={total_tokens}"

            if "choices" in response_json and len(response_json["choices"]) > 0:
                content = response_json["choices"][0].get("message", {}).get("content", "")
                return out(content, status_msg)
            else:
                return out("", "Error: No response content from the model")

        except Exception as e:
            # ComfyUI's cancel signal must propagate, not become a chat error.
            if type(e).__name__ == "InterruptProcessingException":
                raise
            return out("", f"Unexpected Error: {str(e)}")


# Legacy v1 compatibility (for nodes that still use old API)
NODE_CLASS_MAPPINGS = {
    "OpenrouterNode": OpenrouterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenrouterNode": "OpenRouter Chat"
}
