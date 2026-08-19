# ComfyUI-EACloudNodes

A collection of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom nodes for interacting with various cloud services, such as LLM providers Groq and OpenRouter. These nodes are designed to work with any ComfyUI instance, including cloud-hosted environments where users may have limited system access.

**Note:** All nodes use the ComfyUI v3 node spec (`comfy_api.latest`) and are also
registered through the legacy `NODE_CLASS_MAPPINGS` path.

## Installation

Use [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) or to manually install:

1. Clone this repository into your ComfyUI custom_nodes folder:
  ```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EnragedAntelope/ComfyUI-EACloudNodes
  ```
2. Install required packages:
  ```bash
cd ComfyUI-EACloudNodes
pip install -r requirements.txt
  ```
3. Restart ComfyUI

## Current Nodes

### Common Features Across LLM Nodes
The following parameters are available in both OpenRouter and Groq nodes:

#### Common Parameters:
- `api_key`: ⚠️ Your API key (Note: key will be visible in workflows)
- `model`: Model selection (dropdown or identifier)
- `system_prompt`: Optional system context setting
- `user_prompt`: Main prompt/question for the model
- `temperature`: Controls response randomness (0.0-2.0)
- `top_p`: Nucleus sampling threshold (0.0-1.0)
- `frequency_penalty`: Token frequency penalty (-2.0 to 2.0)
- `presence_penalty`: Token presence penalty (-2.0 to 2.0)
- `response_format`: Choose between text or JSON object output
- `seed_mode`: Control reproducibility (Fixed, Random, Increment, Decrement)
- `max_retries`: Maximum retry attempts (0-5) for recoverable errors
- `image_input`: Optional image for vision-capable models
- `additional_params`: Optional JSON object for extra model parameters

#### Common Outputs:
- `response`: The model's generated text or JSON response
- `status`: Detailed information about the request, including model used and token counts
- `help`: Static help text with usage information and repository URL

### Groq Chat (v3)

Interact with Groq's API for ultra-fast inference with various LLM models. **Now fully compatible with ComfyUI v3 spec!**

#### Features:
- **ComfyUI v3 compatible** - Enhanced reliability and validation
- **Live model list** - the dropdown can be rebuilt from the Groq API (5-min cache)
- High-speed inference with Groq's optimized hardware
- Support for vision-capable models
- Real-time token usage tracking
- Automatic retry mechanism with exponential backoff
- Enhanced input validation
- Detailed tooltips for all parameters
- Debug mode for troubleshooting

#### Available Models:

The node ships with a static list of known chat models. To pick up newly released
models, run the node once with a valid API key and then press ComfyUI's **Refresh**
button — the dropdown is rebuilt from `GET /openai/v1/models` (cached for 5 minutes,
falling back to the static list whenever the API is unreachable).

Rows shown as `--- Category ---` are group labels, not selectable models.

**Featured:**
- `openai/gpt-oss-120b` - **Default** - 131K context, 500 T/sec
- `groq/compound` - Agentic system with web search and code execution

**Production: Chat** (stable, recommended for production use):
- `openai/gpt-oss-20b` - Faster and cheaper, 131K context, 1000 T/sec

**Production: Systems:**
- `groq/compound-mini` - Lightweight agentic system

**Preview: Chat** (experimental, may be discontinued at short notice):
- `minimaxai/minimax-m2.7` - 196K context, 131K max completion
- `openai/gpt-oss-safeguard-20b` - Safety-focused reasoning model
- `qwen/qwen3.6-27b` - **Vision**, 131K context, accepts files up to 20 MB

> **Not offered in the dropdown:**
> - Groq's speech models (Whisper, Orpheus) are served by `/audio/transcriptions`
>   and `/audio/speech`, so they cannot answer a chat-completions request at all.
>   Selecting one is rejected with a clear message.
> - The `llama-prompt-guard-2-*` classifiers work over chat completions but have a
>   512-token window and return a safety score rather than prose, so they are left
>   out of the curated list. Reach them with `Manual Input` if you want them.

#### Parameters:
- `api_key`: ⚠️ Your Groq API key (Get from [console.groq.com/keys](https://console.groq.com/keys))
- `model`: Select from available models or choose "Manual Input" for custom models
- `manual_model`: Enter custom model identifier (only used when "Manual Input" is selected)
- `system_prompt`: Optional system context (disable for vision models)
- `user_prompt`: Main prompt/question for the model
- `send_system`: Toggle system prompt sending (must be 'no' for vision models)
- `temperature`: Controls response randomness (0.0-2.0)
  - Lower (0.0-0.3): More focused and deterministic
  - Higher (0.7-2.0): More creative and varied
- `top_p`: Nucleus sampling threshold (0.0-1.0)
  - Lower (0.0-0.3): More focused vocabulary
  - Higher (0.7-1.0): More diverse word selection
- `max_completion_tokens`: Maximum tokens to generate (1-131,072, varies by model)
- `frequency_penalty`: Reduce token frequency repetition (-2.0 to 2.0)
- `presence_penalty`: Encourage topic diversity (-2.0 to 2.0)
- `response_format`: Choose between "text" or "json_object" output
- `seed_mode`: Control reproducibility
  - `fixed`: Use seed_value for consistent outputs
  - `random`: New random seed each time
  - `increment`: Increase seed by 1 each run
  - `decrement`: Decrease seed by 1 each run
- `seed_value`: Seed for 'fixed' mode (0-9007199254740991)
- `max_retries`: Auto-retry attempts for recoverable errors (0-5)
- `debug_mode`: Enable detailed error messages and request debugging
- `image_input`: Optional image for vision-capable models (max 2048x2048)
- `additional_params`: Extra model parameters in JSON format

#### Outputs:
- `response`: The model's generated text or JSON response
- `status`: Detailed request information including model, seed, and token counts
- `help`: Comprehensive help text with usage information

#### Vision Model Usage:

1. Select a vision-capable model — `qwen/qwen3.6-27b` at time of writing, or any
   other via `Manual Input`
2. Connect an image to the `image_input` parameter
3. Set `send_system` to "no" (vision models often reject system prompts)
4. Describe what you want to know about the image in `user_prompt`

**An attached image is always sent.** The node does not refuse a request based on
its own idea of which models accept images — that check would go stale every time
Groq reshuffles its line-up and would block models that actually work. Groq is the
authority; if it refuses the image, the error comes back with a hint naming the
models that did look capable.

Images are capped at 2048 pixels per dimension, and only the first image of a
batch is sent.

#### Production vs Preview Models:
- **Production Models**: Stable, reliable, meet high standards for speed/quality. Recommended for production use.
- **Preview Models**: Experimental, intended for evaluation only. May be deprecated with short notice.

### OpenRouter Chat (v3)

Interact with OpenRouter's API to access various AI models for text and vision tasks. **Now fully compatible with ComfyUI v3 spec!**

#### Features:
- **ComfyUI v3 compatible** - Enhanced reliability and validation
- Access to multiple AI providers through a single API
- Free-model dropdown built live from OpenRouter's public catalogue
- Vision support, with capability read from the catalogue rather than hardcoded
- JSON output support
- Automatic retry mechanism with exponential backoff
- Enhanced input validation
- Detailed tooltips for all parameters
- Debug mode for troubleshooting

#### Model List:

The dropdown is built at load time from OpenRouter's public catalogue
(`GET /api/v1/models`) and lists every model priced at **$0 for both prompt and
completion**, plus a `Manual Input` entry. It is not a hand-maintained list, so it
tracks OpenRouter's free tier as it changes; press ComfyUI's **Refresh** to rebuild
it (cached for 5 minutes).

To use a **paid** model, select `Manual Input` and enter its `provider/model-name`
id in `manual_model`.

Use the **OpenRouter Models** node below to browse what is currently on offer,
including pricing and context lengths.

#### Parameters:
- `api_key`: ⚠️ Your OpenRouter API key (Get from [https://openrouter.ai/keys](https://openrouter.ai/keys))
- `model`: Select from free models or choose "Manual Input" for custom models
- `manual_model`: Enter custom model identifier (only used when "Manual Input" is selected)
- `base_url`: OpenRouter API endpoint URL (default: https://openrouter.ai/api/v1/chat/completions)
- `system_prompt`: Optional system context setting
- `user_prompt`: Main prompt/question for the model (required)
- `send_system`: Toggle system prompt on/off
- `temperature`: Controls response randomness (0.0-2.0)
  - Lower (0.0-0.3): More focused and deterministic
  - Higher (0.7-2.0): More creative and varied
- `top_p`: Nucleus sampling threshold (0.0-1.0)
- `top_k`: Vocabulary limit (1-1000)
- `max_tokens`: Maximum tokens to generate (1-32,768)
- `frequency_penalty`: Reduce token frequency repetition (-2.0 to 2.0)
- `presence_penalty`: Encourage topic diversity (-2.0 to 2.0)
- `repetition_penalty`: OpenRouter-specific repetition penalty (1.0-2.0, 1.0=off)
- `response_format`: Choose between "text" or "json_object" output
- `seed_mode`: Control reproducibility (Fixed, Random, Increment, Decrement)
- `seed_value`: Seed for 'fixed' mode (0-9007199254740991)
- `max_retries`: Auto-retry attempts for recoverable errors (0-5)
- `debug_mode`: Enable detailed error messages and request debugging
- `image_input`: Optional image for vision models (max 2048x2048)
- `additional_params`: Extra model parameters in JSON format

#### Outputs:
- `response`: The model's generated text or JSON response
- `status`: Detailed request information including model, seed, and token counts
- `help`: Comprehensive help text with usage information

#### Vision Model Usage:
1. Select a vision-capable model — from the free dropdown, or via `Manual Input`
   for a paid one such as `openai/gpt-4o`
2. Connect an image to the `image_input` parameter
3. Describe what you want to know about the image in `user_prompt`

Image capability is read from OpenRouter's own catalogue (`architecture.input_modalities`),
so it stays correct as models come and go. A model the catalogue lists as text-only is
rejected before the request is sent; an id the catalogue does not know is passed through
so OpenRouter can answer for itself. Images are capped at 2048 pixels per dimension, and
only the first image of a batch is sent.

### OpenRouter Models Node
Query and filter available models from OpenRouter's API.

#### Features:
- Retrieve complete list of available models
- Filter models using custom search terms (e.g., 'free', 'gpt', 'claude')
- Sort models by name, pricing, or context length
- Detailed model information including pricing and context length
- Easy-to-read formatted output

#### Parameters:
- `api_key`: Optional — OpenRouter's model catalogue is public, so this can be left
  empty (Note: if you do supply a key it will be visible in saved workflows)
- `filter_text`: Text to filter models. `free` is matched against actual pricing
  rather than the model name; all other terms are matched against id, name, and
  description, and multiple terms are AND-ed together
- `sort_by`: Sort models by name, pricing, or context length
- `sort_order`: Choose ascending or descending sort order

## Usage Guide

### Basic Text Generation
1. Add an LLM node (OpenRouter or Groq) to your workflow
2. Set your API key
3. Choose a model
4. (Optional) Set system prompt for context/behavior
5. Enter your prompt in the `user_prompt` field
6. Connect the node's output to view results

### Vision Analysis
1. Add an LLM node to your workflow
2. Choose a vision-capable model
3. Connect an image output to the `image_input`
4. For Groq vision models, set 'send_system' to 'no'
5. Add your prompt about the image in `user_prompt`
6. Connect outputs to view response and status

### Advanced Usage
- Use `system_prompt` to set context or behavior
- Adjust temperature and other parameters to control response style
- Select `json_object` format for structured outputs
- Monitor token usage via the status output
- Chain multiple nodes for complex workflows
- Use seed_mode for reproducible outputs (Fixed) or controlled variation (Increment/Decrement)
- Use `additional_params` to set model-specific parameters in JSON format:
  ```json
  {
    "min_p": 0.1,
    "stop": ["\n\n"]
  }
  ```

### Parameter Optimization Tips
- **Temperature**:
  - Lower (0.1-0.3): More focused, deterministic responses
  - Higher (0.7-1.0): More creative outputs
- **Top-p**:
  - Lower (0.1-0.3): More predictable word choices
  - Higher (0.7-1.0): More diverse vocabulary
- **Penalties**:
  - Use `presence_penalty` to reduce topic repetition
  - Use `frequency_penalty` to reduce word repetition
- **Seed Mode**:
  - `fixed`: Use for reproducible outputs (same seed + params = same output)
  - `random`: Use for varied responses each time
  - `increment/decrement`: Use for controlled variation across runs
- **Token Management**:
  - Monitor token usage in status output to optimize costs
  - Adjust `max_completion_tokens` to control response length

### Error Handling
Both nodes provide detailed error messages for common issues:
- Missing or invalid API keys
- Model compatibility issues
- Image size and format requirements
- JSON format validation
- Token limits and usage
- API rate limits and automatic retries
- Parameter validation errors

Enable `debug_mode` in the Groq node for detailed troubleshooting information.

## Version History

### v2.1.0 (Current)
- **Model list refreshed against Groq's current catalogue**
  - `llama-3.3-70b-versatile` (the node's **default**) and `llama-3.1-8b-instant`
    have been retired by Groq. The default is now `openai/gpt-oss-120b`; before
    this change the node failed out of the box.
  - `meta-llama/llama-4-scout-17b-16e-instruct` and `qwen/qwen3-32b` are gone
  - Added `minimaxai/minimax-m2.7` and `qwen/qwen3.6-27b` (the current vision model)
  - The `llama-prompt-guard-2-*` classifiers left the curated dropdown (512-token
    safety classifiers, not chat models); still reachable via `Manual Input`
- **Built to absorb model churn** (see *Keeping up with model churn* above)
  - An attached image is always sent; the node no longer refuses one on the
    strength of its own capability list, which also fixes a path where a stale
    list would have silently dropped the image instead of sending it
  - Modality metadata outranks name heuristics everywhere, in both directions
  - The default model is resolved against the list that actually loaded, so a
    retired default can no longer break the node on a fresh drop-in
  - OpenRouter: embedding, reranking and speech models are filtered out of the
    free dropdown — they are priced at $0 and so passed a pricing-only test
    straight into a chat model list
- **Correctness**
  - Groq: send `max_completion_tokens` instead of the deprecated `max_tokens`
  - Groq: dropped Whisper/Orpheus from the chat dropdown — they are served by the
    audio endpoints and could never answer a chat-completions request
  - Groq: selecting a `--- Category ---` row is now rejected with a clear message
    instead of being sent to the API as a model id
  - Groq: the model list is now actually fetched from the API — the fetch helper
    was previously only ever called without a key, so the dropdown never updated
  - OpenRouter: image capability is read from the full catalogue, so paid vision
    models entered via `Manual Input` are no longer blocked
  - Both chat nodes: added `fingerprint_inputs()`, without which the `random`,
    `increment`, and `decrement` seed modes were inert on re-queue
  - Both chat nodes: an image batch larger than 1 no longer errors out
  - OpenRouter Models: null `context_length` and non-numeric pricing no longer
    crash sorting and filtering
  - OpenRouter Models: the API key is optional, matching the public endpoint
- **Robustness**
  - Failed model-list fetches back off for 60s instead of retrying on every run
  - `additional_params` must be a JSON object, reported clearly rather than as an
    "Unexpected Error" from inside `dict.update()`
  - A malformed 200 response is reported as such instead of being retried as a
    network error
  - Seed counters are bounded
- **Housekeeping**
  - Removed the `ImportError` fallback that re-imported the same failing modules,
    and the `WEB_DIRECTORY` pointing at a directory that does not exist
  - De-duplicated the Groq help text and the README
  - Added a pytest suite covering all three nodes

### v2.0.0
- **MAJOR UPDATE**: All nodes converted to ComfyUI v3 spec
- **Groq Node v3**:
  - Updated models list to latest production and preview models
  - Added new production models: groq/compound, groq/compound-mini
  - Added new preview models: qwen/qwen3-32b
  - Set llama-3.3-70b-versatile as default model
  - Enhanced input validation with validate_inputs method
  - Improved tooltips with detailed explanations for all parameters
  - Better error messages and debug mode support
  - Fixed output labels to use proper display_name syntax
- **OpenRouter Node v3**:
  - Converted to v3 spec with enhanced validation
  - Updated free models list to current 50+ offerings (January 2025)
  - Organized models by provider: Meta, Google, Mistral, Qwen, Microsoft, DeepSeek, Nvidia, Others
  - Set meta-llama/llama-3.3-70b-instruct:free as default model
  - Added comprehensive tooltips for all parameters
  - Enhanced error handling and debug mode
  - Better vision model detection and validation
  - Updated vision models list with all current vision-capable models
  - Fixed output labels to use proper display_name syntax
- **OpenRouter Models Node v3**:
  - Converted to v3 spec
  - Enhanced validation and error handling
  - Improved tooltip documentation
  - Fixed output labels to use proper display_name syntax
- **Architecture**:
  - All nodes use stateless design with class methods
  - Class-level seed tracking for reproducibility
  - Maintained full backward compatibility with v1 API
  - Combined v3 entry point for all nodes
  - Corrected combo input syntax (removed invalid enum classes)
  - Proper output definition using display_name parameter
- **Documentation**:
  - Comprehensive README updates for all v3 nodes
  - Updated OpenRouter model list with all 50+ current free models
  - Production vs preview model guidance
  - Enhanced parameter optimization tips
  - Detailed vision model usage instructions with current models

### v1.3.0
- Groq node v3 conversion (initial v3 work)

### Previous Versions
- See git history for earlier changes

## Technical Details

### ComfyUI v3 Compatibility
All nodes have been fully migrated to ComfyUI v3 spec:
- Uses `comfy_api.latest` for enhanced reliability
- Implements `define_schema()` with comprehensive input/output definitions
- Stateless design with class methods (`execute()`, `validate_inputs()`)
- Provides both `NODE_CLASS_MAPPINGS` and a `comfy_entrypoint()` extension, so the
  pack registers on whichever path a given ComfyUI build checks
- `fingerprint_inputs()` (v3's `IS_CHANGED`) so the non-fixed seed modes actually
  re-run instead of serving a cached response

These nodes require `comfy_api.latest`, which ships with current ComfyUI builds.

### API Compatibility
- **Groq**: OpenAI-compatible API endpoint
- **OpenRouter**: Multi-provider aggregation API
- Both support standard OpenAI message format
- Vision models use base64-encoded images in message content

## Keeping up with model churn

Groq and OpenRouter change their line-ups constantly, so the nodes are built to
absorb that without edits here. The rule throughout: **provider metadata decides,
hardcoded names are only a fallback, and nothing is refused on a guess.**

| Concern | How it self-heals |
| --- | --- |
| New model released | Appears in the dropdown on the next Refresh. Anything the pack does not recognise is listed under `--- Other ---` rather than hidden. |
| Default model retired | The default is resolved against the list that actually loaded, walking a preference order and then falling back to the first real entry. A retired default can no longer break the node on a fresh drop-in. |
| New vision model | Works immediately. An attached image is always sent and the provider decides — the node never refuses one from its own capability list. |
| A model's name trips a heuristic | Affirmative modality metadata always wins. A chat model called `…-tts-…` or `…-embed-…` stays listed if the API says it emits text. |
| Provider adds modality fields | Read automatically. Several plausible field names are checked, and an entry that reports nothing is treated as "unknown", never as "unsupported". |
| Provider is unreachable | Groq falls back to a static list and backs off for 60s. OpenRouter offers `Manual Input`, which reaches any model id. |

The one genuinely time-sensitive thing in the repo is Groq's `STATIC_FALLBACK_MODELS`,
used only when no API key has been supplied yet. A stale entry there costs a clear
error from Groq, not a broken node.

`tests/test_package.py` simulates a wholesale catalogue reshuffle — invented vendors,
unfamiliar ids, retired defaults — to check these paths keep working.

## Development

Run the test suite (no API keys and no network access required — every HTTP call is
stubbed):

```bash
pip install pytest
python -m pytest tests
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or feature requests:
- Open an issue on [GitHub](https://github.com/EnragedAntelope/ComfyUI-EACloudNodes/issues)
- Check existing issues for solutions
- Enable debug mode for detailed error information

## License

[License](https://github.com/EnragedAntelope/ComfyUI-EACloudNodes/blob/main/LICENSE)
