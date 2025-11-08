# ComfyUI-EACloudNodes

A collection of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom nodes for interacting with various cloud services, such as LLM providers Groq and OpenRouter. These nodes are designed to work with any ComfyUI instance, including cloud-hosted environments where users may have limited system access.

**Note:** The Groq node has been updated to ComfyUI v3 spec for enhanced reliability and features. OpenRouter nodes currently use v1 API but will be migrated in a future update.

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
- High-speed inference with Groq's optimized hardware
- Comprehensive model selection including production and preview models
- Support for vision-capable models (Llama-4 Maverick and Scout)
- Real-time token usage tracking
- Automatic retry mechanism with exponential backoff
- Enhanced input validation
- Detailed tooltips for all parameters
- Debug mode for troubleshooting

#### Available Models:

**Production Models** (Stable, recommended for production use):
- `llama-3.1-8b-instant` - Fast 8B parameter model (560 T/sec)
- `llama-3.3-70b-versatile` - **Default** - Powerful 70B model (280 T/sec)
- `meta-llama/llama-guard-4-12b` - Safety and moderation model (1200 T/sec)
- `openai/gpt-oss-120b` - Large open-source GPT (500 T/sec)
- `openai/gpt-oss-20b` - Efficient open-source GPT (1000 T/sec)
- `whisper-large-v3` - Speech recognition model
- `whisper-large-v3-turbo` - Faster speech recognition

**Production Systems** (Agentic systems with tools):
- `groq/compound` - Multi-model system with tools
- `groq/compound-mini` - Lightweight agentic system

**Preview Models** (Experimental, for evaluation only):
- `meta-llama/llama-4-maverick-17b-128e-instruct` - **Vision** (600 T/sec)
- `meta-llama/llama-4-scout-17b-16e-instruct` - **Vision** (750 T/sec)
- `meta-llama/llama-prompt-guard-2-22m` - Prompt injection detection
- `meta-llama/llama-prompt-guard-2-86m` - Enhanced prompt guard
- `moonshotai/kimi-k2-instruct-0905` - 262K context window (200 T/sec)
- `openai/gpt-oss-safeguard-20b` - Safety-focused model (1000 T/sec)
- `playai-tts` - Text-to-speech model
- `playai-tts-arabic` - Arabic text-to-speech
- `qwen/qwen3-32b` - Qwen 32B model (400 T/sec)

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
- `image_input`: Optional image for vision models (Llama-4 only)
- `additional_params`: Extra model parameters in JSON format

#### Outputs:
- `response`: The model's generated text or JSON response
- `status`: Detailed request information including model, seed, and token counts
- `help`: Comprehensive help text with usage information

#### Vision Model Usage:
1. Select a vision-capable model:
   - `meta-llama/llama-4-maverick-17b-128e-instruct`
   - `meta-llama/llama-4-scout-17b-16e-instruct`
2. Connect an image to the `image_input` parameter
3. Set `send_system` to "no" (vision models don't accept system prompts)
4. Describe what you want to know about the image in `user_prompt`

#### Production vs Preview Models:
- **Production Models**: Stable, reliable, meet high standards for speed/quality. Recommended for production use.
- **Preview Models**: Experimental, intended for evaluation only. May be deprecated with short notice.

### OpenRouter Chat

Interact with OpenRouter's API to access various AI models for text and vision tasks.

#### Model Selection
- Choose from a curated list of free models in the dropdown
- Select "Manual Input" to use any OpenRouter-supported model
- When using Manual Input, enter the full model identifier in the manual_model field
- Supports both text and vision models (vision-capable models indicated in name)

#### Available Free Models
- Google Gemini models (various versions)
- Meta Llama models (including vision-capable versions)
- Microsoft Phi models
- Mistral, Qwen, DeepSeek, and other open models
- Full list viewable in node dropdown

#### Parameters:
- `api_key`: ⚠️ Your OpenRouter API key (Get from [https://openrouter.ai/keys](https://openrouter.ai/keys))
- `model`: Select from available models or choose "Manual Input"
- `manual_model`: Enter custom model name (only used when "Manual Input" is selected)
- `base_url`: OpenRouter API endpoint URL (default: https://openrouter.ai/api/v1/chat/completions)
- `system_prompt`: Optional system context setting
- `user_prompt`: Main prompt/question for the model
- `temperature`: Controls response randomness (0.0-2.0)
- `top_p`: Nucleus sampling threshold (0.0-1.0)
- `top_k`: Vocabulary limit (1-1000)
- `max_tokens`: Maximum number of tokens to generate
- `frequency_penalty`: Token frequency penalty (-2.0 to 2.0)
- `presence_penalty`: Token presence penalty (-2.0 to 2.0)
- `repetition_penalty`: Repetition penalty (1.0-2.0)
- `response_format`: Choose between text or JSON object output
- `seed_mode`: Control reproducibility (Fixed, Random, Increment, Decrement)
- `max_retries`: Number of retry attempts for recoverable errors (0-5)
- `image_input`: Optional image for vision-capable models
- `additional_params`: Optional JSON object for extra model parameters

#### Outputs:
- `response`: The model's generated text or JSON response
- `status`: Detailed information about the request, including model used and token counts
- `help`: Static help text with usage information and repository URL

### OpenRouter Models Node
Query and filter available models from OpenRouter's API.

#### Features:
- Retrieve complete list of available models
- Filter models using custom search terms (e.g., 'free', 'gpt', 'claude')
- Sort models by name, pricing, or context length
- Detailed model information including pricing and context length
- Easy-to-read formatted output

#### Parameters:
- `api_key`: ⚠️ Your OpenRouter API key (Note: key will be visible in workflows)
- `filter_text`: Text to filter models
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

### v1.3.0 (Current)
- **Major Update**: Converted Groq node to ComfyUI v3 spec
- Updated Groq models list to latest production and preview models
- Added new production models: groq/compound, groq/compound-mini
- Added new preview models: qwen/qwen3-32b, and updated model identifiers
- Set llama-3.3-70b-versatile as default model
- Enhanced input validation with validate_inputs method
- Improved tooltips with detailed explanations for all parameters
- Better error messages and debug mode support
- Class-level seed tracking for stateless v3 architecture
- Maintained backward compatibility with v1 API
- Updated documentation with production vs preview model guidance

### Previous Versions
- See git history for earlier changes

## Technical Details

### ComfyUI v3 Compatibility
The Groq node has been fully migrated to ComfyUI v3 spec:
- Uses `comfy_api.latest` for enhanced reliability
- Implements `define_schema()` with comprehensive input/output definitions
- Stateless design with class methods (`execute()`, `validate_inputs()`)
- Proper `comfy_entrypoint()` function for v3 registration
- Maintains v1 compatibility through legacy NODE_CLASS_MAPPINGS

### API Compatibility
- **Groq**: OpenAI-compatible API endpoint
- **OpenRouter**: Multi-provider aggregation API
- Both support standard OpenAI message format
- Vision models use base64-encoded images in message content

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or feature requests:
- Open an issue on [GitHub](https://github.com/EnragedAntelope/ComfyUI-EACloudNodes/issues)
- Check existing issues for solutions
- Enable debug mode for detailed error information

## License

[License](https://github.com/EnragedAntelope/ComfyUI-EACloudNodes/blob/main/LICENSE)
