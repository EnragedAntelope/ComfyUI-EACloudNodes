# ComfyUI-EACloudNodes

A collection of ComfyUI custom nodes for interacting with various cloud services, such as LLM providers Groq and OpenRouter. These nodes are designed to work with any ComfyUI instance, including cloud-hosted environments where users may have limited system access.

## Installation

Use ComfyManager or to manually install:

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
- `seed`: Control reproducibility (0 for random)
- `seed_mode`: Choose between fixed, increment, decrement, or randomize
- `max_retries`: Maximum retry attempts (0-5) for recoverable errors
- `image_input`: Optional image for vision-capable models
- `additional_params`: Optional JSON object for extra model parameters

#### Common Outputs:
- `response`: The model's generated text or JSON response
- `status`: Detailed information about the request, including model used and token counts
- `help`: Static help text with usage information and repository URL

### OpenRouter Chat
Interact with OpenRouter's API to access various AI models for text and vision tasks.

#### OpenRouter Chat Node
Interact with OpenRouter's API to access various AI models for text and vision tasks.

##### Additional Features:
- Access to multiple AI providers through a single API
- `top_k`: Vocabulary limit (1-1000)
- `repetition_penalty`: Repetition penalty (1.0-2.0)
- `base_url`: Configurable OpenRouter API endpoint URL

##### Parameters:
- `api_key`: ⚠️ Your OpenRouter API key (Get from openrouter.ai/settings/keys)
- `model`: Model identifier from OpenRouter
- `base_url`: OpenRouter API endpoint URL
- `user_prompt`: Main prompt/question for the model
- `system_prompt`: Optional system context setting
- `temperature`: Controls response randomness (0.0-2.0)
- `top_p`: Nucleus sampling threshold (0.0-1.0)
- `top_k`: Vocabulary limit (1-1000)
- `max_tokens`: Maximum number of tokens to generate
- `frequency_penalty`: Token frequency penalty (-2.0 to 2.0)
- `presence_penalty`: Token presence penalty (-2.0 to 2.0)
- `repetition_penalty`: Repetition penalty (1.0-2.0)
- `response_format`: Choose between text or JSON object output
- `seed`: Control reproducibility (0 for random)
- `seed_mode`: Choose between fixed, increment, decrement, or randomize
- `max_retries`: Number of retry attempts for recoverable errors (0-5)
- `image_input`: Optional image for vision-capable models
- `additional_params`: Optional JSON object for extra model parameters

##### Outputs:
- `response`: The model's generated text or JSON response
- `status`: Detailed information about the request, including model used and token counts
- `help`: Static help text with usage information and repository URL

#### OpenRouter Models Node
Query and filter available models from OpenRouter's API.

##### Features:
- Retrieve complete list of available models
- Filter models using custom search terms (e.g., 'free', 'gpt', 'claude')
- Sort models by name, pricing, or context length
- Detailed model information including pricing and context length
- Easy-to-read formatted output

##### Parameters:
- `api_key`: ⚠️ Your OpenRouter API key (Note: key will be visible in workflows)
- `filter_text`: Text to filter models
- `sort_by`: Sort models by name, pricing, or context length
- `sort_order`: Choose ascending or descending sort order

### Groq Chat
Interact with Groq's API for ultra-fast inference with various LLM models.

#### Features:
- High-speed inference with Groq's optimized hardware
- Dropdown selection of current Groq models
- Manual model input option for future or custom models
- Support for vision-capable models (models with 'vision' in their name)
- Real-time token usage tracking
- Automatic retry mechanism for recoverable errors
- Toggle for system prompt sending (required for vision models)

#### Usage:
1. Get your API key from [console.groq.com/keys](https://console.groq.com/keys)
2. Add the Groq node to your workflow
3. Enter your API key
4. Select a model from the dropdown or choose "Manual Input" to specify a custom model
5. Configure your prompts and parameters
6. Connect outputs to view responses, status, and help information

#### Parameters:
- `model`: Select from available Groq models or choose "Manual Input"
- `manual_model`: Enter custom model name (only used when "Manual Input" is selected)
- `user_prompt`: Main prompt/question for the model
- `system_prompt`: Optional system context setting
- `send_system`: Toggle system prompt sending (must be off for vision models)
- `temperature`: Controls response randomness (0.0-2.0)
- `top_p`: Nucleus sampling threshold (0.0-1.0)
- `max_completion_tokens`: Maximum number of tokens to generate
- `frequency_penalty`: Token frequency penalty (-2.0 to 2.0)
- `presence_penalty`: Token presence penalty (-2.0 to 2.0)
- `response_format`: Choose between text or JSON object output
- `seed`: Control reproducibility (0 for random)
- `seed_mode`: Choose between fixed, increment, decrement, or randomize
- `max_retries`: Number of retry attempts for recoverable errors (0-5)
- `image_input`: Optional image for vision-capable models
- `additional_params`: Optional JSON object for extra model parameters

#### Outputs:
- `response`: The model's generated text or JSON response
- `status`: Detailed information about the request, including model used and token counts
- `help`: Static help text with usage information and repository URL

### Vision Analysis
1. Add an LLM node to your workflow
2. Choose a vision-capable model
3. Connect an image output to the `image_input`
4. For Groq vision models, toggle 'send_system' to 'no'
5. Add your prompt about the image
6. Connect outputs to view response, status, and help information

## Usage Guide

### Basic Text Generation
1. Add an LLM node (OpenRouter or Groq) to your workflow
2. Set your API key
3. Choose a model
4. (Optional) Set system prompt for context/behavior
5. Enter your prompt in the `user_prompt` field
6. Connect the node's output to a Text node to display results

### Vision Analysis
1. Add an LLM node to your workflow
2. Choose a vision-capable model
3. Connect an image output to the `image_input`
4. Add your prompt about the image
5. Connect outputs to view both the response and status

### Advanced Usage
- Use `system_prompt` to set context or behavior
- Adjust temperature and other parameters to control response style
- Select `json_object` format for structured outputs
- Monitor token usage via the status output
- Chain multiple nodes for complex workflows
- Use seed controls for reproducible outputs
- Use `additional_params` to set model-specific parameters in JSON format:
  ```json
  {
    "min_p": 0.1,
    "top_a": 0.8
  }
  ```

### Parameter Optimization Tips
- Lower temperature (0.1-0.3) for more focused responses
- Higher temperature (0.7-1.0) for more creative outputs
- Lower top_p (0.1-0.3) for more predictable word choices
- Higher top_p (0.7-1.0) for more diverse vocabulary
- Use presence_penalty to reduce repetition
- Monitor token usage to optimize costs
- Save API key in the node for reuse in workflows

### Error Handling
Both nodes provide detailed error messages for common issues:
- Missing or invalid API keys
- Model compatibility issues
- Image size and format requirements
- JSON format validation
- Token limits and usage
- API rate limits and retries

## Changelog

### v0.1.0
- Initial release
- Support for Groq and OpenRouter APIs
- Vision model support
- Automatic retry mechanism
- Token usage tracking
- Seed control options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache License 2.0
Copyright 2024 EnragedAntelope
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
