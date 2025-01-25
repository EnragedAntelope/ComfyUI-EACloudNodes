# ComfyUI-EACloudNodes

A collection of ComfyUI custom nodes for interacting with various cloud services. These nodes are designed to work with any ComfyUI instance, including cloud-hosted environments where users may have limited system access.

## Current Nodes

### OpenRouter Node
Interact with OpenRouter's API to access various AI models for text and vision tasks.

#### Features:
- Text completion with various AI models
- Vision-language model support for image analysis
- Configurable model parameters
- Secure API key handling
- Support for both text and JSON responses
- Detailed status feedback and token usage tracking

#### Parameters:
- `base_url`: OpenRouter API endpoint URL
- `model`: Model identifier from OpenRouter
- `api_key`: Your OpenRouter API key (securely masked in UI)
- `system_prompt`: Optional system context setting
- `user_prompt`: Main prompt/question for the model
- `temperature`: Controls response randomness (0.0-2.0)
- `top_p`: Nucleus sampling threshold (0.0-1.0)
- `top_k`: Vocabulary limit (1-1000)
- `frequency_penalty`: Token frequency penalty (-2.0 to 2.0)
- `presence_penalty`: Token presence penalty (-2.0 to 2.0)
- `repetition_penalty`: Repetition penalty (1.0-2.0)
- `response_format`: Choose between text or JSON object output
- `image_input`: Optional image for vision-capable models
- `additional_params`: Optional JSON object for extra model parameters (e.g., seed, min_p, top_a)

### OpenRouter Models Node
Query and filter available models from OpenRouter's API.

#### Features:
- Retrieve complete list of available models
- Filter models using custom search terms
- Sort models by name, pricing, or context length
- Detailed model information including pricing and context length
- Easy-to-read formatted output

#### Parameters:
- `api_key`: Your OpenRouter API key (securely masked in UI)
- `filter_text`: Text to filter models (e.g., 'free', 'gpt', 'claude')
- `sort_by`: Sort models by name, pricing, or context length
- `sort_order`: Choose ascending or descending sort order

#### Usage:
1. Add the OpenRouter Models node to your workflow
2. Set your OpenRouter API key
3. (Optional) Set filter text to find specific models
4. Choose sorting preferences
5. Connect output to a Text node to view results
6. Copy desired model ID for use in OpenRouter node

#### Tips:
- Use 'free' filter to find free models
- Combine terms like 'free llama' for specific searches
- Sort by pricing to find cost-effective options
- Check context length for your use case needs
- Leave filter empty to see all available models

## Installation

Use ComfyManager or to manually install:

1. Clone this repository into your ComfyUI custom_nodes folder:
bash
cd ComfyUI/custom_nodes
git clone https://github.com/EnragedAntelope/ComfyUI-EACloudNodes

2. Install required packages:
cd ComfyUI-EACloudNodes
pip install -r requirements.txt

3. Restart ComfyUI

## Usage

### Basic Text Generation
1. Add the OpenRouter node to your workflow
2. Set your OpenRouter API key
3. Choose a text model (e.g., `google/gemma-2-9b-it:free`)
4. Enter your prompt in the `user_prompt` field
5. Connect the node's output to a Text node to display results

### Vision Analysis
1. Add the OpenRouter node to your workflow
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
- Use `additional_params` to set model-specific parameters in JSON format:
  ```json
  {
    "seed": 42,
    "min_p": 0.1,
    "top_a": 0.8
  }
  ```

### Tips
- Lower temperature (0.1-0.3) for more focused responses
- Higher temperature (0.7-1.0) for more creative outputs
- Use presence_penalty to reduce repetition
- Monitor token usage to optimize costs
- Save API key in the node for reuse in workflows


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
