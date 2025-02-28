import base64
import json
import requests
from PIL import Image
import io
import torch
from torchvision.transforms import ToPILImage
import random

class OpenrouterNode:
    """
    A node for interacting with OpenRouter API.
    Supports text and vision-language models through OpenRouter's API.
    """
    # Default models list
    DEFAULT_MODELS = [
        "cognitivecomputations/dolphin3.0-mistral-24b:free",
        "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
        "deepseek/deepseek-chat:free",
        "deepseek/deepseek-r1-distill-llama-70b:free",
        "deepseek/deepseek-r1:free",
        "google/gemini-2.0-flash-exp:free",
        "google/gemini-2.0-flash-lite-preview-02-05:free",
        "google/gemini-2.0-flash-thinking-exp-1219:free",
        "google/gemini-2.0-flash-thinking-exp:free",
        "google/gemini-2.0-pro-exp-02-05:free",
        "google/gemini-exp-1206:free",
        "google/gemma-2-9b-it:free",
        "google/learnlm-1.5-pro-experimental:free",
        "gryphe/mythomax-l2-13b:free",
        "huggingfaceh4/zephyr-7b-beta:free",
        "meta-llama/llama-3-8b-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "meta-llama/llama-3.2-1b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "microsoft/phi-3-medium-128k-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "mistralai/mistral-nemo:free",
        "mistralai/mistral-small-24b-instruct-2501:free",
        "moonshotai/moonlight-16b-a3b-instruct:free",
        "nousresearch/deephermes-3-llama-3-8b-preview:free",
        "nvidia/llama-3.1-nemotron-70b-instruct:free",
        "openchat/openchat-7b:free",
        "qwen/qwen-2-7b-instruct:free",
        "qwen/qwen-2.5-coder-32b-instruct:free",
        "qwen/qwen-vl-plus:free",
        "qwen/qwen2.5-vl-72b-instruct:free",
        "sophosympatheia/rogue-rose-103b-v0.2:free",
        "undi95/toppy-m-7b:free",
        "Manual Input"  # Add manual input option
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "⚠️ Your OpenRouter API key from https://openrouter.ai/keys (Note: key will be visible - take care when sharing workflows)",
                    "password": True,
                    "sensitive": True
                }),
                "model": (cls.DEFAULT_MODELS, {
                    "default": "google/gemma-2-9b-it:free",
                    "tooltip": "Select a model from the list or choose 'Manual Input' to specify a custom model"
                }),
                "manual_model": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Enter a custom model identifier (only used when 'Manual Input' is selected above)",
                }),
                "base_url": ("STRING", {
                    "multiline": False,
                    "default": "https://openrouter.ai/api/v1/chat/completions",
                    "tooltip": "OpenRouter API endpoint URL"
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Main prompt/question for the model",
                    "lines": 8
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful AI assistant. Please provide clear, accurate, and ethical responses.",
                    "tooltip": "Optional system prompt to set context/behavior",
                    "lines": 4
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Controls randomness (0.0 = deterministic, 2.0 = very random)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 2,
                    "tooltip": "Controls diversity of word choices (0.0 = focused, 1.0 = more varied)"
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Limits vocabulary to top K tokens"
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 32768,
                    "step": 1,
                    "tooltip": "Maximum number of tokens to generate (1-32768)"
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "round": 2,
                    "tooltip": "Penalizes frequent tokens (-2.0 to 2.0)"
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "round": 2,
                    "tooltip": "Penalizes repeated tokens (-2.0 to 2.0)"
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.1,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.01,
                    "round": 2,
                    "tooltip": "Penalizes repetition (1.0 = off, >1.0 = more penalty)"
                }),
                "response_format": (["text", "json_object"], {
                    "default": "text",
                    "tooltip": "Format of the model's response"
                }),
                "seed_mode": (["fixed", "random", "increment", "decrement"], {
                    "default": "random",
                    "tooltip": "Control seed behavior: fixed (use seed value), random (new seed each time), increment/decrement (change by 1)"
                }),
                "seed_value": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9007199254740991,  # 2^53 - 1, max safe integer in JavaScript
                    "tooltip": "Seed value for 'fixed' mode. Limited to JS safe integer range."
                }),
                "max_retries": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Maximum number of retry attempts for recoverable errors"
                }),
            },
            "optional": {
                "image_input": ("IMAGE", {
                    "tooltip": "Optional image input for vision-capable models"
                }),
                "additional_params": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional OpenRouter parameters in JSON format",
                    "lines": 6
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("response", "status", "help",)
    FUNCTION = "chat_completion"
    CATEGORY = "OpenRouter"
    OUTPUT_NODE = True

    def chat_completion(
        self, api_key: str, model: str, manual_model: str,
        base_url: str,
        user_prompt: str, system_prompt: str,
        temperature: float, top_p: float, top_k: int,
        max_tokens: int, frequency_penalty: float,
        presence_penalty: float, repetition_penalty: float,
        response_format: str, seed_mode: str, seed_value: int,
        max_retries: int, image_input=None, additional_params=None
    ) -> tuple[str, str, str]:
        help_text = """ComfyUI-EACloudNodes - OpenRouter Chat
Repository: https://github.com/EnragedAntelope/ComfyUI-EACloudNodes

Key Settings:
- API Key: Get from https://openrouter.ai/keys
- Model: Choose from dropdown or use Manual Input
- Manual Model: Custom model ID (when Manual Input selected)
- System Prompt: Set behavior/context
- User Prompt: Main input for the model
- Temperature: 0.0 (focused) to 2.0 (creative)
- Top-p: Nucleus sampling threshold
- Top-k: Vocabulary limit
- Max Tokens: Limit response length
- Frequency Penalty: Control token frequency
- Presence Penalty: Control token presence
- Repetition Penalty: Control repetition
- Response Format: Text or JSON output
- Seed Mode: Fixed/random/increment/decrement
- Max Retries: Auto-retry on errors (0-5)

Optional:
- Image Input: For vision-capable models
- Additional Params: Extra model parameters

For vision models:
1. Select a vision-capable model
2. Connect image to 'image_input'
3. Describe or ask about the image in user_prompt"""

        try:
            # Use manual_model if "Manual Input" is selected
            actual_model = manual_model if model == "Manual Input" else model

            # Validate model
            if model == "Manual Input" and not manual_model.strip():
                return "", "🔑 Error: Manual model identifier is required when 'Manual Input' is selected.\n- Enter a valid model ID in the manual_model field", help_text

            # Add user prompt validation
            if not user_prompt.strip():
                return "", "❌ Error: User prompt is required.\n- Please enter a prompt in the user_prompt field", help_text

            # Handle seed based on mode, limiting to JS safe integer range (2^53 - 1)
            # This avoids the "seed has type number" error with the API
            MAX_SAFE_INT = 9007199254740991  # 2^53 - 1, max safe integer in JavaScript
            
            if seed_mode == "random":
                seed = random.randint(0, MAX_SAFE_INT)
            elif seed_mode == "increment":
                seed = (seed_value + 1) % MAX_SAFE_INT
            elif seed_mode == "decrement":
                seed = (seed_value - 1) if seed_value > 0 else MAX_SAFE_INT
            else:  # "fixed"
                # Ensure seed value is within safe range
                seed = min(seed_value, MAX_SAFE_INT)

            if not api_key.strip():
                return "", "🔑 Error: OpenRouter API key is required.\n- Get one at https://openrouter.ai/keys\n- Add it to the api_key field", help_text

            if not actual_model.strip():
                return "", "❌ Error: Model identifier required.\n- Choose a model from the dropdown\n- Or specify a custom model (e.g., 'anthropic/claude-3-opus')", help_text

            if not base_url.strip():
                return "", "❌ Error: OpenRouter API endpoint URL is required.\n- The default URL should work in most cases\n- Only change if you know what you're doing", help_text

            if not base_url.startswith(("http://", "https://")):
                return "", "❌ Error: Invalid API endpoint URL format.\n- URL must start with http:// or https://\n- Use the default URL if unsure: https://openrouter.ai/api/v1/chat/completions", help_text

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            # Initialize messages list
            messages = []
            
            # Add system prompt if provided
            if system_prompt.strip():
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Prepare user message content
            messages.append({
                "role": "user",
                "content": user_prompt  # Direct text for non-image messages
            })

            # Handle image input if provided
            if image_input is not None:
                try:
                    if isinstance(image_input, torch.Tensor):
                        # Handle 4D tensors (batch dimension)
                        if image_input.dim() == 4:
                            image_input = image_input.squeeze(0)
                        if image_input.dim() != 3:
                            return "", "❌ Error: Image tensor must be 3D after squeezing.\n- Check your image processing chain\n- Ensure the image has valid dimensions", help_text
                        
                        # Ensure correct channel order
                        if image_input.shape[-1] in [1, 3, 4]:  # HWC format
                            image_input = image_input.permute(2, 0, 1)  # Convert to CHW
                        
                        pil_image = ToPILImage()(image_input)
                    elif isinstance(image_input, Image.Image):
                        pil_image = image_input
                    else:
                        return "", "❌ Error: Unsupported image input type.\n- Make sure you're connecting a valid image output\n- Try using a LoadImage or other image-producing node", help_text

                    # Add size validation
                    if pil_image.size[0] * pil_image.size[1] > 2048 * 2048:
                        return "", "❌ Error: Image too large.\n- Maximum size is 2048x2048\n- Please resize your image\n- Try using a ResizeImage node", help_text
                    
                    # Convert image to base64
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    # Add image to user content
                    user_content = [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ]
                    
                    # Use this message structure instead of adding duplicate
                    messages = [
                        *({"role": "system", "content": system_prompt} 
                          for _ in [1] if system_prompt.strip()),
                        {"role": "user", "content": user_content}
                    ]
                except Exception as img_err:
                    return "", f"🖼️ Image Processing Error: {str(img_err)}.\n- Check that your image is valid\n- Try using a different image\n- If using a tensor, ensure valid format", help_text

            # Prepare request body with standard parameters
            body = {
                "model": actual_model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "repetition_penalty": repetition_penalty,
                "max_tokens": max_tokens,
                "seed": seed
            }

            # Add response format if json_object is selected
            if response_format == "json_object":
                body["response_format"] = {"type": "json_object"}

            # Parse and add additional parameters if provided
            if additional_params and additional_params.strip():
                try:
                    extra_params = json.loads(additional_params)
                    body.update(extra_params)
                except json.JSONDecodeError:
                    return "", "❌ Error: Invalid JSON in additional parameters.\n- Check your JSON syntax\n- Example format: {\"top_k\": 50}", help_text

            # Add detailed error handling for the API response
            retries = 0
            while True:
                try:
                    response = requests.post(base_url, headers=headers, json=body, timeout=120)
                    
                    # Define retryable status codes
                    retryable_codes = {429, 500, 502, 503, 504}
                    
                    if response.status_code in retryable_codes and retries < max_retries:
                        retries += 1
                        # Add exponential backoff
                        import time
                        time.sleep(2 ** retries)  # 2, 4, 8, 16... seconds
                        continue
                    
                    # Enhanced error handling with user guidance
                    if response.status_code == 401:
                        return "", "🔑 Error: Invalid API key or unauthorized access.\n- Check your API key at https://openrouter.ai/keys\n- Ensure it's entered correctly\n- Generate a new key if necessary", help_text
                    elif response.status_code == 429:
                        return "", f"⚠️ Error: Rate limit exceeded. Tried {retries} times.\n- Your account has reached its request limit\n- Consider upgrading your plan or using paid models\n- Try again later", help_text
                    elif response.status_code == 500:
                        return "", f"🔧 Error: OpenRouter service error.\n- This is a problem with OpenRouter's servers, not your request\n- Check OpenRouter status: https://status.openrouter.ai/\n- Try again later", help_text
                    elif response.status_code == 400:
                        error_msg = "Bad request"
                        try:
                            error_data = response.json()
                            if "error" in error_data:
                                error_msg = error_data.get("error", {}).get("message", "Bad request")
                        except:
                            pass
                        return "", f"❌ Error: {error_msg}\n- Check your parameters (especially model name)\n- Verify prompt format and content\n- Ensure inputs are correctly formatted", help_text
                    elif response.status_code == 404:
                        return "", f"❓ Error: Model '{actual_model}' not found.\n- Check if the model name is correct\n- Select a different model from the dropdown\n- Use the OpenRouterModels node to see available models", help_text
                    elif response.status_code == 413:
                        return "", "⚠️ Error: Payload too large.\n- Try reducing prompt length\n- Resize your image to smaller dimensions\n- Use a more compressed image format", help_text
                    elif response.status_code == 502:
                        return "", f"🔌 Error: Model provider is unavailable.\n- The specific model provider ({actual_model.split('/')[0]}) may be down\n- Try a different model from another provider\n- Check OpenRouter status: https://status.openrouter.ai/", help_text
                    elif response.status_code == 504:
                        return "", f"⏱️ Error: Gateway timeout.\n- The request took too long to process\n- Try with a smaller prompt, smaller image, or fewer tokens\n- Use a faster model\n- Try again later", help_text
                    elif response.status_code != 200:
                        error_msg = "Unknown error"
                        try:
                            error_data = response.json()
                            if "error" in error_data:
                                error_msg = error_data.get("error", {}).get("message", "Unknown error")
                        except:
                            pass
                        return "", f"⚠️ Error: API returned status {response.status_code}: {error_msg}.\n- Tried {retries} times\n- Check OpenRouter status: https://status.openrouter.ai/\n- Try again with different parameters", help_text
                    
                    response.raise_for_status()
                    response_json = response.json()

                    # Extract useful information for status
                    model_used = response_json.get("model", "unknown")
                    tokens = response_json.get("usage", {})
                    prompt_tokens = tokens.get("prompt_tokens", 0)
                    completion_tokens = tokens.get("completion_tokens", 0)
                    status_msg = f"✅ Success: Used {model_used} | Tokens: {prompt_tokens}+{completion_tokens}={prompt_tokens+completion_tokens}"

                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        assistant_message = response_json["choices"][0].get("message", {}).get("content", "")
                        return (assistant_message, status_msg, help_text)
                    else:
                        return "", "❓ Error: No response content from the model.\n- Try simplifying your request\n- Use a different model\n- Check if your prompt follows model guidelines", help_text

                except requests.exceptions.Timeout:
                    return "", "⏱️ Error: Request timed out.\n- OpenRouter or the model provider's servers may be overloaded\n- Try again later\n- Consider using a different model", help_text
                except requests.exceptions.ConnectionError:
                    return "", "📶 Error: Connection failed.\n- Check your internet connection\n- OpenRouter's servers may be unreachable\n- Try again later", help_text
                except requests.exceptions.RequestException as req_err:
                    return "", f"🌐 Request Error: {str(req_err)}\n- This is likely a network issue\n- Check your internet connection\n- Try again later", help_text
                except json.JSONDecodeError:
                    return "", "⚠️ Error: Invalid JSON response from OpenRouter.\n- The service returned malformed data\n- Try again later or with a different model", help_text
                except Exception as e:
                    return "", f"⚠️ Unexpected Error: {str(e)}\n- Try again with different parameters\n- If the error persists, try a different model", help_text

        except Exception as e:
            return "", f"⚠️ Unexpected Error: {str(e)}\n- Check all input parameters\n- Verify your API key and model selection\n- If the error persists, report the issue", help_text

# Node registration
NODE_CLASS_MAPPINGS = {
    "OpenrouterNode": OpenrouterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenrouterNode": "OpenRouter Chat"
}
