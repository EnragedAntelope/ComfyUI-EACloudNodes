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
    # Default models list - updated April 2025
    DEFAULT_MODELS = [
        "google/gemini-2.5-pro-exp:free",
        "google/gemini-2.0-flash-exp:free", 
        "google/gemini-2.0-flash-thinking-exp:free",
        "google/gemma-2-9b-it:free",
        "meta-llama/llama-4-scout-17b-16e-instruct:free",
        "meta-llama/llama-4-maverick-17b-128e-instruct:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "meta-llama/llama-3.2-90b-vision-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "meta-llama/llama-3.1-70b-instruct:free",
        "mistralai/mistral-small-3.1:free",
        "mistralai/mistral-small-3:free",
        "mistralai/mistral-saba-24b:free",
        "mistralai/mistral-7b-instruct:free",
        "microsoft/phi-3-medium-128k-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "qwen/qwen-2.5-7b-instruct:free",
        "qwen/qwen-2-7b-instruct:free",
        "deepseek/deephermes-3:free",
        "deepseek/deepseek-r1-distill-qwen-32b:free", 
        "deepseek/deepseek-r1-distill-llama-70b:free",
        "openchat/openchat-7b:free",
        "sophosympatheia/rogue-rose-103b-v0.2:free",
        "Manual Input"  # Add manual input option
    ]
    
    def __init__(self):
        # Initialize parameter values with defaults - this is needed for UI interactions
        self.temperature = 0.7
        self.top_p = 0.7
        self.top_k = 50
        self.max_tokens = 1000
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        self.repetition_penalty = 1.1
        self.seed_value = 0
        self.last_seed = 0
        self.max_retries = 3

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
                    "tooltip": "Enter a custom model identifier (only used when 'Manual Input' is selected above)"
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
                    "tooltip": "Penalizes frequent tokens (-2.0 to 2.0)"
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Penalizes repeated tokens (-2.0 to 2.0)"
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.1,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.01,
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
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed value for 'fixed' mode. Ignored in other modes."
                }),
                "max_retries": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Maximum number of retry attempts for recoverable errors"
                })
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
                })
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
            # Validate and sanitize numeric inputs with error handling
            try:
                temperature = max(0.0, min(2.0, float(temperature)))
                top_p = max(0.0, min(1.0, float(top_p)))
                top_k = max(1, min(1000, int(top_k)))
                max_tokens = max(1, min(32768, int(max_tokens)))
                frequency_penalty = max(-2.0, min(2.0, float(frequency_penalty)))
                presence_penalty = max(-2.0, min(2.0, float(presence_penalty)))
                repetition_penalty = max(1.0, min(2.0, float(repetition_penalty)))
                max_retries = max(0, min(5, int(max_retries)))
            except (ValueError, TypeError) as e:
                return "", f"Error: Invalid parameter value - {str(e)}", help_text
                
            # Store current parameter values
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            self.max_tokens = max_tokens
            self.frequency_penalty = frequency_penalty
            self.presence_penalty = presence_penalty
            self.repetition_penalty = repetition_penalty
            self.seed_value = seed_value
            self.max_retries = max_retries
            
            # Use manual_model if "Manual Input" is selected
            actual_model = manual_model if model == "Manual Input" else model

            # Validate model
            if model == "Manual Input" and not manual_model.strip():
                return "", "Error: Manual model identifier is required when 'Manual Input' is selected", help_text

            # Add user prompt validation
            if not user_prompt.strip():
                return ("", "Error: User prompt is required", help_text)

            # Handle seed based on mode
            if seed_mode == "random":
                seed = random.randint(0, 0xffffffffffffffff)
            elif seed_mode == "increment":
                seed = (self.last_seed + 1) % 0xffffffffffffffff
            elif seed_mode == "decrement":
                seed = (self.last_seed - 1) if self.last_seed > 0 else 0xffffffffffffffff
            else:  # "fixed"
                seed = seed_value
                
            # Store the seed we're using
            self.last_seed = seed

            if not api_key.strip():
                return ("", "Error: OpenRouter API key is required. Get one at https://openrouter.ai/keys", help_text)

            if not actual_model.strip():
                return ("", "Error: Model identifier required (e.g., 'anthropic/claude-3-opus')", help_text)

            if not base_url.strip():
                return ("", "Error: OpenRouter API endpoint URL is required", help_text)

            if not base_url.startswith(("http://", "https://")):
                return ("", "Error: Invalid API endpoint URL format", help_text)

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
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
            user_content = user_prompt  # Default to direct text for non-image messages

            # Handle image input if provided
            if image_input is not None:
                try:
                    if isinstance(image_input, torch.Tensor):
                        # Handle 4D tensors (batch dimension)
                        if image_input.dim() == 4:
                            image_input = image_input.squeeze(0)
                        if image_input.dim() != 3:
                            return ("", "Error: Image tensor must be 3D after squeezing", help_text)
                        
                        # Ensure correct channel order
                        if image_input.shape[-1] in [1, 3, 4]:  # HWC format
                            image_input = image_input.permute(2, 0, 1)  # Convert to CHW
                        
                        pil_image = ToPILImage()(image_input)
                    elif isinstance(image_input, Image.Image):
                        pil_image = image_input
                    else:
                        return ("", "Error: Unsupported image input type", help_text)

                    # Add size validation
                    if pil_image.size[0] * pil_image.size[1] > 2048 * 2048:
                        return ("", "Error: Image too large. Maximum size is 2048x2048. Please resize your image.", help_text)
                    
                    # Convert image to base64
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    # Add image to user content
                    user_content = [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ]
                except Exception as img_err:
                    return ("", f"Image Processing Error: {str(img_err)}", help_text)

            # Add user message with the appropriate content format
            if isinstance(user_content, list):
                # We've already constructed the formatted content for vision models
                messages.append({
                    "role": "user",
                    "content": user_content
                })
            else:
                # Simple text message
                messages.append({
                    "role": "user",
                    "content": user_content
                })

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
                    return ("", "Error: Invalid JSON in additional parameters. Example format: {\"top_k\": 50}", help_text)

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
                    
                    # Handle different response status codes
                    if response.status_code == 401:
                        return ("", "Error: Invalid API key or unauthorized access", help_text)
                    elif response.status_code == 429:
                        return ("", f"Error: Rate limit exceeded. Tried {retries} times", help_text)
                    elif response.status_code == 500:
                        return ("", f"Error: OpenRouter service error. Tried {retries} times", help_text)
                    elif response.status_code == 400:
                        return ("", "Error: Bad request - check model name and parameters", help_text)
                    elif response.status_code == 413:
                        return ("", "Error: Payload too large - try reducing prompt or image size", help_text)
                    
                    response.raise_for_status()
                    response_json = response.json()

                    # Extract useful information for status
                    model_used = response_json.get("model", "unknown")
                    tokens = response_json.get("usage", {})
                    prompt_tokens = tokens.get("prompt_tokens", 0)
                    completion_tokens = tokens.get("completion_tokens", 0)
                    status_msg = f"Success: Used {model_used} | Tokens: {prompt_tokens}+{completion_tokens}={prompt_tokens+completion_tokens}"

                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        assistant_message = response_json["choices"][0].get("message", {}).get("content", "")
                        return (assistant_message, status_msg, help_text)
                    else:
                        return ("", "Error: No response content from the model", help_text)

                except requests.exceptions.Timeout:
                    return ("", "Error: Request timed out. Please try again", help_text)
                except requests.exceptions.RequestException as req_err:
                    return ("", f"Request Error: {str(req_err)}", help_text)
                except json.JSONDecodeError:
                    return ("", "Error: Invalid JSON response from OpenRouter", help_text)
                except Exception as e:
                    return ("", f"Network Error: {str(e)}", help_text)
                
                # Break out of retry loop if we reach here
                break

        except Exception as e:
            return ("", f"Unexpected Error: {str(e)}", help_text)

# Node registration
NODE_CLASS_MAPPINGS = {
    "OpenrouterNode": OpenrouterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenrouterNode": "OpenRouter Chat"
}
