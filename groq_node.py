import json
import requests
import base64
from PIL import Image
import io
import torch
from torchvision.transforms import ToPILImage
import random

class GroqNode:
    """
    A node for interacting with Groq's API.
    Supports text and vision-language models through Groq's API.
    """
    
    # Default models list from Groq documentation
    DEFAULT_MODELS = [
        # Production Models
        "distil-whisper-large-v3-en",
        "gemma2-9b-it",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-guard-3-8b",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "whisper-large-v3",
        "whisper-large-v3-turbo",
        # Preview Models
        "deepseek-r1-distill-llama-70b",
        "deepseek-r1-distill-llama-70b-specdec",
        "deepseek-r1-distill-qwen-32b",
        "llama-3.3-70b-specdec",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
        "mistral-saba-24b",
        "qwen-2.5-32b",
        "qwen-2.5-coder-32b",
        "Manual Input"  # Add this option at the end
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "⚠️ Your Groq API key from https://console.groq.com/keys (Note: key will be visible - take care when sharing workflows)",
                    "password": True,
                    "sensitive": True
                }),
                "model": (cls.DEFAULT_MODELS, {
                    "default": "llama-3.3-70b-versatile",
                    "tooltip": "Select a Groq model or choose 'Manual Input' to specify a custom model"
                }),
                "manual_model": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Enter a custom model identifier (only used when 'Manual Input' is selected above)",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful AI assistant. Please provide clear, accurate, and ethical responses.",
                    "tooltip": "Optional system prompt to set context/behavior",
                    "lines": 4
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Main prompt/question for the model",
                    "lines": 8
                }),
                "send_system": (["yes", "no"], {
                    "default": "yes",
                    "tooltip": "Some models (especially vision models) don't accept system prompts. Toggle 'no' to skip sending the system prompt."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "round": 2,
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
                "max_completion_tokens": ("INT", {
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
                    "tooltip": "Additional Groq parameters in JSON format",
                    "lines": 6
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("response", "status", "help",)
    FUNCTION = "chat_completion"
    CATEGORY = "Groq"
    OUTPUT_NODE = True

    def chat_completion(
        self, api_key: str, model: str, manual_model: str,
        user_prompt: str, system_prompt: str, send_system: str,
        temperature: float, top_p: float, max_completion_tokens: int,
        frequency_penalty: float, presence_penalty: float,
        response_format: str, seed_mode: str, seed_value: int,
        max_retries: int, image_input=None, additional_params=None
    ) -> tuple[str, str, str]:
        """
        Handles chat completion requests to Groq API
        """
        help_text = """ComfyUI-EACloudNodes - Groq Chat
Repository: https://github.com/EnragedAntelope/ComfyUI-EACloudNodes

Key Settings:
- API Key: Get from https://console.groq.com/keys
- Model: Choose from dropdown or use Manual Input
- System Prompt: Set behavior/context
- User Prompt: Main input for the model
- Send System: Toggle system prompt (off for vision)
- Temperature: 0.0 (focused) to 2.0 (creative)
- Top-p: Nucleus sampling threshold
- Max Tokens: Limit response length
- Frequency Penalty: Control token frequency
- Presence Penalty: Control token presence
- Response Format: Text or JSON output
- Seed Mode: Fixed/random/increment/decrement
- Seed Value: Seed value for 'fixed' mode
- Max Retries: Auto-retry on errors (0-5)

Optional:
- Image Input: For vision-capable models
- Additional Params: Extra model parameters

For vision models:
1. Select a vision-capable model
2. Toggle 'send_system' to 'no'
3. Connect image to 'image_input'
4. Describe or ask about the image in user_prompt"""

        try:
            # Use manual_model if "Manual Input" is selected
            actual_model = manual_model if model == "Manual Input" else model

            # Validate model
            if model == "Manual Input" and not manual_model.strip():
                return "", "🔑 Error: Manual model identifier is required when 'Manual Input' is selected.\n- Enter a valid model ID in the manual_model field", help_text

            # Validate user prompt
            if not user_prompt.strip():
                return "", "❌ Error: User prompt is required.\n- Please enter a prompt in the user_prompt field", help_text

            # Handle seed based on mode, limiting to JS safe integer range (2^53 - 1)
            # This avoids the "seed has type number" error with the Groq API
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

            # Validate API key
            if not api_key.strip():
                return "", "🔑 Error: Groq API key is required.\n- Get one at console.groq.com/keys\n- Add it to the api_key field", help_text

            # Vision model validation
            if image_input is not None and "vision" not in model.lower():
                return "", f"❌ Error: Model '{model}' does not support vision inputs.\n- Please select a vision-capable model (has 'vision' in the name)\n- Or remove the image input connection", help_text

            # Initialize messages list
            messages = []
            
            # Add system prompt if provided and enabled
            if system_prompt.strip() and send_system == "yes":
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Prepare user message content
            user_content = [{"type": "text", "text": user_prompt}]

            # Handle image input if provided
            if image_input is not None:
                try:
                    if isinstance(image_input, torch.Tensor):
                        if image_input.dim() == 4:
                            image_input = image_input.squeeze(0)
                        if image_input.dim() != 3:
                            return "", "❌ Error: Image tensor must be 3D after squeezing.\n- Check your image processing chain\n- Ensure the image has valid dimensions", help_text
                        
                        if image_input.shape[-1] in [1, 3, 4]:
                            image_input = image_input.permute(2, 0, 1)
                        
                        pil_image = ToPILImage()(image_input)
                    elif isinstance(image_input, Image.Image):
                        pil_image = image_input
                    else:
                        return "", "❌ Error: Unsupported image input type.\n- Make sure you're connecting a valid image output\n- Try using a LoadImage or other image-producing node", help_text

                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    })
                except Exception as img_err:
                    return "", f"🖼️ Image Processing Error: {str(img_err)}.\n- Check that your image is valid\n- Try using a different image\n- If using a tensor, ensure valid format", help_text

            # Add user message
            messages.append({
                "role": "user",
                "content": user_content
            })

            # Prepare request body
            body = {
                "model": actual_model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_completion_tokens,
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
                        # Add exponential backoff
                        import time
                        time.sleep(2 ** retries)  # 2, 4, 8, 16... seconds
                        continue

                    # Enhanced error handling with user guidance
                    if response.status_code == 401:
                        return "", "🔑 Error: Invalid API key.\n- Check your API key at https://console.groq.com/keys\n- Ensure it's entered correctly\n- Create a new key if necessary", help_text
                    elif response.status_code == 429:
                        return "", f"⚠️ Error: Rate limit exceeded. Tried {retries} times.\n- Your account has reached its request limit\n- Consider upgrading your plan\n- Try again later", help_text
                    elif response.status_code == 400:
                        error_msg = "Bad request"
                        try:
                            error_data = response.json()
                            if "error" in error_data:
                                error_msg = error_data["error"].get("message", "Bad request")
                        except:
                            pass
                        return "", f"❌ Error: {error_msg}\n- Check your parameters (especially model name)\n- Verify prompt format and content\n- Ensure inputs are correctly formatted", help_text
                    elif response.status_code == 404:
                        return "", f"❓ Error: Model '{actual_model}' not found.\n- Check if the model name is correct\n- Select a different model from the dropdown\n- Verify the model exists in Groq's catalog", help_text
                    elif response.status_code == 500:
                        return "", f"🔧 Error: Groq service error.\n- This is a problem with Groq's servers, not your request\n- Check Groq status page: https://status.groq.com/\n- Try again later", help_text
                    elif response.status_code == 503:
                        return "", f"🚧 Error: Groq service temporarily unavailable.\n- The service may be down for maintenance\n- Check Groq status page: https://status.groq.com/\n- Try again later", help_text
                    elif response.status_code != 200:
                        error_msg = "Unknown error"
                        try:
                            error_data = response.json()
                            if "error" in error_data:
                                error_msg = error_data["error"].get("message", "Unknown error")
                        except:
                            pass
                        return "", f"⚠️ Error: API returned status {response.status_code}: {error_msg}.\n- Tried {retries} times\n- Check Groq status page: https://status.groq.com/\n- Try again with different parameters", help_text

                    response_json = response.json()
                    
                    # Extract useful information for status
                    model_used = response_json.get("model", "unknown")
                    tokens = response_json.get("usage", {})
                    prompt_tokens = tokens.get("prompt_tokens", 0)
                    completion_tokens = tokens.get("completion_tokens", 0)
                    status_msg = f"✅ Success: Used {model_used} | Tokens: {prompt_tokens}+{completion_tokens}={prompt_tokens+completion_tokens}"

                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        content = response_json["choices"][0].get("message", {}).get("content", "")
                        return (content, status_msg, help_text)
                    else:
                        return "", "❓ Error: No response content from model.\n- Try simplifying your request\n- Use a different model\n- Check if your prompt follows model guidelines", help_text

                except requests.exceptions.Timeout:
                    return "", "⏱️ Error: Request timed out.\n- Groq's servers may be overloaded\n- Try again later\n- Consider using a different model", help_text
                except requests.exceptions.ConnectionError:
                    return "", "📶 Error: Connection failed.\n- Check your internet connection\n- Groq's servers may be unreachable\n- Try again later", help_text
                except requests.exceptions.RequestException as req_err:
                    return "", f"🌐 Request Error: {str(req_err)}\n- This is likely a network issue\n- Check your internet connection\n- Try again later", help_text
                except Exception as e:
                    return "", f"⚠️ Unexpected Error: {str(e)}\n- Try again with different parameters\n- If the error persists, try a different model", help_text

        except Exception as e:
            return "", f"⚠️ Unexpected Error: {str(e)}\n- Check all input parameters\n- Verify your API key and model selection\n- If the error persists, report the issue", help_text

# Node registration
NODE_CLASS_MAPPINGS = {
    "GroqNode": GroqNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroqNode": "Groq Chat"
}
