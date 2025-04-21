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
    
    # JavaScript safe integer limit (2^53 - 1)
    MAX_SAFE_INTEGER = 9007199254740991
    
    # Models that support vision capabilities
    VISION_MODELS = [
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]
    
    # Default models list from Groq documentation - updated based on current availability
    DEFAULT_MODELS = [
        # Production Models
        "llama-3.3-70b-versatile",  # Default model
        "llama-3.1-8b-instant",
        "llama-guard-3-8b",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "gemma2-9b-it",
        "distil-whisper-large-v3-en",
        "whisper-large-v3",
        "whisper-large-v3-turbo",
        # Preview Models
        "allam-2-7b",
        "deepseek-r1-distill-llama-70b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",  # Vision-capable
        "meta-llama/llama-4-scout-17b-16e-instruct",      # Vision-capable
        "mistral-saba-24b",
        "playai-tts",
        "playai-tts-arabic",
        "qwen-qwq-32b",
        "Manual Input"  # Add this option at the end
    ]
    
    def __init__(self):
        # Initialize parameter values with defaults
        self.temperature = 0.7
        self.top_p = 0.7
        self.max_completion_tokens = 1000
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
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
                    "tooltip": "Enter a custom model identifier (only used when 'Manual Input' is selected above)"
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
                    "tooltip": "Controls randomness (0.0 = deterministic, 2.0 = very random)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
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
                    "tooltip": "Penalizes frequent tokens (-2.0 to 2.0)"
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
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
                    "max": 9007199254740991,  # JavaScript safe integer limit
                    "tooltip": "Seed value for 'fixed' mode. Ignored in other modes. (0-9007199254740991)"
                }),
                "max_retries": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Maximum number of retry attempts for recoverable errors"
                }),
                "debug_mode": (["off", "on"], {
                    "default": "off",
                    "tooltip": "Enable debug mode to get more detailed error messages"
                })
            },
            "optional": {
                "image_input": ("IMAGE", {
                    "tooltip": "Optional image input for vision-capable models (currently only Llama-4 models support vision)"
                }),
                "additional_params": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional Groq parameters in JSON format",
                    "lines": 6
                })
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
        max_retries: int, debug_mode: str, image_input=None, additional_params=None
    ) -> tuple[str, str, str]:
        """
        Handles chat completion requests to Groq API
        """
        # Store current parameter values
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed_value = seed_value
        self.max_retries = max_retries
        
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
- Seed Value: Seed for 'fixed' mode (0-9007199254740991)
- Max Retries: Auto-retry on errors (0-5)
- Debug Mode: Enable to get detailed error messages

Optional:
- Image Input: Only for Llama-4 models (Scout and Maverick)
- Additional Params: Extra model parameters

Vision Support:
Currently only these models support vision inputs:
- meta-llama/llama-4-maverick-17b-128e-instruct
- meta-llama/llama-4-scout-17b-16e-instruct"""

        try:
            # Validate and sanitize numeric inputs
            try:
                temperature = max(0.0, min(2.0, float(temperature)))
                top_p = max(0.0, min(1.0, float(top_p))) 
                max_completion_tokens = max(1, min(32768, int(max_completion_tokens)))
                frequency_penalty = max(-2.0, min(2.0, float(frequency_penalty)))
                presence_penalty = max(-2.0, min(2.0, float(presence_penalty)))
                max_retries = max(0, min(5, int(max_retries)))
                # Ensure seed is within JavaScript safe integer limits
                seed_value = max(0, min(self.MAX_SAFE_INTEGER, int(seed_value)))
            except (ValueError, TypeError) as e:
                return "", f"Error: Invalid parameter value - {str(e)}", help_text
            
            # Update instance variables with sanitized values
            self.temperature = temperature
            self.top_p = top_p
            self.max_completion_tokens = max_completion_tokens
            self.frequency_penalty = frequency_penalty
            self.presence_penalty = presence_penalty
            self.max_retries = max_retries
            self.seed_value = seed_value

            # Use manual_model if "Manual Input" is selected
            actual_model = manual_model if model == "Manual Input" else model

            # Validate model
            if model == "Manual Input" and not manual_model.strip():
                return "", "Error: Manual model identifier is required when 'Manual Input' is selected", help_text

            # Validate user prompt
            if not user_prompt.strip():
                return "", "Error: User prompt is required", help_text

            # Handle seed based on mode
            if seed_mode == "random":
                seed = random.randint(0, self.MAX_SAFE_INTEGER)
            elif seed_mode == "increment":
                seed = (self.last_seed + 1) % self.MAX_SAFE_INTEGER
            elif seed_mode == "decrement":
                seed = (self.last_seed - 1) if self.last_seed > 0 else self.MAX_SAFE_INTEGER
            else:  # "fixed"
                seed = seed_value
            
            # Store the seed we're using
            self.last_seed = seed

            # Validate API key
            if not api_key.strip():
                return "", "Error: Groq API key is required. Get one at console.groq.com/keys", help_text

            # Check if model supports vision capabilities
            is_vision_model = actual_model in self.VISION_MODELS

            # Vision model validation
            if image_input is not None and not is_vision_model:
                return "", f"Error: Model '{actual_model}' does not support vision inputs. Only the following Groq models support vision: {', '.join(self.VISION_MODELS)}", help_text

            # Initialize messages list
            messages = []
            
            # Add system prompt if provided and enabled
            if system_prompt.strip() and send_system == "yes":
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Handle different message formats based on whether it's a vision model with image
            if image_input is not None and is_vision_model:
                try:
                    # Process image for vision models
                    if isinstance(image_input, torch.Tensor):
                        if image_input.dim() == 4:
                            image_input = image_input.squeeze(0)
                        if image_input.dim() != 3:
                            return "", "Error: Image tensor must be 3D after squeezing", help_text
                        
                        if image_input.shape[-1] in [1, 3, 4]:
                            image_input = image_input.permute(2, 0, 1)
                        
                        pil_image = ToPILImage()(image_input)
                    elif isinstance(image_input, Image.Image):
                        pil_image = image_input
                    else:
                        return "", "Error: Unsupported image input type", help_text

                    # Convert image to base64
                    buffered = io.BytesIO()
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
                    return "", f"Image Processing Error: {str(img_err)}", help_text
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
                "max_tokens": max_completion_tokens
            }
            
            # Only add seed if it's supported (it is on most Groq models)
            if seed is not None:
                body["seed"] = seed
                
            # Only add frequency_penalty and presence_penalty if they're non-zero
            # as not all Groq models support these parameters
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
                    body.update(extra_params)
                except json.JSONDecodeError:
                    return "", "Error: Invalid JSON in additional parameters. Example format: {\"top_k\": 50}", help_text

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
                    
                    # For 400 errors, try to get detailed error information
                    if response.status_code == 400:
                        try:
                            error_json = response.json()
                            error_message = error_json.get("error", {}).get("message", "Unknown error")
                            
                            # If debug mode is on, provide more detailed error info
                            if debug_mode == "on":
                                return "", f"Error 400: {error_message}\nRequest body: {json.dumps(body, indent=2)}", help_text
                            else:
                                return "", f"Error 400: {error_message}", help_text
                        except Exception:
                            # If we can't parse the error, fall back to basic message
                            return "", "Error: Bad request - check model name and parameters (enable debug mode for details)", help_text

                    # Handle other response codes
                    if response.status_code == 401:
                        return "", "Error: Invalid API key", help_text
                    elif response.status_code == 429:
                        return "", f"Error: Rate limit exceeded. Tried {retries} times", help_text
                    elif response.status_code != 200:
                        return "", f"Error: API returned status {response.status_code}. Tried {retries} times", help_text

                    response_json = response.json()
                    
                    # Extract useful information for status
                    model_used = response_json.get("model", "unknown")
                    tokens = response_json.get("usage", {})
                    prompt_tokens = tokens.get("prompt_tokens", 0)
                    completion_tokens = tokens.get("completion_tokens", 0)
                    
                    # Add seed to status message so user can see what was used
                    status_msg = f"Success: Used {model_used} | Seed: {seed} | Tokens: {prompt_tokens}+{completion_tokens}={prompt_tokens+completion_tokens}"

                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        content = response_json["choices"][0].get("message", {}).get("content", "")
                        return (content, status_msg, help_text)
                    else:
                        return "", "Error: No response content from model", help_text

                except requests.exceptions.RequestException as req_err:
                    # Only retry network-related errors
                    if retries < max_retries:
                        retries += 1
                        import time
                        time.sleep(2 ** retries)
                        continue
                    return "", f"Network Error: {str(req_err)}. Tried {retries} times.", help_text
                
                # Break out of retry loop if we reach here
                break

        except Exception as e:
            return "", f"Unexpected Error: {str(e)}", help_text

# Node registration
NODE_CLASS_MAPPINGS = {
    "GroqNode": GroqNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroqNode": "Groq Chat"
}
