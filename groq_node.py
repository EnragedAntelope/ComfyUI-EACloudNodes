import json
import requests
import base64
from PIL import Image
import io
import torch
from torchvision.transforms import ToPILImage

class GroqNode:
    """
    A node for interacting with Groq's API.
    Supports text and vision-language models through Groq's API.
    """
    
    # Default models list from Groq documentation
    DEFAULT_MODELS = [
        # Production Models
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-guard-3-8b",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        # Preview Models
        "llama-3.3-70b-specdec",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "⚠️ Your Groq API key from console.groq.com/keys (Note: key will be visible - take care when sharing workflows)",
                    "password": True,
                    "sensitive": True
                }),
                "model": ("STRING", {
                    "default": "llama-3.3-70b-versatile",
                    "options": cls.DEFAULT_MODELS,
                    "tooltip": "Model identifier from Groq. Vision features available in models with 'vision' in their name. Default options shown but custom values allowed."
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": """You are a helpful AI assistant. Provide clear, accurate, and concise responses. If you're unsure about something, say so. Avoid harmful or unethical content.""",
                    "tooltip": "Optional system prompt to set context/behavior",
                    "lines": 4
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": """Overwrite this with your user prompt.

Get your Groq API key at: https://console.groq.com/keys
Model info at: https://console.groq.com/docs/models

Additional parameters can be set via the additional_params field.
Examples:

{
    "temperature": 0.5,
    "top_p": 0.7,
    "max_tokens": 2000,
    "presence_penalty": 0.1,
    "frequency_penalty": 0.1
}

or

{
    "stop": ["\n\n", "Human:", "Assistant:"],
    "response_format": {"type": "json_object"}
}""",
                    "tooltip": "Main prompt/question for the model",
                    "lines": 8
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
                    "tooltip": "Nucleus sampling threshold"
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
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for reproducible outputs. 0 means random seed."
                }),
                "seed_mode": (["fixed", "increment", "decrement", "randomize"], {
                    "default": "fixed",
                    "tooltip": "Controls how seed changes between runs"
                }),
                "max_completion_tokens": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 32768,
                    "step": 1,
                    "tooltip": "Maximum number of tokens to generate (1-32768)"
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

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "status",)
    FUNCTION = "chat_completion"
    CATEGORY = "Groq"
    OUTPUT_NODE = True  # Add this for green coloring

    def chat_completion(
        self, api_key: str, model: str, system_prompt: str, user_prompt: str,
        temperature: float, top_p: float, frequency_penalty: float,
        presence_penalty: float, response_format: str, seed: int,
        seed_mode: str, max_completion_tokens: int,
        image_input=None, additional_params=None
    ) -> tuple[str, str]:
        """
        Handles chat completion requests to Groq API
        """
        try:
            # Validate user prompt
            if not user_prompt.strip():
                return "", "Error: User prompt is required"

            # Handle seed based on mode
            if seed_mode == "randomize":
                import random
                seed = random.randint(0, 0xffffffffffffffff)
            elif seed_mode == "increment":
                seed = (seed + 1) % 0xffffffffffffffff
            elif seed_mode == "decrement":
                seed = (seed - 1) if seed > 0 else 0xffffffffffffffff

            # Validate API key
            if not api_key.strip():
                return "", "Error: API key is required"

            # Initialize messages list
            messages = []
            
            # Add system prompt if provided
            if system_prompt.strip():
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
                            return "", "Error: Image tensor must be 3D after squeezing"
                        
                        if image_input.shape[-1] in [1, 3, 4]:
                            image_input = image_input.permute(2, 0, 1)
                        
                        pil_image = ToPILImage()(image_input)
                    elif isinstance(image_input, Image.Image):
                        pil_image = image_input
                    else:
                        return "", "Error: Unsupported image input type"

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
                    return "", f"Image Processing Error: {str(img_err)}"

            # Add user message
            messages.append({
                "role": "user",
                "content": user_content
            })

            # Prepare request body
            body = {
                "model": model,
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
                    return "", "Error: Invalid JSON in additional parameters"

            # Make API request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=120
            )

            # Handle response
            if response.status_code == 401:
                return "", "Error: Invalid API key"
            elif response.status_code == 429:
                return "", "Error: Rate limit exceeded"
            elif response.status_code != 200:
                return "", f"Error: API returned status {response.status_code}"

            response_json = response.json()
            
            # Extract useful information for status
            model_used = response_json.get("model", "unknown")
            tokens = response_json.get("usage", {})
            prompt_tokens = tokens.get("prompt_tokens", 0)
            completion_tokens = tokens.get("completion_tokens", 0)
            status_msg = f"Success: Used {model_used} | Tokens: {prompt_tokens}+{completion_tokens}={prompt_tokens+completion_tokens}"

            if "choices" in response_json and len(response_json["choices"]) > 0:
                content = response_json["choices"][0].get("message", {}).get("content", "")
                return (content, status_msg)
            else:
                return "", "Error: No response content from model"

        except Exception as e:
            return "", f"Unexpected Error: {str(e)}"

# Node registration
NODE_CLASS_MAPPINGS = {
    "GroqNode": GroqNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroqNode": "Groq Chat"
} 
