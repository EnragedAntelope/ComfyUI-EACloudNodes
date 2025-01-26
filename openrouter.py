import base64
import json
import requests
from PIL import Image
import io
import torch
from torchvision.transforms import ToPILImage

class OpenrouterNode:
    """
    A node for interacting with OpenRouter API.
    Supports text and vision-language models through OpenRouter's API.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {
                    "multiline": False,
                    "default": "https://openrouter.ai/api/v1/chat/completions",
                    "tooltip": "OpenRouter API endpoint URL"
                }),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "google/gemma-2-9b-it:free",
                    "tooltip": "Model identifier from OpenRouter (e.g., anthropic/claude-3-opus, google/gemini-pro)"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Your OpenRouter API key from openrouter.ai",
                    "password": True
                }),
                "system_prompt": ("STRING", {
    "multiline": True,
    "default": "You are a helpful AI assistant. Provide clear, accurate, and concise responses. If you're unsure about something, say so. Avoid harmful or unethical content.",
    "tooltip": "Optional system prompt to set context/behavior",
    "lines": 3  # Reduced from 4 since it's shorter
}),
"user_prompt": ("STRING", {
    "multiline": True,
    "default": """Overwrite this with your user prompt.

Get OpenRouter API key at https://openrouter.ai/settings/keys

Review parameter details and options here:
https://openrouter.ai/docs/parameters

Available models here:
https://openrouter.ai/models""",
    "tooltip": "Main prompt/question for the model",
    "lines": 10  # Increased from 8 for better visibility
}),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "round": 2,
                    "tooltip": "Controls randomness in responses (0.0 = deterministic, 2.0 = very random)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 2,
                    "tooltip": "Nucleus sampling threshold"
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Limits vocabulary to top K tokens"
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
                "max_tokens": ("INT", {
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
                    "tooltip": "Additional OpenRouter parameters in JSON format",
                    "lines": 6
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "status",)
    FUNCTION = "get_completion"
    CATEGORY = "OpenRouter"
    OUTPUT_NODE = True

    def get_completion(
        self, base_url: str, model: str, api_key: str, system_prompt: str,
        user_prompt: str, temperature: float, top_p: float, top_k: int,
        frequency_penalty: float, presence_penalty: float,
        repetition_penalty: float, response_format: str, seed: int,
        seed_mode: str, max_tokens: int,
        image_input=None, additional_params=None
    ) -> tuple[str, str]:
        try:
            # Add user prompt validation
            if not user_prompt.strip():
                return ("", "Error: User prompt is required")

            # Handle seed based on mode
            if seed_mode == "randomize":
                import random
                seed = random.randint(0, 0xffffffffffffffff)
            elif seed_mode == "increment":
                seed = (seed + 1) % 0xffffffffffffffff
            elif seed_mode == "decrement":
                seed = (seed - 1) if seed > 0 else 0xffffffffffffffff
            # "fixed" mode doesn't modify the seed

            if not api_key.strip():
                return ("", "Error: API key is required")

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
                            return ("", "Error: Image tensor must be 3D after squeezing")
                        
                        # Ensure correct channel order
                        if image_input.shape[-1] in [1, 3, 4]:  # HWC format
                            image_input = image_input.permute(2, 0, 1)  # Convert to CHW
                        
                        pil_image = ToPILImage()(image_input)
                    elif isinstance(image_input, Image.Image):
                        pil_image = image_input
                    else:
                        return ("", "Error: Unsupported image input type")

                    # Add size validation
                    if pil_image.size[0] * pil_image.size[1] > 2048 * 2048:
                        return ("", "Error: Image too large. Maximum size is 2048x2048")
                    
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
                    return ("", f"Image Processing Error: {str(img_err)}")

            # Prepare request body with standard parameters
            body = {
                "model": model,
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
                    return ("", "Error: Invalid JSON in additional parameters")

            # Add detailed error handling for the API response
            try:
                response = requests.post(base_url, headers=headers, json=body, timeout=120)
                
                # Handle different response status codes
                if response.status_code == 401:
                    return ("", "Error: Invalid API key or unauthorized access")
                elif response.status_code == 429:
                    return ("", "Error: Rate limit exceeded. Please wait before trying again")
                elif response.status_code == 500:
                    return ("", "Error: OpenRouter service error. Please try again later")
                elif response.status_code == 400:
                    return ("", "Error: Bad request - check model name and parameters")
                elif response.status_code == 413:
                    return ("", "Error: Payload too large - try reducing prompt or image size")
                
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
                    return (assistant_message, status_msg)
                else:
                    return ("", "Error: No response content from the model")

            except requests.exceptions.Timeout:
                return ("", "Error: Request timed out. Please try again")
            except requests.exceptions.RequestException as req_err:
                return ("", f"Request Error: {str(req_err)}")
            except json.JSONDecodeError:
                return ("", "Error: Invalid JSON response from OpenRouter")

        except Exception as e:
            return ("", f"Unexpected Error: {str(e)}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "OpenrouterNode": OpenrouterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenrouterNode": "OpenRouter Node"
}
