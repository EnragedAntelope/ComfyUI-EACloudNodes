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
                    "lines": 4  # Make the text box taller
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
                    "lines": 8  # Make the text box much taller for visibility
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "round": 2,
                    "tooltip": "Controls randomness in responses (0.0 = deterministic, 2.0 = very random)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
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
                    "default": 0.2,
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
            },
            "optional": {
                "image_input": ("IMAGE", {
                    "tooltip": "Optional image input for vision-capable models"
                }),
                "additional_params": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional OpenRouter parameters in JSON format. Example:\n{\n  \"seed\": 42,\n  \"min_p\": 0.1,\n  \"top_a\": 0.8\n}",
                    "lines": 6
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "status",)
    FUNCTION = "get_completion"
    CATEGORY = "OpenRouter"

    def get_completion(self, base_url, model, api_key, system_prompt, user_prompt, 
                      temperature, top_p, top_k, frequency_penalty, presence_penalty, 
                      repetition_penalty, response_format, image_input=None, additional_params=None):
        try:
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
            user_content = [{"type": "text", "text": user_prompt}]

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

                    # Convert image to base64
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    # Add image to user content
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    })
                except Exception as img_err:
                    return ("", f"Image Processing Error: {str(img_err)}")

            # Add user message with text and optional image
            messages.append({
                "role": "user",
                "content": user_content
            })

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
                
                response.raise_for_status()
                response_json = response.json()

                # Extract useful information for status
                model_used = response_json.get("model", "unknown")
                tokens_used = response_json.get("usage", {}).get("total_tokens", 0)
                status_msg = f"Success: Used model {model_used} | Tokens: {tokens_used}"

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