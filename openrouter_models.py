import json
import requests
from typing import Tuple, List, Dict, Any

class OpenRouterModels:
    """
    A node for retrieving and filtering available models from OpenRouter API.
    Provides functionality to:
    - Fetch complete model list from OpenRouter
    - Filter models using custom search terms
    - Sort models by various criteria
    - Format model information for easy viewing
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Your OpenRouter API key from openrouter.ai (required for API access)",
                    "password": True
                }),
                "filter_text": ("STRING", {
                    "multiline": False,
                    "default": "free",
                    "tooltip": "Filter models by text. Examples:\n- 'free' for free models\n- 'gpt' for GPT models\n- 'free claude' for free Claude models\nLeave empty to show all models"
                }),
                "sort_by": (["name", "pricing", "context_length"], {
                    "default": "name",
                    "tooltip": "Sort models by:\n- name: alphabetically\n- pricing: cost per token\n- context_length: maximum input size"
                }),
                "sort_order": (["ascending", "descending"], {
                    "default": "ascending",
                    "tooltip": "Sort order (ascending = A-Z, low-high; descending = Z-A, high-low)"
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("models", "status",)
    FUNCTION = "get_models"
    CATEGORY = "OpenRouter"

    def get_models(self, api_key: str, filter_text: str, sort_by: str, sort_order: str) -> Tuple[str, str]:
        """
        Retrieves, filters, and sorts OpenRouter models based on user parameters.

        Args:
            api_key: OpenRouter API key
            filter_text: Text to filter models (space-separated terms)
            sort_by: Field to sort by (name/pricing/context_length)
            sort_order: Sort direction (ascending/descending)

        Returns:
            Tuple containing:
            - Formatted string of model information
            - Status message indicating success/failure
        """
        try:
            # Validate API key
            if not api_key.strip():
                return ("", "Error: API key is required")

            # Setup API request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Make API request with timeout and error handling
            try:
                response = requests.get(
                    "https://openrouter.ai/api/v1/models",
                    headers=headers,
                    timeout=30
                )
            except requests.exceptions.Timeout:
                return ("", "Error: Request timed out. Please try again")
            except requests.exceptions.ConnectionError:
                return ("", "Error: Connection failed. Please check your internet connection")
            except requests.exceptions.RequestException as e:
                return ("", f"Error: Request failed - {str(e)}")

            # Handle API response status codes
            if response.status_code == 401:
                return ("", "Error: Invalid API key or unauthorized access")
            elif response.status_code == 429:
                return ("", "Error: Rate limit exceeded. Please wait before trying again")
            elif response.status_code != 200:
                return ("", f"Error: API returned status code {response.status_code}")

            # Parse JSON response
            try:
                models_data = response.json().get("data", [])
            except json.JSONDecodeError:
                return ("", "Error: Invalid JSON response from API")

            # Apply filters if provided
            if filter_text.strip():
                filter_terms = filter_text.lower().split()
                filtered_models = []
                for model in models_data:
                    model_text = (
                        f"{model.get('id', '')} "
                        f"{model.get('name', '')} "
                        f"{model.get('description', '')}"
                    ).lower()
                    if all(term in model_text for term in filter_terms):
                        filtered_models.append(model)
                models_data = filtered_models

            # Sort models based on user preferences
            try:
                if sort_by == "pricing":
                    models_data.sort(
                        key=lambda x: float(x.get("pricing", {}).get("prompt", "0")),
                        reverse=(sort_order == "descending")
                    )
                elif sort_by == "context_length":
                    models_data.sort(
                        key=lambda x: x.get("context_length", 0),
                        reverse=(sort_order == "descending")
                    )
                else:  # sort by name
                    models_data.sort(
                        key=lambda x: x.get("name", ""),
                        reverse=(sort_order == "descending")
                    )
            except Exception as sort_err:
                return ("", f"Error during sorting: {str(sort_err)}")

            # Format output with detailed model information
            model_list = []
            for model in models_data:
                model_info = (
                    f"ID: {model.get('id')}\n"
                    f"Name: {model.get('name')}\n"
                    f"Context Length: {model.get('context_length')}\n"
                    f"Pricing (per token):\n"
                    f"  Prompt: ${model.get('pricing', {}).get('prompt', 'N/A')}\n"
                    f"  Completion: ${model.get('pricing', {}).get('completion', 'N/A')}\n"
                    f"{'=' * 40}\n"
                )
                model_list.append(model_info)

            # Return results with status
            if not model_list:
                return ("No models found matching your criteria.", "Warning: No models found")
            
            return ("\n".join(model_list), f"Success: Found {len(models_data)} models")

        except Exception as e:
            return ("", f"Unexpected Error: {str(e)}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "OpenRouterModels": OpenRouterModels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterModels": "OpenRouter Models"
} 