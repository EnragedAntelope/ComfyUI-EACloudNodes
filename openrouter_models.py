"""
OpenRouter Models Node for ComfyUI v3
Query and filter available models from OpenRouter's API.
"""

import json
import requests
from typing import Tuple

from comfy_api.latest import ComfyExtension, io


class SortByEnum(io.ComboInput):
    """Enum for sort field selection"""
    OPTIONS = ["name", "pricing", "context_length"]


class SortOrderEnum(io.ComboInput):
    """Enum for sort order selection"""
    OPTIONS = ["ascending", "descending"]


class OpenRouterModels(io.ComfyNode):
    """
    A node for retrieving and filtering available models from OpenRouter API.
    Provides functionality to:
    - Fetch complete model list from OpenRouter
    - Filter models using custom search terms
    - Sort models by various criteria
    - Format model information for easy viewing
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OpenRouterModels",
            display_name="OpenRouter Models",
            category="OpenRouter",
            description="Query and filter available models from OpenRouter's API. Filter by text, sort by name/pricing/context length, and view detailed model information.",
            inputs=[
                io.String.Input(
                    "api_key",
                    default="",
                    multiline=False,
                    tooltip="⚠️ Your OpenRouter API key from https://openrouter.ai/keys (Note: key will be visible - take care when sharing workflows)"
                ),
                io.String.Input(
                    "filter_text",
                    default="free",
                    multiline=False,
                    tooltip="Filter models by text. Examples: 'free' for free models (pricing=$0), 'gpt' for GPT models, 'free claude' for free Claude models. Leave empty to show all models. Multiple terms are AND-ed together."
                ),
                SortByEnum.Input(
                    "sort_by",
                    default="name",
                    tooltip="Sort models by: 'name' (alphabetically), 'pricing' (cost per token), or 'context_length' (maximum input size in tokens)."
                ),
                SortOrderEnum.Input(
                    "sort_order",
                    default="ascending",
                    tooltip="Sort order: 'ascending' (A-Z, low to high) or 'descending' (Z-A, high to low)."
                )
            ],
            outputs=[
                io.String.Output(
                    "models",
                    tooltip="Formatted list of models with ID, name, context length, and pricing information"
                ),
                io.String.Output(
                    "status",
                    tooltip="Status message indicating success, number of models found, or error details"
                )
            ]
        )

    @classmethod
    def validate_inputs(cls, api_key, **kwargs):
        """Validate inputs before execution"""
        if not api_key or not api_key.strip():
            return "OpenRouter API key is required. Get one at https://openrouter.ai/keys"
        return True

    @classmethod
    def execute(
        cls,
        api_key: str,
        filter_text: str,
        sort_by: str,
        sort_order: str
    ) -> io.NodeOutput:
        """
        Retrieves, filters, and sorts OpenRouter models based on user parameters.

        Args:
            api_key: OpenRouter API key
            filter_text: Text to filter models (space-separated terms)
            sort_by: Field to sort by (name/pricing/context_length)
            sort_order: Sort direction (ascending/descending)

        Returns:
            NodeOutput containing:
            - Formatted string of model information
            - Status message indicating success/failure
        """
        try:
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
                return io.NodeOutput(
                    "",
                    "⏱️ Error: Request timed out.\n- OpenRouter's servers may be busy\n- Try again later"
                )
            except requests.exceptions.ConnectionError:
                return io.NodeOutput(
                    "",
                    "📶 Error: Connection failed.\n- Check your internet connection\n- OpenRouter servers may be unreachable"
                )
            except requests.exceptions.RequestException as e:
                return io.NodeOutput(
                    "",
                    f"🌐 Error: Request failed.\n- {str(e)}\n- Check your network connection"
                )

            # Handle API response status codes with clear user guidance
            if response.status_code == 401:
                return io.NodeOutput(
                    "",
                    "🔑 Error: Invalid API key or unauthorized access.\n- Check your API key at https://openrouter.ai/keys\n- Ensure it's entered correctly\n- Generate a new key if necessary"
                )
            elif response.status_code == 429:
                return io.NodeOutput(
                    "",
                    "⚠️ Error: Rate limit exceeded.\n- You've made too many requests\n- Please wait before trying again\n- Consider upgrading your plan"
                )
            elif response.status_code == 500:
                return io.NodeOutput(
                    "",
                    "🔧 Error: OpenRouter service error.\n- This is a problem with OpenRouter's servers\n- Check status page: https://status.openrouter.ai/\n- Try again later"
                )
            elif response.status_code != 200:
                return io.NodeOutput(
                    "",
                    f"⚠️ Error: API returned status code {response.status_code}.\n- Unexpected error from OpenRouter\n- Try again later"
                )

            # Parse JSON response
            try:
                models_data = response.json().get("data", [])
            except json.JSONDecodeError:
                return io.NodeOutput(
                    "",
                    "⚠️ Error: Invalid JSON response from API.\n- OpenRouter returned malformed data\n- Try again later"
                )

            # Apply filters if provided
            if filter_text.strip():
                filter_terms = filter_text.lower().split()
                filtered_models = []
                for model in models_data:
                    # Get all relevant text fields for text search
                    model_text = (
                        f"{model.get('id', '')} "
                        f"{model.get('name', '')} "
                        f"{model.get('description', '')}"
                    ).lower()

                    # Check if model is actually free (pricing = 0)
                    pricing = model.get('pricing', {})
                    is_free = (
                        float(pricing.get('prompt', '0')) == 0 and
                        float(pricing.get('completion', '0')) == 0 and
                        float(pricing.get('image', '0')) == 0 and
                        float(pricing.get('request', '0')) == 0
                    )

                    # For each filter term, check if it matches either:
                    # 1. The term is "free" and the model is actually free (pricing = 0)
                    # 2. OR the term appears in the model text
                    matches_all_terms = all(
                        (term == 'free' and is_free) or  # Check pricing if term is "free"
                        (term != 'free' and term in model_text)  # Otherwise check text
                        for term in filter_terms
                    )

                    if matches_all_terms:
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
                return io.NodeOutput(
                    "",
                    f"⚠️ Error during sorting: {str(sort_err)}.\n- Try a different sort method\n- This may be due to inconsistent model data"
                )

            # Format output with detailed model information and clear formatting
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
                return io.NodeOutput(
                    "No models found matching your criteria.",
                    "⚠️ Warning: No models found.\n- Try broadening your filter terms\n- Or check if OpenRouter has available models"
                )

            return io.NodeOutput(
                "\n".join(model_list),
                f"✅ Success: Found {len(models_data)} models"
            )

        except Exception as e:
            return io.NodeOutput(
                "",
                f"⚠️ Unexpected Error: {str(e)}.\n- Check all input parameters\n- If the error persists, report the issue"
            )


class OpenRouterModelsExtension(ComfyExtension):
    """Extension class for OpenRouter Models node"""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [OpenRouterModels]


async def comfy_entrypoint() -> ComfyExtension:
    """Entry point for ComfyUI v3"""
    return OpenRouterModelsExtension()


# Legacy v1 compatibility
NODE_CLASS_MAPPINGS = {
    "OpenRouterModels": OpenRouterModels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterModels": "OpenRouter Models"
}
