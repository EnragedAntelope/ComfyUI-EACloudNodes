"""
ComfyUI-EACloudNodes
A collection of nodes for interacting with various cloud services.

Repository: https://github.com/EnragedAntelope/ComfyUI-EACloudNodes
"""

# Import v3 Groq node
try:
    from .groq_node import comfy_entrypoint as groq_entrypoint, GroqNode, NODE_CLASS_MAPPINGS as GROQ_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as GROQ_DISPLAY_MAPPINGS
    GROQ_V3_AVAILABLE = True
except ImportError:
    # Fallback to v1 if v3 is not available
    from .groq_node import GroqNode, NODE_CLASS_MAPPINGS as GROQ_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as GROQ_DISPLAY_MAPPINGS
    GROQ_V3_AVAILABLE = False

# Import v1 OpenRouter nodes (will be converted to v3 in future)
from .openrouter import OpenrouterNode, NODE_CLASS_MAPPINGS as OPENROUTER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as OPENROUTER_DISPLAY_MAPPINGS
from .openrouter_models import OpenRouterModels, NODE_CLASS_MAPPINGS as OPENROUTER_MODELS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as OPENROUTER_MODELS_DISPLAY_MAPPINGS

# Initialize mappings for v1 compatibility
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenrouterNode": "OpenRouter Chat",
    "GroqNode": "Groq Chat",
    "OpenRouterModels": "OpenRouter Models"
}

# Update with all nodes
NODE_CLASS_MAPPINGS.update(OPENROUTER_MAPPINGS)
NODE_CLASS_MAPPINGS.update(OPENROUTER_MODELS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(GROQ_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS.update(OPENROUTER_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(OPENROUTER_MODELS_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(GROQ_DISPLAY_MAPPINGS)

# V1 compatibility function
def get_node_mapping():
    """V1 API compatibility function"""
    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# V3 entry point
async def comfy_entrypoint():
    """
    ComfyUI v3 entry point
    Returns a combined extension with all available nodes
    """
    if GROQ_V3_AVAILABLE:
        from comfy_api.latest import ComfyExtension, io

        class EACloudNodesExtension(ComfyExtension):
            """Combined extension for all EACloudNodes"""

            async def get_node_list(self) -> list[type[io.ComfyNode]]:
                # Get Groq v3 nodes
                groq_ext = await groq_entrypoint()
                groq_nodes = await groq_ext.get_node_list()

                # Future: Add v3 OpenRouter nodes when converted
                # openrouter_ext = await openrouter_entrypoint()
                # openrouter_nodes = await openrouter_ext.get_node_list()

                return groq_nodes  # + openrouter_nodes (when converted)

        return EACloudNodesExtension()
    else:
        # Fallback for environments without v3 support
        raise ImportError("ComfyUI v3 API not available. Using v1 compatibility mode.")

# Optional: Provide web extension directory
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "comfy_entrypoint"]
