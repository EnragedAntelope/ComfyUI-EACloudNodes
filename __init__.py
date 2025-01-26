"""
ComfyUI-EACloudNodes
A collection of nodes for interacting with various cloud services.

Repository: https://github.com/EnragedAntelope/ComfyUI-EACloudNodes
"""

# Import nodes - flat structure
from .openrouter import OpenrouterNode, NODE_CLASS_MAPPINGS as OPENROUTER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as OPENROUTER_DISPLAY_MAPPINGS
from .openrouter_models import OpenRouterModels, NODE_CLASS_MAPPINGS as OPENROUTER_MODELS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as OPENROUTER_MODELS_DISPLAY_MAPPINGS
from .groq_node import GroqNode, NODE_CLASS_MAPPINGS as GROQ_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as GROQ_DISPLAY_MAPPINGS

# Initialize mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Update with all nodes
NODE_CLASS_MAPPINGS.update(OPENROUTER_MAPPINGS)
NODE_CLASS_MAPPINGS.update(OPENROUTER_MODELS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(GROQ_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS.update(OPENROUTER_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(OPENROUTER_MODELS_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(GROQ_DISPLAY_MAPPINGS)

# This function is called by ComfyUI to get the mappings
def get_node_mapping():
    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Optional: Provide web extension directory
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 
