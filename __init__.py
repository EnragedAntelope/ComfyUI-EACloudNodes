"""
ComfyUI-EACloudNodes
A collection of ComfyUI v3 nodes for interacting with cloud LLM services.

Repository: https://github.com/EnragedAntelope/ComfyUI-EACloudNodes
"""

from comfy_api.latest import ComfyExtension, io

from .groq_node import (
    GroqNode,
    NODE_CLASS_MAPPINGS as GROQ_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as GROQ_DISPLAY_MAPPINGS,
)
from .openrouter import (
    OpenrouterNode,
    NODE_CLASS_MAPPINGS as OPENROUTER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as OPENROUTER_DISPLAY_MAPPINGS,
)
from .openrouter_models import (
    OpenRouterModels,
    NODE_CLASS_MAPPINGS as OPENROUTER_MODELS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as OPENROUTER_MODELS_DISPLAY_MAPPINGS,
)

# ComfyUI registers a pack through NODE_CLASS_MAPPINGS when it is present, and only
# falls back to a `comfy_entrypoint` extension otherwise. These nodes subclass the v3
# `io.ComfyNode`, which exposes the v1 class attributes ComfyUI needs, so the mapping
# below registers the v3 nodes through the path ComfyUI checks first.
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(GROQ_MAPPINGS)
NODE_CLASS_MAPPINGS.update(OPENROUTER_MAPPINGS)
NODE_CLASS_MAPPINGS.update(OPENROUTER_MODELS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(GROQ_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(OPENROUTER_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(OPENROUTER_MODELS_DISPLAY_MAPPINGS)


class EACloudNodesExtension(ComfyExtension):
    """Combined extension for all EACloudNodes"""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [GroqNode, OpenrouterNode, OpenRouterModels]


async def comfy_entrypoint() -> ComfyExtension:
    """
    ComfyUI v3 entry point, kept for ComfyUI builds that prefer it over
    NODE_CLASS_MAPPINGS. Returns one extension covering every node in the pack.
    """
    return EACloudNodesExtension()


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "comfy_entrypoint",
]
