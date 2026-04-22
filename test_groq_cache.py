"""
Unit tests for Groq model fetching cache behavior.
Run: python test_groq_cache.py

Note: These tests verify the caching and fallback logic.
Full integration requires ComfyUI environment.
"""
import sys
import os

# Mock the comfy_api import for standalone testing
class MockIO:
    class Input:
        def __init__(self, name, **kwargs):
            self.name = name
            self.__dict__.update(kwargs)
    class Schema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class NodeOutput:
        def __init__(self, response, status, help_text=None):
            self.response = response
            self.status = status
            self.help_text = help_text
    class ComfyNode:
        pass

class MockComfyExtension:
    pass

sys.modules['comfy_api'] = type(sys)('comfy_api')
sys.modules['comfy_api.latest'] = type(sys)('comfy_api.latest')
sys.modules['comfy_api.latest'].ComfyExtension = MockComfyExtension
sys.modules['comfy_api.latest'].io = MockIO

# Now import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groq_node

def test_cache_initial_state():
    """Cache starts empty."""
    assert groq_node._groq_model_cache["models"] is None
    assert groq_node._groq_model_cache["vision_models"] is None
    print("[PASS] Cache initial state test passed")

def test_static_fallback_without_api_key():
    """Returns static fallback when no API key provided."""
    models, vision_models = groq_node._fetch_groq_models(api_key=None)
    assert models is not None
    assert len(models) > 0
    assert "Manual Input" in models
    assert any("--- Featured ---" in m for m in models)
    print(f"[PASS] Static fallback test passed ({len(models)} models)")

def test_static_fallback_function():
    """_get_static_fallback_models returns proper structure."""
    models, vision = groq_node._get_static_fallback_models()
    assert isinstance(models, list)
    assert isinstance(vision, list)
    assert len(models) > 0
    assert len(vision) > 0
    assert "Manual Input" in models
    print(f"[PASS] Static fallback function test passed ({len(models)} models, {len(vision)} vision)")

def test_categorization_structure():
    """Model list has category headers."""
    models, _ = groq_node._fetch_groq_models(api_key=None)
    categories_found = [m for m in models if m.startswith("---")]
    assert len(categories_found) >= 5
    # Manual Input section exists
    assert "Manual Input" in models
    print(f"[PASS] Categorization structure test passed ({len(categories_found)} categories)")

def test_model_categories_mapping():
    """MODEL_CATEGORIES has expected structure."""
    assert isinstance(groq_node.MODEL_CATEGORIES, dict)
    assert "Featured" in groq_node.MODEL_CATEGORIES
    assert "Production: Chat" in groq_node.MODEL_CATEGORIES
    assert len(groq_node.MODEL_CATEGORIES) >= 7
    print(f"[PASS] Model categories mapping test passed ({len(groq_node.MODEL_CATEGORIES)} categories)")

def test_vision_patterns():
    """VISION_PATTERNS has expected patterns."""
    assert isinstance(groq_node.VISION_PATTERNS, list)
    assert len(groq_node.VISION_PATTERNS) > 0
    assert "vision" in groq_node.VISION_PATTERNS
    assert "vl" in groq_node.VISION_PATTERNS
    print(f"[PASS] Vision patterns test passed ({len(groq_node.VISION_PATTERNS)} patterns)")

def test_known_vision_models():
    """KNOWN_VISION_MODELS has expected models."""
    assert isinstance(groq_node.KNOWN_VISION_MODELS, list)
    assert len(groq_node.KNOWN_VISION_MODELS) > 0
    assert "meta-llama/llama-4-scout-17b-16e-instruct" in groq_node.KNOWN_VISION_MODELS
    print(f"[PASS] Known vision models test passed ({len(groq_node.KNOWN_VISION_MODELS)} models)")

if __name__ == "__main__":
    print("Running Groq Cache Tests...\n")
    test_cache_initial_state()
    test_static_fallback_without_api_key()
    test_static_fallback_function()
    test_categorization_structure()
    test_model_categories_mapping()
    test_vision_patterns()
    test_known_vision_models()
    print("\n[SUCCESS] All cache tests passed!")
