"""
Unit tests for Groq vision model detection.
Run: python test_groq_vision.py

Note: These tests verify the vision detection logic.
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

def test_known_vision_models_detected():
    """Known vision models are detected."""
    mock_api_response = [
        {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "active": True},
        {"id": "llama-3.3-70b-versatile", "active": True},
    ]
    vision_models = groq_node._detect_vision_models(mock_api_response)
    assert "meta-llama/llama-4-scout-17b-16e-instruct" in vision_models
    assert "llama-3.3-70b-versatile" not in vision_models
    print("[PASS] Known vision models test passed")

def test_pattern_detection():
    """Pattern-based detection catches vision models."""
    mock_api_response = [
        {"id": "some-vl-model", "active": True},
        {"id": "vision-model-v2", "active": True},
        {"id": "llama-4-test", "active": True},
        {"id": "regular-model", "active": True},
    ]
    vision_models = groq_node._detect_vision_models(mock_api_response)
    assert "some-vl-model" in vision_models
    assert "vision-model-v2" in vision_models
    assert "llama-4-test" in vision_models
    assert "regular-model" not in vision_models
    print("[PASS] Pattern detection test passed")

def test_inactive_models_excluded():
    """Inactive models excluded from vision list."""
    mock_api_response = [
        {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "active": False},
    ]
    vision_models = groq_node._detect_vision_models(mock_api_response)
    assert "meta-llama/llama-4-scout-17b-16e-instruct" not in vision_models
    print("[PASS] Inactive model exclusion test passed")

def test_empty_response():
    """Empty API response returns empty vision list."""
    mock_api_response = []
    vision_models = groq_node._detect_vision_models(mock_api_response)
    assert vision_models == []
    print("[PASS] Empty response test passed")

def test_case_insensitive_pattern():
    """Pattern matching is case insensitive."""
    mock_api_response = [
        {"id": "MODEL-VL-UPPERCASE", "active": True},
        {"id": "Vision-Mixed-Case", "active": True},
    ]
    vision_models = groq_node._detect_vision_models(mock_api_response)
    assert "MODEL-VL-UPPERCASE" in vision_models
    assert "Vision-Mixed-Case" in vision_models
    print("[PASS] Case insensitive pattern test passed")

if __name__ == "__main__":
    print("Running Groq Vision Detection Tests...\n")
    test_known_vision_models_detected()
    test_pattern_detection()
    test_inactive_models_excluded()
    test_empty_response()
    test_case_insensitive_pattern()
    print("\n[SUCCESS] All vision detection tests passed!")
