"""Tests for the Groq Chat node."""
import json

import pytest
import torch

import groq_node
from conftest import FakeResponse, chat_payload

GroqNode = groq_node.GroqNode


# --------------------------------------------------------------------------
# Schema / dropdown invariants
# --------------------------------------------------------------------------

def _model_options():
    schema = GroqNode.define_schema()
    return schema.input_by_id("model").options


def test_default_model_is_selectable():
    schema = GroqNode.define_schema()
    model_input = schema.input_by_id("model")
    assert model_input.default in model_input.options


def test_dropdown_has_no_audio_models():
    """Whisper/Orpheus use the audio endpoints and can never work in a chat node."""
    assert [m for m in _model_options() if groq_node._is_audio_model(m)] == []


def test_dropdown_ends_with_manual_input():
    assert _model_options()[-1] == "Manual Input"


def test_every_categorized_model_appears_in_the_static_list():
    listed = set(_model_options())
    for models in groq_node.MODEL_CATEGORIES.values():
        for model_id in models:
            assert model_id in listed


def test_help_output_is_not_duplicated(groq_call):
    help_text = groq_call().args[2]
    assert help_text.count("Key Settings:") == 1
    assert help_text.count("Repository:") == 1


# --------------------------------------------------------------------------
# Model list fetching
# --------------------------------------------------------------------------

def test_fetch_without_key_returns_static_fallback():
    models, vision = groq_node._fetch_groq_models(api_key=None)
    assert models == groq_node.STATIC_FALLBACK_MODELS
    assert vision == groq_node.KNOWN_VISION_MODELS


def test_fetch_with_key_uses_the_api(monkeypatch):
    payload = {"data": [
        {"id": "openai/gpt-oss-120b", "active": True},
        {"id": "brand-new-model", "active": True},
        {"id": "whisper-large-v3", "active": True},
        {"id": "retired-model", "active": False},
    ]}
    monkeypatch.setattr(groq_node.requests, "get", lambda *a, **k: FakeResponse(200, payload))

    models, _ = groq_node._fetch_groq_models(api_key="test-key")
    assert "brand-new-model" in models          # unknown models surface under "Other"
    assert "whisper-large-v3" not in models     # audio models filtered out
    assert "retired-model" not in models        # inactive models filtered out


def test_fetch_falls_back_when_api_errors(monkeypatch):
    monkeypatch.setattr(groq_node.requests, "get", lambda *a, **k: FakeResponse(500, {}))
    models, _ = groq_node._fetch_groq_models(api_key="test-key")
    assert models == groq_node.STATIC_FALLBACK_MODELS


def test_execute_warms_the_model_cache(monkeypatch, groq_call):
    """Running the node with a key is what makes a later Refresh show live models."""
    payload = {"data": [{"id": "live-only-model", "active": True}]}
    monkeypatch.setattr(groq_node.requests, "get", lambda *a, **k: FakeResponse(200, payload))

    assert groq_node._groq_model_cache["models"] is None
    groq_call()
    assert "live-only-model" in groq_node._groq_model_cache["models"]


def test_vision_detection_ignores_inactive_and_audio_models():
    detected = groq_node._detect_vision_models([
        {"id": "some-vl-model", "active": True},
        {"id": "openai/gpt-oss-120b", "active": True},
        {"id": "hidden-vision-model", "active": False},
        {"id": "whisper-large-v3", "active": True},
    ])
    assert "some-vl-model" in detected
    assert "openai/gpt-oss-120b" not in detected
    assert "hidden-vision-model" not in detected
    assert "whisper-large-v3" not in detected


@pytest.mark.parametrize("entry,expected", [
    ({"input_modalities": ["text", "image"]}, True),
    ({"input_modalities": ["text"]}, False),
    ({"modalities": ["text", "image"]}, True),
    ({"supported_modalities": ["text"]}, False),
    ({"modalities": "text+image->text"}, True),
    ({"modalities": "text->text"}, False),
    ({}, None),
])
def test_modality_metadata_is_read_when_present(entry, expected):
    """If Groq ever publishes modality data, it must outrank the name heuristics."""
    assert groq_node._model_reports_image_input(entry) is expected


def test_api_metadata_outranks_the_seed_list():
    """A seeded model the API says is text-only must not stay on the vision list."""
    seeded = groq_node.KNOWN_VISION_MODELS[0]
    detected = groq_node._detect_vision_models([
        {"id": seeded, "active": True, "input_modalities": ["text"]},
        {"id": "brand/new-model", "active": True, "input_modalities": ["text", "image"]},
    ])
    assert seeded not in detected
    assert "brand/new-model" in detected      # unknown name, but the API vouched for it


def test_known_vision_models_are_offered_in_the_dropdown():
    """A vision model users cannot select is no use; keep the two lists in step."""
    options = _model_options()
    for model_id in groq_node.KNOWN_VISION_MODELS:
        assert model_id in options, model_id


def test_a_pattern_matching_model_is_treated_as_vision(groq_call):
    """A new vision model must work through 'Manual Input' with no code change."""
    out = groq_call(model="Manual Input", manual_model="groq/some-new-vision-model",
                    send_system="no", image_input=torch.rand(1, 16, 16, 3))
    assert out.args[0] == "hello"


# --------------------------------------------------------------------------
# Request body
# --------------------------------------------------------------------------

def test_request_uses_max_completion_tokens(groq_call):
    """Groq deprecated max_tokens for chat completions."""
    groq_call(max_completion_tokens=321)
    body = groq_call.calls["calls"][0]["body"]
    assert body["max_completion_tokens"] == 321
    assert "max_tokens" not in body


def test_request_targets_the_chat_completions_endpoint(groq_call):
    groq_call()
    call = groq_call.calls["calls"][0]
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["timeout"] == 120


def test_penalties_are_omitted_at_zero_and_sent_otherwise(groq_call):
    groq_call()
    assert "frequency_penalty" not in groq_call.calls["calls"][0]["body"]

    groq_call(frequency_penalty=0.5, presence_penalty=-0.5)
    body = groq_call.calls["calls"][0]["body"]
    assert body["frequency_penalty"] == 0.5
    assert body["presence_penalty"] == -0.5


def test_out_of_range_values_are_clamped(groq_call):
    groq_call(temperature=9.0, top_p=-1.0)
    body = groq_call.calls["calls"][0]["body"]
    assert body["temperature"] == 2.0
    assert body["top_p"] == 0.0


def test_system_prompt_can_be_suppressed(groq_call):
    groq_call(send_system="no")
    roles = [m["role"] for m in groq_call.calls["calls"][0]["body"]["messages"]]
    assert "system" not in roles


def test_json_response_format(groq_call):
    groq_call(response_format="json_object")
    assert groq_call.calls["calls"][0]["body"]["response_format"] == {"type": "json_object"}


def test_additional_params_merge_into_the_body(groq_call):
    groq_call(additional_params=json.dumps({"stop": ["\n"]}))
    assert groq_call.calls["calls"][0]["body"]["stop"] == ["\n"]


def test_additional_params_must_be_an_object(groq_call):
    out = groq_call(additional_params="[1, 2, 3]")
    assert out.args[0] == ""
    assert "must be a JSON object" in out.args[1]


def test_additional_params_rejects_invalid_json(groq_call):
    out = groq_call(additional_params="{oops")
    assert "Invalid JSON" in out.args[1]


# --------------------------------------------------------------------------
# Model selection guards
# --------------------------------------------------------------------------

def test_category_separator_is_rejected(groq_call):
    out = groq_call(model="--- Featured ---")
    assert out.args[0] == ""
    assert "category label" in out.args[1]
    assert groq_call.calls["calls"] == []


def test_audio_model_is_rejected(groq_call):
    out = groq_call(model="Manual Input", manual_model="whisper-large-v3")
    assert "audio endpoints" in out.args[1]
    assert groq_call.calls["calls"] == []


def test_manual_model_is_used_when_selected(groq_call):
    groq_call(model="Manual Input", manual_model="  custom/model  ")
    assert groq_call.calls["calls"][0]["body"]["model"] == "custom/model"


def test_empty_user_prompt_is_rejected(groq_call):
    out = groq_call(user_prompt="   ")
    assert "User prompt is required" in out.args[1]
    assert groq_call.calls["calls"] == []


@pytest.mark.parametrize("model,manual,expected", [
    ("--- Featured ---", "", "category label"),
    ("Manual Input", "", "Manual model identifier is required"),
    ("Manual Input", "whisper-large-v3", "audio endpoints"),
    ("Manual Input", "canopylabs/orpheus-v1-english", "audio endpoints"),
])
def test_validate_inputs_rejects_bad_models(model, manual, expected):
    result = GroqNode.validate_inputs(
        api_key="k", model=model, manual_model=manual, user_prompt="hi")
    assert isinstance(result, str) and expected in result


def test_validate_inputs_accepts_a_valid_selection():
    assert GroqNode.validate_inputs(
        api_key="k", model="openai/gpt-oss-120b", manual_model="", user_prompt="hi") is True


def test_validate_inputs_requires_an_api_key():
    result = GroqNode.validate_inputs(
        api_key="", model="openai/gpt-oss-120b", manual_model="", user_prompt="hi")
    assert "API key is required" in result


def test_validate_inputs_rejects_non_object_additional_params():
    result = GroqNode.validate_inputs(
        api_key="k", model="openai/gpt-oss-120b", manual_model="",
        user_prompt="hi", additional_params="[1,2]")
    assert "must be a JSON object" in result


# --------------------------------------------------------------------------
# Seeds and caching
# --------------------------------------------------------------------------

def test_fixed_seed_is_forwarded(groq_call):
    groq_call(seed_mode="fixed", seed_value=1234)
    assert groq_call.calls["calls"][0]["body"]["seed"] == 1234


def test_increment_and_decrement_walk_the_seed(groq_call):
    """Counters are keyed by (model, seed_value), so they carry across runs."""
    groq_call(seed_mode="increment", seed_value=10)
    assert groq_call.calls["calls"][0]["body"]["seed"] == 11
    groq_call(seed_mode="increment", seed_value=10)
    assert groq_call.calls["calls"][0]["body"]["seed"] == 12

    groq_call(seed_mode="decrement", seed_value=500)
    assert groq_call.calls["calls"][0]["body"]["seed"] == 499
    groq_call(seed_mode="decrement", seed_value=500)
    assert groq_call.calls["calls"][0]["body"]["seed"] == 498


def test_decrement_floors_at_zero(groq_call):
    groq_call(seed_mode="decrement", seed_value=0)
    assert groq_call.calls["calls"][0]["body"]["seed"] == GroqNode.MAX_SAFE_INTEGER


def test_random_seed_stays_in_range(groq_call):
    groq_call(seed_mode="random")
    seed = groq_call.calls["calls"][0]["body"]["seed"]
    assert 0 <= seed <= GroqNode.MAX_SAFE_INTEGER


def test_fingerprint_marks_the_node_dirty_for_moving_seeds():
    """Without this, ComfyUI serves a cached response and the seed never moves."""
    for mode in ("random", "increment", "decrement"):
        assert GroqNode.fingerprint_inputs(seed_mode=mode, seed_value=5) != \
            GroqNode.fingerprint_inputs(seed_mode=mode, seed_value=5)  # NaN != NaN
    assert GroqNode.fingerprint_inputs(seed_mode="fixed", seed_value=5) == 5


def test_seed_counter_dict_is_bounded(groq_call):
    for i in range(GroqNode.MAX_TRACKED_SEEDS + 20):
        groq_call(seed_mode="increment", seed_value=i)
    assert len(GroqNode._last_seed) <= GroqNode.MAX_TRACKED_SEEDS


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

VISION_MODEL = "qwen/qwen3.6-27b"  # Groq's current multimodal chat model


def _image_content(call):
    return call["body"]["messages"][-1]["content"]


def test_single_image_is_sent_as_a_data_url(groq_call):
    groq_call(model=VISION_MODEL, send_system="no", image_input=torch.rand(1, 32, 32, 3))
    content = _image_content(groq_call.calls["calls"][0])
    assert content[0]["type"] == "text"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_batch_uses_the_first_frame(groq_call):
    """A batch of 2 used to fail with 'must be 3D after squeezing'."""
    out = groq_call(model=VISION_MODEL, send_system="no", image_input=torch.rand(4, 32, 32, 3))
    assert out.args[0] == "hello"
    assert len(_image_content(groq_call.calls["calls"][0])) == 2


def test_unbatched_image_tensor_is_accepted(groq_call):
    out = groq_call(model=VISION_MODEL, send_system="no", image_input=torch.rand(32, 32, 3))
    assert out.args[0] == "hello"


def test_oversized_image_is_rejected(groq_call):
    out = groq_call(model=VISION_MODEL, send_system="no", image_input=torch.rand(1, 2049, 32, 3))
    assert "Image too large" in out.args[1]
    assert groq_call.calls["calls"] == []


def test_an_image_is_never_silently_dropped(groq_call):
    """
    The node must not decide from a capability list that an image should not be
    sent. A stale list would otherwise drop the user's image without a word.
    """
    out = groq_call(model="a-model-nothing-knows-about",
                    image_input=torch.rand(1, 32, 32, 3))
    assert out.args[0] == "hello"
    content = _image_content(groq_call.calls["calls"][0])
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_an_unlisted_vision_model_is_not_blocked(groq_call):
    """
    A Groq vision model released after this code shipped must work with no edit
    here. Blocking on KNOWN_VISION_MODELS is what used to prevent that.
    """
    out = groq_call(model="Manual Input", manual_model="groq/model-from-the-future",
                    send_system="no", image_input=torch.rand(1, 16, 16, 3))
    assert out.args[0] == "hello"
    assert groq_call.calls["calls"][0]["body"]["model"] == "groq/model-from-the-future"


def test_groq_400_on_an_image_gains_a_hint(groq_call):
    """Groq stays the authority on capability; we only make its refusal actionable."""
    out = groq_call(responses=[FakeResponse(400, {"error": {"message": "no image support"}})],
                    model="openai/gpt-oss-120b", image_input=torch.rand(1, 32, 32, 3))
    assert "no image support" in out.args[1]
    assert "did not look vision-capable" in out.args[1]
    assert "qwen/qwen3.6-27b" in out.args[1]


def test_the_hint_reads_cleanly_when_nothing_looks_capable(groq_call, monkeypatch):
    """Groq has emptied its vision line-up before, when Llama 4 Scout was retired."""
    monkeypatch.setattr(groq_node, "KNOWN_VISION_MODELS", [])
    out = groq_call(responses=[FakeResponse(400, {"error": {"message": "nope"}})],
                    model="openai/gpt-oss-120b", image_input=torch.rand(1, 32, 32, 3))
    assert "No model in the current list advertises image input." in out.args[1]
    assert not out.args[1].rstrip().endswith(":")


def test_no_hint_when_the_model_looks_capable(groq_call):
    out = groq_call(responses=[FakeResponse(400, {"error": {"message": "rate limited"}})],
                    model=VISION_MODEL, image_input=torch.rand(1, 32, 32, 3))
    assert "did not look vision-capable" not in out.args[1]


def test_out_of_range_pixels_do_not_crash_conversion(groq_call):
    out = groq_call(model=VISION_MODEL, send_system="no",
                    image_input=torch.rand(1, 16, 16, 3) * 4 - 2)
    assert out.args[0] == "hello"


def test_uint8_image_tensor_survives_conversion(groq_call):
    """Integer tensors are already 0..255 and must not be clamped to 0..1."""
    import base64
    import io as py_io

    from PIL import Image as PILImage

    pixels = torch.full((1, 8, 8, 3), 200, dtype=torch.uint8)
    out = groq_call(model=VISION_MODEL, send_system="no", image_input=pixels)
    assert out.args[0] == "hello"

    url = _image_content(groq_call.calls["calls"][0])[1]["image_url"]["url"]
    decoded = PILImage.open(py_io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert decoded.getpixel((0, 0)) == (200, 200, 200)


# --------------------------------------------------------------------------
# HTTP handling
# --------------------------------------------------------------------------

def test_success_status_reports_model_and_tokens(groq_call):
    out = groq_call()
    assert out.args[0] == "hello"
    assert "Model=test-model" in out.args[1]
    assert "3+4=7" in out.args[1]


def test_invalid_key_is_reported(groq_call):
    out = groq_call(responses=[FakeResponse(401, {})])
    assert "Invalid API key" in out.args[1]


def test_400_surfaces_the_api_message(groq_call):
    payload = {"error": {"message": "model_not_found"}}
    out = groq_call(responses=[FakeResponse(400, payload)])
    assert "model_not_found" in out.args[1]
    assert "Request body" not in out.args[1]


def test_debug_mode_includes_the_request_body(groq_call):
    payload = {"error": {"message": "model_not_found"}}
    out = groq_call(responses=[FakeResponse(400, payload)], debug_mode="on")
    assert "Request body" in out.args[1]


def test_rate_limit_is_retried_then_reported(groq_call, no_sleep):
    responses = [FakeResponse(429, {}), FakeResponse(429, {}), FakeResponse(429, {})]
    out = groq_call(responses=responses, max_retries=2)
    assert "Rate limit exceeded" in out.args[1]
    assert len(groq_call.calls["calls"]) == 3
    # Jittered exponential backoff: two sleeps inside the 1..4s band.
    assert len(no_sleep) == 2
    assert all(1 <= s <= 4 for s in no_sleep)


def test_server_error_recovers_on_retry(groq_call, no_sleep):
    out = groq_call(responses=[FakeResponse(503, {}), FakeResponse(200, chat_payload())],
                    max_retries=2)
    assert out.args[0] == "hello"


def test_network_error_is_retried_then_reported(monkeypatch, no_sleep):
    import requests
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(groq_node.requests, "post", boom)
    out = GroqNode.execute(
        api_key="k", model="openai/gpt-oss-120b", manual_model="", system_prompt="",
        user_prompt="hi", send_system="no", temperature=0.7, top_p=0.7,
        max_completion_tokens=10, frequency_penalty=0.0, presence_penalty=0.0,
        response_format="text", seed_mode="fixed", seed_value=0, max_retries=2,
        debug_mode="off")
    assert "Network Error" in out.args[1]
    assert len(calls) == 3


def test_malformed_success_body_is_not_retried(groq_call, no_sleep):
    out = groq_call(responses=[FakeResponse(200, raise_json=True)], max_retries=3)
    assert "Invalid JSON response" in out.args[1]
    assert len(groq_call.calls["calls"]) == 1


def test_empty_choices_is_reported(groq_call):
    out = groq_call(responses=[FakeResponse(200, {"model": "m", "choices": []})])
    assert "No response content" in out.args[1]


def test_api_key_is_never_echoed_into_outputs(groq_call):
    out = groq_call(responses=[FakeResponse(400, {"error": {"message": "bad"}})], debug_mode="on")
    assert "test-key" not in "".join(out.args)


# --------------------------------------------------------------------------
# v2.2.0: shared-image pipeline, redaction, eviction, interrupts
# --------------------------------------------------------------------------


def test_jpeg_format_produces_a_jpeg_data_url(groq_call):
    import base64
    import io as py_io

    from PIL import Image as PILImage

    out = groq_call(model=VISION_MODEL, send_system="no",
                    image_input=torch.rand(1, 16, 16, 3), image_format="jpeg")
    assert out.args[0] == "hello"
    url = _image_content(groq_call.calls["calls"][0])[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    decoded = PILImage.open(py_io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert decoded.format == "JPEG"


def test_debug_body_redacts_image_data(groq_call):
    """A multi-megabyte base64 blob must not land in the UI status field."""
    out = groq_call(responses=[FakeResponse(400, {"error": {"message": "bad"}})],
                    model=VISION_MODEL, send_system="no",
                    image_input=torch.rand(1, 32, 32, 3), debug_mode="on")
    assert "Request body" in out.args[1]
    assert "redacted data URI" in out.args[1]
    # The PNG magic constant as it appears once base64-encoded.
    assert "iVBORw0KGgo" not in out.args[1]


def test_seed_eviction_keeps_the_newest_counters(groq_call):
    for i in range(GroqNode.MAX_TRACKED_SEEDS + 5):
        groq_call(seed_mode="increment", seed_value=i)
    assert len(GroqNode._last_seed) <= GroqNode.MAX_TRACKED_SEEDS
    key = ("openai/gpt-oss-120b", GroqNode.MAX_TRACKED_SEEDS + 4)
    groq_call(seed_mode="increment", seed_value=GroqNode.MAX_TRACKED_SEEDS + 4)
    assert GroqNode._last_seed[key] == GroqNode.MAX_TRACKED_SEEDS + 6


def test_interrupt_is_not_swallowed(monkeypatch, groq_call):
    """Cancelling the queue must propagate, not turn into a chat error."""
    import sys
    import types

    class InterruptProcessingException(Exception):
        pass

    comfy_pkg = types.ModuleType("comfy")
    model_management = types.ModuleType("comfy.model_management")
    model_management.throw_exception_if_processing_interrupted = (
        lambda: (_ for _ in ()).throw(InterruptProcessingException()))
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    with pytest.raises(InterruptProcessingException):
        groq_call()
