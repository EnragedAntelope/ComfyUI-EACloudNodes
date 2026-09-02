"""Tests for the OpenRouter Chat node."""
import json

import pytest
import torch

import openrouter
from conftest import FakeResponse, chat_payload

OpenrouterNode = openrouter.OpenrouterNode

CATALOGUE = {"data": [
    {
        "id": "free/text-model:free",
        "name": "Free Text Model",
        "pricing": {"prompt": "0", "completion": "0"},
        "architecture": {"input_modalities": ["text"], "modality": "text->text"},
    },
    {
        "id": "free/vision-model:free",
        "name": "Free Vision Model",
        "pricing": {"prompt": "0", "completion": "0"},
        "architecture": {"input_modalities": ["text", "image"], "modality": "text+image->text"},
    },
    {
        "id": "paid/vision-model",
        "name": "Paid Vision Model",
        "pricing": {"prompt": "0.000005", "completion": "0.00001"},
        "architecture": {"input_modalities": ["text", "image"], "modality": "text+image->text"},
    },
    {
        "id": "paid/text-model",
        "name": "Paid Text Model",
        "pricing": {"prompt": "0.000005", "completion": "0.00001"},
        "architecture": {"input_modalities": ["text"], "modality": "text->text"},
    },
]}


@pytest.fixture
def catalogue(monkeypatch):
    monkeypatch.setattr(openrouter.requests, "get", lambda *a, **k: FakeResponse(200, CATALOGUE))
    return CATALOGUE


# --------------------------------------------------------------------------
# Model catalogue
# --------------------------------------------------------------------------

def test_dropdown_lists_only_free_models(catalogue):
    models, _ = openrouter._fetch_openrouter_free_models()
    assert models == ["free/text-model:free", "free/vision-model:free", "Manual Input"]


def test_vision_list_covers_paid_models_too(catalogue):
    """Paid vision models are reachable via Manual Input and must not be blocked."""
    _, vision = openrouter._fetch_openrouter_free_models()
    assert "paid/vision-model" in vision
    assert "free/vision-model:free" in vision
    assert "paid/text-model" not in vision


def test_catalogue_result_is_cached(catalogue, monkeypatch):
    openrouter._fetch_openrouter_free_models()
    calls = []
    monkeypatch.setattr(openrouter.requests, "get",
                        lambda *a, **k: calls.append(1) or FakeResponse(200, CATALOGUE))
    openrouter._fetch_openrouter_free_models()
    assert calls == []


def test_failed_fetch_is_not_retried_immediately(monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("offline")

    monkeypatch.setattr(openrouter.requests, "get", boom)
    assert openrouter._fetch_openrouter_free_models() == (None, None)
    assert openrouter._fetch_openrouter_free_models() == (None, None)
    assert len(calls) == 1


@pytest.mark.parametrize("model,expected", [
    ({"architecture": {"input_modalities": ["text", "image"]}}, True),
    ({"architecture": {"input_modalities": ["text"]}}, False),
    ({"architecture": {"modality": "text+image->text"}}, True),
    ({"architecture": {"modality": "text->text+image"}}, False),
    ({"id": "qwen/qwen2.5-vl-32b", "architecture": {}}, True),
    ({"name": "Something Vision", "architecture": {}}, True),
    ({"id": "plain/model", "architecture": {}}, False),
    ({}, False),
])
def test_image_capability_detection(model, expected):
    assert openrouter._model_accepts_images(model) is expected


def test_schema_default_is_selectable(catalogue):
    schema = OpenrouterNode.define_schema()
    model_input = schema.input_by_id("model")
    assert model_input.default in model_input.options


def test_schema_offers_manual_input_when_offline():
    schema = OpenrouterNode.define_schema()
    assert schema.input_by_id("model").options == ["Manual Input"]


# --------------------------------------------------------------------------
# Request body
# --------------------------------------------------------------------------

def test_request_hits_the_configured_base_url(openrouter_call):
    openrouter_call()
    call = openrouter_call.calls["calls"][0]
    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer test-key"


def test_default_body_fields(openrouter_call):
    openrouter_call()
    body = openrouter_call.calls["calls"][0]["body"]
    assert body["model"] == "meta-llama/llama-3.3-70b-instruct:free"
    assert body["max_tokens"] == 1000
    assert body["repetition_penalty"] == 1.1
    assert "top_k" not in body          # only sent when moved off the default


def test_top_k_is_sent_when_changed(openrouter_call):
    openrouter_call(top_k=20)
    assert openrouter_call.calls["calls"][0]["body"]["top_k"] == 20


def test_values_are_clamped(openrouter_call):
    openrouter_call(temperature=99, top_p=99, repetition_penalty=99, max_tokens=10 ** 9)
    body = openrouter_call.calls["calls"][0]["body"]
    assert body["temperature"] == 2.0
    assert body["top_p"] == 1.0
    assert body["repetition_penalty"] == 2.0
    assert body["max_tokens"] == 32768


def test_additional_params_must_be_an_object(openrouter_call):
    out = openrouter_call(additional_params='"just a string"')
    assert "must be a JSON object" in out.args[1]


def test_additional_params_merge(openrouter_call):
    openrouter_call(additional_params=json.dumps({"top_a": 0.5}))
    assert openrouter_call.calls["calls"][0]["body"]["top_a"] == 0.5


def test_empty_user_prompt_is_rejected(openrouter_call):
    out = openrouter_call(user_prompt="")
    assert "User prompt is required" in out.args[1]
    assert openrouter_call.calls["calls"] == []


# --------------------------------------------------------------------------
# Vision gating
# --------------------------------------------------------------------------

def test_paid_vision_model_accepts_an_image(catalogue, openrouter_call):
    """This used to be blocked because only free models were considered."""
    out = openrouter_call(manual_model="paid/vision-model", image_input=torch.rand(1, 32, 32, 3))
    assert out.args[0] == "hello"
    content = openrouter_call.calls["calls"][0]["body"]["messages"][-1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_known_text_only_model_rejects_an_image(catalogue, openrouter_call):
    out = openrouter_call(manual_model="paid/text-model", image_input=torch.rand(1, 32, 32, 3))
    assert out.args[0] == ""
    assert "does not accept image input" in out.args[1]
    assert openrouter_call.calls["calls"] == []


def test_unknown_model_with_an_image_is_passed_through(catalogue, openrouter_call):
    """An id the catalogue does not list is the API's call to make, not ours."""
    out = openrouter_call(manual_model="brand/new-model", image_input=torch.rand(1, 32, 32, 3))
    assert out.args[0] == "hello"


def test_image_batch_uses_the_first_frame(catalogue, openrouter_call):
    out = openrouter_call(manual_model="paid/vision-model", image_input=torch.rand(3, 32, 32, 3))
    assert out.args[0] == "hello"


def test_oversized_image_is_rejected(catalogue, openrouter_call):
    out = openrouter_call(manual_model="paid/vision-model",
                          image_input=torch.rand(1, 32, 3000, 3))
    assert "Image too large" in out.args[1]


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------

def test_fingerprint_marks_the_node_dirty_for_moving_seeds():
    for mode in ("random", "increment", "decrement"):
        assert OpenrouterNode.fingerprint_inputs(seed_mode=mode, seed_value=1) != \
            OpenrouterNode.fingerprint_inputs(seed_mode=mode, seed_value=1)
    assert OpenrouterNode.fingerprint_inputs(seed_mode="fixed", seed_value=7) == 7


def test_seed_counter_dict_is_bounded(openrouter_call):
    for i in range(OpenrouterNode.MAX_TRACKED_SEEDS + 10):
        openrouter_call(seed_mode="increment", seed_value=i)
    assert len(OpenrouterNode._last_seed) <= OpenrouterNode.MAX_TRACKED_SEEDS


# --------------------------------------------------------------------------
# Validation and HTTP handling
# --------------------------------------------------------------------------

def test_missing_api_key_is_reported(openrouter_call):
    out = openrouter_call(api_key="")
    assert out.args[0] == ""
    assert "OpenRouter API key is required" in out.args[1]
    assert openrouter_call.calls["calls"] == []


def test_manual_model_required_when_selected(openrouter_call):
    out = openrouter_call(model="Manual Input", manual_model="")
    assert "Manual model identifier is required" in out.args[1]
    assert openrouter_call.calls["calls"] == []


def test_empty_base_url_is_rejected(openrouter_call):
    out = openrouter_call(base_url="")
    assert "endpoint URL is required" in out.args[1]
    assert openrouter_call.calls["calls"] == []


def test_additional_params_rejects_invalid_json(openrouter_call):
    out = openrouter_call(additional_params="{oops")
    assert "Invalid JSON" in out.args[1]
    assert openrouter_call.calls["calls"] == []

def test_payload_too_large_is_reported(openrouter_call):
    out = openrouter_call(responses=[FakeResponse(413, {})])
    assert "Payload too large" in out.args[1]


def test_service_error_is_reported_after_retries(openrouter_call, no_sleep):
    out = openrouter_call(responses=[FakeResponse(502, {})], max_retries=1)
    assert "service error" in out.args[1]
    assert len(openrouter_call.calls["calls"]) == 2


def test_400_surfaces_the_api_message(openrouter_call):
    out = openrouter_call(responses=[FakeResponse(400, {"error": {"message": "no such model"}})])
    assert "no such model" in out.args[1]


def test_malformed_success_body_is_not_retried(openrouter_call, no_sleep):
    out = openrouter_call(responses=[FakeResponse(200, raise_json=True)], max_retries=3)
    assert "Invalid JSON response" in out.args[1]
    assert len(openrouter_call.calls["calls"]) == 1


def test_success_reports_model_and_tokens(openrouter_call):
    out = openrouter_call(responses=[FakeResponse(200, chat_payload("hi", "some/model"))])
    assert out.args[0] == "hi"
    assert "Model=some/model" in out.args[1]
    assert "3+4=7" in out.args[1]


def test_help_output_is_not_duplicated(openrouter_call):
    help_text = openrouter_call().args[2]
    assert help_text.count("Key Settings:") == 1


# --------------------------------------------------------------------------
# v2.2.0: endpoint policy, image format, redaction, Retry-After
# --------------------------------------------------------------------------



def test_custom_endpoint_warns_in_the_status(catalogue, openrouter_call):
    """A shared workflow pointing base_url elsewhere must be loud about it."""
    out = openrouter_call(base_url="https://proxy.example.com/v1")
    assert out.args[0] == "hello"
    assert "Custom endpoint" in out.args[1]
    assert "proxy.example.com" in out.args[1]


def test_default_endpoint_does_not_warn(openrouter_call):
    out = openrouter_call()
    assert "Custom endpoint" not in out.args[1]


def test_jpeg_format_produces_a_jpeg_data_url(catalogue, openrouter_call):
    openrouter_call(manual_model="paid/vision-model",
                    image_input=torch.rand(1, 16, 16, 3), image_format="jpeg")
    url = openrouter_call.calls["calls"][0]["body"]["messages"][-1]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_debug_body_redacts_image_data(catalogue, openrouter_call):
    out = openrouter_call(responses=[FakeResponse(400, {"error": {"message": "no"}})],
                          manual_model="paid/vision-model",
                          image_input=torch.rand(1, 32, 32, 3), debug_mode="on")
    assert "Request body" in out.args[1]
    assert "redacted data URI" in out.args[1]
    assert "iVBORw0KGgo" not in out.args[1]


def test_retry_after_header_is_honoured(openrouter_call, no_sleep):
    """A server-provided Retry-After beats our backoff curve."""
    responses = [FakeResponse(429, {}, headers={"Retry-After": "7"}),
                 FakeResponse(200, chat_payload())]
    out = openrouter_call(responses=responses, max_retries=2)
    assert out.args[0] == "hello"
    assert no_sleep == [7]


def test_execute_refuses_plain_http_even_without_validate_inputs(openrouter_call):
    """
    validate_inputs only sees a literal widget value. Once base_url is converted
    to an input socket its value is unknown until the graph runs, so execute()
    must enforce the endpoint policy itself - it is where the key is sent.
    """
    out = openrouter_call(base_url="http://evil.example.com/v1")
    assert out.args[0] == ""
    assert "plain http://" in out.args[1]
    assert openrouter_call.calls["calls"] == []   # nothing left the machine


def test_execute_still_allows_a_localhost_http_proxy(openrouter_call):
    out = openrouter_call(base_url="http://127.0.0.1:8080/v1")
    assert out.args[0] == "hello"
    assert openrouter_call.calls["calls"][0]["url"] == "http://127.0.0.1:8080/v1"


def test_execute_rejects_a_non_http_scheme(openrouter_call):
    out = openrouter_call(base_url="file:///etc/passwd")
    assert out.args[0] == ""
    assert "must start with http:// or https://" in out.args[1]
    assert openrouter_call.calls["calls"] == []


def test_custom_endpoint_warning_does_not_claim_an_unsent_key(openrouter_call):
    """
    An early validation failure sends nothing. Saying the key 'was sent' there
    would be a false alarm on the one message a user must be able to trust.
    """
    out = openrouter_call(base_url="https://proxy.example.com/v1", user_prompt="  ")
    assert openrouter_call.calls["calls"] == []
    assert "were sent to" not in out.args[1]
    assert "proxy.example.com" in out.args[1]     # still flags the endpoint
