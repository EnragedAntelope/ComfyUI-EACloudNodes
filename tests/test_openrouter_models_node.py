"""Tests for the OpenRouter Models query node."""
import pytest

import openrouter_models
from conftest import FakeResponse

OpenRouterModels = openrouter_models.OpenRouterModels

CATALOGUE = {"data": [
    {"id": "z/free-model", "name": "Zed Free", "description": "a free one",
     "context_length": 8000, "pricing": {"prompt": "0", "completion": "0",
                                         "image": "0", "request": "0"}},
    {"id": "a/paid-gpt", "name": "Alpha GPT", "description": "a paid gpt",
     "context_length": 128000, "pricing": {"prompt": "0.00003", "completion": "0.00006",
                                           "image": "0", "request": "0"}},
    {"id": "m/odd-model", "name": "Odd", "description": "broken metadata",
     "context_length": None, "pricing": {"prompt": None, "completion": "0"}},
]}


def run(payload=CATALOGUE, status=200, **overrides):
    kwargs = dict(api_key="", filter_text="", sort_by="name", sort_order="ascending")
    kwargs.update(overrides)
    return OpenRouterModels.execute(**kwargs)


@pytest.fixture
def catalogue(monkeypatch):
    seen = {}

    def fake_get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers or {}
        return FakeResponse(200, CATALOGUE)

    monkeypatch.setattr(openrouter_models.requests, "get", fake_get)
    return seen


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------

def test_api_key_is_optional(catalogue):
    """The models catalogue is public; requiring a key was a false barrier."""
    out = run()
    assert out.args[0] != ""
    assert "Authorization" not in catalogue["headers"]


def test_api_key_is_sent_when_supplied(catalogue):
    run(api_key="  secret  ")
    assert catalogue["headers"]["Authorization"] == "Bearer secret"


def test_validate_inputs_does_not_require_a_key():
    assert OpenRouterModels.validate_inputs(api_key="") is True


def test_endpoint(catalogue):
    run()
    assert catalogue["url"] == "https://openrouter.ai/api/v1/models"


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

def test_free_filter_uses_pricing_not_text(catalogue):
    out = run(filter_text="free")
    assert "z/free-model" in out.args[0]
    assert "a/paid-gpt" not in out.args[0]


def test_text_filter_matches_id_name_and_description(catalogue):
    assert "a/paid-gpt" in run(filter_text="gpt").args[0]
    assert "z/free-model" in run(filter_text="Zed").args[0]
    assert "a/paid-gpt" in run(filter_text="a paid").args[0]


def test_filter_terms_are_anded(catalogue):
    out = run(filter_text="free gpt")
    assert "No models found" in out.args[0]


def test_empty_filter_returns_everything(catalogue):
    out = run(filter_text="   ")
    assert "Found 3 models" in out.args[1]


# --------------------------------------------------------------------------
# Sorting — must survive nulls in the catalogue
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sort_by", ["name", "pricing", "context_length"])
@pytest.mark.parametrize("sort_order", ["ascending", "descending"])
def test_sorting_never_crashes_on_null_fields(catalogue, sort_by, sort_order):
    out = run(sort_by=sort_by, sort_order=sort_order)
    assert "Error" not in out.args[1]
    assert "Found 3 models" in out.args[1]


def test_sort_by_name_ascending(catalogue):
    body = run(sort_by="name").args[0]
    assert body.index("Alpha GPT") < body.index("Odd") < body.index("Zed Free")


def test_sort_by_context_length_descending(catalogue):
    body = run(sort_by="context_length", sort_order="descending").args[0]
    assert body.index("Alpha GPT") < body.index("Zed Free")


def test_null_pricing_does_not_crash_the_free_filter(catalogue):
    out = run(filter_text="free")
    assert "Error" not in out.args[1]


# --------------------------------------------------------------------------
# Coercion helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [("1.5", 1.5), (None, 0.0), ("", 0.0), ("x", 0.0), (2, 2.0)])
def test_as_float(value, expected):
    assert openrouter_models._as_float(value) == expected


@pytest.mark.parametrize("value,expected", [(5, 5), ("7", 7), (None, 0), ("x", 0)])
def test_as_int(value, expected):
    assert openrouter_models._as_int(value) == expected


def test_is_free_model():
    assert openrouter_models._is_free_model({"pricing": {"prompt": "0", "completion": "0"}})
    assert not openrouter_models._is_free_model({"pricing": {"prompt": "0.1", "completion": "0"}})
    assert openrouter_models._is_free_model({})


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    (401, "Invalid API key"),
    (429, "Rate limit exceeded"),
    (500, "service error"),
    (418, "status code 418"),
])
def test_http_errors_are_reported(monkeypatch, status, expected):
    monkeypatch.setattr(openrouter_models.requests, "get",
                        lambda *a, **k: FakeResponse(status, {}))
    assert expected in run().args[1]


def test_timeout_is_reported(monkeypatch):
    import requests

    def timeout(*args, **kwargs):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(openrouter_models.requests, "get", timeout)
    assert "timed out" in run().args[1]


def test_connection_error_is_reported(monkeypatch):
    import requests

    def down(*args, **kwargs):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(openrouter_models.requests, "get", down)
    assert "Connection failed" in run().args[1]


def test_malformed_body_is_reported(monkeypatch):
    monkeypatch.setattr(openrouter_models.requests, "get",
                        lambda *a, **k: FakeResponse(200, raise_json=True))
    assert "Invalid JSON response" in run().args[1]


def test_null_data_is_handled(monkeypatch):
    monkeypatch.setattr(openrouter_models.requests, "get",
                        lambda *a, **k: FakeResponse(200, {"data": None}))
    out = run()
    assert "No models found" in out.args[0]
