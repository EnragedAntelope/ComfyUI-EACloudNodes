import importlib.util
import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)

# Only tests/ goes on sys.path, for comfy_stub. The repository root deliberately
# does NOT: ComfyUI never puts a custom-node folder on sys.path either, so adding
# it here would let a sibling module imported absolutely pass the suite and then
# fail at registration inside ComfyUI. See load_pack() below.
sys.path.insert(0, TESTS_DIR)

import comfy_stub  # noqa: E402

comfy_stub.install()

PACKAGE_NAME = "eacloudnodes_under_test"


def load_pack(module_name=PACKAGE_NAME, path=REPO_ROOT):
    """
    Import the repository the way ComfyUI's load_custom_node() does.

    ComfyUI builds a spec from the folder's __init__.py under a synthetic module
    name and executes it; the folder itself is never importable by name. Loading
    the same way here means the suite exercises the real registration path.
    """
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(path, "__init__.py"),
        submodule_search_locations=[path],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


eacloudnodes = load_pack()

# Bind the submodules under their bare names so the test modules can `import
# groq_node` and still be testing the package-loaded objects. chat_common is
# deliberately left unbound: nothing outside the package may reach it by a bare
# name, and test_package.py re-runs the whole load in a clean interpreter.
groq_node = sys.modules[f"{PACKAGE_NAME}.groq_node"]
openrouter = sys.modules[f"{PACKAGE_NAME}.openrouter"]
openrouter_models = sys.modules[f"{PACKAGE_NAME}.openrouter_models"]

sys.modules["groq_node"] = groq_node
sys.modules["openrouter"] = openrouter
sys.modules["openrouter_models"] = openrouter_models


class FakeResponse:
    """Stand-in for requests.Response."""

    def __init__(self, status_code=200, payload=None, raise_json=False, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self._raise_json = raise_json
        self.headers = headers or {}

    def json(self):
        if self._raise_json:
            import requests
            raise requests.exceptions.JSONDecodeError("bad", "", 0)
        return self._payload


def chat_payload(content="hello", model="test-model"):
    return {
        "model": model,
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        "choices": [{"message": {"content": content}}],
    }


@pytest.fixture(autouse=True)
def clear_caches():
    """Keep module-level caches and seed counters from leaking between tests."""
    for cache in (groq_node._groq_model_cache, openrouter._openrouter_model_cache):
        cache["models"] = None
        cache["vision_models"] = None
        cache["last_fetch"] = 0
        cache["last_failure"] = 0
        if "known_models" in cache:
            cache["known_models"] = None
    groq_node.GroqNode._last_seed.clear()
    openrouter.OpenrouterNode._last_seed.clear()
    yield


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No test may touch the real network; model fetches look like an outage."""
    def blocked(*args, **kwargs):
        import requests
        raise requests.exceptions.ConnectionError("network disabled in tests")

    for module in (groq_node, openrouter, openrouter_models):
        monkeypatch.setattr(module.requests, "get", blocked)
        if hasattr(module.requests, "post"):
            monkeypatch.setattr(module.requests, "post", blocked)


@pytest.fixture
def no_sleep(monkeypatch):
    """Make retry backoff instant and record how long it would have slept."""
    slept = []
    monkeypatch.setattr(groq_node.time, "sleep", slept.append)
    monkeypatch.setattr(openrouter.time, "sleep", slept.append)
    return slept


@pytest.fixture
def groq_call(monkeypatch):
    """Run GroqNode.execute with sane defaults and capture the outgoing request."""
    captured = {}

    def run(responses=None, **overrides):
        queue = list(responses) if responses is not None else [FakeResponse(200, chat_payload())]
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append({"url": url, "headers": headers, "body": json, "timeout": timeout})
            return queue.pop(0) if len(queue) > 1 else queue[0]

        monkeypatch.setattr(groq_node.requests, "post", fake_post)
        captured["calls"] = calls

        kwargs = dict(
            api_key="test-key",
            model="openai/gpt-oss-120b",
            manual_model="",
            system_prompt="be helpful",
            user_prompt="hello there",
            send_system="yes",
            temperature=0.7,
            top_p=0.7,
            max_completion_tokens=1000,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            response_format="text",
            seed_mode="fixed",
            seed_value=42,
            max_retries=0,
            debug_mode="off",
        )
        kwargs.update(overrides)
        return groq_node.GroqNode.execute(**kwargs)

    run.calls = captured
    return run


@pytest.fixture
def openrouter_call(monkeypatch):
    """Run OpenrouterNode.execute with sane defaults and capture the outgoing request."""
    captured = {}

    def run(responses=None, **overrides):
        queue = list(responses) if responses is not None else [FakeResponse(200, chat_payload())]
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append({"url": url, "headers": headers, "body": json, "timeout": timeout})
            return queue.pop(0) if len(queue) > 1 else queue[0]

        monkeypatch.setattr(openrouter.requests, "post", fake_post)
        captured["calls"] = calls

        kwargs = dict(
            api_key="test-key",
            model="Manual Input",
            manual_model="meta-llama/llama-3.3-70b-instruct:free",
            base_url="https://openrouter.ai/api/v1/chat/completions",
            system_prompt="be helpful",
            user_prompt="hello there",
            send_system="yes",
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            max_tokens=1000,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.1,
            response_format="text",
            seed_mode="fixed",
            seed_value=42,
            max_retries=0,
            debug_mode="off",
        )
        kwargs.update(overrides)
        return openrouter.OpenrouterNode.execute(**kwargs)

    run.calls = captured
    return run
