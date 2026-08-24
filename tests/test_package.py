"""Tests for how the pack presents itself to ComfyUI."""
import inspect
import os
import subprocess
import sys
import textwrap

import groq_node
import openrouter
import openrouter_models

from conftest import REPO_ROOT, TESTS_DIR

NODE_MODULES = (groq_node, openrouter, openrouter_models)
NODE_CLASSES = (groq_node.GroqNode, openrouter.OpenrouterNode, openrouter_models.OpenRouterModels)


def test_the_pack_loads_the_way_comfyui_loads_it():
    """
    ComfyUI executes a custom node's __init__.py under a synthetic module name and
    never adds the pack folder to sys.path. A sibling module imported absolutely
    (`import chat_common`) therefore resolves in a test process that has the repo
    root on sys.path, and raises ModuleNotFoundError inside ComfyUI - taking the
    registration of every node in the pack with it.

    Run the load in a clean interpreter so nothing already in this process's
    sys.modules can hide the failure.
    """
    script = textwrap.dedent(
        """
        import importlib.util, json, os, sys
        tests_dir, repo_root = sys.argv[1], sys.argv[2]
        sys.path.insert(0, tests_dir)
        import comfy_stub
        comfy_stub.install()
        sys.path.remove(tests_dir)
        # Mirror ComfyUI: the pack folder is not importable by name.
        sys.path[:] = [p for p in sys.path
                       if os.path.abspath(p or os.getcwd()) != os.path.abspath(repo_root)]
        spec = importlib.util.spec_from_file_location(
            "comfyui_loaded_pack", os.path.join(repo_root, "__init__.py"),
            submodule_search_locations=[repo_root])
        module = importlib.util.module_from_spec(spec)
        sys.modules["comfyui_loaded_pack"] = module
        spec.loader.exec_module(module)
        print(json.dumps(sorted(module.NODE_CLASS_MAPPINGS)))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, TESTS_DIR, REPO_ROOT],
        capture_output=True, text=True, cwd=os.path.dirname(REPO_ROOT),
    )
    assert result.returncode == 0, (
        "the pack failed to load the way ComfyUI loads it:\n" + result.stderr)
    assert "GroqNode" in result.stdout
    assert "OpenrouterNode" in result.stdout
    assert "OpenRouterModels" in result.stdout


def test_sibling_imports_are_relative():
    """
    The guard above catches this at load time; this one names the cause, so a
    reviewer sees which line to change rather than a ModuleNotFoundError.
    """
    siblings = {"chat_common", "groq_node", "openrouter", "openrouter_models"}
    for name in ("__init__.py", "groq_node.py", "openrouter.py",
                 "openrouter_models.py", "chat_common.py"):
        source = open(os.path.join(REPO_ROOT, name), encoding="utf-8").read()
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped.startswith("import "):
                continue
            imported = stripped[len("import "):].split(" as ")[0].split(".")[0].strip()
            assert imported not in siblings, (
                f"{name}: `{stripped}` must be a relative import "
                f"(`from . import {imported}`) - ComfyUI does not put the pack "
                "folder on sys.path")


def test_every_module_exposes_v1_mappings():
    for module in NODE_MODULES:
        assert module.NODE_CLASS_MAPPINGS
        assert module.NODE_DISPLAY_NAME_MAPPINGS


def test_mapping_keys_match_the_schema_node_ids():
    for module in NODE_MODULES:
        for node_id, cls in module.NODE_CLASS_MAPPINGS.items():
            assert cls.define_schema().node_id == node_id
            assert node_id in module.NODE_DISPLAY_NAME_MAPPINGS


def test_display_names_match_the_schema():
    for module in NODE_MODULES:
        for node_id, cls in module.NODE_CLASS_MAPPINGS.items():
            assert module.NODE_DISPLAY_NAME_MAPPINGS[node_id] == cls.define_schema().display_name


def test_node_ids_are_unique():
    ids = [c.define_schema().node_id for c in NODE_CLASSES]
    assert len(ids) == len(set(ids))


def test_input_ids_are_unique_per_node():
    for cls in NODE_CLASSES:
        ids = [i.id for i in cls.define_schema().inputs]
        assert len(ids) == len(set(ids)), cls.__name__


def test_execute_signature_covers_every_declared_input():
    """A schema input with no matching execute parameter would fail at runtime."""
    for cls in NODE_CLASSES:
        params = inspect.signature(cls.execute).parameters
        for declared in cls.define_schema().inputs:
            assert declared.id in params, f"{cls.__name__}.execute is missing {declared.id}"


def test_combo_defaults_are_in_their_options():
    for cls in NODE_CLASSES:
        for declared in cls.define_schema().inputs:
            options = getattr(declared, "options", None)
            if options is None:
                continue
            assert declared.default in options, f"{cls.__name__}.{declared.id}"


def test_numeric_defaults_sit_inside_their_bounds():
    for cls in NODE_CLASSES:
        for declared in cls.define_schema().inputs:
            default = getattr(declared, "default", None)
            low = getattr(declared, "min", None)
            high = getattr(declared, "max", None)
            if not isinstance(default, (int, float)) or isinstance(default, bool):
                continue
            if low is not None:
                assert default >= low, f"{cls.__name__}.{declared.id}"
            if high is not None:
                assert default <= high, f"{cls.__name__}.{declared.id}"


def test_every_input_has_a_tooltip():
    for cls in NODE_CLASSES:
        for declared in cls.define_schema().inputs:
            assert getattr(declared, "tooltip", None), f"{cls.__name__}.{declared.id}"


def test_output_counts_match_what_execute_returns():
    """execute() returns NodeOutput(response, status, help) — the schema must agree."""
    expected = {"GroqNode": 3, "OpenrouterNode": 3, "OpenRouterModels": 2}
    for cls in NODE_CLASSES:
        schema = cls.define_schema()
        assert len(schema.outputs) == expected[schema.node_id]


def test_declared_dependencies_are_actually_imported():
    """requirements.txt should list only what the modules import beyond what ComfyUI provides."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "requirements.txt"), encoding="utf-8") as handle:
        declared = {
            line.split(">=")[0].split("==")[0].strip().lower()
            for line in handle
            if line.strip() and not line.strip().startswith("#")
        }
    # torch/torchvision belong to ComfyUI's environment; declaring them risks
    # pip replacing a working CUDA build with a CPU-only wheel.
    assert "torch" not in declared
    assert "torchvision" not in declared
    # Pillow and requests are in ComfyUI's own requirements.txt, so every real
    # ComfyUI environment has them; declaring them only adds install steps.
    assert "pillow" not in declared
    assert "requests" not in declared
    sources = "".join(
        open(os.path.join(root, name), encoding="utf-8").read()
        for name in ("groq_node.py", "openrouter.py", "openrouter_models.py",
                     "chat_common.py")
    )
    import_names = {"pillow": "from PIL", "requests": "import requests",
                    "torch": "import torch"}
    for package in declared:
        assert package in import_names, f"undeclared package in requirements: {package}"
        assert import_names[package] in sources, f"{package} is listed but never imported"


def test_pyproject_version_is_semver():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as handle:
        text = handle.read()
    version = [line for line in text.splitlines() if line.startswith("version")][0]
    number = version.split("=")[1].strip().strip('"')
    parts = number.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts), number


# --------------------------------------------------------------------------
# Resilience to provider catalogue churn
#
# Groq and OpenRouter retire and rename models constantly. These tests simulate
# that churn so the pack keeps working without an edit here.
# --------------------------------------------------------------------------

def test_groq_default_survives_its_preferred_models_being_retired(monkeypatch):
    """The shipped default vanishing is what broke this node once already."""
    survivors = ["--- Other ---", "groq/something-nobody-predicted", "Manual Input"]
    monkeypatch.setattr(groq_node, "STATIC_FALLBACK_MODELS", survivors)

    schema = groq_node.GroqNode.define_schema()
    model_input = schema.input_by_id("model")
    assert model_input.default == "groq/something-nobody-predicted"
    assert model_input.default in model_input.options


def test_groq_default_never_lands_on_a_separator_or_manual_input(monkeypatch):
    for listing in (
        ["--- Other ---", "a/model", "Manual Input"],
        ["a/model", "Manual Input"],
        ["--- Other ---", "Manual Input"],
        ["Manual Input"],
    ):
        monkeypatch.setattr(groq_node, "STATIC_FALLBACK_MODELS", listing)
        default = groq_node.GroqNode.define_schema().input_by_id("model").default
        assert not default.startswith("---")
        assert default in listing


def test_groq_default_prefers_the_first_available_preference(monkeypatch):
    second = groq_node.PREFERRED_DEFAULT_MODELS[1]
    monkeypatch.setattr(groq_node, "STATIC_FALLBACK_MODELS", [second, "z/other", "Manual Input"])
    assert groq_node.GroqNode.define_schema().input_by_id("model").default == second


def test_a_wholly_unfamiliar_groq_catalogue_still_yields_a_usable_dropdown(monkeypatch):
    """Every id here is invented; none match any constant in the module."""
    payload = {"data": [
        {"id": "vendor-x/thing-1", "active": True},
        {"id": "vendor-y/thing-2", "active": True},
        {"id": "vendor-z/speech-tts-1", "active": True},
    ]}
    monkeypatch.setattr(groq_node.requests, "get",
                        lambda *a, **k: _Response(200, payload))
    models, _ = groq_node._fetch_groq_models(api_key="key")

    offered = [m for m in models if not m.startswith("---") and m != "Manual Input"]
    assert offered == ["vendor-x/thing-1", "vendor-y/thing-2"]  # tts excluded
    assert _pick(models) in offered


def test_a_wholly_unfamiliar_openrouter_catalogue_still_filters_correctly(monkeypatch):
    """Only pricing and modality decide; no model name is known in advance."""
    payload = {"data": [
        {"id": "vendor-a/chat-1", "pricing": {"prompt": "0", "completion": "0"},
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}},
        {"id": "vendor-b/speaks-1", "pricing": {"prompt": "0", "completion": "0"},
         "architecture": {"input_modalities": ["text"], "output_modalities": ["audio"]}},
        {"id": "vendor-c/sees-1", "pricing": {"prompt": "0", "completion": "0"},
         "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]}},
        {"id": "vendor-d/costs-1", "pricing": {"prompt": "0.01", "completion": "0.01"},
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}},
    ]}
    monkeypatch.setattr(openrouter.requests, "get", lambda *a, **k: _Response(200, payload))

    free, vision = openrouter._fetch_openrouter_free_models()
    assert free == ["vendor-a/chat-1", "vendor-c/sees-1", "Manual Input"]
    assert "vendor-b/speaks-1" not in free      # emits audio, cannot chat
    assert "vendor-c/sees-1" in vision          # capability from metadata alone
    assert "vendor-d/costs-1" not in free       # priced


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _pick(models):
    return groq_node._pick_default_model(models)


def test_metadata_outranks_naming_conventions_in_both_nodes():
    """
    A future chat model whose name happens to trip a heuristic must not be
    hidden. Affirmative modality metadata always wins over the name backstop.
    """
    # Groq: an id containing "tts" that the API says emits text
    assert groq_node._is_audio_model("acme/tts-reasoning-chat") is True   # name only
    assert groq_node._is_audio_model(
        "acme/tts-reasoning-chat", {"output_modalities": ["text"]}) is False

    # OpenRouter: an id containing "embed" that the API says emits text
    assert openrouter._model_supports_chat({"id": "acme/embedded-reasoner"}) is False
    assert openrouter._model_supports_chat({
        "id": "acme/embedded-reasoner",
        "architecture": {"output_modalities": ["text"]}}) is True


def test_name_heuristics_still_apply_without_metadata():
    assert groq_node._is_audio_model("openai/whisper-large-v9", {}) is True
    assert groq_node._is_audio_model("openai/gpt-oss-999b", {}) is False
    assert openrouter._model_supports_chat({"id": "x/nemotron-embed-1b"}) is False
    assert openrouter._model_supports_chat({"id": "x/plain-chat"}) is True


def test_no_provider_lookup_happens_at_import_time():
    """
    define_schema() may call out; importing the modules must not. A pack that
    blocks on a provider outage at import time takes ComfyUI's startup with it.
    """
    import ast

    for module in (groq_node, openrouter, openrouter_models):
        tree = ast.parse(open(module.__file__, encoding="utf-8").read())
        # Only module-level statements run at import; skip function and class bodies.
        toplevel = [n for n in tree.body
                    if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        for node in toplevel:
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                name = ast.unparse(call.func)
                assert "requests" not in name, f"{module.__name__} calls {name} at import"
