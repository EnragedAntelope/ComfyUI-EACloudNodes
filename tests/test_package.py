"""Tests for how the pack presents itself to ComfyUI."""
import inspect
import os

import groq_node
import openrouter
import openrouter_models

NODE_MODULES = (groq_node, openrouter, openrouter_models)
NODE_CLASSES = (groq_node.GroqNode, openrouter.OpenrouterNode, openrouter_models.OpenRouterModels)


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
    """requirements.txt should list what the modules import, and nothing stale."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "requirements.txt")) as handle:
        declared = {
            line.split(">=")[0].split("==")[0].strip().lower()
            for line in handle if line.strip()
        }
    sources = "".join(
        open(os.path.join(root, name)).read()
        for name in ("groq_node.py", "openrouter.py", "openrouter_models.py")
    )
    import_names = {"pillow": "from PIL", "requests": "import requests",
                    "torch": "import torch", "torchvision": "from torchvision"}
    for package in declared:
        assert package in import_names, f"undeclared package in requirements: {package}"
        assert import_names[package] in sources, f"{package} is listed but never imported"


def test_pyproject_version_is_semver():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml")) as handle:
        text = handle.read()
    version = [l for l in text.splitlines() if l.startswith("version")][0]
    number = version.split("=")[1].strip().strip('"')
    parts = number.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts), number
