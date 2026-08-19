"""Minimal stand-in for ComfyUI's `comfy_api.latest` so the nodes can be
imported and exercised outside a running ComfyUI install."""
import sys
import types


class _InputBase:
    def __init__(self, id=None, **kwargs):
        self.id = id
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"<Input {self.id}>"


def _make_io_type(name):
    class _IOType:
        Input = type(f"{name}Input", (_InputBase,), {})
        Output = type(
            f"{name}Output",
            (object,),
            {"__init__": lambda self, id=None, **kw: (setattr(self, "id", id), self.__dict__.update(kw))[0]},
        )
    _IOType.__name__ = name
    return _IOType


class Schema:
    def __init__(self, node_id=None, display_name=None, category=None,
                 description=None, inputs=None, outputs=None, **kwargs):
        self.node_id = node_id
        self.display_name = display_name
        self.category = category
        self.description = description
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.__dict__.update(kwargs)

    def input_by_id(self, input_id):
        for i in self.inputs:
            if i.id == input_id:
                return i
        raise KeyError(input_id)


class NodeOutput:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)

    def __getitem__(self, i):
        return self.args[i]


class ComfyNode:
    pass


class ComfyExtension:
    async def get_node_list(self):
        raise NotImplementedError


class io:
    String = _make_io_type("String")
    Int = _make_io_type("Int")
    Float = _make_io_type("Float")
    Combo = _make_io_type("Combo")
    Image = _make_io_type("Image")
    Schema = Schema
    NodeOutput = NodeOutput
    ComfyNode = ComfyNode


def install():
    """Register the stub modules in sys.modules (idempotent)."""
    if "comfy_api.latest" in sys.modules:
        return
    pkg = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ComfyExtension = ComfyExtension
    pkg.latest = latest
    sys.modules["comfy_api"] = pkg
    sys.modules["comfy_api.latest"] = latest
