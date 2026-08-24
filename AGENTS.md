# AGENTS.md

ComfyUI custom-node pack exposing Groq Chat, OpenRouter Chat, and an OpenRouter
model browser as ComfyUI v3 nodes (`comfy_api.latest`), also registered through
legacy `NODE_CLASS_MAPPINGS`.

## Current state

_Last verified: 2026-08-24_

- **Status**: v2.2.0 on `main`. External audit remediated (endpoint policy, packaging, retry loop, dedupe into `chat_common.py`), then reviewed: the sibling import was absolute and the pack did not register inside ComfyUI at all — fixed, with the suite now loading the pack ComfyUI's way.
- **Works**: all three nodes; offline pytest suite (no network or API keys needed — `python -m pytest tests` reports the count); vision via tensor→PIL in `chat_common.py` (no torchvision).
- **In progress**: nothing.
- **Known gaps**: seed counters are keyed by `(model, seed_value)` — two nodes sharing both advance one counter, because the v3 execute API exposes no per-instance id. Accepted.
- **Deep docs**: `PROJECT_NOTES.md` (version history with rationale), README (user-facing).

## Build / test

```bash
pip install pytest pillow requests torch numpy   # torch comes from ComfyUI normally
python -m pytest tests                            # rootdir pinned to tests/ by tests/pytest.ini
```

## Conventions

- **Sibling modules must be imported relatively** (`from . import chat_common`).
  ComfyUI executes a pack's `__init__.py` under a synthetic module name and never
  puts the folder on `sys.path`, so an absolute `import chat_common` raises
  `ModuleNotFoundError` and takes every node's registration with it. The test
  suite loads the pack the same way; `tests/test_package.py` also re-runs that
  load in a clean interpreter, where no in-process alias can mask it.
- `requirements.txt` must never list `torch`/`torchvision` — ComfyUI's environment
  owns them. `tests/test_package.py` enforces this.
- Shared chat-node plumbing lives in `chat_common.py`; do not re-duplicate image
  encoding, seeds, or the retry loop into the node modules.
- OpenRouter `base_url`: https-only except localhost hosts, enforced in **both**
  `validate_inputs` and `execute` — a widget converted to an input socket has no
  value at validation time, and `execute` is where the key goes on the wire.
  Non-`openrouter.ai` endpoints must keep their visible status warning, and that
  warning must not claim the key "was sent" on a path that sent nothing. Don't
  weaken this without a replacement mitigation — workflows are shared JSON and
  carry the user's key.
- Retries go through `chat_common.post_with_retries` and must keep checking
  interrupts (`check_interrupted`) — never swallow `InterruptProcessingException`.
- Every schema input needs a tooltip; README ↔ pyproject version/description stay
  in sync per release.
- Provider model lists are time-sensitive. Check a claimed retirement against
  Groq's deprecation table (console.groq.com/docs/deprecations) before acting on
  it — the models page lags it.
- The publish workflow pins both actions to commit SHAs. `Comfy-Org/publish-node-action@v1`
  is a **mutable branch**, and the `1.0.1` tag calls `comfy --yes env`, which
  current comfy-cli rejects; pin to main-branch HEAD instead.
