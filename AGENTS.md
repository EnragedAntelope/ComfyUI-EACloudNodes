# AGENTS.md

ComfyUI custom-node pack exposing Groq Chat, OpenRouter Chat, and an OpenRouter
model browser as ComfyUI v3 nodes (`comfy_api.latest`), also registered through
legacy `NODE_CLASS_MAPPINGS`.

## Current state

_Last verified: 2026-08-23_

- **Status**: v2.2.0 on `claude/repo-audit-validation-5vorx4`; external audit remediated (endpoint policy, packaging, retry loop, dedupe into `chat_common.py`). Not yet merged to main.
- **Works**: all three nodes; full offline pytest suite (174 tests, no network/keys needed); vision via tensor→PIL in `chat_common.py` (no torchvision).
- **In progress**: nothing.
- **Known gaps**: seed counters are keyed by `(model, seed_value)` — two nodes sharing both advance one counter, because the v3 execute API exposes no per-instance id. Accepted.
- **Deep docs**: `PROJECT_NOTES.md` (version history with rationale), README (user-facing).

## Build / test

```bash
pip install pytest pillow requests torch numpy   # torch comes from ComfyUI normally
python -m pytest tests                            # rootdir pinned to tests/ by tests/pytest.ini
```

## Conventions

- `requirements.txt` must never list `torch`/`torchvision` — ComfyUI's environment owns them. `tests/test_package.py` enforces this.
- Shared chat-node plumbing lives in `chat_common.py`; do not re-duplicate image encoding, seeds, or the retry loop into the node modules.
- OpenRouter `base_url`: https-only except localhost hosts; non-`openrouter.ai` endpoints must keep their visible status warning. Don't weaken this without a replacement mitigation — workflows are shared JSON and carry the user's key.
- Retries go through `chat_common.post_with_retries` and must keep checking interrupts (`check_interrupted`) — never swallow `InterruptProcessingException`.
- Every schema input needs a tooltip; README ↔ pyproject version/description stay in sync per release.
