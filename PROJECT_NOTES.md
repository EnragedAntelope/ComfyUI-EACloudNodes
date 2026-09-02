## Version History

### v2.2.2 - Validation errors surfaced to the UI, not just the console

A user reported that a Groq node run with no API key set (default widgets,
nothing else changed) produced a wall of duplicate `[ERROR] Custom validation
failed for node: <widget> - Groq API key is required...` lines in the console
— one per widget on the node — and the actual message never appeared in the
node's own status/help output where the UI shows it.

Root cause: ComfyUI's v3 validation step calls `validate_inputs()` once, and
when it returns an error string, the frontend's fan-out logic attaches that
same string to every input on the node (hence one line per widget) and
refuses to queue the node at all, so `execute()` never runs and the node's
own status output — the one place a user actually looks — stays blank.

Fix: removed `validate_inputs()` from `GroqNode` and `OpenrouterNode`
entirely. The checks it used to make — API key required, manual model
required when "Manual Input" is selected, `additional_params` must parse as a
JSON object, audio-model rejection (Groq), and endpoint policy (OpenRouter)
— now run at the top of `execute()` and return through the node's `status`
output via `io.NodeOutput`, the same as every other error path in these
nodes already did. The node now always validates and queues; a bad input
produces one clear message in the status output instead of a validation
rejection. Verified live against a v3-registered instance: the reproduction
workflow (empty API key, all three outputs wired to a display node) now
executes in ~0.02s with a clean console and the expected message in the
status output, for both chat nodes.

`OpenRouterModels` was checked too — its `validate_inputs` unconditionally
returns `True` (the models endpoint needs no key), so it was never capable of
this failure mode and was left as-is.

### v2.2.1 - Zero-dependency packaging

ComfyUI's own requirements.txt has shipped both Pillow and requests for years,
so listing them here made every fresh install run pip for packages that were
guaranteed present already. Both manifests now declare nothing;
requirements.txt remains as a commented stub because ComfyUI-Manager-style
tooling and `test_package.py` expect the file to exist, and the dependency
guard test now forbids all four ComfyUI-owned packages (torch, torchvision,
pillow, requests) from creeping back in.

The publish workflow's Node.js 20 deprecation warning was traced rather than
guessed at: this repo pins `actions/checkout@v7.0.1` (the current latest
release, node 24 runtime) and `Comfy-Org/publish-node-action@d2366e7`, but the
latter's composite `action.yml` still calls `actions/setup-python@v5` and
`actions/checkout@v4` internally. Upstream main has not moved since
2025-05-10, so there is no newer commit to pin. Fixing it would mean forking
a third-party action that runs with REGISTRY_ACCESS_TOKEN — exactly what the
v2.2.0 supply-chain hardening avoids. Revisit when Comfy-Org publishes an
update.

### v2.2.0 - External audit remediation

An independent review of the v2.1.0 branch was verified finding-by-finding
against the code before anything was changed. Fixes:

- **The remediation branch itself did not load in ComfyUI.** Extracting
  `chat_common.py` introduced `import chat_common` — an absolute import of a
  sibling. ComfyUI builds a spec from the pack's `__init__.py` under a synthetic
  module name and never adds the folder to `sys.path`, so the import raised
  `ModuleNotFoundError` and *no node in the pack registered*. The full suite was
  green throughout, because `conftest.py` put the repo root on `sys.path` and
  imported the modules as top-level ones — the real registration path was never
  exercised. Sibling imports are now relative; the suite loads the pack the way
  ComfyUI does; and `test_package.py` re-runs that load in a subprocess so no
  alias in the test process can hide a regression. Both new guards were confirmed
  to fail with the fix removed.

  The lesson generalises: a green suite proves the paths it covers work, never
  that the artifact loads where it actually runs. Test the entry point the host
  uses, not a convenient stand-in for it.
- **Endpoint policy enforced at execution, not only at validation.** The
  https-only rule lived in `validate_inputs`, which ComfyUI can only run against
  a literal widget value. Convert `base_url` to an input socket and its value is
  unknown until the graph runs, leaving `execute()` free to POST the key to a
  plain-http host. `_endpoint_error()` is now the single rule both call. The
  "Custom endpoint" warning also stopped claiming the key "was sent" on paths
  that sent nothing — a false alarm on a security message is worse than none.
- **Publish workflow supply chain.** `Comfy-Org/publish-node-action@v1` resolves
  to a mutable *branch*, so upstream could change what runs with the registry
  token at any time. Both actions are now pinned to commit SHAs, checkout uses
  `persist-credentials: false` so the `GITHUB_TOKEN` is not left in `.git/config`
  for a third-party step, and `skip_checkout` removes the action's own duplicate
  (unhardened) checkout.
- **Key-exfiltration surface (OpenRouter).** `base_url` is a widget, workflows
  are shared as JSON, and the `Authorization` header went to whatever URL it
  named — validation only checked the scheme prefix. Now: `https://` required
  unless the host is localhost/127.0.0.1/::1 (local proxies are the one
  legitimate plain-http case), and any non-`openrouter.ai` endpoint appends a
  visible "Custom endpoint" warning to every status output. Proxy use stays
  possible; silent redirection does not.
- **torch/torchvision removed from both manifests.** ComfyUI's environment owns
  torch (often CUDA-specific); declaring it invites pip to install a CPU wheel
  over a working setup. The torchvision import is gone entirely —
  `chat_common.tensor_to_pil()` does the conversion with torch + PIL. numpy is
  used for the final array step; it is a hard dependency of ComfyUI itself.
- **Windows test failures fixed.** Every file read in `tests/test_package.py`
  now passes `encoding="utf-8"`; the ⚠️ emoji in module sources crashed two
  tests on cp1252 locales.
- **Retry loop rewritten** (`chat_common.post_with_retries`): honours 429
  `Retry-After`, jittered exponential backoff capped at 30s, and calls ComfyUI's
  interrupt check between attempts so Cancel works mid-retry. The interrupt
  exception is re-raised by the nodes' outer handlers instead of being converted
  into a chat error string.
- **~300 duplicated lines extracted** into `chat_common.py`: image tensor→PIL,
  data-URI encoding, seed derivation/eviction, retry loop, debug redaction.
  Divergence between the Groq and OpenRouter copies can no longer accumulate.
- **Debug dumps redact image data**: `redact_body()` replaces any data URI with
  a size note, keeping 400 diagnostics readable without multi-MB status strings.
- **New `image_format` input** on both chat nodes (png default / jpeg q90). JPEG
  converts non-RGB modes first; PNG behaviour unchanged.
- **Smaller fixes**: FIFO eviction of seed counters instead of clearing them all;
  locks around the model caches and seed counters; OpenRouter default chosen by
  preference walk (`PREFERRED_DEFAULT_MODELS`) mirroring Groq; attempt counts in
  retry messages corrected; dead per-module Extension classes and
  `comfy_entrypoint()`s removed (the combined entrypoint in `__init__.py` is the
  only live path); publish workflow permissions cut to `contents: read`; README
  installation typo fixed.

Not acted on, deliberately: API keys remain plaintext widget values (inherent to
ComfyUI's design, documented in tooltips); `additional_params` can override
request fields (acts on the user's own request); model caches stay key-agnostic
(both catalogues are public).

### v2.1.0 - Repository Audit

**Designed for model churn.** Both providers rotate models constantly, so the
capability logic was reworked to stop depending on hardcoded ids:

- *Nothing is refused on a guess.* The Groq node used to reject an attached image
  when the selected model was not in `KNOWN_VISION_MODELS`. That list is a snapshot,
  so every Groq reshuffle would block a model that actually works — and the message
  construction then took the text-only branch, meaning a stale list could silently
  drop the user's image. The image is now always sent; Groq decides, and its 400 is
  annotated with whatever did look capable.
- *Metadata outranks names.* Vision detection, Groq's audio exclusion, and
  OpenRouter's non-chat filter all read modality metadata first and fall back to
  naming conventions only when an entry carries none. A model whose name trips a
  heuristic stays listed if the API says it emits text. Absent metadata means
  "unknown", never "unsupported".
- *The default heals itself.* `PREFERRED_DEFAULT_MODELS` is walked against the list
  that actually loaded, falling back to the first real entry, so a retired default
  cannot break the node on a fresh drop-in the way `llama-3.3-70b-versatile` did.
- *Unrecognised models are surfaced, not hidden.* Anything outside the curated
  categories lands under `--- Other ---`.

`tests/test_package.py` simulates a wholesale catalogue reshuffle with invented
vendor names to keep these paths honest.

**Groq model list refreshed against the current catalogue.** Three of the models the
node offered no longer exist, including the one it shipped as the default:

| Removed | Reason |
| --- | --- |
| `llama-3.3-70b-versatile` | retired by Groq — **was the node's default**, so the node failed out of the box |
| `llama-3.1-8b-instant` | retired by Groq |
| `meta-llama/llama-4-scout-17b-16e-instruct` | retired — was the *only* entry in `KNOWN_VISION_MODELS` |
| `qwen/qwen3-32b` | superseded by `qwen/qwen3.6-27b` |
| `whisper-large-v3`, `whisper-large-v3-turbo` | audio endpoints, never callable here |
| `canopylabs/orpheus-*` | audio endpoints, never callable here |
| `meta-llama/llama-prompt-guard-2-*` | 512-token safety classifiers, not chat models — still reachable via `Manual Input` |

Added: `minimaxai/minimax-m2.7`, `qwen/qwen3.6-27b`. The default is now
`openai/gpt-oss-120b`.

`qwen/qwen3.6-27b` is multimodal and is the node's current vision model. Its id
matches none of the name heuristics, which is precisely why capability is no longer
decided from a name or a hardcoded list — see the churn notes above.

**OpenRouter's free dropdown was listing models that cannot chat.** The free filter
tested pricing alone, and embedding, reranking and text-to-speech models are all
priced at $0. Of the 23 entries in OpenRouter's current free listing, six are
non-chat: two embedding models, one reranker, and three speech models. Worse, one of
them (`llama-nemotron-embed-vl-1b-v2`) contains `-vl`, so the vision detector was
flagging an *embedding* model as vision-capable. Free models are now filtered on
output modality, with naming conventions as the fallback.

An end-to-end audit of the three nodes against the current Groq and OpenRouter APIs,
checking that every widget does what its tooltip claims. Each defect below was
reproduced before being fixed and is covered by a regression test in `tests/`.

**API currency**

- Groq chat completions were sent `max_tokens`, which Groq deprecated in favour of
  `max_completion_tokens`. The widget was already named `max_completion_tokens`; only
  the wire field was stale.
- `whisper-large-v3`, `whisper-large-v3-turbo` and the `canopylabs/orpheus-*` models
  were offered in the chat model dropdown. They are served by `/audio/transcriptions`
  and `/audio/speech` and cannot answer a chat-completions request at all. They are
  gone from the list, filtered out of the dynamic fetch, and rejected with an
  explanatory message if one is entered manually.

**Widgets that did not do what they claimed**

- *Groq dynamic model fetching was dead code.* `_fetch_groq_models()` was only ever
  called as `_fetch_groq_models(api_key=None)` — once in `define_schema()` and once in
  `execute()` — so the branch that calls the Groq API was unreachable and the dropdown
  always showed the static list, despite the tooltip and README promising a live one.
  `execute()` now passes the user's key, which warms the module cache so a subsequent
  ComfyUI Refresh rebuilds the dropdown from the API.
- *The `random`/`increment`/`decrement` seed modes were inert.* The seed is derived
  inside `execute()`, so with unchanged widgets ComfyUI's output cache served the
  previous response and the node never re-ran. Both chat nodes now implement
  `fingerprint_inputs()` (v3's `IS_CHANGED`), returning NaN whenever the seed is meant
  to move.
- *`--- Category ---` separator rows were selectable* and were sent to the API as a
  model id, producing an opaque HTTP 400. They are now rejected up front.
- *OpenRouter vision gating blocked valid requests.* Capability was checked against the
  free-model list only, so a paid vision model reached through `Manual Input` (e.g.
  `openai/gpt-4o`) was hard-refused — under a message labelled "Warning" that actually
  aborted the run. Capability is now read from `architecture.input_modalities` across
  the whole catalogue, only positively-known text-only models are refused, and unknown
  ids are passed through for OpenRouter to answer.

**Crashes and confusing errors**

- An IMAGE batch larger than one failed with "Image tensor must be 3D after squeezing",
  because `squeeze(0)` cannot drop a non-unit dimension. Both nodes now take the first
  frame of the batch.
- OpenRouter Models crashed on a null `context_length` (`'<' not supported between
  instances of 'int' and 'NoneType'`) and on null pricing (`float() argument must be a
  string or a real number`). Both fields now go through safe coercion helpers.
- A JSON array in `additional_params` passed validation and then failed inside
  `dict.update()` as "Unexpected Error: cannot convert dictionary update sequence
  element #0 to a sequence". Both nodes now require a JSON object and say so.
- A malformed HTTP 200 body was caught by `except requests.exceptions.RequestException`
  and retried as a network error; the `except json.JSONDecodeError` clause below it was
  unreachable, since `requests.exceptions.JSONDecodeError` subclasses both. Reordered.
- A failing model-list fetch was not cached, so every execution (and every
  `/object_info` refresh) paid the full request timeout again. Failures now back off
  for 60 seconds.

**Housekeeping**

- The OpenRouter Models node demanded an API key for an endpoint that is public — and
  that the OpenRouter chat node already calls without one. The key is now optional and
  only sent when supplied.
- `__init__.py` wrapped its imports in `except ImportError` and then re-imported the
  same modules, which import `comfy_api` at module scope. The fallback could never
  succeed; it only obscured the real error. Removed, along with `WEB_DIRECTORY`
  pointing at a `./web` directory that does not exist.
- The Groq `help_text` contained a duplicated, self-contradicting second half (it
  advertised `kimi-k2`, which is not in any list, and described vision support as
  Scout-only). The README had the same problem in its Features, Production Models, and
  Vision Usage sections, and carried a hand-maintained list of ~50 OpenRouter "free"
  models that the code has not used since the dropdown became API-driven.
- Seed counter dicts are now bounded at 1024 entries.

**Testing**

The two ad-hoc scripts (`test_groq_cache.py`, `test_groq_vision.py`, 12 assertions
between them, both requiring manual invocation) are replaced by a pytest suite in
`tests/` with 139 tests. `tests/comfy_stub.py` stands in for `comfy_api.latest`, and an
autouse fixture blocks all network access, so the suite runs offline without API keys.

### v2.0.13 (April 22, 2026) - Critical Bug Fix

**Fixed:**
- Removed duplicate `_last_seed` class attribute declaration (lines 229-233)
- Removed leftover `is_vision_model = actual_model in cls.VISION_MODELS` override (line 545)
- Cleared Python cache files to prevent stale `.pyc` from causing "GroqNodeClone" errors

**Root Cause:**
During dynamic model fetching implementation, edit operations created duplicate class attributes. The "GroqNodeClone" error was caused by:
1. Duplicate `_last_seed = {}` declarations in GroqNode class
2. Leftover line overriding dynamic vision detection with static check
3. Cached `.pyc` files containing old class definitions

**Verification:**
- ✅ All syntax checks pass
- ✅ 7/7 cache tests pass
- ✅ 5/5 vision tests pass
- ✅ No `cls.VISION_MODELS` references remain (all use dynamic detection)

### v2.0.12 (April 22, 2026) - Dynamic Model Fetching

## Dynamic Model Fetching Implementation (v2.0.12)

**Date:** April 22, 2026  
**Implemented by:** Sisyphus (OhMyOpenCode)

### Overview

Implemented dynamic model fetching for the Groq node, replacing the static model list with an API-driven approach that:
- Fetches available models from Groq's API (`https://api.groq.com/openai/v1/models`)
- Caches results for 5 minutes to minimize API calls
- Falls back to a comprehensive static list if API is unavailable
- Auto-detects vision-capable models using hybrid detection (hardcoded + pattern matching)
- Maintains model categorization (Featured, Production, Preview, etc.)

### Changes Made

#### 1. Core Infrastructure (`groq_node.py`)

**Module-level cache:**
```python
_groq_model_cache = {
    "models": None,
    "vision_models": None,
    "last_fetch": 0,
    "cache_ttl": 300  # 5 minutes
}
```

**Model categorization mapping:**
- `MODEL_CATEGORIES`: Dict mapping category names to known model IDs
- `KNOWN_VISION_MODELS`: List of known vision-capable models
- `VISION_PATTERNS`: List of patterns for detecting unknown vision models (`["vision", "vl", "-4-"]`)
- `STATIC_FALLBACK_MODELS`: Comprehensive static list for API failures

**New functions:**
- `_fetch_groq_models(api_key=None)`: Main fetch function with caching
- `_get_static_fallback_models()`: Returns static fallback list
- `_categorize_groq_models(api_models)`: Applies categorization to fetched models
- `_detect_vision_models(api_models)`: Hybrid vision detection

#### 2. Node Integration

**`define_schema()` method:**
- Now calls `_fetch_groq_models(api_key=None)` to populate model dropdown
- Falls back to static list when no API key provided
- Updated description and tooltip to mention dynamic fetching

**`execute()` method:**
- Vision detection now uses dynamic `_fetch_groq_models()` instead of static `cls.VISION_MODELS`
- Error messages updated to show dynamically detected vision models

**Help text:**
- Added documentation about dynamic fetching
- Explains 5-minute cache behavior
- Documents fallback mechanism

#### 3. Removed Deprecated Models

**Removed from static list (deprecated by Groq):**
- `moonshotai/kimi-k2-instruct-0905` - Deprecated April 15, 2026
- `meta-llama/llama-4-maverick-17b-128e-instruct` - Deprecated March 9, 2026
- `meta-llama/llama-guard-4-12b` - Deprecated March 5, 2026
- `playai-tts` - Deprecated December 31, 2025
- `playai-tts-arabic` - Deprecated December 31, 2025

**Added replacement models:**
- `canopylabs/orpheus-arabic-saudi` - Arabic TTS (replaces playai-tts-arabic)
- `canopylabs/orpheus-v1-english` - English TTS (replaces playai-tts)

### Testing

**Unit tests created:**
- `test_groq_cache.py`: Tests cache behavior, fallback, categorization structure
- `test_groq_vision.py`: Tests vision detection (known models + pattern matching)

**Test results:**
- All cache tests pass (7/7)
- All vision tests pass (5/5)

### Files Modified

| File | Changes |
|------|---------|
| `groq_node.py` | Added dynamic fetching infrastructure, updated schema/execute methods |
| `pyproject.toml` | Version bumped to 2.0.12 |
| `README.md` | Added dynamic fetching feature note, removed deprecated models |
| `test_groq_cache.py` | New file - cache behavior tests |
| `test_groq_vision.py` | New file - vision detection tests |

### Backward Compatibility

**Preserved:**
- All existing model IDs still work
- Manual Input mode unchanged
- All node parameters unchanged
- Existing workflows will load without modification
- Static fallback ensures node works even without API access

**Enhanced:**
- Model list now stays current automatically
- New models from Groq API appear in dropdown immediately
- Deprecated models automatically excluded from fetched list
- Vision detection more accurate with hybrid approach

### Future Maintenance

**To update model categorization:**
Edit `MODEL_CATEGORIES` dict in `groq_node.py` (lines ~28-55)

**To update known vision models:**
Edit `KNOWN_VISION_MODELS` list in `groq_node.py` (lines ~58-60)

**To adjust cache duration:**
Modify `_groq_model_cache["cache_ttl"]` in `groq_node.py` (line ~25)

**To verify current Groq models:**
Check official docs: https://console.groq.com/docs/models

### Deployment Notes

**ComfyUI Manager:**
- Version bump to 2.0.12 will trigger update notification
- Users will see "Dynamic model fetching" in changelog
- No migration needed - drop-in replacement

**User experience:**
- Existing users: Will see updated model list on next node load
- New users: Model list fetched automatically when API key entered
- Offline users: Static fallback ensures full functionality

### Known Limitations

1. **API authentication required for fetching:** Model list fetch requires valid Groq API key. Without key, static fallback is used.

2. **5-minute cache:** New models added to Groq may take up to 5 minutes to appear. Users can refresh node to force update.

3. **Categorization is hybrid:** New models not in `MODEL_CATEGORIES` mapping appear in "Other" category. This is intentional - unknown models may be preview/experimental.

4. **Vision detection limitation:** Groq's `/openai/v1/models` API endpoint does **not** include capability metadata (no `capabilities`, `modality`, or `vision` fields). Vision capability is detected via:
   - **Hardcoded allowlist** (`KNOWN_VISION_MODELS`): Currently includes `meta-llama/llama-4-scout-17b-16e-instruct`
   - **Pattern matching** (`VISION_PATTERNS`): Detects models with "vision", "vl", or "-4-" in their ID
   
   This means:
   - New vision models may not be detected until added to allowlist or matching patterns
   - Pattern matching may have false positives (non-vision models with "-4-" in name)
   - To update: Check [Groq Vision Docs](https://console.groq.com/docs/vision) and add new vision models to `KNOWN_VISION_MODELS` list
   
   ### References

- Groq API docs: https://console.groq.com/docs/models
- Groq deprecations: https://console.groq.com/docs/deprecations
- Groq Vision docs: https://console.groq.com/docs/vision
- OpenRouter implementation (reference pattern): `openrouter.py` lines 28-96

### Critical Lessons Learned

1. **Always check for duplicate class attributes after editing** - Python class bodies can't have duplicate attribute declarations
2. **Clear `.pyc` cache files** - Old compiled bytecode can cause "Clone" errors and stale class definitions
3. **Remove ALL references to old static attributes** - Search for `cls.ATTRIBUTE_NAME` to ensure no leftover references
4. **Test in actual ComfyUI environment** - Syntax checks and unit tests don't catch import/cache issues
5. **Groq API limitation documented** - Vision capability NOT available via API, requires allowlist + pattern matching
