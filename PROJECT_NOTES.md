## Version History

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
