"""
Shared helpers for the EACloudNodes chat nodes.

Everything here is provider-neutral: tensor conversion, image encoding,
seed bookkeeping, and the retrying POST loop. The Groq and OpenRouter
nodes stay thin so a fix only ever lands once.
"""

import base64
import threading
import time
import io as python_io
import random

import requests
import torch
from PIL import Image

# One lock guards every mutable module-level structure (model caches and
# seed counters) across both node modules. Contention is negligible next
# to the network calls these protect.
LOCK = threading.RLock()

MAX_IMAGE_DIMENSION = 2048


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """
    Convert a ComfyUI IMAGE tensor to a PIL image without torchvision.

    Accepts [batch, height, width, channels] or [height, width, channels].
    Float tensors are expected in 0..1 (clamped); integer tensors are already
    0..255 and are passed through. Raises ValueError on impossible shapes.
    """
    if image.dim() == 4:
        image = image[0]
    if image.dim() != 3:
        raise ValueError(f"Expected a 3D or 4D image tensor, got {image.dim()}D")

    # ComfyUI hands us HWC; PIL wants channel-first until the very end.
    if image.shape[-1] in (1, 3, 4):
        image = image.permute(2, 0, 1)

    image = image.cpu()
    if image.is_floating_point():
        image = image.clamp(0, 1).mul(255).round().to(torch.uint8)
    else:
        image = image.to(torch.uint8)

    array = image.numpy()
    mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(array.shape[0])
    if mode is None:
        raise ValueError(f"Unsupported channel count: {array.shape[0]}")
    if mode == "L":
        array = array[0]
    else:
        array = array.transpose(1, 2, 0)
    return Image.fromarray(array, mode=mode)


def encode_image_message(pil_image: Image.Image, user_prompt: str,
                         image_format: str = "png") -> dict:
    """
    Build the multimodal user message carrying the image as a data URI.

    PNG is lossless; JPEG trades fidelity for a much smaller payload on
    photographic content. Raises ValueError when the image exceeds the
    size cap so the caller can surface it like any other image problem.
    """
    width, height = pil_image.size
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image too large ({width}x{height}). Maximum is "
            f"{MAX_IMAGE_DIMENSION} pixels in either dimension. Please resize your image."
        )

    fmt = "JPEG" if image_format == "jpeg" else "PNG"
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    if fmt == "JPEG" and pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")

    buffered = python_io.BytesIO()
    if fmt == "JPEG":
        pil_image.save(buffered, format=fmt, quality=90)
    else:
        pil_image.save(buffered, format=fmt)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_str}"}},
        ],
    }


def redact_body(body) -> object:
    """
    Copy a request body with data URIs replaced by a size note, safe for
    debug output. A full-resolution PNG base64 blob can be several MB;
    dropping it from the dumped JSON keeps the UI responsive while the
    rest of the body stays fully visible.
    """
    if isinstance(body, dict):
        return {k: redact_body(v) for k, v in body.items()}
    if isinstance(body, (list, tuple)):
        return [redact_body(v) for v in body]
    if isinstance(body, str) and body.startswith("data:image/"):
        return f"<redacted data URI, {len(body)} chars>"
    return body


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def derive_seed(counters: dict, key, mode: str, seed_value: int,
                max_safe_integer: int) -> int:
    """Resolve the seed for this run according to the selected mode."""
    if mode == "random":
        return random.randint(0, max_safe_integer)
    if mode == "increment":
        last = counters.get(key, seed_value)
        return (last + 1) % max_safe_integer
    if mode == "decrement":
        last = counters.get(key, seed_value)
        return last - 1 if last > 0 else max_safe_integer
    return seed_value


def store_seed(counters: dict, key, seed: int, max_tracked: int) -> None:
    """
    Remember the seed just used, evicting the oldest entries first.

    Counters are keyed by (model, starting seed); two nodes sharing a model
    and starting seed advance one shared counter because the v3 execute API
    exposes no per-instance id to key on. Eviction is FIFO rather than a
    blanket clear so long-lived counters survive unrelated churn.
    """
    while len(counters) >= max_tracked:
        del counters[next(iter(counters))]
    counters[key] = seed


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def check_interrupted() -> None:
    """
    Re-raise ComfyUI's interrupt signal if the user cancelled the queue.

    Imported lazily: outside a running ComfyUI (the test suite) the module
    does not exist and there is nothing to check. The raised exception is
    deliberately NOT caught anywhere in this pack.
    """
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def _backoff_seconds(attempt: int, retry_after=None) -> float:
    """
    Exponential backoff with jitter (full jitter halves the floor to spread
    concurrent clients), honouring a server-provided Retry-After when given.
    """
    if retry_after is not None:
        try:
            return min(60.0, max(1.0, float(retry_after)))
        except (TypeError, ValueError):
            pass
    base = min(30.0, 2.0 ** (attempt + 1))
    return random.uniform(base / 2.0, base)


def _retry_after_seconds(response) -> float | None:
    value = getattr(response, "headers", None)
    value = value.get("Retry-After") if value else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def post_with_retries(url: str, headers: dict, body: dict, *,
                      max_retries: int, timeout: float = 120):
    """
    POST the request, retrying rate limits, 5xx responses, timeouts and
    network errors with interrupt-aware exponential backoff.

    Returns (response, error_message): exactly one is None. A response comes
    back for every answered request regardless of status so callers keep
    their own status-code messaging; error_message carries exhausted
    transport failures (timeout / network) already worded for the user.
    """
    retries = 0
    while True:
        check_interrupted()
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)

            retryable = response.status_code in {429, 500, 502, 503, 504}
            if retryable and retries < max_retries:
                time.sleep(_backoff_seconds(retries,
                                            _retry_after_seconds(response)
                                            if response.status_code == 429 else None))
                retries += 1
                continue
            return response, None

        except requests.exceptions.Timeout:
            if retries < max_retries:
                time.sleep(_backoff_seconds(retries))
                retries += 1
                continue
            attempts = retries + 1
            return None, (f"Error: Request timed out after {attempts} "
                          f"attempt{'s' if attempts != 1 else ''}. Please try again")
        except requests.exceptions.RequestException as req_err:
            if retries < max_retries:
                time.sleep(_backoff_seconds(retries))
                retries += 1
                continue
            attempts = retries + 1
            return None, (f"Network Error: {str(req_err)} after {attempts} "
                          f"attempt{'s' if attempts != 1 else ''}.")
