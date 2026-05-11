"""
Server-side API routes for the interactive Mask Editor widget.

Registers two endpoints on the ComfyUI ``PromptServer``:

* **GET  /api/vae-noise-fix/editor-data/{node_id}**
  Returns the stored image (base64 JPEG) and mask (base64 PNG grayscale)
  for the given node so the JavaScript editor can render them.

* **POST /api/vae-noise-fix/editor-data/{node_id}**
  Receives an edited mask (base64 PNG) from the frontend and stores it
  so the next workflow execution picks up the user's edits.

A module-level dictionary ``_editor_store`` holds per-node data.  It is
intentionally **not** persisted to disk — data lives only for the
current ComfyUI session, which matches the ephemeral nature of the
editing workflow (detect → edit → inpaint → done).
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("VAENoiseMaskEditor.server")

# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

@dataclass
class EditorData:
    """Snapshot of the image and mask for one node instance."""

    image_rgb_u8: np.ndarray            # [H, W, 3] uint8 RGB
    mask_u8: np.ndarray                 # [H, W]    uint8 {0, 255}
    edited_mask_u8: Optional[np.ndarray] = field(default=None)  # user edit


# node_id → EditorData
_editor_store: Dict[str, EditorData] = {}
_MAX_STORE_ENTRIES = 32  # prevent unbounded memory growth


def store_editor_data(
    node_id: str,
    image_rgb_u8: np.ndarray,
    mask_u8: np.ndarray,
) -> None:
    """Called by the Python node during ``execute()`` to stage data."""
    _editor_store[node_id] = EditorData(
        image_rgb_u8=image_rgb_u8,
        mask_u8=mask_u8,
        edited_mask_u8=None,
    )
    # Evict oldest entries when the store exceeds the limit.
    while len(_editor_store) > _MAX_STORE_ENTRIES:
        _editor_store.pop(next(iter(_editor_store)))


def get_edited_mask(node_id: str) -> Optional[np.ndarray]:
    """Return the user-edited mask if available, else ``None``."""
    entry = _editor_store.get(node_id)
    if entry is not None and entry.edited_mask_u8 is not None:
        return entry.edited_mask_u8
    return None


def clear_edited_mask(node_id: str) -> None:
    """Reset the edited mask so the next run re-stages fresh data."""
    entry = _editor_store.get(node_id)
    if entry is not None:
        entry.edited_mask_u8 = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ndarray_to_base64_jpeg(arr: np.ndarray, quality: int = 90) -> str:
    """Encode an RGB uint8 array as a base64 JPEG string."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_to_base64_png(mask_u8: np.ndarray) -> str:
    """Encode a [H, W] uint8 mask as a base64 RGBA PNG.

    The mask value is stored in the **alpha channel** (RGB are zeroed),
    so the frontend can rely on alpha > 0 ↔ masked.  This matches how
    ``mask_editor.js`` interprets the canvas pixel data.
    """
    h, w = mask_u8.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = mask_u8
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _base64_png_to_mask(b64: str) -> np.ndarray:
    """Decode a base64 PNG into a [H, W] uint8 mask.

    Accepts either:
      - RGBA — alpha channel is taken as the mask (matches the editor's
        save format).
      - Grayscale / RGB — luminance is taken as the mask (backward-compat).
    """
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "LA"):
        return np.array(img.split()[-1], dtype=np.uint8)
    return np.array(img.convert("L"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Route registration (called at import time)
# ---------------------------------------------------------------------------

def _register_routes() -> None:
    """Register API endpoints on the ComfyUI PromptServer."""
    try:
        from server import PromptServer        # type: ignore[import-untyped]
        from aiohttp import web                 # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "PromptServer not available — mask editor API routes not registered. "
            "This is expected when running outside of ComfyUI."
        )
        return

    try:
        routes = PromptServer.instance.routes
    except (AttributeError, TypeError):
        logger.warning(
            "PromptServer.instance not ready — mask editor API routes not registered."
        )
        return

    # -- GET: serve image + mask to frontend ---------------------------------

    @routes.get("/api/vae-noise-fix/editor-data/{node_id}")
    async def _get_editor_data(request: web.Request) -> web.Response:
        node_id = request.match_info["node_id"]
        entry = _editor_store.get(node_id)
        if entry is None:
            return web.json_response(
                {"error": "No data staged for this node. Run the workflow first."},
                status=404,
            )

        # Use edited mask if available, else original
        mask_to_send = (
            entry.edited_mask_u8
            if entry.edited_mask_u8 is not None
            else entry.mask_u8
        )

        payload = {
            "image":  _ndarray_to_base64_jpeg(entry.image_rgb_u8),
            "mask":   _mask_to_base64_png(mask_to_send),
            "width":  int(entry.image_rgb_u8.shape[1]),
            "height": int(entry.image_rgb_u8.shape[0]),
        }
        return web.json_response(payload)

    # -- POST: receive edited mask from frontend -----------------------------

    @routes.post("/api/vae-noise-fix/editor-data/{node_id}")
    async def _post_editor_data(request: web.Request) -> web.Response:
        node_id = request.match_info["node_id"]
        entry = _editor_store.get(node_id)
        if entry is None:
            return web.json_response(
                {"error": "No data staged for this node."},
                status=404,
            )

        try:
            body = await request.json()
            mask_b64 = body["mask"]
        except Exception as exc:
            return web.json_response(
                {"error": f"Invalid request body: {exc}"},
                status=400,
            )

        edited = _base64_png_to_mask(mask_b64)

        # Validate dimensions match
        h, w = entry.image_rgb_u8.shape[:2]
        if edited.shape != (h, w):
            return web.json_response(
                {
                    "error": (
                        f"Mask dimensions {edited.shape} do not match "
                        f"image dimensions ({h}, {w})."
                    )
                },
                status=400,
            )

        entry.edited_mask_u8 = edited
        logger.info("Mask editor: saved edited mask for node %s (%dx%d)", node_id, w, h)
        return web.json_response({"status": "ok"})

    logger.info("Mask editor API routes registered.")


# Auto-register on import
_register_routes()
