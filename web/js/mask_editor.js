/**
 * VAE Noise Mask Editor — Interactive mask editor frontend extension.
 *
 * Registers with ComfyUI's extension system to add an "Edit Mask" button
 * to the VAENoiseMaskEditor node.  When clicked, a full-screen popup
 * canvas editor opens, showing the original image with a semi-transparent
 * red mask overlay.  The user can paint / erase mask regions, then save.
 *
 * Features:
 *   - Brush tool (add mask)
 *   - Eraser tool (remove mask)
 *   - Adjustable brush size (slider + mouse wheel)
 *   - Adjustable overlay opacity
 *   - Undo / Redo (Ctrl+Z / Ctrl+Shift+Z)
 *   - Clear / Invert mask
 *   - Zoom in/out (Ctrl+wheel, centred on cursor)
 *   - Pan (Space + drag)
 *   - Save & Close / Cancel
 */

import { app } from "/scripts/app.js";

// ─── Constants ──────────────────────────────────────────────────────────────
const NODE_NAME = "VAENoiseMaskEditor";
const API_BASE  = "/api/vae-noise-fix/editor-data";

const MASK_COLOR        = [255, 0, 0];   // red overlay
const DEFAULT_OPACITY   = 0.4;
const DEFAULT_BRUSH     = 20;
const MIN_BRUSH         = 1;
const MAX_BRUSH         = 200;
const UNDO_LIMIT        = 50;
const MIN_ZOOM          = 0.1;
const MAX_ZOOM          = 10;
const ZOOM_STEP         = 0.1;

// ─── Extension ──────────────────────────────────────────────────────────────

console.log("[VAENoiseFix] Loading Mask Editor extension...");

app.registerExtension({
    name: "VAENoiseFix.MaskEditor",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_NAME) return;
        
        console.log(`[VAENoiseFix] Registering UI for node: ${nodeData.name}`);

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            // Add "Edit Mask" button. The node_id is auto-injected on the
            // Python side via UNIQUE_ID, so the frontend just needs to use
            // this LiteGraph node's runtime id when talking to the API.
            this.addWidget("button", "✏️ Edit Mask", null, () => {
                console.log(`[VAENoiseFix] Opening editor for node ${this.id}`);
                openMaskEditor(String(this.id));
            });
        };
    },
});

// ─── Editor ─────────────────────────────────────────────────────────────────

async function openMaskEditor(nodeId) {
    // 1. Fetch staged data from backend
    let data;
    try {
        const res = await fetch(`${API_BASE}/${nodeId}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert(err.error || "No data available. Please run the workflow first.");
            return;
        }
        data = await res.json();
    } catch (e) {
        alert("Failed to fetch editor data: " + e.message);
        return;
    }

    const imgWidth  = data.width;
    const imgHeight = data.height;

    // 2. Decode image and mask
    const bgImage = await loadImage("data:image/jpeg;base64," + data.image);
    const maskImg = await loadImage("data:image/png;base64,"  + data.mask);

    // 3. Create off-screen mask canvas (source of truth)
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width  = imgWidth;
    maskCanvas.height = imgHeight;
    const maskCtx = maskCanvas.getContext("2d");
    maskCtx.drawImage(maskImg, 0, 0, imgWidth, imgHeight);

    // 4. Build UI
    buildEditorUI(nodeId, bgImage, maskCanvas, imgWidth, imgHeight);
}

/**
 * Build and show the full-screen editor overlay.
 */
function buildEditorUI(nodeId, bgImage, maskCanvas, imgW, imgH) {
    // ── State ───────────────────────────────────────────────────────────────
    let tool       = "brush";      // "brush" | "eraser"
    let brushSize  = DEFAULT_BRUSH;
    let opacity    = DEFAULT_OPACITY;
    let zoom       = 1.0;
    let panX       = 0;
    let panY       = 0;
    let isPanning  = false;
    let isDrawing  = false;
    let spaceDown  = false;
    let lastX      = null;
    let lastY      = null;

    // Reusable temporary canvas for drawColorizedMask (avoids per-frame allocation).
    const _tmpCanvas = document.createElement("canvas");
    _tmpCanvas.width  = imgW;
    _tmpCanvas.height = imgH;
    const _tmpCtx = _tmpCanvas.getContext("2d");

    // Undo / Redo stacks (store mask ImageData snapshots)
    const undoStack = [];
    const redoStack = [];
    function pushUndo() {
        const maskCtx = maskCanvas.getContext("2d");
        undoStack.push(maskCtx.getImageData(0, 0, imgW, imgH));
        if (undoStack.length > UNDO_LIMIT) undoStack.shift();
        redoStack.length = 0;
    }
    function undo() {
        if (undoStack.length === 0) return;
        const maskCtx = maskCanvas.getContext("2d");
        redoStack.push(maskCtx.getImageData(0, 0, imgW, imgH));
        maskCtx.putImageData(undoStack.pop(), 0, 0);
        render();
    }
    function redo() {
        if (redoStack.length === 0) return;
        const maskCtx = maskCanvas.getContext("2d");
        undoStack.push(maskCtx.getImageData(0, 0, imgW, imgH));
        maskCtx.putImageData(redoStack.pop(), 0, 0);
        render();
    }

    // ── Overlay container ───────────────────────────────────────────────────
    const overlay = document.createElement("div");
    overlay.id = "vae-mask-editor-overlay";
    Object.assign(overlay.style, {
        position: "fixed", top: 0, left: 0, width: "100vw", height: "100vh",
        background: "rgba(0,0,0,0.85)", zIndex: 99999,
        display: "flex", flexDirection: "column",
        fontFamily: "'Segoe UI', Arial, sans-serif", color: "#eee",
        userSelect: "none",
    });

    // ── Toolbar ─────────────────────────────────────────────────────────────
    const toolbar = document.createElement("div");
    Object.assign(toolbar.style, {
        display: "flex", alignItems: "center", gap: "12px",
        padding: "8px 16px", background: "#1a1a2e",
        borderBottom: "1px solid #333", flexShrink: 0, flexWrap: "wrap",
    });

    // Helper: create a toolbar button
    function tbBtn(label, onClick, accent = false) {
        const btn = document.createElement("button");
        btn.textContent = label;
        Object.assign(btn.style, {
            padding: "6px 14px", border: "1px solid #555", borderRadius: "4px",
            background: accent ? "#e94560" : "#16213e", color: "#eee",
            cursor: "pointer", fontSize: "13px", whiteSpace: "nowrap",
        });
        btn.addEventListener("mouseenter", () => btn.style.background = accent ? "#c0392b" : "#0f3460");
        btn.addEventListener("mouseleave", () => btn.style.background = accent ? "#e94560" : "#16213e");
        btn.addEventListener("click", onClick);
        return btn;
    }

    // Helper: create a labeled slider
    function tbSlider(label, min, max, value, step, onChange) {
        const wrap = document.createElement("div");
        wrap.style.display = "flex";
        wrap.style.alignItems = "center";
        wrap.style.gap = "6px";
        const lbl = document.createElement("span");
        lbl.style.fontSize = "12px";
        lbl.style.minWidth = "90px";
        lbl.textContent = `${label}: ${value}`;
        const slider = document.createElement("input");
        Object.assign(slider, { type: "range", min, max, value, step });
        slider.style.width = "120px";
        slider.style.accentColor = "#e94560";
        slider.addEventListener("input", () => {
            const v = parseFloat(slider.value);
            lbl.textContent = `${label}: ${Math.round(v * (step < 1 ? 100 : 1)) / (step < 1 ? 100 : 1)}`;
            onChange(v);
        });
        wrap.appendChild(lbl);
        wrap.appendChild(slider);
        wrap._slider = slider;
        wrap._label  = lbl;
        return wrap;
    }

    // Tool buttons
    const brushBtn  = tbBtn("🖌️ Brush",  () => { tool = "brush";  updateToolHighlight(); });
    const eraserBtn = tbBtn("🧹 Eraser", () => { tool = "eraser"; updateToolHighlight(); });

    function updateToolHighlight() {
        brushBtn.style.outline  = tool === "brush"  ? "2px solid #e94560" : "none";
        eraserBtn.style.outline = tool === "eraser" ? "2px solid #e94560" : "none";
    }
    updateToolHighlight();

    // Sliders
    const brushSlider = tbSlider("Brush", MIN_BRUSH, MAX_BRUSH, brushSize, 1, v => { brushSize = v; render(); });
    const opacitySlider = tbSlider("Opacity", 0, 1, opacity, 0.05, v => { opacity = v; render(); });

    // Action buttons
    const undoBtn    = tbBtn("↩ Undo",   undo);
    const redoBtn    = tbBtn("↪ Redo",   redo);
    const clearBtn   = tbBtn("🗑 Clear",  () => { pushUndo(); const ctx = maskCanvas.getContext("2d"); ctx.clearRect(0,0,imgW,imgH); render(); });
    const invertBtn  = tbBtn("🔄 Invert", () => {
        pushUndo();
        const ctx = maskCanvas.getContext("2d");
        const id  = ctx.getImageData(0, 0, imgW, imgH);
        for (let i = 3; i < id.data.length; i += 4) {
            id.data[i] = 255 - id.data[i]; // invert alpha
        }
        ctx.putImageData(id, 0, 0);
        render();
    });

    // Zoom display
    const zoomLabel = document.createElement("span");
    zoomLabel.style.fontSize = "12px";
    zoomLabel.style.minWidth = "60px";
    const updateZoomLabel = () => { zoomLabel.textContent = `Zoom: ${Math.round(zoom * 100)}%`; };
    updateZoomLabel();

    const fitBtn = tbBtn("⊡ Fit", () => { fitView(); render(); });

    // Save / Cancel
    const saveBtn   = tbBtn("💾 Save & Close", () => saveAndClose(), true);
    const cancelBtn = tbBtn("✖ Cancel",        () => closeEditor());

    // Assemble toolbar
    [brushBtn, eraserBtn, brushSlider, opacitySlider,
     undoBtn, redoBtn, clearBtn, invertBtn,
     zoomLabel, fitBtn,
     saveBtn, cancelBtn,
    ].forEach(el => toolbar.appendChild(el));

    // ── Canvas area ─────────────────────────────────────────────────────────
    const canvasWrap = document.createElement("div");
    Object.assign(canvasWrap.style, {
        flex: 1, overflow: "hidden", position: "relative", cursor: "crosshair",
    });

    const displayCanvas = document.createElement("canvas");
    const displayCtx    = displayCanvas.getContext("2d");

    canvasWrap.appendChild(displayCanvas);

    // ── Assemble ────────────────────────────────────────────────────────────
    overlay.appendChild(toolbar);
    overlay.appendChild(canvasWrap);
    document.body.appendChild(overlay);

    // Size canvas to available area
    function resizeDisplay() {
        const rect = canvasWrap.getBoundingClientRect();
        displayCanvas.width  = rect.width;
        displayCanvas.height = rect.height;
        render();
    }
    resizeDisplay();
    window.addEventListener("resize", resizeDisplay);

    // Fit initial view
    function fitView() {
        const cw = displayCanvas.width;
        const ch = displayCanvas.height;
        const scaleX = cw / imgW;
        const scaleY = ch / imgH;
        zoom = Math.min(scaleX, scaleY) * 0.95;
        panX = (cw - imgW * zoom) / 2;
        panY = (ch - imgH * zoom) / 2;
        updateZoomLabel();
    }
    fitView();

    // ── Render ───────────────────────────────────────────────────────────────
    function render() {
        const cw = displayCanvas.width;
        const ch = displayCanvas.height;
        displayCtx.clearRect(0, 0, cw, ch);

        displayCtx.save();
        displayCtx.translate(panX, panY);
        displayCtx.scale(zoom, zoom);

        // Draw background image
        displayCtx.drawImage(bgImage, 0, 0, imgW, imgH);

        // Draw mask overlay with user opacity
        // We need to colorize the mask: wherever mask alpha > 0, draw red
        displayCtx.globalAlpha = opacity;
        drawColorizedMask(displayCtx, maskCanvas, imgW, imgH);
        displayCtx.globalAlpha = 1.0;

        displayCtx.restore();

        // Draw brush cursor
        if (lastX !== null && lastY !== null && !isPanning) {
            displayCtx.beginPath();
            displayCtx.arc(lastX, lastY, brushSize * zoom / 2, 0, Math.PI * 2);
            displayCtx.strokeStyle = tool === "brush" ? "rgba(255,0,0,0.8)" : "rgba(255,255,255,0.8)";
            displayCtx.lineWidth = 1.5;
            displayCtx.stroke();
        }
    }

    /**
     * Draw the mask canvas onto displayCtx, colorized as red.
     * The mask canvas stores mask as alpha channel only.
     */
    function drawColorizedMask(ctx, mCanvas, w, h) {
        // Reuse closure-scoped temporary canvas to avoid per-frame allocation.
        _tmpCtx.clearRect(0, 0, w, h);

        // Draw mask
        _tmpCtx.drawImage(mCanvas, 0, 0);

        // Get pixel data and colorize
        const id = _tmpCtx.getImageData(0, 0, w, h);
        const d  = id.data;
        for (let i = 0; i < d.length; i += 4) {
            const a = d[i + 3];
            if (a > 0) {
                d[i]     = MASK_COLOR[0]; // R
                d[i + 1] = MASK_COLOR[1]; // G
                d[i + 2] = MASK_COLOR[2]; // B
                d[i + 3] = 255;            // Full alpha (overall opacity is from globalAlpha)
            }
        }
        _tmpCtx.putImageData(id, 0, 0);
        ctx.drawImage(_tmpCanvas, 0, 0, w, h);
    }

    // ── Convert screen coords to image coords ────────────────────────────────
    function screenToImage(sx, sy) {
        return [(sx - panX) / zoom, (sy - panY) / zoom];
    }

    // ── Drawing ──────────────────────────────────────────────────────────────
    function drawAt(ix, iy) {
        const maskCtx = maskCanvas.getContext("2d");
        maskCtx.save();
        maskCtx.beginPath();
        maskCtx.arc(ix, iy, brushSize / 2, 0, Math.PI * 2);
        if (tool === "brush") {
            maskCtx.globalCompositeOperation = "source-over";
            // RGB values are irrelevant — only the alpha channel is used
            // by drawColorizedMask and saveAndClose to determine mask regions.
            maskCtx.fillStyle = "rgba(255,255,255,1)";
        } else {
            maskCtx.globalCompositeOperation = "destination-out";
            maskCtx.fillStyle = "rgba(0,0,0,1)";
        }
        maskCtx.fill();
        maskCtx.restore();
    }

    function drawLine(x0, y0, x1, y1) {
        const dist = Math.hypot(x1 - x0, y1 - y0);
        const steps = Math.max(1, Math.ceil(dist / (brushSize / 4)));
        for (let i = 0; i <= steps; i++) {
            const t  = i / steps;
            const ix = x0 + (x1 - x0) * t;
            const iy = y0 + (y1 - y0) * t;
            drawAt(ix, iy);
        }
    }

    // ── Mouse events ────────────────────────────────────────────────────────
    let prevImgX = null;
    let prevImgY = null;
    let panStartX = 0;
    let panStartY = 0;

    canvasWrap.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;

        const rect = displayCanvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        if (spaceDown) {
            isPanning = true;
            panStartX = e.clientX - panX;
            panStartY = e.clientY - panY;
            canvasWrap.style.cursor = "grabbing";
            return;
        }

        isDrawing = true;
        pushUndo();
        const [ix, iy] = screenToImage(sx, sy);
        drawAt(ix, iy);
        prevImgX = ix;
        prevImgY = iy;
        render();
    });

    canvasWrap.addEventListener("mousemove", (e) => {
        const rect = displayCanvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        lastX = sx;
        lastY = sy;

        if (isPanning) {
            panX = e.clientX - panStartX;
            panY = e.clientY - panStartY;
            render();
            return;
        }

        if (isDrawing) {
            const [ix, iy] = screenToImage(sx, sy);
            drawLine(prevImgX, prevImgY, ix, iy);
            prevImgX = ix;
            prevImgY = iy;
        }
        render();
    });

    const endDraw = () => {
        isDrawing = false;
        isPanning = false;
        prevImgX = null;
        prevImgY = null;
        canvasWrap.style.cursor = spaceDown ? "grab" : "crosshair";
    };

    canvasWrap.addEventListener("mouseup",    endDraw);
    canvasWrap.addEventListener("mouseleave", () => { endDraw(); lastX = null; lastY = null; render(); });

    // ── Wheel: brush size (no modifier) / zoom (Ctrl) ────────────────────────
    canvasWrap.addEventListener("wheel", (e) => {
        e.preventDefault();

        if (e.ctrlKey) {
            // Zoom centred on mouse cursor
            const rect = displayCanvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;

            const oldZoom = zoom;
            const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
            zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom + delta * zoom));

            // Adjust pan so the point under cursor stays fixed
            panX = mx - (mx - panX) * (zoom / oldZoom);
            panY = my - (my - panY) * (zoom / oldZoom);

            updateZoomLabel();
        } else {
            // Adjust brush size
            const delta = e.deltaY > 0 ? -2 : 2;
            brushSize = Math.min(MAX_BRUSH, Math.max(MIN_BRUSH, brushSize + delta));
            brushSlider._slider.value = brushSize;
            brushSlider._label.textContent = `Brush: ${brushSize}`;
        }
        render();
    }, { passive: false });

    // ── Keyboard ─────────────────────────────────────────────────────────────
    function onKeyDown(e) {
        if (e.code === "Space") {
            e.preventDefault();
            spaceDown = true;
            if (!isDrawing) canvasWrap.style.cursor = "grab";
        }
        if (e.ctrlKey && e.code === "KeyZ") {
            e.preventDefault();
            if (e.shiftKey) redo(); else undo();
        }
        // Quick tool switch
        if (e.code === "KeyB" && !e.ctrlKey) { tool = "brush";  updateToolHighlight(); }
        if (e.code === "KeyE" && !e.ctrlKey) { tool = "eraser"; updateToolHighlight(); }
    }
    function onKeyUp(e) {
        if (e.code === "Space") {
            spaceDown = false;
            canvasWrap.style.cursor = "crosshair";
        }
    }
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup",   onKeyUp);

    // ── Save / Close ─────────────────────────────────────────────────────────
    async function saveAndClose() {
        // Extract mask as grayscale PNG (alpha → luminance)
        const maskCtx = maskCanvas.getContext("2d");
        const id = maskCtx.getImageData(0, 0, imgW, imgH);

        // Create a grayscale canvas
        const grayCanvas = document.createElement("canvas");
        grayCanvas.width  = imgW;
        grayCanvas.height = imgH;
        const gCtx = grayCanvas.getContext("2d");
        const gId  = gCtx.createImageData(imgW, imgH);
        for (let i = 0; i < id.data.length; i += 4) {
            const a = id.data[i + 3];              // mask is in alpha channel
            const v = a > 0 ? 255 : 0;
            gId.data[i]     = v;    // R
            gId.data[i + 1] = v;    // G
            gId.data[i + 2] = v;    // B
            gId.data[i + 3] = v;    // A (Server reads Alpha channel if RGBA)
        }
        gCtx.putImageData(gId, 0, 0);

        // Convert to base64 PNG
        const b64 = grayCanvas.toDataURL("image/png").split(",")[1];

        try {
            const res = await fetch(`${API_BASE}/${nodeId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mask: b64 }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                alert("Failed to save mask: " + (err.error || res.statusText));
                return;
            }
        } catch (e) {
            alert("Failed to save mask: " + e.message);
            return;
        }
        closeEditor();
    }

    function closeEditor() {
        window.removeEventListener("keydown", onKeyDown);
        window.removeEventListener("keyup",   onKeyUp);
        window.removeEventListener("resize",  resizeDisplay);
        overlay.remove();
    }

    // Initial render
    render();
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload  = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}
