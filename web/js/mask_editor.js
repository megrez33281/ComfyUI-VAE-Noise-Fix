/**
 * VAE Noise Mask Editor — Interactive mask editor frontend extension.
 *
 * Registers with ComfyUI's extension system to add an "Edit Mask" button
 * and an inline mask+image preview to the VAENoiseMaskEditor node.
 * When the button is clicked, a full-screen popup canvas editor opens.
 *
 * Features:
 *   - Inline node preview: image + mask overlay (updates after each save)
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
import { api } from "/scripts/api.js";

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

// ─── SVG icons (Lucide style) ────────────────────────────────────────────────

function makeSvgIcon(paths, size = 14) {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
        stroke-linejoin="round" style="vertical-align:middle;pointer-events:none">${paths}</svg>`;
}

// Paintbrush (Lucide paintbrush)
const SVG_BRUSH = makeSvgIcon(
    `<path d="M18.37 2.63 14 7l-1.59-1.59a2 2 0 0 0-2.82 0L8 7l9 9 1.59-1.59a2 2 0 0 0 0-2.82L17 10l4.37-4.37a2.12 2.12 0 1 0-3-3Z"/>
     <path d="M9 8c-2 3-4 3.5-7 4l8 8c1-.5 3.5-2.5 4-7"/>
     <path d="M14.5 17.5 4.5 15"/>`
);

// Eraser (Lucide eraser)
const SVG_ERASER = makeSvgIcon(
    `<path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/>
     <path d="M22 21H7"/>
     <path d="m5 11 9 9"/>`
);

// ─── Preview helpers ─────────────────────────────────────────────────────────

/**
 * Fetch image+mask from the API and render a composite onto `previewCanvas`.
 * Returns true on success, false if no data is available yet.
 */
async function renderNodePreview(nodeId, previewCanvas) {
    const res = await fetch(`${API_BASE}/${nodeId}`);
    if (!res.ok) return false;
    const data = await res.json();

    const bgImg   = await loadImage("data:image/jpeg;base64," + data.image);
    const maskImg = await loadImage("data:image/png;base64,"  + data.mask);

    const W = data.width;
    const H = data.height;
    previewCanvas.width  = W;
    previewCanvas.height = H;

    const ctx = previewCanvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(bgImg, 0, 0, W, H);

    // Colorize mask as semi-transparent red overlay
    const tmpC = document.createElement("canvas");
    tmpC.width = W; tmpC.height = H;
    const tmpCtx = tmpC.getContext("2d");
    tmpCtx.drawImage(maskImg, 0, 0, W, H);
    const imgData = tmpCtx.getImageData(0, 0, W, H);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
        if (d[i + 3] > 0) {
            d[i] = 255; d[i + 1] = 0; d[i + 2] = 0; d[i + 3] = 220;
        } else {
            d[i + 3] = 0;
        }
    }
    tmpCtx.putImageData(imgData, 0, 0);
    ctx.globalAlpha = 0.5;
    ctx.drawImage(tmpC, 0, 0, W, H);
    ctx.globalAlpha = 1.0;

    return true;
}

// ─── Extension ──────────────────────────────────────────────────────────────

console.log("[VAENoiseFix] Loading Mask Editor extension...");

// ── Helpers for looking up our node in the graph ────────────────────────────

function findVaeNode(nodeId) {
    // Prefer the official LiteGraph API; fall back to _nodes array scan.
    const byId = app.graph?.getNodeById?.(parseInt(nodeId));
    if (byId?.type === NODE_NAME) return byId;
    return app.graph?._nodes?.find(
        n => n.type === NODE_NAME && String(n.id) === String(nodeId)
    ) ?? null;
}

function refreshAllVaeNodes() {
    const nodes = app.graph?._nodes ?? [];
    for (const n of nodes) {
        if (n.type === NODE_NAME && typeof n._refreshMaskPreview === "function") {
            n._refreshMaskPreview();
        }
    }
}

// ── Event listeners ──────────────────────────────────────────────────────────

// Per-node: fires when a specific node finishes executing.
api.addEventListener("executed", ({ detail }) => {
    if (!detail?.node) return;
    const node = findVaeNode(detail.node);
    if (node) node._refreshMaskPreview?.();
});

// Full-prompt fallback: fires when the entire prompt finishes successfully.
// Catches cases where the per-node "executed" event is missed.
api.addEventListener("execution_success", () => {
    refreshAllVaeNodes();
});

app.registerExtension({
    name: "VAENoiseFix.MaskEditor",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_NAME) return;

        console.log(`[VAENoiseFix] Registering UI for node: ${nodeData.name}`);

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            const self = this;

            // ── "Edit Mask" button (no icon/emoji on the node) ──────────────
            this.addWidget("button", "Edit Mask", null, () => {
                console.log(`[VAENoiseFix] Opening editor for node ${self.id}`);
                openMaskEditor(String(self.id));
            });

            // ── Inline preview DOM widget ────────────────────────────────────
            const previewWrapper = document.createElement("div");
            Object.assign(previewWrapper.style, {
                width:          "100%",
                height:         "140px",
                background:     "#0d0d0d",
                display:        "flex",
                justifyContent: "center",
                alignItems:     "center",
                overflow:       "hidden",
                boxSizing:      "border-box",
                borderTop:      "1px solid #222",
            });

            const placeholder = document.createElement("span");
            placeholder.textContent = "Run workflow to preview";
            Object.assign(placeholder.style, {
                color:      "#444",
                fontSize:   "11px",
                fontFamily: "'Segoe UI', Arial, sans-serif",
            });
            previewWrapper.appendChild(placeholder);

            const previewCanvas = document.createElement("canvas");
            Object.assign(previewCanvas.style, {
                maxWidth:  "100%",
                maxHeight: "100%",
                display:   "none",
            });
            previewWrapper.appendChild(previewCanvas);

            this.addDOMWidget("mask_preview", "mask_preview", previewWrapper, {
                serialize: false,
            });

            // Attach refresh function to the node instance so the editor can
            // call it by looking up the node in app.graph._nodes.
            self._refreshMaskPreview = async () => {
                try {
                    const ok = await renderNodePreview(String(self.id), previewCanvas);
                    if (ok) {
                        placeholder.style.display = "none";
                        previewCanvas.style.display = "block";
                    }
                } catch (_) {
                    // No data yet — leave placeholder visible.
                }
            };

            // On page reload, the node may already have a valid id and staged data —
            // attempt a preview load. Guard against -1 (id not yet assigned by LiteGraph).
            if (self.id > 0) {
                self._refreshMaskPreview();
            }
        };

        // Third-layer fallback: ComfyUI calls onExecuted directly on the node
        // instance after receiving the backend "executed" message.
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);
            this._refreshMaskPreview?.();
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

    // Reusable temporary canvas for drawColorizedMask.
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
    // Fully opaque — no background bleed-through outside the image.
    const overlay = document.createElement("div");
    overlay.id = "vae-mask-editor-overlay";
    Object.assign(overlay.style, {
        position:       "fixed",
        top:            0,
        left:           0,
        width:          "100vw",
        height:         "100vh",
        background:     "#111111",   // solid, no transparency
        zIndex:         99999,
        display:        "flex",
        flexDirection:  "column",
        fontFamily:     "'Segoe UI', Arial, sans-serif",
        color:          "#eee",
        userSelect:     "none",
    });

    // ── Toolbar ─────────────────────────────────────────────────────────────
    const toolbar = document.createElement("div");
    Object.assign(toolbar.style, {
        display:         "flex",
        alignItems:      "center",
        gap:             "10px",
        padding:         "8px 16px",
        background:      "#1a1a2e",
        borderBottom:    "1px solid #2a2a3e",
        flexShrink:      0,
        flexWrap:        "wrap",
    });

    // Helper: plain text / HTML toolbar button
    function tbBtn(htmlContent, onClick, accent = false) {
        const btn = document.createElement("button");
        btn.innerHTML = htmlContent;
        Object.assign(btn.style, {
            display:        "inline-flex",
            alignItems:     "center",
            gap:            "5px",
            padding:        "5px 13px",
            border:         "1px solid " + (accent ? "#c0392b" : "#3a3a5a"),
            borderRadius:   "5px",
            background:     accent ? "#e94560" : "#16213e",
            color:          "#eee",
            cursor:         "pointer",
            fontSize:       "12px",
            fontWeight:     "500",
            whiteSpace:     "nowrap",
            transition:     "background 0.15s, border-color 0.15s",
            letterSpacing:  "0.02em",
        });
        btn.addEventListener("mouseenter", () => {
            btn.style.background    = accent ? "#c0392b" : "#0f3460";
            btn.style.borderColor   = accent ? "#a93226" : "#4a4a7a";
        });
        btn.addEventListener("mouseleave", () => {
            btn.style.background    = accent ? "#e94560" : "#16213e";
            btn.style.borderColor   = accent ? "#c0392b" : "#3a3a5a";
        });
        btn.addEventListener("click", onClick);
        return btn;
    }

    // Helper: labeled slider group
    function tbSlider(label, min, max, value, step, onChange) {
        const wrap = document.createElement("div");
        Object.assign(wrap.style, { display: "flex", alignItems: "center", gap: "6px" });

        const lbl = document.createElement("span");
        lbl.style.fontSize = "11px";
        lbl.style.minWidth = "88px";
        lbl.style.color    = "#aaa";
        lbl.textContent    = `${label}: ${value}`;

        const slider = document.createElement("input");
        Object.assign(slider, { type: "range", min, max, value, step });
        Object.assign(slider.style, { width: "110px", accentColor: "#e94560" });

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

    // Divider
    function tbDivider() {
        const d = document.createElement("div");
        Object.assign(d.style, {
            width:      "1px",
            height:     "22px",
            background: "#2a2a3e",
            flexShrink: 0,
        });
        return d;
    }

    // ── Tool buttons (with SVG icons) ────────────────────────────────────────
    const brushBtn  = tbBtn(`${SVG_BRUSH} Brush`,   () => { tool = "brush";  updateToolHighlight(); });
    const eraserBtn = tbBtn(`${SVG_ERASER} Eraser`, () => { tool = "eraser"; updateToolHighlight(); });

    function updateToolHighlight() {
        brushBtn.style.outline  = tool === "brush"  ? "2px solid #e94560" : "none";
        eraserBtn.style.outline = tool === "eraser" ? "2px solid #e94560" : "none";
    }
    updateToolHighlight();

    // ── Sliders ──────────────────────────────────────────────────────────────
    const brushSlider   = tbSlider("Brush",   MIN_BRUSH, MAX_BRUSH, brushSize, 1,    v => { brushSize = v; render(); });
    const opacitySlider = tbSlider("Opacity", 0,         1,         opacity,   0.05, v => { opacity   = v; render(); });

    // ── Action buttons ───────────────────────────────────────────────────────
    const undoBtn   = tbBtn("↩ Undo",   undo);
    const redoBtn   = tbBtn("↪ Redo",   redo);
    const clearBtn  = tbBtn("Clear",    () => { pushUndo(); const ctx = maskCanvas.getContext("2d"); ctx.clearRect(0, 0, imgW, imgH); render(); });
    const invertBtn = tbBtn("Invert",   () => {
        pushUndo();
        const ctx = maskCanvas.getContext("2d");
        const id  = ctx.getImageData(0, 0, imgW, imgH);
        for (let i = 3; i < id.data.length; i += 4) id.data[i] = 255 - id.data[i];
        ctx.putImageData(id, 0, 0);
        render();
    });

    // ── Zoom label & fit ─────────────────────────────────────────────────────
    const zoomLabel = document.createElement("span");
    Object.assign(zoomLabel.style, { fontSize: "11px", minWidth: "58px", color: "#aaa" });
    const updateZoomLabel = () => { zoomLabel.textContent = `Zoom: ${Math.round(zoom * 100)}%`; };
    updateZoomLabel();

    const fitBtn = tbBtn("Fit", () => { fitView(); render(); });

    // ── Save / Cancel ─────────────────────────────────────────────────────────
    const saveBtn   = tbBtn("Save & Close", () => saveAndClose(), true);
    const cancelBtn = tbBtn("Cancel",       () => closeEditor());

    // ── Assemble toolbar ─────────────────────────────────────────────────────
    [
        brushBtn, eraserBtn,
        tbDivider(),
        brushSlider, opacitySlider,
        tbDivider(),
        undoBtn, redoBtn,
        tbDivider(),
        clearBtn, invertBtn,
        tbDivider(),
        zoomLabel, fitBtn,
        tbDivider(),
        saveBtn, cancelBtn,
    ].forEach(el => toolbar.appendChild(el));

    // ── Canvas area ──────────────────────────────────────────────────────────
    const canvasWrap = document.createElement("div");
    Object.assign(canvasWrap.style, {
        flex:       1,
        overflow:   "hidden",
        position:   "relative",
        cursor:     "crosshair",
        background: "#111111",   // solid dark — no bleed from behind
    });

    const displayCanvas = document.createElement("canvas");
    const displayCtx    = displayCanvas.getContext("2d");

    canvasWrap.appendChild(displayCanvas);

    // ── Assemble ─────────────────────────────────────────────────────────────
    overlay.appendChild(toolbar);
    overlay.appendChild(canvasWrap);
    document.body.appendChild(overlay);

    // ── Size canvas to available area ─────────────────────────────────────────
    function resizeDisplay() {
        const rect = canvasWrap.getBoundingClientRect();
        displayCanvas.width  = rect.width;
        displayCanvas.height = rect.height;
        render();
    }
    resizeDisplay();
    window.addEventListener("resize", resizeDisplay);

    // ── Fit initial view ──────────────────────────────────────────────────────
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

    // ── Render ────────────────────────────────────────────────────────────────
    function render() {
        const cw = displayCanvas.width;
        const ch = displayCanvas.height;
        displayCtx.clearRect(0, 0, cw, ch);

        displayCtx.save();
        displayCtx.translate(panX, panY);
        displayCtx.scale(zoom, zoom);

        // Draw background image
        displayCtx.drawImage(bgImage, 0, 0, imgW, imgH);

        // Draw colorized mask overlay
        displayCtx.globalAlpha = opacity;
        drawColorizedMask(displayCtx, maskCanvas, imgW, imgH);
        displayCtx.globalAlpha = 1.0;

        displayCtx.restore();

        // Draw brush cursor
        if (lastX !== null && lastY !== null && !isPanning) {
            displayCtx.beginPath();
            displayCtx.arc(lastX, lastY, brushSize * zoom / 2, 0, Math.PI * 2);
            displayCtx.strokeStyle = tool === "brush" ? "rgba(255,80,80,0.9)" : "rgba(220,220,220,0.9)";
            displayCtx.lineWidth = 1.5;
            displayCtx.stroke();
        }
    }

    function drawColorizedMask(ctx, mCanvas, w, h) {
        _tmpCtx.clearRect(0, 0, w, h);
        _tmpCtx.drawImage(mCanvas, 0, 0);
        const id = _tmpCtx.getImageData(0, 0, w, h);
        const d  = id.data;
        for (let i = 0; i < d.length; i += 4) {
            if (d[i + 3] > 0) {
                d[i]     = MASK_COLOR[0];
                d[i + 1] = MASK_COLOR[1];
                d[i + 2] = MASK_COLOR[2];
                d[i + 3] = 255;
            }
        }
        _tmpCtx.putImageData(id, 0, 0);
        ctx.drawImage(_tmpCanvas, 0, 0, w, h);
    }

    // ── Coordinate conversion ─────────────────────────────────────────────────
    function screenToImage(sx, sy) {
        return [(sx - panX) / zoom, (sy - panY) / zoom];
    }

    // ── Drawing ───────────────────────────────────────────────────────────────
    function drawAt(ix, iy) {
        const maskCtx = maskCanvas.getContext("2d");
        maskCtx.save();
        maskCtx.beginPath();
        maskCtx.arc(ix, iy, brushSize / 2, 0, Math.PI * 2);
        if (tool === "brush") {
            maskCtx.globalCompositeOperation = "source-over";
            maskCtx.fillStyle = "rgba(255,255,255,1)";
        } else {
            maskCtx.globalCompositeOperation = "destination-out";
            maskCtx.fillStyle = "rgba(0,0,0,1)";
        }
        maskCtx.fill();
        maskCtx.restore();
    }

    function drawLine(x0, y0, x1, y1) {
        const dist  = Math.hypot(x1 - x0, y1 - y0);
        const steps = Math.max(1, Math.ceil(dist / (brushSize / 4)));
        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            drawAt(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
        }
    }

    // ── Mouse events ──────────────────────────────────────────────────────────
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
        prevImgX  = null;
        prevImgY  = null;
        canvasWrap.style.cursor = spaceDown ? "grab" : "crosshair";
    };

    canvasWrap.addEventListener("mouseup",    endDraw);
    canvasWrap.addEventListener("mouseleave", () => { endDraw(); lastX = null; lastY = null; render(); });

    // ── Wheel: brush size / zoom (Ctrl) ───────────────────────────────────────
    canvasWrap.addEventListener("wheel", (e) => {
        e.preventDefault();
        if (e.ctrlKey) {
            const rect    = displayCanvas.getBoundingClientRect();
            const mx      = e.clientX - rect.left;
            const my      = e.clientY - rect.top;
            const oldZoom = zoom;
            const delta   = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
            zoom  = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom + delta * zoom));
            panX  = mx - (mx - panX) * (zoom / oldZoom);
            panY  = my - (my - panY) * (zoom / oldZoom);
            updateZoomLabel();
        } else {
            const delta = e.deltaY > 0 ? -2 : 2;
            brushSize = Math.min(MAX_BRUSH, Math.max(MIN_BRUSH, brushSize + delta));
            brushSlider._slider.value    = brushSize;
            brushSlider._label.textContent = `Brush: ${brushSize}`;
        }
        render();
    }, { passive: false });

    // ── Keyboard ──────────────────────────────────────────────────────────────
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

    // ── Save / Close ──────────────────────────────────────────────────────────
    async function saveAndClose() {
        // Extract mask as grayscale PNG
        const maskCtx = maskCanvas.getContext("2d");
        const id = maskCtx.getImageData(0, 0, imgW, imgH);

        const grayCanvas = document.createElement("canvas");
        grayCanvas.width  = imgW;
        grayCanvas.height = imgH;
        const gCtx = grayCanvas.getContext("2d");
        const gId  = gCtx.createImageData(imgW, imgH);
        for (let i = 0; i < id.data.length; i += 4) {
            const v = id.data[i + 3] > 0 ? 255 : 0;
            gId.data[i]     = v;
            gId.data[i + 1] = v;
            gId.data[i + 2] = v;
            gId.data[i + 3] = v;
        }
        gCtx.putImageData(gId, 0, 0);

        const b64 = grayCanvas.toDataURL("image/png").split(",")[1];

        try {
            const res = await fetch(`${API_BASE}/${nodeId}`, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ mask: b64 }),
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

        // Refresh the inline node preview after a successful save.
        const targetNode = app.graph?._nodes?.find(n => String(n.id) === String(nodeId));
        if (targetNode?._refreshMaskPreview) {
            targetNode._refreshMaskPreview();
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

// ─── Helpers ─────────────────────────────────────────────────────────────────

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload  = () => resolve(img);
        img.onerror = reject;
        img.src     = src;
    });
}
