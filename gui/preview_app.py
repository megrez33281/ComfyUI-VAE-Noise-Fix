"""
Interactive parameter-tuning GUI — tkinter edition.

Replaces the old OpenCV HighGUI window with a proper two-panel layout:

  LEFT   — image display canvas (fills remaining space).
  RIGHT  — scrollable control panel:
              Navigation · Parameters · View Mode · Brush Edit ·
              Tools · Actions · Statistics

Key design decisions
--------------------
* All UI chrome (labels, sliders, buttons, stats) lives in tkinter widgets
  — nothing is baked into the image pixels.  Saved images are therefore
  always clean, pixel-perfect views of the algorithm output.
* "Save Repaired" applies the current brush deltas to the detected mask,
  runs Telea inpainting on the *full-resolution* source image, and saves
  the result — fixing the previous gap where brush edits had no effect on
  the output.
* Parameter changes are debounced (80 ms) so dragging a slider does not
  trigger a re-run on every pixel of movement.
* Requires Pillow: ``pip install Pillow``
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit(
        "The GUI requires tkinter and Pillow.\n"
        "Install with:  pip install Pillow\n"
        f"(Original error: {exc})"
    ) from exc

from core import NoiseFixPipeline, PreviewMode, TeleaInpainter
from core.overlay import DebugOverlayRenderer
from core.pipeline import DetectionResult

from .brush_editor import BrushEditor
from .canvas_zoom import CanvasZoom
from .image_io import imread_safe, imwrite_safe
from .zoom_lens import ZoomLens


# ---------------------------------------------------------------------------
# View-mode metadata
# ---------------------------------------------------------------------------

_VIEW_MODES: List[PreviewMode] = list(PreviewMode)

# Listbox display labels.  A None entry in _LB_TO_MODE marks a separator row.
_VIEW_LABELS: List[str] = [
    "1   Original",
    "2   Mask Overlay  ✦",
    "3   Mask Only",
    "4   Repaired  ✦",
    "5   Side-by-Side",
    "─── debug ──────────",
    "6   Laplacian Energy",
    "7   Median Residual",
    "8   Seed Mask",
    "9   Context Mask",
    "0   Filtered Candidates",
    "-   Final Verified Mask",
]

# Maps listbox row index → PreviewMode list index (None = separator row)
_LB_TO_MODE: List[Optional[int]] = [0, 1, 2, 3, 4, None, 5, 6, 7, 8, 9, 10]

_SAVE_SUFFIXES: List[str] = [
    "_original", "_mask_overlay", "_mask_only", "_fixed", "_compare",
    "_laplacian", "_median_residual", "_seed", "_context", "_filtered", "_final",
]

# ---------------------------------------------------------------------------
# Dark colour palette
# ---------------------------------------------------------------------------

_BG     = "#1e1e1e"
_BG2    = "#2a2a2a"
_BG3    = "#373737"
_BG4    = "#444444"
_ACCENT = "#00c8a0"
_FG     = "#d4d4d4"
_DIM    = "#707070"
_DANGER = "#e06c75"
_GREEN  = "#1a3a28"
_RED    = "#3a1515"
_PANEL  = 304        # right panel fixed width in px
_DEBOUNCE = 80       # ms debounce before reprocessing after a slider move


# ---------------------------------------------------------------------------
# PreviewApp
# ---------------------------------------------------------------------------

class PreviewApp:
    """Tkinter-based interactive preview / parameter-tuning application."""

    def __init__(self, image_paths: List[str]) -> None:
        self._paths   = image_paths
        self._idx     = 0
        self._bgr:    Optional[np.ndarray]         = None
        self._result: Optional[DetectionResult]    = None
        self._dirty   = True
        self._pending: Optional[str]               = None  # after() handle

        # -- Root window -------------------------------------------------------
        self._root = tk.Tk()
        self._root.title("VAE Noise Fix  —  Preview")
        self._root.configure(bg=_BG)
        self._root.minsize(960, 600)
        self._root.geometry("1420x880")

        # -- Tk variables ------------------------------------------------------
        self._sens_var   = tk.DoubleVar(value=0.35)
        self._size_var   = tk.IntVar(value=6)
        self._dilate_var = tk.IntVar(value=2)
        self._view_var   = tk.IntVar(value=1)   # listbox row index
        self._lens_var   = tk.BooleanVar(value=False)
        self._brush_var  = tk.BooleanVar(value=False)   # brush toggle state
        self._brush_r    = tk.IntVar(value=8)

        # -- Tools -------------------------------------------------------------
        self._zoom_lens   = ZoomLens()
        self._canvas_zoom = CanvasZoom()
        self._brush       = BrushEditor()

        # -- Display cache -----------------------------------------------------
        self._photo:      Optional[ImageTk.PhotoImage] = None
        self._clean_view: Optional[np.ndarray]         = None  # for saving
        self._lb_x = 0; self._lb_y = 0; self._lb_s = 1.0
        self._mx   = 0; self._my   = 0

        self._build_ui()
        self._bind_keys()

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self) -> None:
        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(0, weight=1)

        paned = tk.PanedWindow(self._root, orient=tk.HORIZONTAL,
                               bg=_BG, bd=0, sashwidth=4,
                               sashrelief="flat", sashpad=0)
        paned.grid(row=0, column=0, sticky="nsew")

        left = tk.Frame(paned, bg=_BG)
        paned.add(left, stretch="always")
        self._build_canvas(left)

        right = tk.Frame(paned, bg=_BG, width=_PANEL)
        right.pack_propagate(False)
        paned.add(right, stretch="never")
        self._build_panel(right)

    # -- Image canvas ----------------------------------------------------------

    def _build_canvas(self, parent: tk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(parent, bg="#111111",
                                  highlightthickness=0, cursor="crosshair")
        self._canvas.grid(row=0, column=0, sticky="nsew")

        self._canvas.bind("<Configure>",       self._on_resize)
        self._canvas.bind("<Motion>",          self._on_move)
        self._canvas.bind("<ButtonPress-1>",   self._on_lb_down)
        self._canvas.bind("<B1-Motion>",       self._on_lb_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_lb_up)
        self._canvas.bind("<ButtonPress-3>",   self._on_rb_down)
        self._canvas.bind("<B3-Motion>",       self._on_rb_drag)
        self._canvas.bind("<ButtonRelease-3>", self._on_rb_up)
        # Windows/macOS scroll
        self._canvas.bind("<MouseWheel>", self._on_scroll_win)
        # Linux scroll
        self._canvas.bind("<Button-4>",
            lambda e: self._do_scroll(1,  bool(e.state & 4), e.x, e.y))
        self._canvas.bind("<Button-5>",
            lambda e: self._do_scroll(-1, bool(e.state & 4), e.x, e.y))

    # -- Right control panel ---------------------------------------------------

    def _build_panel(self, parent: tk.Frame) -> None:
        # Thin accent bar at the top edge of the panel
        tk.Frame(parent, bg=_ACCENT, height=2).pack(fill="x")

        outer = tk.Frame(parent, bg=_BG)
        outer.pack(fill="both", expand=True)

        pc = tk.Canvas(outer, bg=_BG, highlightthickness=0,
                       width=_PANEL - 16)
        sb = tk.Scrollbar(outer, orient="vertical", command=pc.yview,
                          bg=_BG3, troughcolor=_BG2, bd=0,
                          highlightthickness=0)
        self._inner = tk.Frame(pc, bg=_BG)
        self._inner.bind("<Configure>",
            lambda _e: pc.configure(scrollregion=pc.bbox("all")))
        pc.create_window((0, 0), window=self._inner, anchor="nw")
        pc.configure(yscrollcommand=sb.set)

        sb.pack(side="right", fill="y")
        pc.pack(side="left", fill="both", expand=True)
        pc.bind("<MouseWheel>",
            lambda e: pc.yview_scroll(-1 if e.delta > 0 else 1, "units"))

        self._build_nav(self._inner)
        self._build_params(self._inner)
        self._build_viewmode(self._inner)
        self._build_brush(self._inner)
        self._build_tools(self._inner)
        self._build_actions(self._inner)
        self._build_stats(self._inner)
        tk.Frame(self._inner, bg=_BG, height=20).pack()

    # -- Helper: labelled section frame ----------------------------------------

    def _section(self, parent: tk.Frame, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=_BG)
        outer.pack(fill="x", padx=8, pady=(8, 0))

        hdr = tk.Frame(outer, bg=_BG)
        hdr.pack(fill="x")
        tk.Label(hdr, text=title, bg=_BG, fg=_ACCENT,
                 font=("Helvetica", 8, "bold")).pack(side="left")
        tk.Frame(hdr, bg=_BG3, height=1).pack(
            side="left", fill="x", expand=True, padx=(6, 0), pady=5)

        body = tk.Frame(outer, bg=_BG2, padx=8, pady=6)
        body.pack(fill="x")
        return body

    # -- Helper: flat button ---------------------------------------------------

    @staticmethod
    def _btn(parent: tk.Frame, text: str, cmd,
             bg: str = _BG3, fg: str = _FG,
             font: tuple = ("Helvetica", 9)) -> tk.Button:
        return tk.Button(
            parent, text=text, command=cmd,
            bg=bg, fg=fg, font=font,
            relief="flat", bd=0, padx=10, pady=5,
            activebackground=_BG4, activeforeground=_FG,
            cursor="hand2")

    # -- Sections --------------------------------------------------------------

    def _build_nav(self, p: tk.Frame) -> None:
        body = self._section(p, "Navigation")

        self._fname_lbl = tk.Label(
            body, text="—", bg=_BG2, fg=_ACCENT,
            font=("Helvetica", 8), wraplength=260,
            justify="left", anchor="w")
        self._fname_lbl.pack(fill="x", pady=(0, 4))

        row = tk.Frame(body, bg=_BG2)
        row.pack(fill="x")
        self._btn(row, "◄", lambda: self._switch(-1)).pack(
            side="left", fill="x", expand=True, padx=(0, 2))
        self._nav_lbl = tk.Label(row, text="—", bg=_BG2, fg=_DIM,
                                  font=("Helvetica", 8))
        self._nav_lbl.pack(side="left", padx=4)
        self._btn(row, "►", lambda: self._switch(1)).pack(
            side="right", fill="x", expand=True, padx=(2, 0))

    def _build_params(self, p: tk.Frame) -> None:
        body = self._section(p, "Parameters")

        def slider_row(label: str, var: tk.Variable,
                       from_: float, to: float,
                       res: float, fmt: str) -> None:
            r = tk.Frame(body, bg=_BG2)
            r.pack(fill="x", pady=3)

            tk.Label(r, text=label, bg=_BG2, fg=_FG,
                     font=("Helvetica", 8), width=14,
                     anchor="w").pack(side="left")

            val_lbl = tk.Label(r, text=fmt.format(var.get()),
                                bg=_BG2, fg=_ACCENT,
                                font=("Courier", 8), width=5, anchor="e")
            val_lbl.pack(side="right")

            def _changed(*_):
                val_lbl.config(text=fmt.format(var.get()))
                self._dirty = True
                self._schedule()

            var.trace_add("write", _changed)

            tk.Scale(r, variable=var, from_=from_, to=to,
                     resolution=res, orient="horizontal", showvalue=False,
                     bg=_BG2, fg=_FG, troughcolor=_BG3,
                     highlightthickness=0, bd=0,
                     sliderlength=12, sliderrelief="flat",
                     activebackground=_ACCENT,
                     ).pack(side="left", fill="x", expand=True, padx=(6, 6))

        slider_row("Sensitivity",    self._sens_var,   0.01, 1.0,  0.01, "{:.2f}")
        slider_row("Max Noise Size", self._size_var,   1,    100,  1,    "{:d}")
        slider_row("Mask Dilate",    self._dilate_var, 0,    10,   1,    "{:d}")

    def _build_viewmode(self, p: tk.Frame) -> None:
        body = self._section(p, "View Mode")

        lbf = tk.Frame(body, bg=_BG2)
        lbf.pack(fill="x")

        sb = tk.Scrollbar(lbf, orient="vertical", bg=_BG3,
                           troughcolor=_BG2, bd=0, highlightthickness=0)
        self._lb = tk.Listbox(
            lbf,
            height=len(_VIEW_LABELS),
            selectmode="single",
            bg=_BG2, fg=_FG,
            selectbackground=_ACCENT, selectforeground="#000",
            activestyle="none",
            relief="flat", bd=0, highlightthickness=0,
            font=("Courier", 8),
            yscrollcommand=sb.set,
        )
        sb.config(command=self._lb.yview)

        for i, lbl in enumerate(_VIEW_LABELS):
            self._lb.insert(tk.END, f"  {lbl}")
            if _LB_TO_MODE[i] is None:   # separator — dim it
                self._lb.itemconfig(i, fg=_DIM,
                                    selectbackground=_BG2,
                                    selectforeground=_DIM)

        self._lb.select_set(1)
        self._lb.bind("<<ListboxSelect>>", self._on_lb_select)

        self._lb.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")

    def _build_brush(self, p: tk.Frame) -> None:
        body = self._section(p, "Brush Edit")

        # Toggle button: indicatoron=False → behaves like a push-button,
        # selectcolor is the background when checked (ON state).
        self._brush_cb = tk.Checkbutton(
            body,
            text="✏  Edit Mask  [E]",
            variable=self._brush_var,
            indicatoron=False,          # hide the checkbox diamond
            command=self._toggle_brush,
            bg=_BG3, fg=_FG,
            selectcolor=_GREEN,         # bg when ON
            activebackground=_BG4, activeforeground=_FG,
            font=("Helvetica", 9),
            relief="flat", bd=0, padx=10, pady=6,
            cursor="hand2",
        )
        self._brush_cb.pack(fill="x")

        # Radius
        rr = tk.Frame(body, bg=_BG2)
        rr.pack(fill="x", pady=(6, 0))
        tk.Label(rr, text="Radius  [  /  ]", bg=_BG2, fg=_FG,
                 font=("Helvetica", 8), width=14,
                 anchor="w").pack(side="left")
        self._br_lbl = tk.Label(rr, text=str(self._brush_r.get()),
                                  bg=_BG2, fg=_ACCENT,
                                  font=("Courier", 8), width=3, anchor="e")
        self._br_lbl.pack(side="right")

        def _r_changed(*_):
            r = int(self._brush_r.get())
            self._brush.adjust_radius(r - self._brush.radius)
            self._br_lbl.config(text=str(r))

        self._brush_r.trace_add("write", _r_changed)
        tk.Scale(rr, variable=self._brush_r, from_=1, to=100,
                 resolution=1, orient="horizontal", showvalue=False,
                 bg=_BG2, troughcolor=_BG3,
                 highlightthickness=0, bd=0, sliderlength=12,
                 sliderrelief="flat", activebackground=_ACCENT,
                 ).pack(side="left", fill="x", expand=True, padx=(6, 6))

        self._btn(body, "Clear Edits  [C]", self._clear_brush,
                  bg=_RED, fg=_DANGER).pack(fill="x", pady=(6, 0))

        tk.Label(body, text="Left-drag: paint    Right-drag: erase",
                 bg=_BG2, fg=_DIM, font=("Helvetica", 7)).pack(pady=(4, 0))

    def _build_tools(self, p: tk.Frame) -> None:
        body = self._section(p, "Tools")

        tk.Checkbutton(body, text="Zoom Lens  [Z]",
                        variable=self._lens_var, command=self._toggle_lens,
                        bg=_BG2, fg=_FG, selectcolor=_BG3,
                        activebackground=_BG2, activeforeground=_FG,
                        font=("Helvetica", 8),
                        highlightthickness=0, bd=0,
                        ).pack(anchor="w")

        self._btn(body, "Reset Canvas Zoom  [R]",
                  self._reset_zoom).pack(fill="x", pady=(6, 0))

        self._zoom_lbl = tk.Label(body, text="Canvas: 1.0×  (Ctrl + Scroll)",
                                   bg=_BG2, fg=_DIM, font=("Helvetica", 7))
        self._zoom_lbl.pack(anchor="w", pady=(2, 0))

    def _build_actions(self, p: tk.Frame) -> None:
        body = self._section(p, "Actions")

        self._btn(body, "💾  Save Current View  [S]",
                  self._save_view).pack(fill="x", pady=(0, 4))

        self._btn(body, "✅  Apply Edits & Save Repaired",
                  self._save_repaired,
                  bg=_GREEN, fg=_ACCENT,
                  font=("Helvetica", 9, "bold")).pack(fill="x")

        self._status_lbl = tk.Label(body, text="", bg=_BG2, fg=_ACCENT,
                                     font=("Helvetica", 7),
                                     wraplength=260, justify="left")
        self._status_lbl.pack(fill="x", pady=(4, 0))

    def _build_stats(self, p: tk.Frame) -> None:
        body = self._section(p, "Statistics")

        self._stat: dict = {}
        rows = [
            ("res",      "Resolution"),
            ("blobs",    "Noise Blobs"),
            ("pixels",   "Noise Pixels"),
            ("coverage", "Coverage"),
            ("area",     "Area Range"),
            ("elapsed",  "Detection Time"),
        ]
        for key, label in rows:
            r = tk.Frame(body, bg=_BG2)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=f"{label}:", bg=_BG2, fg=_DIM,
                     font=("Helvetica", 7), width=14,
                     anchor="w").pack(side="left")
            lbl = tk.Label(r, text="—", bg=_BG2, fg=_FG,
                            font=("Courier", 7), anchor="w")
            lbl.pack(side="left", fill="x")
            self._stat[key] = lbl

    # =========================================================================
    # Keyboard shortcuts
    # =========================================================================

    def _bind_keys(self) -> None:
        r = self._root
        for i, k in enumerate(["1", "2", "3", "4", "5",
                                "6", "7", "8", "9", "0", "minus"]):
            r.bind(f"<Key-{k}>", lambda _e, idx=i: self._set_view_by_mode(idx))
        r.bind("<e>",            lambda _e: self._toggle_brush())
        r.bind("<E>",            lambda _e: self._toggle_brush())
        r.bind("<z>",            lambda _e: self._toggle_lens())
        r.bind("<Z>",            lambda _e: self._toggle_lens())
        r.bind("<s>",            lambda _e: self._save_view())
        r.bind("<S>",            lambda _e: self._save_view())
        r.bind("<a>",            lambda _e: self._switch(-1))
        r.bind("<A>",            lambda _e: self._switch(-1))
        r.bind("<d>",            lambda _e: self._switch(1))
        r.bind("<D>",            lambda _e: self._switch(1))
        r.bind("<r>",            lambda _e: self._reset_zoom())
        r.bind("<R>",            lambda _e: self._reset_zoom())
        r.bind("<c>",            lambda _e: self._clear_brush())
        r.bind("<C>",            lambda _e: self._clear_brush())
        r.bind("<bracketleft>",  lambda _e: self._adj_brush(-2))
        r.bind("<bracketright>", lambda _e: self._adj_brush(+2))
        r.bind("<q>",            lambda _e: self._root.destroy())
        r.bind("<Escape>",       lambda _e: self._root.destroy())

    # =========================================================================
    # Image loading & navigation
    # =========================================================================

    def run(self) -> None:
        self._load(0)
        self._root.mainloop()

    def _load(self, idx: int) -> None:
        self._idx = idx % len(self._paths)
        bgr = imread_safe(self._paths[self._idx])
        if bgr is None:
            messagebox.showerror("Error", "Cannot read image.")
            return
        self._bgr = bgr
        self._canvas_zoom.reset()
        self._brush.attach_to(bgr.shape[:2])
        self._dirty = True
        self._fname_lbl.config(
            text=os.path.basename(self._paths[self._idx]), fg=_ACCENT)
        self._nav_lbl.config(text=f"{self._idx + 1} / {len(self._paths)}")
        self._schedule(immediate=True)

    def _switch(self, delta: int) -> None:
        if len(self._paths) > 1:
            self._load((self._idx + delta) % len(self._paths))

    # =========================================================================
    # Processing pipeline
    # =========================================================================

    def _schedule(self, immediate: bool = False) -> None:
        """Debounced refresh: cancel any pending call, wait, then fire."""
        if self._pending is not None:
            self._root.after_cancel(self._pending)
        delay = 0 if immediate else _DEBOUNCE
        self._pending = self._root.after(delay, self._refresh)

    def _refresh(self) -> None:
        self._pending = None
        if self._bgr is None:
            return
        if self._dirty:
            self._process()
            self._dirty = False
        self._redraw()

    def _process(self) -> None:
        pipeline = NoiseFixPipeline(
            gradient_sensitivity=float(self._sens_var.get()),
            max_noise_size=int(self._size_var.get()),
            mask_dilate=int(self._dilate_var.get()),
        )
        self._result = pipeline.run(self._bgr)
        self._update_stats()

    def _update_stats(self) -> None:
        if self._result is None:
            return
        s = self._result.stats.as_dict()
        self._stat["res"].config(text=str(s["resolution"]))
        self._stat["blobs"].config(text=str(s["noise_blobs"]))
        self._stat["pixels"].config(text=str(s["noise_pixels"]))
        self._stat["coverage"].config(text=f"{s['coverage_pct']:.4f}%")
        self._stat["area"].config(
            text=f"{s['area_min']}–{s['area_max']}  avg {s['area_avg']:.1f}")
        self._stat["elapsed"].config(text=f"{s['elapsed_ms']:.1f} ms")

    # =========================================================================
    # Rendering
    # =========================================================================

    def _current_mode(self) -> PreviewMode:
        lb_row   = self._view_var.get()
        mode_idx = _LB_TO_MODE[lb_row] if lb_row < len(_LB_TO_MODE) else 1
        return _VIEW_MODES[mode_idx if mode_idx is not None else 1]

    def _get_clean_view(self) -> np.ndarray:
        """Compute the current view — no zoom-lens, no brush cursor baked in."""
        if self._bgr is None or self._result is None:
            return np.zeros((400, 600, 3), dtype=np.uint8)

        mode      = self._current_mode()
        algo_mask = self._result.final_mask
        eff_mask  = (self._brush.apply(algo_mask)
                     if self._brush.has_edits() else algo_mask)

        if self._brush.enabled:
            # Always render live overlay so painted edits appear immediately.
            view = DebugOverlayRenderer.render(self._bgr, eff_mask)
        elif self._brush.has_edits() and mode == PreviewMode.MASK_OVERLAY:
            view = DebugOverlayRenderer.render(self._bgr, eff_mask)
        elif self._brush.has_edits() and mode == PreviewMode.REPAIRED:
            # Live repaired preview: re-inpaint with the edited mask.
            view = (TeleaInpainter.inpaint(
                        self._bgr, eff_mask, int(self._size_var.get()))
                    if eff_mask.any() else self._bgr.copy())
        else:
            view = self._result.select_view(mode).copy()

        return self._canvas_zoom.apply(view)

    def _redraw(self) -> None:
        view             = self._get_clean_view()
        self._clean_view = view   # ← clean copy for saving (no overlays)

        # Add zoom-lens overlay on a separate display copy only.
        # Brush cursor is drawn as a tk oval in _draw_cursor_on_canvas —
        # no PIL roundtrip needed on mouse moves.
        display = view.copy()
        if self._zoom_lens.enabled:
            self._zoom_lens.update_position(self._mx, self._my)
            display = self._zoom_lens.render(display)

        # BGR → RGB → PIL → ImageTk, letterboxed into the canvas widget.
        rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)

        cw = max(1, self._canvas.winfo_width())
        ch = max(1, self._canvas.winfo_height())
        iw, ih = pil.size
        scale  = min(cw / iw, ch / ih)
        nw     = max(1, int(iw * scale))
        nh     = max(1, int(ih * scale))

        pil    = pil.resize((nw, nh),
                             Image.LANCZOS if scale < 1.0 else Image.NEAREST)

        self._photo = ImageTk.PhotoImage(pil)
        self._canvas.delete("all")

        xoff = (cw - nw) // 2
        yoff = (ch - nh) // 2
        self._lb_x, self._lb_y, self._lb_s = xoff, yoff, scale
        self._canvas.create_image(xoff, yoff, anchor="nw", image=self._photo)

        # Re-draw the brush cursor on top (it was wiped by delete("all")).
        self._draw_cursor_on_canvas(self._mx, self._my)

        self._zoom_lbl.config(
            text=f"Canvas: {self._canvas_zoom.zoom:.1f}×  (Ctrl + Scroll)")

    def _on_resize(self, _event) -> None:
        if self._result is not None:
            self._redraw()

    def _draw_cursor_on_canvas(self, x: int, y: int) -> None:
        """Draw the brush cursor ring directly on the canvas — no PIL roundtrip."""
        self._canvas.delete("cursor")
        if not self._brush.enabled:
            return
        zoom = self._canvas_zoom.zoom
        r = max(1, round(self._brush.radius * zoom * self._lb_s))
        # Black outer ring for contrast on any background.
        self._canvas.create_oval(
            x - r - 1, y - r - 1, x + r + 1, y + r + 1,
            outline="#000000", width=1, tags="cursor")
        # Cyan inner ring.
        self._canvas.create_oval(
            x - r, y - r, x + r, y + r,
            outline="#00ffff", width=1, tags="cursor")

    # =========================================================================
    # Mouse events
    # =========================================================================

    def _canvas_to_img(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        """Map a canvas pixel → source-image pixel (undoes letterbox + zoom)."""
        if self._bgr is None:
            return None
        s  = self._lb_s
        ix = int((cx - self._lb_x) / s) if s > 0 else cx
        iy = int((cy - self._lb_y) / s) if s > 0 else cy
        h, w = self._bgr.shape[:2]
        zoom = self._canvas_zoom.zoom
        if zoom > 1.01:
            vw, vh = w / zoom, h / zoom
            ncx, ncy = self._canvas_zoom.center
            x0 = max(0, min(int(ncx * w - vw / 2), w - int(vw)))
            y0 = max(0, min(int(ncy * h - vh / 2), h - int(vh)))
            ix = x0 + int(ix / zoom)
            iy = y0 + int(iy / zoom)
        if 0 <= ix < w and 0 <= iy < h:
            return ix, iy
        return None

    def _on_move(self, e) -> None:
        self._mx, self._my = e.x, e.y
        if self._brush.enabled:
            self._draw_cursor_on_canvas(e.x, e.y)  # instant — no PIL roundtrip
        elif self._zoom_lens.enabled:
            self._redraw()

    def _on_lb_down(self, e) -> None:
        if self._brush.enabled:
            self._brush.begin_stroke()
            p = self._canvas_to_img(e.x, e.y)
            if p:
                self._brush.stroke(*p, mode="add")
                self._redraw()

    def _on_lb_drag(self, e) -> None:
        self._mx, self._my = e.x, e.y
        if self._brush.enabled:
            p = self._canvas_to_img(e.x, e.y)
            if p:
                self._brush.stroke(*p, mode="add")
                self._redraw()

    def _on_lb_up(self, _e) -> None:
        self._brush.end_stroke()

    def _on_rb_down(self, e) -> None:
        if self._brush.enabled:
            self._brush.begin_stroke()
            p = self._canvas_to_img(e.x, e.y)
            if p:
                self._brush.stroke(*p, mode="erase")
                self._redraw()

    def _on_rb_drag(self, e) -> None:
        self._mx, self._my = e.x, e.y
        if self._brush.enabled:
            p = self._canvas_to_img(e.x, e.y)
            if p:
                self._brush.stroke(*p, mode="erase")
                self._redraw()

    def _on_rb_up(self, _e) -> None:
        self._brush.end_stroke()

    def _on_scroll_win(self, e) -> None:
        self._do_scroll(1 if e.delta > 0 else -1,
                        bool(e.state & 4), e.x, e.y)

    def _do_scroll(self, delta: int, ctrl: bool, x: int, y: int) -> None:
        if self._brush.enabled:
            self._adj_brush(delta * 2)
            return
        if ctrl:
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            aw = max(1, cw - 2 * self._lb_x)
            ah = max(1, ch - 2 * self._lb_y)
            nx = max(0.0, min((x - self._lb_x) / aw, 1.0))
            ny = max(0.0, min((y - self._lb_y) / ah, 1.0))
            self._canvas_zoom.adjust(delta, nx, ny)
            self._redraw()
        else:
            self._zoom_lens.adjust_magnification(delta)
            if self._zoom_lens.enabled:
                self._redraw()

    # =========================================================================
    # Control actions
    # =========================================================================

    def _set_view_by_mode(self, mode_idx: int) -> None:
        """Select a view by PreviewMode list index, syncing the listbox."""
        for lb_row, mi in enumerate(_LB_TO_MODE):
            if mi == mode_idx:
                self._view_var.set(lb_row)
                self._lb.select_clear(0, tk.END)
                self._lb.select_set(lb_row)
                self._lb.see(lb_row)
                break
        if self._result is not None:
            self._redraw()

    def _on_lb_select(self, _event) -> None:
        sel = self._lb.curselection()
        if not sel:
            return
        row = sel[0]
        if _LB_TO_MODE[row] is None:          # separator row — deselect
            self._lb.selection_clear(row, row)
            return
        self._view_var.set(row)
        if self._result is not None:
            self._redraw()

    def _toggle_brush(self) -> None:
        self._brush.toggle()
        self._brush_var.set(self._brush.enabled)   # keep Checkbutton in sync
        if self._brush.enabled:
            self._brush_cb.config(text="✏  Editing Mask  [E to exit]",
                                   fg=_ACCENT)
            self._set_view_by_mode(1)   # force Mask Overlay; calls _redraw
        else:
            self._brush_cb.config(text="✏  Edit Mask  [E]",
                                   fg=_FG)
            self._canvas.delete("cursor")
            if self._brush.has_edits():
                # Auto-switch to Repaired so the user sees the fix immediately.
                self._set_view_by_mode(3)  # calls _redraw
            else:
                self._redraw()

    def _clear_brush(self) -> None:
        self._brush.clear()
        if self._result is not None:
            self._redraw()

    def _toggle_lens(self) -> None:
        self._zoom_lens.toggle()
        self._lens_var.set(self._zoom_lens.enabled)
        self._redraw()

    def _reset_zoom(self) -> None:
        self._canvas_zoom.reset()
        if self._result is not None:
            self._redraw()

    def _adj_brush(self, delta: int) -> None:
        self._brush_r.set(max(1, min(self._brush.radius + delta, 100)))

    # =========================================================================
    # Save operations
    # =========================================================================

    def _save_view(self) -> None:
        """Save the current view — guaranteed clean (no UI text, no zoom-lens)."""
        if self._clean_view is None or self._bgr is None:
            return

        src  = self._paths[self._idx]
        base, ext = os.path.splitext(src)

        lb_row   = self._view_var.get()
        mode_idx = _LB_TO_MODE[lb_row] if lb_row < len(_LB_TO_MODE) else 1
        mode_idx = mode_idx if mode_idx is not None else 1

        if self._brush.enabled or self._brush.has_edits():
            suffix = "_mask_edited_overlay"
        else:
            suffix = (_SAVE_SUFFIXES[mode_idx]
                      if mode_idx < len(_SAVE_SUFFIXES) else "_view")

        out = f"{base}{suffix}{ext}"
        if imwrite_safe(out, self._clean_view):
            self._toast(f"Saved: {os.path.basename(out)}")
        else:
            messagebox.showerror("Error", f"Could not write:\n{out}")

        # Also dump the raw binary mask when brush edits are active.
        if self._brush.has_edits() and self._result is not None:
            mask     = self._brush.apply(self._result.final_mask)
            mask_out = f"{base}_mask_edited.png"
            if imwrite_safe(mask_out, mask):
                self._toast(f"Mask saved: {os.path.basename(mask_out)}")

    def _save_repaired(self) -> None:
        """Apply brush edits → Telea inpaint on full-res source → save."""
        if self._bgr is None or self._result is None:
            return

        effective = (self._brush.apply(self._result.final_mask)
                     if self._brush.has_edits()
                     else self._result.final_mask)

        if effective.any():
            repaired = TeleaInpainter.inpaint(
                self._bgr, effective, int(self._size_var.get()))
        else:
            repaired = self._bgr.copy()
            self._toast("Mask is empty — saving original unchanged.")

        src  = self._paths[self._idx]
        base, ext = os.path.splitext(src)
        out  = f"{base}_repaired{ext}"

        if imwrite_safe(out, repaired):
            self._toast(f"Repaired saved: {os.path.basename(out)}")
        else:
            messagebox.showerror("Error", f"Could not write:\n{out}")

    # -------------------------------------------------------------------------

    def _toast(self, msg: str) -> None:
        """Briefly display a status message inside the Actions section."""
        self._status_lbl.config(text=f"✓  {msg}", fg=_ACCENT)
        self._root.after(4000,
            lambda: self._status_lbl.config(text="", fg=_ACCENT))


# ---------------------------------------------------------------------------
# Path helpers — re-exported via gui/__init__.py
# ---------------------------------------------------------------------------

def collect_image_paths(target: str) -> List[str]:
    """Resolve a file / directory / glob to a sorted list of image paths."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        paths: List[str] = []
        for root, _, files in os.walk(target):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(root, f))
        paths.sort()
        return paths
    paths = sorted(glob.glob(target, recursive=True))
    return [p for p in paths if os.path.splitext(p)[1].lower() in exts]


def open_file_dialog() -> Optional[str]:
    """Open a native file-selection dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select an image or folder",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files",   "*.*"),
            ],
        )
        root.destroy()
        return path if path else None
    except ImportError:
        return input("Enter image or folder path: ").strip().strip('"')
