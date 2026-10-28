"""Spatial Measure desktop app for Windows, macOS and Linux (Tkinter, ships with Python).

    python app.py                       # live webcam (the camera saved in settings.json, default 0)
    python app.py --source photo.jpg    # a still image or a video file instead of the camera
    run_windows.bat                     # Windows: sets up .venv on first run, then starts this app

The same engine as the native macOS app (engine_server.Engine) runs in-process on a worker thread. The camera is
read on its own thread and shown at its full frame rate; the engine always takes the newest frame when it is free
(late frames are dropped, never queued), and its results are drawn over the video as a vector overlay.
Settings (marker size, calibration, mode, ...) are saved to settings.json and restored on the next start.

Click an object to outline and measure it; Esc clears. Ctrl+L (Cmd+L on a Mac) logs to measurements.csv,
Ctrl+P pauses.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import traceback

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, font as tkfont, ttk
from PIL import Image, ImageTk

from engine_server import Engine
from measure import open_camera
from spatial_measure import settings as saved_settings

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
GREEN, ORANGE, CYAN, YELLOW = "#30d158", "#ff9f0a", "#32ade6", "#ffd60a"


class FrameSource(threading.Thread):
    """Webcam, video file or still image. Keeps only the newest frame and hands each one to `on_frame`."""

    def __init__(self, source, width, height, on_frame):
        super().__init__(daemon=True)
        self.source, self.width, self.height, self.on_frame = source, width, height, on_frame
        self.frame, self.seq, self.fps, self.error = None, 0, 0.0, None
        self.running = True

    def run(self):
        src = self.source
        if isinstance(src, str) and src.lower().endswith(IMAGE_EXT):
            img = cv2.imread(src)
            if img is None:
                self.error = f"Could not load image {src}"
                return
            while self.running:  # a still image is fed like a 30 fps camera
                self._deliver(img)
                time.sleep(1 / 30)
            return
        is_file = isinstance(src, str)
        cap = cv2.VideoCapture(src) if is_file else open_camera(src, self.width, self.height)
        if not cap.isOpened():
            self.error = f"Could not open {'video ' + src if is_file else f'camera {src}'}"
            return
        delay = 1 / (cap.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0
        last = time.perf_counter()
        while self.running:
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
                    continue
                self.error = f"Camera {src} stopped sending frames"
                break
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-6)
            last = now
            self._deliver(frame)
            if delay:
                time.sleep(delay)
        cap.release()

    def _deliver(self, frame):
        self.frame, self.seq = frame, self.seq + 1
        self.on_frame(frame)

    def stop(self):
        self.running = False


class EngineWorker(threading.Thread):
    """Runs Engine on its own thread. Commands (config, tap, log) are applied in order before the next frame,
    and only the newest submitted frame is processed, so results never fall behind the video."""

    def __init__(self, out: queue.Queue):
        super().__init__(daemon=True)
        self.out, self.cmds = out, queue.Queue()
        self.cond = threading.Condition()
        self.frame, self.running = None, True

    def submit(self, frame):
        with self.cond:
            self.frame = frame
            self.cond.notify()

    def command(self, name: str, arg: dict):
        self.cmds.put((name, arg))
        with self.cond:
            self.cond.notify()

    def stop(self):
        with self.cond:
            self.running = False
            self.cond.notify()

    def run(self):
        engine = Engine(emit=self.out.put)
        self.out.put({"type": "ready"})
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.frame is not None or not self.cmds.empty() or not self.running)
                if not self.running:
                    return
                frame, self.frame = self.frame, None
            while not self.cmds.empty():
                name, arg = self.cmds.get_nowait()
                try:
                    getattr(engine, name)(arg)
                except Exception as e:
                    traceback.print_exc()
                    if name != "tap":
                        self.out.put({"type": "logged" if name == "log" else "config", "ok": False, "message": str(e)})
            if frame is not None:
                try:
                    engine.process(frame)
                except Exception as e:
                    traceback.print_exc()
                    h, w = frame.shape[:2]
                    self.out.put({"type": "result", "frame_size": [w, h], "marker_found": False, "error": str(e)})


class App:
    def __init__(self, root: tk.Tk, args):
        self.root, self.args = root, args
        self.s = saved_settings.load()
        self.result, self.status, self.calibrated = None, "Starting…", False
        self.paused, self.selecting = False, False
        self.engine_fps, self.results_counted, self.fps_t0 = 0.0, 0, time.perf_counter()
        self.shown_seq, self.shown_result, self.photo, self.map_photo = -1, None, None, None
        self.view = (1.0, 0.0, 0.0)  # scale, x offset, y offset: raw frame pixels -> canvas

        self.messages = queue.Queue()
        self.worker = EngineWorker(self.messages)
        self.worker.start()
        self.source = None

        self.tag_font = tkfont.nametofont("TkDefaultFont").copy()
        self.tag_font.configure(weight="bold")
        self.mono = tkfont.nametofont("TkFixedFont")
        self.build_ui()
        self.push_config()
        if self.s["ai"]:
            self.status = "Loading Smart outline (the first run downloads the model)…"
        self.start_source()
        self.root.after(15, self.tick)

    # ---------- layout ----------

    def build_ui(self):
        root = self.root
        root.title("Spatial Measure")
        root.minsize(1000, 620)
        root.geometry("1360x800")
        style = ttk.Style()
        if sys.platform == "win32" and "vista" in style.theme_names():
            style.theme_use("vista")

        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        bar = ttk.Frame(left, padding=(8, 6))
        bar.pack(side="top", fill="x")
        self.pause_btn = ttk.Button(bar, text="Pause", command=self.toggle_pause)
        self.pause_btn.pack(side="left")
        self.clear_btn = ttk.Button(bar, text="Clear (Esc)", command=self.clear_selection, state="disabled")
        self.clear_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Log measurement", command=self.log_measurement).pack(side="left")
        self.status_dot = tk.Canvas(bar, width=12, height=12, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(14, 6))
        self.status_var = tk.StringVar(value=self.status)
        ttk.Label(bar, textvariable=self.status_var).pack(side="left", fill="x", expand=True)
        self.fps_var = tk.StringVar()
        ttk.Label(bar, textvariable=self.fps_var, foreground="gray").pack(side="right")

        self.canvas = tk.Canvas(left, bg="black", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Configure>", lambda e: self.render(force=True))
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")

        side = self.scrollable_sidebar(main)
        self.build_sidebar(side)

        mod = "Command" if sys.platform == "darwin" else "Control"
        root.bind("<Escape>", lambda e: self.clear_selection())
        root.bind(f"<{mod}-l>", lambda e: self.log_measurement())
        root.bind(f"<{mod}-p>", lambda e: self.toggle_pause())
        root.protocol("WM_DELETE_WINDOW", self.shutdown)

    def scrollable_sidebar(self, parent) -> ttk.Frame:
        outer = ttk.Frame(parent, width=330)
        outer.pack(side="right", fill="y")
        outer.pack_propagate(False)
        cv = tk.Canvas(outer, highlightthickness=0, width=310)
        sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        inner = ttk.Frame(cv, padding=(10, 8))
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        win = cv.create_window(0, 0, window=inner, anchor="nw")
        cv.bind("<Configure>", lambda e: cv.itemconfigure(win, width=e.width))
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)

        def wheel(e):
            if e.num == 4 or e.delta > 0:
                step = -1
            elif e.num == 5 or e.delta < 0:
                step = 1
            else:
                return
            cv.yview_scroll(step * (1 if sys.platform == "darwin" else 3), "units")

        def bind_wheel(_):
            cv.bind_all("<MouseWheel>", wheel)
            cv.bind_all("<Button-4>", wheel)
            cv.bind_all("<Button-5>", wheel)

        def unbind_wheel(_):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                cv.unbind_all(seq)

        outer.bind("<Enter>", bind_wheel)
        outer.bind("<Leave>", unbind_wheel)
        return inner

    def section(self, parent, title) -> ttk.LabelFrame:
        f = ttk.LabelFrame(parent, text=title, padding=(8, 6))
        f.pack(fill="x", pady=(0, 10))
        f.columnconfigure(1, weight=1)
        return f

    def number_entry(self, parent, row, label, key, unit, scale=1.0):
        """A numeric field bound to self.s[key] (shown divided by `scale`); applied on Enter or leaving the field."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=f"{self.s[key] / scale:g}")
        e = ttk.Entry(parent, textvariable=var, width=8, justify="right")
        e.grid(row=row, column=1, sticky="e", pady=2)
        ttk.Label(parent, text=unit, foreground="gray").grid(row=row, column=2, sticky="w", padx=(4, 0))

        def apply(_=None):
            try:
                v = float(var.get().replace(",", "."))
                if v <= 0:
                    raise ValueError
            except ValueError:
                var.set(f"{self.s[key] / scale:g}")
                return
            if abs(v * scale - self.s[key]) > 1e-9:
                self.set_setting(key, v * scale)

        e.bind("<Return>", apply)
        e.bind("<FocusOut>", apply)
        return e

    def build_sidebar(self, side):
        f = self.section(side, "Objects")
        self.objects_var = tk.StringVar()
        ttk.Label(f, textvariable=self.objects_var, font=self.mono, justify="left", wraplength=280).grid(
            row=0, column=0, columnspan=3, sticky="w")
        self.map_label = ttk.Label(f)
        self.map_label.grid(row=1, column=0, columnspan=3, pady=(4, 0))

        f = self.section(side, "Camera")
        self.cam_vars = {}
        for i, (key, label) in enumerate([("pnp", "Distance (PnP)"), ("sim", "Distance (W·f/P)"), ("tilt", "Plane tilt")]):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w")
            self.cam_vars[key] = tk.StringVar(value="—")
            ttk.Label(f, textvariable=self.cam_vars[key]).grid(row=i, column=1, columnspan=2, sticky="e")
        self.calib_note = ttk.Label(f, text="Distances need a calibration file.", foreground="gray")
        self.calib_note.grid(row=3, column=0, columnspan=3, sticky="w")
        self.warn_var = tk.StringVar()
        ttk.Label(f, textvariable=self.warn_var, foreground="#c85a00", wraplength=280, justify="left").grid(
            row=4, column=0, columnspan=3, sticky="w")

        f = self.section(side, "Mode")
        self.mode_var = tk.StringVar(value=self.s["mode"])
        for i, (val, text) in enumerate([("table", "Table"), ("hand", "Hand + marker"), ("nomarker", "Hand, no marker")]):
            ttk.Radiobutton(f, text=text, value=val, variable=self.mode_var, command=self.on_mode).grid(
                row=0, column=i, sticky="w")
        self.mode_help = ttk.Label(f, foreground="gray", wraplength=280, justify="left")
        self.mode_help.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 2))
        self.nomarker_box = ttk.Frame(f)
        self.nomarker_box.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.nomarker_box.columnconfigure(1, weight=1)
        self.number_entry(self.nomarker_box, 0, "Distance to object", "distance_mm", "cm", scale=10.0)
        self.number_entry(self.nomarker_box, 1, "Camera field of view", "hfov_deg", "°")

        f = self.section(side, "Settings (saved)")
        ttk.Label(f, text="Camera").grid(row=0, column=0, sticky="w")
        self.cam_combo = ttk.Combobox(f, values=[f"Camera {i}" for i in range(6)], width=10, state="readonly")
        self.cam_combo.set(f"Camera {self.s['camera']}")
        self.cam_combo.grid(row=0, column=1, columnspan=2, sticky="e", pady=2)
        self.cam_combo.bind("<<ComboboxSelected>>", self.on_camera)
        if self.args.source is not None:
            self.cam_combo.configure(state="disabled")
        self.number_entry(f, 1, "Marker size", "marker_mm", "mm")
        ttk.Label(f, text="Calibration").grid(row=2, column=0, sticky="w")
        self.calib_var = tk.StringVar()
        ttk.Label(f, textvariable=self.calib_var, foreground="gray").grid(row=2, column=1, columnspan=2, sticky="e")
        btns = ttk.Frame(f)
        btns.grid(row=3, column=0, columnspan=3, sticky="e", pady=(2, 4))
        ttk.Button(btns, text="Choose…", command=self.choose_calibration).pack(side="left")
        ttk.Button(btns, text="None", command=lambda: self.set_setting("calibration", None)).pack(side="left", padx=(4, 0))
        self.flags = {}
        for i, (key, text) in enumerate([("scan", "Scan large objects (follow beyond the frame)"),
                                         ("ai", "Smart outline (AI, tracks only the object)"),
                                         ("smoothing", "Smooth readings"),
                                         ("use_clahe", "Shadow suppression (CLAHE)"),
                                         ("refine", "Sub-pixel edge refinement")]):
            var = tk.BooleanVar(value=bool(self.s[key]))
            cb = ttk.Checkbutton(f, text=text, variable=var, command=lambda k=key, v=var: self.set_setting(k, v.get()))
            cb.grid(row=4 + i, column=0, columnspan=3, sticky="w")
            self.flags[key] = (var, cb)
        ttk.Label(f, text=f"Saved in {os.path.basename(saved_settings.PATH)}", foreground="gray").grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(4, 0))

        f = self.section(side, "Benchmark log")
        self.bench = {}
        for i, (key, label, hint) in enumerate([("label", "Object label", "Credit card"),
                                                ("truth", "Caliper size", "85.60x53.98"),
                                                ("truth_distance", "True distance (mm)", "600")]):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w")
            self.bench[key] = tk.StringVar()
            ttk.Entry(f, textvariable=self.bench[key], width=14).grid(row=i, column=1, columnspan=2, sticky="e", pady=2)
            ttk.Label(f, text=f"e.g. {hint}", foreground="gray").grid(row=i, column=3, sticky="w", padx=(4, 0))
        ttk.Button(f, text="Log measurement", command=self.log_measurement).grid(row=3, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        self.log_var = tk.StringVar()
        ttk.Label(f, textvariable=self.log_var, foreground="gray", wraplength=280).grid(row=4, column=0, columnspan=4, sticky="w")

        self.refresh_settings_ui()

    def refresh_settings_ui(self):
        mode = self.s["mode"]
        self.mode_help.configure(text=(
            "No marker: hold the object at the distance set below (measure it with a ruler once) and facing the "
            "camera, then click it. Sizes are estimates — they scale directly with that distance."
            if mode == "nomarker" else
            "Hold the printed marker next to the object, at the same distance from the camera and facing it, then "
            "click the object. It is followed as you move it."
            if mode == "hand" else
            "Lay the marker flat beside the object, click the object, and slide the camera along it if it is "
            "bigger than the frame."))
        if mode == "nomarker":
            self.nomarker_box.grid()
        else:
            self.nomarker_box.grid_remove()
        self.flags["scan"][1].configure(state="disabled" if mode != "table" else "normal")
        self.calib_var.set(os.path.basename(self.s["calibration"]) if self.s["calibration"] else "None")
        if self.calibrated:
            self.calib_note.grid_remove()
        else:
            self.calib_note.grid()

    # ---------- settings & engine commands ----------

    def set_setting(self, key, value):
        self.s[key] = value
        try:
            saved_settings.save(self.s)
        except OSError as e:
            self.status = f"Could not save settings: {e}"
        if key != "camera":
            self.push_config()
        self.refresh_settings_ui()

    def push_config(self):
        cfg = {k: self.s[k] for k in ("marker_mm", "calibration", "use_clahe", "refine", "smoothing", "scan",
                                      "mode", "ai", "distance_mm", "hfov_deg")}
        self.worker.command("configure", cfg)

    def on_mode(self):
        self.set_setting("mode", self.mode_var.get())
        self.clear_selection()

    def on_camera(self, _=None):
        self.set_setting("camera", int(self.cam_combo.get().split()[-1]))
        self.clear_selection()
        self.start_source()

    def choose_calibration(self):
        path = filedialog.askopenfilename(title="Choose a calibration.npz produced by calibrate.py",
                                          initialdir=saved_settings.PROJECT_DIR,
                                          filetypes=[("Calibration", "*.npz"), ("All files", "*.*")])
        if path:
            self.set_setting("calibration", os.path.normpath(path))

    def start_source(self):
        if self.source is not None:
            self.source.stop()
        src = self.args.source if self.args.source is not None else int(self.s["camera"])
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.source = FrameSource(src, self.args.width, self.args.height, self.on_frame)
        self.source.start()

    def on_frame(self, frame):  # camera thread
        if not self.paused:
            self.worker.submit(frame)

    def on_click(self, e):
        fs = (self.result or {}).get("frame_size")
        if not fs:
            return
        s, ox, oy = self.view
        x, y = (e.x - ox) / s, (e.y - oy) / s
        if not (0 <= x < fs[0] and 0 <= y < fs[1]):
            self.clear_selection()
            return
        self.worker.command("tap", {"x": x, "y": y})
        self.selecting = True

    def clear_selection(self):
        self.worker.command("tap", {"clear": True})
        self.selecting = False

    def log_measurement(self):
        self.log_var.set("Logging…")
        self.worker.command("log", {k: v.get() for k, v in self.bench.items()})

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.configure(text="Resume" if self.paused else "Pause")

    def shutdown(self):
        if self.source is not None:
            self.source.stop()
        self.worker.stop()
        self.root.after(150, self.root.destroy)  # let the camera thread release the device

    # ---------- engine messages ----------

    def tick(self):
        while True:
            try:
                msg = self.messages.get_nowait()
            except queue.Empty:
                break
            self.handle(msg)
        if self.source is not None and self.source.error:
            self.status = self.source.error
        self.status_var.set(self.status)
        self.render()
        self.root.after(15, self.tick)

    def handle(self, msg):
        kind = msg.get("type")
        if kind == "ready":
            self.status = "Engine ready"
        elif kind == "config":
            if "calibrated" in msg:
                self.calibrated = bool(msg["calibrated"])
            if msg.get("ok") is False:
                self.status = msg.get("message") or "Config error"
                if msg.get("ai_failed"):
                    self.s["ai"] = False
                    self.flags["ai"][0].set(False)
                else:
                    self.s["calibration"] = None
                saved_settings.save(self.s)
            elif self.status.startswith("Loading Smart outline"):
                self.status = "Engine ready"
            self.refresh_settings_ui()
        elif kind == "logged":
            self.log_var.set(msg.get("message") or "")
        elif kind == "result":
            self.results_counted += 1  # counted, not timed: several results can be drained in one tick
            now = time.perf_counter()
            if now - self.fps_t0 >= 0.5:
                self.engine_fps = self.results_counted / (now - self.fps_t0)
                self.results_counted, self.fps_t0 = 0, now
            self.status = self.status_for(msg)
            self.selecting = bool(msg.get("selecting"))
            self.result = msg
            self.update_sidebar(msg)

    def status_for(self, msg) -> str:
        mode, in_hand = self.s["mode"], self.s["mode"] != "table"
        state, selecting = msg.get("selection_state"), msg.get("selecting")
        if msg.get("error"):
            return f"Engine error: {msg['error']}"
        if mode == "nomarker":
            d = round(self.s["distance_mm"] / 10)
            return {"ok": f"Estimating at {d} cm — hold the object at that distance",
                    "nothing": "Couldn't outline anything there — click the object itself",
                    "lost": "Lost the object — hold it in view, then click it again"}.get(
                state, f"Hold the object {d} cm from the camera and click it")
        if not msg.get("marker_found") and msg.get("plane_source") == "tracked":
            n = (msg.get("scan") or {}).get("frames_since_marker") or 0
            return f"Marker out of view — following the table ({n} frames). Keep moving along the object"
        if not msg.get("marker_found"):
            if not selecting:
                return "Looking for the ArUco marker…"
            return ("Keep the marker in view, held beside the object at the same distance" if in_hand
                    else "Tracking lost — move back to the marker or to a part you already scanned")
        if state == "no_marker":
            return "Show the ArUco marker first, then click the object"
        if state == "marker":
            return "That's the reference marker — click an object next to it"
        if state == "nothing":
            return "Couldn't outline anything there — click on the object itself"
        if state == "lost" and selecting:
            return ("Lost the object — hold it in view, then click it again" if in_hand
                    else "Lost the object — keep it still or click it again")
        if state == "ok":
            objs = msg.get("objects") or []
            if in_hand:
                return "Following the object in your hand"
            if objs and objs[0].get("open"):
                return "Object continues — keep moving the camera along it"
            return "Whole object scanned" if msg.get("scan") else "Measuring the clicked object"
        return "Tracking — click an object to measure it"

    def update_sidebar(self, r):
        objects, lines = r.get("objects") or [], []
        if self.s["mode"] == "nomarker" and not self.selecting:
            lines.append("Click the object you are holding")
        elif not r.get("marker_found") and not self.selecting:
            lines.append("Place the ArUco marker flat in view")
        elif not objects and not self.selecting:
            lines.append("No objects found next to the marker")
        elif not self.selecting:
            lines.append("Click an object in the video to\noutline and measure it")
        for o in objects:
            if o.get("area_mm2") is not None:
                sign = "≥" if o.get("open") else "≈"
                L, W, A = o["length_mm"], o["width_mm"], o["area_mm2"]
                lines += [f"Length {sign} {L:7.1f} mm  ({L / 10:.1f} cm)",
                          f"Width  {sign} {W:7.1f} mm  ({W / 10:.1f} cm)",
                          f"Area   {sign} {A:7.0f} mm² ({A / 100:.1f} cm²)"]
                if o.get("open"):
                    lines.append("Not fully seen yet — pan along the\nobject until the ends are in view")
            else:
                lines.append(f"#{o['id']:<3} {o['length_mm']:7.1f} × {o['width_mm']:.1f} mm")
        for p in r.get("pairs") or []:
            lines.append(f"Spacing {p['mm']:.0f} mm")
        self.objects_var.set("\n".join(lines))
        b64 = (r.get("scan") or {}).get("map") if objects else None
        if b64:
            if b64 != getattr(self, "_map_b64", None):
                self._map_b64, self.map_photo = b64, tk.PhotoImage(data=b64)
                self.map_label.configure(image=self.map_photo)
        elif self.map_photo is not None:
            self._map_b64, self.map_photo = None, None
            self.map_label.configure(image="")

        fmt = lambda v, u: "—" if v is None else (f"{v:.0f}°" if u == "°" else f"{v:.0f} {u}")
        self.cam_vars["pnp"].set(fmt(r.get("distance_pnp_mm"), "mm"))
        self.cam_vars["sim"].set(fmt(r.get("distance_similar_mm"), "mm"))
        self.cam_vars["tilt"].set(fmt(r.get("tilt_deg"), "°"))
        self.warn_var.set("\n".join(f"⚠ {w}" for w in r.get("warnings") or []))
        self.clear_btn.configure(state="normal" if self.selecting else "disabled")

    # ---------- drawing ----------

    def render(self, force=False):
        src = self.source
        frame = src.frame if src is not None else None
        cam_fps = f"Camera {src.fps:.0f} fps   " if src is not None and src.fps else ""
        ms = (self.result or {}).get("ms")
        self.fps_var.set(f"{cam_fps}Engine {self.engine_fps:.0f} fps" + (f"   {ms:.0f} ms" if ms else ""))
        self.status_dot.delete("all")
        self.status_dot.create_oval(2, 2, 11, 11, fill=GREEN if (self.result or {}).get("marker_found") else ORANGE, outline="")
        if frame is None or (not force and self.paused and self.photo is not None):
            return
        if not force and src.seq == self.shown_seq and self.result is self.shown_result:
            return
        self.shown_seq, self.shown_result = src.seq, self.result

        cw, ch = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        h, w = frame.shape[:2]
        s = min(cw / w, ch / h)
        dw, dh = max(int(w * s), 1), max(int(h * s), 1)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.view = (s, ox, oy)
        img = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

        r = self.result if self.result and self.result.get("frame_size") == [w, h] else None
        pt = lambda p: (ox + p[0] * s, oy + p[1] * s)
        if r:  # translucent fills go into the image itself; Tk canvas items cannot be transparent
            fills = [(o["outline"], 0.22) if o.get("outline") and len(o["outline"]) >= 3 else (o["box"], 0.12)
                     for o in r.get("objects") or []]
            for poly, alpha in fills:
                layer = img.copy()
                cv2.fillPoly(layer, [np.round(np.array(poly) * s).astype(np.int32)], (88, 209, 48))
                cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)

        rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.photo is None or self.photo.width() != dw or self.photo.height() != dh:
            self.photo = ImageTk.PhotoImage(rgb)
            self.canvas.itemconfigure(self.image_item, image=self.photo)
        else:
            self.photo.paste(rgb)
        self.canvas.coords(self.image_item, ox, oy)

        c = self.canvas
        c.delete("ov")
        if not r:
            return
        flat = lambda pts: [v for p in pts for v in pt(p)]
        if r.get("marker") and len(r["marker"]) == 4:
            c.create_polygon(flat(r["marker"]), outline=CYAN, fill="", width=2, tags="ov")
            top = min((pt(p) for p in r["marker"]), key=lambda q: q[1])
            self.tag(f"REF {r.get('marker_id') or 0}", top[0], top[1] - 14, CYAN)
        for p in r.get("pairs") or []:
            a, b = pt(p["a"]), pt(p["b"])
            c.create_line(*a, *b, fill=ORANGE, width=1.5, dash=(6, 4), tags="ov")
            self.tag(f"{p['mm']:.0f} mm", (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, ORANGE)
        for o in r.get("objects") or []:
            top = min((pt(p) for p in o["box"]), key=lambda q: q[1])
            cx = pt(o["center"])[0]
            if o.get("outline") and len(o["outline"]) >= 3:
                c.create_polygon(flat(o["box"]), outline=YELLOW, fill="", width=1, dash=(5, 4), tags="ov")
                c.create_polygon(flat(o["outline"]), outline=GREEN, fill="", width=3, joinstyle="round", tags="ov")
                open_ = bool(o.get("open"))
                text = (f"≥ {o['length_mm']:.0f} × {o['width_mm']:.0f} mm so far" if open_
                        else f"≈ {o['length_mm']:.1f} × {o['width_mm']:.1f} mm")
                self.tag(text, cx, top[1] - 16, ORANGE if open_ else "#1f9d45")
            else:
                c.create_polygon(flat(o["box"]), outline=GREEN, fill="", width=2, tags="ov")
                self.tag(f"#{o['id']}  {o['length_mm']:.1f} × {o['width_mm']:.1f} mm", cx, top[1] - 14, "#1f9d45")

    def tag(self, text, x, y, color):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        x, y = min(max(x, 90), cw - 90), min(max(y, 24), ch - 24)
        t = self.canvas.create_text(x, y, text=text, fill="white", font=self.tag_font, tags="ov")
        x0, y0, x1, y1 = self.canvas.bbox(t)
        bg = self.canvas.create_rectangle(x0 - 6, y0 - 2, x1 + 6, y1 + 2, fill=color, outline="", tags="ov")
        self.canvas.tag_lower(bg, t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="webcam index, video file or image file (default: the saved camera)")
    ap.add_argument("--width", type=int, default=1920, help="requested camera resolution (match your calibration)")
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()
    if args.source is not None and not args.source.isdigit():
        args.source = os.path.abspath(args.source)
    os.chdir(saved_settings.PROJECT_DIR)  # measurements.csv, snapshots/ and models/ live in the project folder
    if sys.platform == "win32":
        try:  # sharp text and video on scaled (125 %, 150 %) Windows displays
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
