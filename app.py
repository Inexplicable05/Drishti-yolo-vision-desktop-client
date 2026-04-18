import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ultralytics. Install requirements before running the app."
    ) from exc


import webbrowser

class DrishtiDesktopApp:
    BG_TOP = "#EEF3FF"
    BG_BOTTOM = "#DDE8FF"
    GLASS_PANEL = "#F8FBFF"
    GLASS_CARD = "#FFFFFF"
    GLASS_BORDER = "#CDDDF5"
    TEXT = "#1F2937"
    MUTED = "#64748B"
    ACCENT = "#2563EB"
    ACCENT_SOFT = "#DBEAFE"
    WARN = "#B45309"
    DANGER = "#B91C1C"
    CANVAS_BG = "#0F172A"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Drishti Studio")
        self.root.geometry("1320x780")
        self.root.minsize(1080, 680)
        self.root.configure(bg=self.BG_TOP)

        self.model = None
        self.model_path = ""
        self.capture = None
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.stream_generation = 0
        self.current_stream_kind = None

        self.last_rgb_frame = None
        self.canvas_image = None
        self.resize_after_id = None
        self.background_after_id = None

        self.fullscreen = False
        self.windowed_geometry = self.root.geometry()
        self.app_closing = False
        self.last_frame_timestamp = None

        self.confidence = tk.DoubleVar(value=0.25)
        self.confidence_text = tk.StringVar(value="Confidence threshold: 0.25")
        self.model_text = tk.StringVar(value="No model loaded")
        self.source_text = tk.StringVar(value="Source: none")
        self.metrics_text = tk.StringVar(value="Inference: -- ms    FPS: --")
        self.status_text = tk.StringVar(value="Ready")

        self._build_ui()
        self._bind_events()
        self._set_inference_controls(False)
        self._draw_background()
        self._draw_placeholder()
        self._pump_frames()

    def _build_ui(self) -> None:
        self.bg_canvas = tk.Canvas(self.root, highlightthickness=0, bd=0)
        self.bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.shell = tk.Frame(self.root, bg=self.BG_TOP)
        self.shell.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.shell.grid_rowconfigure(0, weight=1)
        self.shell.grid_columnconfigure(1, weight=1)

        self.sidebar = tk.Frame(
            self.shell,
            bg=self.GLASS_PANEL,
            width=320,
            highlightthickness=1,
            highlightbackground=self.GLASS_BORDER,
            bd=0,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(18, 10), pady=18)
        self.sidebar.grid_propagate(False)

        self.main = tk.Frame(self.shell, bg=self.BG_TOP)
        self.main.grid(row=0, column=1, sticky="nsew", padx=(8, 18), pady=18)
        self.main.grid_rowconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        self._build_topbar()
        self._build_canvas_shell()
        self._build_footer()
        self._build_sidebar()

    def _build_topbar(self) -> None:
        topbar = tk.Frame(self.main, bg=self.BG_TOP)
        topbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        topbar.grid_columnconfigure(0, weight=1)

        tk.Label(
            topbar,
            text="Drishti Studio",
            bg=self.BG_TOP,
            fg=self.TEXT,
            font=("Segoe UI", 19, "bold"),
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            topbar,
            text="Yolo vision interface",
            bg=self.BG_TOP,
            fg=self.MUTED,
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.status_label = tk.Label(
            topbar,
            textvariable=self.status_text,
            bg=self.ACCENT_SOFT,
            fg=self.ACCENT,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=6,
            bd=0,
            relief="flat",
        )
        self.status_label.grid(row=0, column=1, sticky="e")

        github_link = tk.Label(
            topbar,
            text="GitHub Profile",
            bg=self.BG_TOP,
            fg=self.ACCENT,
            font=("Segoe UI", 10, "underline"),
            cursor="hand2"
        )
        github_link.grid(row=1, column=1, sticky="e", pady=(5, 0))
        github_link.bind("<Button-1>", lambda e: webbrowser.open_new("https://github.com/your-username"))

    def _build_canvas_shell(self) -> None:
        canvas_shell = tk.Frame(
            self.main,
            bg=self.GLASS_CARD,
            highlightthickness=1,
            highlightbackground=self.GLASS_BORDER,
            bd=0,
        )
        canvas_shell.grid(row=1, column=0, sticky="nsew")
        canvas_shell.grid_rowconfigure(0, weight=1)
        canvas_shell.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_shell,
            bg=self.CANVAS_BG,
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

    def _build_footer(self) -> None:
        footer = tk.Frame(self.main, bg=self.BG_TOP)
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        footer.grid_columnconfigure(1, weight=1)

        tk.Label(
            footer,
            textvariable=self.source_text,
            bg=self.BG_TOP,
            fg=self.MUTED,
            font=("Segoe UI", 10),
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            footer,
            textvariable=self.metrics_text,
            bg=self.BG_TOP,
            fg=self.MUTED,
            font=("Segoe UI", 10),
        ).grid(row=0, column=1, sticky="e")

    def _build_sidebar(self) -> None:
        tk.Label(
            self.sidebar,
            text="DRISHTI",
            bg=self.GLASS_PANEL,
            fg=self.TEXT,
            font=("Segoe UI", 22, "bold"),
            anchor="w",
        ).pack(fill="x", padx=20, pady=(20, 2))

        tk.Label(
            self.sidebar,
            text="Desktop client",
            bg=self.GLASS_PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 10),
            anchor="w",
        ).pack(fill="x", padx=20, pady=(0, 14))

        model_card = self._card("Model")
        self.btn_load_model = self._make_button(
            model_card,
            text="Load Weight",
            command=self.load_model_dialog,
            bg=self.ACCENT_SOFT,
            fg=self.ACCENT,
        )
        self.btn_load_model.pack(fill="x", pady=(0, 10))

        tk.Label(
            model_card,
            textvariable=self.model_text,
            bg=self.GLASS_CARD,
            fg=self.MUTED,
            justify="left",
            wraplength=260,
            font=("Segoe UI", 9),
        ).pack(fill="x")

        detection_card = self._card("Detection")
        tk.Label(
            detection_card,
            textvariable=self.confidence_text,
            bg=self.GLASS_CARD,
            fg=self.MUTED,
            anchor="w",
            font=("Segoe UI", 10),
        ).pack(fill="x")

        self.conf_slider = tk.Scale(
            detection_card,
            from_=10,
            to=90,
            orient="horizontal",
            bg=self.GLASS_CARD,
            fg=self.TEXT,
            activebackground="#B8CCFA",
            troughcolor="#E2EAFB",
            highlightthickness=0,
            bd=0,
            command=self._on_confidence_change,
        )
        self.conf_slider.set(25)
        self.conf_slider.pack(fill="x", pady=(4, 0))

        source_card = self._card("Source")
        self.btn_open_image = self._make_button(
            source_card,
            text="Open Image",
            command=self.open_image,
            bg="#EFF4FF",
            fg=self.TEXT,
        )
        self.btn_open_image.pack(fill="x", pady=(0, 8))

        self.btn_open_video = self._make_button(
            source_card,
            text="Open Video",
            command=self.open_video,
            bg="#EFF4FF",
            fg=self.TEXT,
        )
        self.btn_open_video.pack(fill="x", pady=(0, 8))

        self.btn_open_webcam = self._make_button(
            source_card,
            text="Open Webcam",
            command=self.open_webcam,
            bg="#EFF4FF",
            fg=self.TEXT,
        )
        self.btn_open_webcam.pack(fill="x", pady=(0, 8))

        self.btn_stop = self._make_button(
            source_card,
            text="Stop Stream",
            command=self.stop_stream,
            bg="#FDECEC",
            fg=self.DANGER,
        )
        self.btn_stop.pack(fill="x")

        view_card = self._card("View")
        self.btn_fullscreen = self._make_button(
            view_card,
            text="Toggle Fullscreen (F11)",
            command=self.toggle_fullscreen,
            bg="#EFF4FF",
            fg=self.TEXT,
        )
        self.btn_fullscreen.pack(fill="x")

        tk.Label(
            self.sidebar,
            text="Esc exits fullscreen",
            bg=self.GLASS_PANEL,
            fg=self.MUTED,
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(fill="x", padx=20, pady=(10, 0))

    def _card(self, title: str) -> tk.Frame:
        card = tk.Frame(
            self.sidebar,
            bg=self.GLASS_CARD,
            highlightthickness=1,
            highlightbackground=self.GLASS_BORDER,
            bd=0,
        )
        card.pack(fill="x", padx=20, pady=(0, 12))

        tk.Label(
            card,
            text=title,
            bg=self.GLASS_CARD,
            fg=self.TEXT,
            anchor="w",
            font=("Segoe UI", 11, "bold"),
        ).pack(fill="x", padx=12, pady=(10, 8))

        body = tk.Frame(card, bg=self.GLASS_CARD)
        body.pack(fill="x", padx=12, pady=(0, 12))
        return body

    def _make_button(self, parent: tk.Widget, text: str, command, bg: str, fg: str) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bd=0,
            relief="flat",
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=9,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=self.GLASS_BORDER,
            highlightcolor=self.GLASS_BORDER,
        )

    def _bind_events(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)
        self.root.bind("<Configure>", self.on_root_resize)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def on_root_resize(self, event) -> None:
        if event.widget is not self.root:
            return
        if self.background_after_id is not None:
            self.root.after_cancel(self.background_after_id)
        self.background_after_id = self.root.after(50, self._draw_background)

    def _mix_hex(self, color1: str, color2: str, ratio: float) -> str:
        ratio = max(0.0, min(1.0, ratio))
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _draw_background(self) -> None:
        self.background_after_id = None
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        if width < 2 or height < 2:
            return

        self.bg_canvas.delete("all")
        steps = 60
        for i in range(steps):
            ratio = i / float(max(steps - 1, 1))
            color = self._mix_hex(self.BG_TOP, self.BG_BOTTOM, ratio)
            y0 = int(height * i / steps)
            y1 = int(height * (i + 1) / steps) + 1
            self.bg_canvas.create_rectangle(0, y0, width, y1, fill=color, outline=color)

        self.bg_canvas.create_oval(
            int(width * -0.10),
            int(height * -0.25),
            int(width * 0.35),
            int(height * 0.35),
            fill="#D8E8FF",
            outline="",
        )
        self.bg_canvas.create_oval(
            int(width * 0.65),
            int(height * -0.18),
            int(width * 1.08),
            int(height * 0.45),
            fill="#E8F0FF",
            outline="",
        )
        self.bg_canvas.create_oval(
            int(width * 0.78),
            int(height * 0.72),
            int(width * 1.18),
            int(height * 1.20),
            fill="#DCEBFF",
            outline="",
        )

        self.bg_canvas.lower(self.shell)

    def _on_confidence_change(self, slider_value: str) -> None:
        value = float(slider_value) / 100.0
        self.confidence.set(value)
        self.confidence_text.set(f"Confidence threshold: {value:.2f}")

    def _set_status(self, text: str, color: str = None) -> None:
        self.status_text.set(text)
        fg = color or self.ACCENT
        bg = self.ACCENT_SOFT if fg == self.ACCENT else "#FFF4D6" if fg == self.WARN else "#FDECEC"
        self.status_label.configure(fg=fg, bg=bg)

    def _set_inference_controls(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_open_image.configure(state=state)
        self.btn_open_video.configure(state=state)
        self.btn_open_webcam.configure(state=state)
        if not enabled:
            self.btn_stop.configure(state=tk.DISABLED)

    def load_model_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Select model weight",
            filetypes=[
                ("YOLO weights", "*.pt *.onnx *.engine *.tflite *.pb *.xml"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.stop_stream(update_status=False)
        self._set_status("Loading model...", self.WARN)
        self.model_text.set(f"Loading: {os.path.basename(path)}")
        self.btn_load_model.configure(state=tk.DISABLED, text="Loading...")

        threading.Thread(target=self._load_model_worker, args=(path,), daemon=True).start()

    def _load_model_worker(self, path: str) -> None:
        try:
            model = YOLO(path)
        except Exception as exc:
            self.root.after(0, lambda: self._on_model_load_error(str(exc)))
            return
        self.root.after(0, lambda: self._on_model_loaded(path, model))

    def _on_model_loaded(self, path: str, model) -> None:
        self.model = model
        self.model_path = path
        self.model_text.set(f"Loaded: {os.path.basename(path)}")
        self.btn_load_model.configure(state=tk.NORMAL, text="Load Weight")
        self._set_inference_controls(True)
        self._set_status("Model ready", self.ACCENT)

    def _on_model_load_error(self, error: str) -> None:
        self.model = None
        self.model_path = ""
        self.model_text.set("Model load failed")
        self.btn_load_model.configure(state=tk.NORMAL, text="Load Weight")
        self._set_inference_controls(False)
        self._set_status("Model load failed", self.DANGER)
        messagebox.showerror("Model Load Error", error)

    def _run_inference(self, bgr_frame):
        start = time.perf_counter()
        results = self.model.predict(
            source=bgr_frame,
            conf=float(self.confidence.get()),
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return results[0].plot(), elapsed_ms

    def open_image(self) -> None:
        if not self.model:
            messagebox.showinfo("Model Required", "Load a weight file before opening a source.")
            return

        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return

        self.stop_stream(update_status=False)
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Image Error", "Unable to open selected image.")
            return

        self.source_text.set(f"Source: image - {os.path.basename(path)}")
        self._set_status("Running image inference...", self.WARN)
        self.root.config(cursor="watch")
        self.root.update_idletasks()

        try:
            rendered_frame, infer_ms = self._run_inference(frame)
        except Exception as exc:
            self.root.config(cursor="")
            self._set_status("Inference error", self.DANGER)
            messagebox.showerror("Inference Error", str(exc))
            return

        self.root.config(cursor="")
        self._draw_bgr_frame(rendered_frame)
        self.metrics_text.set(f"Inference: {infer_ms:.1f} ms    FPS: --")
        self._set_status("Image rendered", self.ACCENT)

    def _open_capture(self, source, prefer_directshow: bool = False):
        captures = []
        if prefer_directshow and os.name == "nt":
            captures.append(cv2.VideoCapture(source, cv2.CAP_DSHOW))
        captures.append(cv2.VideoCapture(source))

        for cap in captures:
            if cap.isOpened():
                return cap
            cap.release()
        return None

    def _start_stream(self, capture, source_text: str, stream_kind: str) -> None:
        self.stop_stream(update_status=False)

        self.capture = capture
        self.current_stream_kind = stream_kind
        self.stop_event.clear()
        self.stream_generation += 1
        generation = self.stream_generation
        self.last_frame_timestamp = None

        self.source_text.set(source_text)
        self._set_status("Streaming...", self.ACCENT)
        self.btn_stop.configure(state=tk.NORMAL)

        self.worker_thread = threading.Thread(
            target=self._stream_worker,
            args=(generation,),
            daemon=True,
        )
        self.worker_thread.start()

    def open_video(self) -> None:
        if not self.model:
            messagebox.showinfo("Model Required", "Load a weight file before opening a source.")
            return

        path = filedialog.askopenfilename(
            title="Open video",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.webm"), ("All files", "*.*")],
        )
        if not path:
            return

        capture = self._open_capture(path)
        if capture is None:
            messagebox.showerror("Video Error", "Unable to open selected video.")
            return

        self._start_stream(
            capture,
            source_text=f"Source: video - {os.path.basename(path)}",
            stream_kind="video",
        )

    def open_webcam(self) -> None:
        if not self.model:
            messagebox.showinfo("Model Required", "Load a weight file before opening a source.")
            return

        selected_capture = None
        selected_index = None
        for index in range(4):
            capture = self._open_capture(index, prefer_directshow=True)
            if capture is None:
                continue
            ok, frame = capture.read()
            if ok and frame is not None:
                selected_capture = capture
                selected_index = index
                break
            capture.release()

        if selected_capture is None:
            messagebox.showerror("Webcam Error", "No webcam was detected.")
            return

        self._start_stream(
            selected_capture,
            source_text=f"Source: webcam - camera {selected_index}",
            stream_kind="webcam",
        )

    def _stream_worker(self, generation: int) -> None:
        while (
            not self.stop_event.is_set()
            and generation == self.stream_generation
            and self.capture is not None
            and self.capture.isOpened()
        ):
            ok, frame = self.capture.read()
            if not ok:
                break

            try:
                rendered_frame, infer_ms = self._run_inference(frame)
            except Exception as exc:
                self.root.after(0, lambda: self._on_stream_error(str(exc)))
                break

            timestamp = time.perf_counter()

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait((rendered_frame, infer_ms, timestamp, generation))
            except queue.Full:
                pass

        self.root.after(0, lambda: self._on_stream_finished(generation))

    def _on_stream_error(self, error: str) -> None:
        self._set_status("Stream inference failed", self.DANGER)
        messagebox.showerror("Inference Error", error)

    def _on_stream_finished(self, generation: int) -> None:
        if generation != self.stream_generation:
            return

        self.btn_stop.configure(state=tk.DISABLED)
        if not self.stop_event.is_set():
            if self.current_stream_kind == "webcam":
                self._set_status("Webcam disconnected", self.WARN)
            else:
                self._set_status("Video finished", self.ACCENT)
        self._release_capture()

    def _pump_frames(self) -> None:
        if self.app_closing:
            return

        latest = None
        while True:
            try:
                latest = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            rendered_frame, infer_ms, timestamp, generation = latest
            if generation == self.stream_generation:
                self._draw_bgr_frame(rendered_frame)
                fps_text = "--"
                if self.last_frame_timestamp is not None and timestamp > self.last_frame_timestamp:
                    fps_text = f"{1.0 / (timestamp - self.last_frame_timestamp):.1f}"
                self.last_frame_timestamp = timestamp
                self.metrics_text.set(f"Inference: {infer_ms:.1f} ms    FPS: {fps_text}")

        self.root.after(15, self._pump_frames)

    def _draw_bgr_frame(self, bgr_frame) -> None:
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self.last_rgb_frame = rgb_frame
        self._render_rgb_to_canvas(rgb_frame)

    def _render_rgb_to_canvas(self, rgb_frame) -> None:
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            return

        frame_h, frame_w = rgb_frame.shape[:2]
        scale = min(canvas_w / frame_w, canvas_h / frame_h)
        draw_w = max(1, int(frame_w * scale))
        draw_h = max(1, int(frame_h * scale))

        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(rgb_frame, (draw_w, draw_h), interpolation=interpolation)

        image = Image.fromarray(resized)
        self.canvas_image = ImageTk.PhotoImage(image=image)

        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, canvas_w, canvas_h, fill=self.CANVAS_BG, outline="")

        x = (canvas_w - draw_w) // 2
        y = (canvas_h - draw_h) // 2
        self.canvas.create_image(x, y, image=self.canvas_image, anchor="nw")
        self.canvas.create_rectangle(x, y, x + draw_w, y + draw_h, outline="#334155")

    def _draw_placeholder(self) -> None:
        canvas_w = max(self.canvas.winfo_width(), 200)
        canvas_h = max(self.canvas.winfo_height(), 200)
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, canvas_w, canvas_h, fill=self.CANVAS_BG, outline="")
        self.canvas.create_text(
            canvas_w // 2,
            canvas_h // 2 - 14,
            text="Load a weight, then open image, video, or webcam",
            fill="#E2E8F0",
            font=("Segoe UI", 13, "bold"),
        )
        self.canvas.create_text(
            canvas_w // 2,
            canvas_h // 2 + 12,
            text="F11 fullscreen  |  Esc restore",
            fill="#94A3B8",
            font=("Segoe UI", 10),
        )

    def on_canvas_resize(self, _event) -> None:
        if self.resize_after_id is not None:
            self.root.after_cancel(self.resize_after_id)
        self.resize_after_id = self.root.after(60, self._redraw_latest_frame)

    def _redraw_latest_frame(self) -> None:
        self.resize_after_id = None
        if self.last_rgb_frame is None:
            self._draw_placeholder()
            return
        self._render_rgb_to_canvas(self.last_rgb_frame)

    def toggle_fullscreen(self, _event=None):
        if not self.fullscreen:
            self.windowed_geometry = self.root.geometry()

        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

        if not self.fullscreen:
            self.root.geometry(self.windowed_geometry)

        self.root.after(140, self._redraw_latest_frame)
        return "break"

    def exit_fullscreen(self, _event=None):
        if self.fullscreen:
            self.toggle_fullscreen()
        return "break"

    def _release_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.current_stream_kind = None

    def stop_stream(self, update_status: bool = True) -> None:
        self.stop_event.set()
        self.stream_generation += 1
        self.btn_stop.configure(state=tk.DISABLED)
        self._release_capture()
        self.last_frame_timestamp = None

        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if update_status:
            self._set_status("Stream stopped", self.WARN)

    def on_close(self) -> None:
        self.app_closing = True
        self.stop_stream(update_status=False)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    DrishtiDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
