"""Desktop control panel built with the standard library's ``tkinter`` (no extra dependencies).

Launch it with::

    open-ai-hci gui                      # or:  python examples/06_desktop_gui.py
    open-ai-hci gui --config my.yaml     # edit / save a different profile
    open-ai-hci gui --autostart          # start the engine straight away

Tabs:

* **Engine Controls**: Start / Stop, Pause / Resume and a live ``Running`` / ``Paused`` / ``Stopped`` status.
* **Core Vision & Gestures**: hand cursor, pinch left-click, middle-pinch right-click, open-palm pause toggle,
  diagnostic HUD, cursor smoothing and pinch-threshold sliders (applied live).
* **Custom ONNX AI Model**: enable a trained model, pick the ``.onnx`` file, confidence slider, and one-click
  *Record Gesture* / *Train Model* launchers.
* **LLM Integration**: provider dropdown (Ollama, LM Studio, custom OpenAI-compatible, OpenAI, Anthropic, Gemini),
  base URL, model, masked API key, **Test Connection** badge and a box to try a command.

**Save Profile** writes everything to ``config.yaml`` (``open-ai-hci --config config.yaml`` runs with it). The API key is
only written if you tick *remember key*; otherwise use the ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / ``GEMINI_API_KEY``
environment variables.

All behaviour lives in :mod:`open_ai_hci.ui.gui_model` (tested without a display); this file is the thin tkinter skin.
"""
from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .gui_model import FLAGS, PROVIDERS, SLIDERS, ControlPanelModel

__all__ = ["ControlPanel", "main", "tkinter_available"]

STATUS_COLORS = {"Running": "#2e9e44", "Paused": "#d9a300", "Stopped": "#8a8a8a", "Starting": "#3b82f6"}
BADGE_COLORS = {"ok": "#2e9e44", "fail": "#c62828", "idle": "#8a8a8a", "busy": "#3b82f6"}


def tkinter_available() -> bool:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        return False
    return True


class ControlPanel:
    """The tabbed window. Create it with a ``tk.Tk()`` root and a :class:`ControlPanelModel`.

    Args:
        root: The ``tkinter.Tk`` window.
        model: Settings + engine controller (default: loads ``config.yaml`` if present).
        poll_ms: How often the status line refreshes.
    """

    def __init__(self, root: Any, model: ControlPanelModel | None = None, *, poll_ms: int = 250) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root = root
        self.model = model or ControlPanelModel()
        self.poll_ms = poll_ms
        self._busy_setting = False  # True while we write model values back into widgets
        self._test_result: Any = None
        self._test_running = False
        self._after_id: str | None = None
        self.flag_vars: dict[str, Any] = {}
        self.slider_vars: dict[str, Any] = {}
        self.slider_labels: dict[str, Any] = {}

        root.title("open-ai-hci Control Panel")
        root.minsize(560, 520)
        root.protocol("WM_DELETE_WINDOW", self.close)
        if sys.platform.startswith("linux"):  # the default Linux theme is hard to read; Windows / macOS keep the native one
            try:
                ttk.Style().theme_use("clam")
            except tk.TclError:  # pragma: no cover - theme missing
                pass

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self.tabs = {}
        for key, title, builder in (
            ("engine", "Engine Controls", self._build_engine_tab),
            ("vision", "Core Vision & Gestures", self._build_vision_tab),
            ("onnx", "Custom ONNX AI Model", self._build_onnx_tab),
            ("llm", "LLM Integration", self._build_llm_tab),
        ):
            frame = ttk.Frame(self.notebook, padding=14)
            self.notebook.add(frame, text=title)
            self.tabs[key] = frame
            builder(frame)
        self._build_footer(outer)
        self.refresh_from_model()
        self._tick()

    # ================================================================== tab: engine
    def _build_engine_tab(self, parent: Any) -> None:
        tk, ttk = self.tk, self.ttk
        box = ttk.Frame(parent)
        box.pack(fill="x")
        self.status_dot = tk.Canvas(box, width=26, height=26, highlightthickness=0)
        self.status_dot.grid(row=0, column=0, padx=(0, 10))
        self._dot = self.status_dot.create_oval(3, 3, 23, 23, fill=STATUS_COLORS["Stopped"], outline="")
        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(box, textvariable=self.status_var, font=("TkDefaultFont", 20, "bold")).grid(row=0, column=1, sticky="w")
        self.fps_var = tk.StringVar(value="")
        ttk.Label(box, textvariable=self.fps_var).grid(row=0, column=2, sticky="e", padx=(20, 0))
        box.columnconfigure(2, weight=1)

        buttons = ttk.Frame(parent)
        buttons.pack(fill="x", pady=(18, 6))
        self.start_btn = ttk.Button(buttons, text="Start", command=self.on_start)
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.on_stop)
        self.pause_btn = ttk.Button(buttons, text="Pause / Resume", command=self.on_pause)
        for b in (self.start_btn, self.stop_btn, self.pause_btn):
            b.pack(side="left", padx=(0, 8))

        cam = ttk.Frame(parent)
        cam.pack(fill="x", pady=(14, 0))
        ttk.Label(cam, text="Camera (index, video file or URL):").pack(side="left")
        self.camera_var = tk.StringVar()
        entry = ttk.Entry(cam, textvariable=self.camera_var, width=22)
        entry.pack(side="left", padx=8)
        self.camera_var.trace_add("write", lambda *_: None if self._busy_setting else self.model.set_camera(self.camera_var.get()))

        self.error_var = tk.StringVar()
        ttk.Label(parent, textvariable=self.error_var, foreground="#c62828", wraplength=500, justify="left").pack(anchor="w", pady=(14, 0))
        self.restart_var = tk.StringVar()
        ttk.Label(parent, textvariable=self.restart_var, foreground="#b26a00", wraplength=500, justify="left").pack(anchor="w", pady=(6, 0))
        ttk.Label(
            parent,
            text="Left click: pinch thumb + index.  Right click: thumb + middle finger.\nHold an open palm for a second to pause or resume.",
            foreground="#666666",
        ).pack(anchor="w", pady=(18, 0))

    # ============================================================ tab: core vision
    def _build_vision_tab(self, parent: Any) -> None:
        ttk = self.ttk
        for name in ("hand_cursor", "left_click", "right_pinch", "palm_pause", "hud"):
            self._add_flag(parent, name)
        if not self.model.hud_supported:
            self.hud_check.state(["disabled"])
            ttk.Label(parent, text="The HUD window cannot run next to this panel on macOS.", foreground="#666666").pack(anchor="w")
        ttk.Separator(parent).pack(fill="x", pady=12)
        self._add_slider(parent, "smoothing")
        self._add_slider(parent, "pinch")
        ttk.Label(parent, text="Slider changes apply instantly to a running engine.", foreground="#666666").pack(anchor="w", pady=(8, 0))

    def _add_flag(self, parent: Any, name: str) -> Any:
        tk, ttk = self.tk, self.ttk
        var = tk.BooleanVar()
        self.flag_vars[name] = var
        check = ttk.Checkbutton(parent, text=FLAGS[name], variable=var, command=lambda n=name: self._on_flag(n))
        check.pack(anchor="w", pady=3)
        if name == "hud":
            self.hud_check = check
        return check

    def _add_slider(self, parent: Any, name: str) -> None:
        tk, ttk = self.tk, self.ttk
        label, lo, hi, _ = SLIDERS[name]
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text=label).pack(side="left")
        value_label = ttk.Label(row, width=7, anchor="e")
        value_label.pack(side="right")
        self.slider_labels[name] = value_label
        var = tk.DoubleVar(value=lo)
        self.slider_vars[name] = var
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, command=lambda v, n=name: self._on_slider(n, float(v)))
        scale.pack(fill="x", pady=(2, 4))

    def _on_flag(self, name: str) -> None:
        if self._busy_setting:
            return
        problem = self.model.set_flag(name, self.flag_vars[name].get())
        self._say(problem)

    def _on_slider(self, name: str, value: float) -> None:
        shown = f"{value:.2f}" if SLIDERS[name][3] < 1 else f"{value:.0f}"
        self.slider_labels[name].configure(text=shown)
        if self._busy_setting:
            return
        self._say(self.model.set_value(name, value))

    # =========================================================== tab: ONNX model
    def _build_onnx_tab(self, parent: Any) -> None:
        tk, ttk = self.tk, self.ttk
        self._add_flag(parent, "onnx_enabled")
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(10, 0))
        ttk.Label(row, text="Model file (.onnx):").pack(side="left")
        self.onnx_path_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.onnx_path_var).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(row, text="Browse...", command=self.on_browse_onnx).pack(side="left")
        self.onnx_path_var.trace_add("write", lambda *_: None if self._busy_setting else self._say(self.model.set_text("onnx_path", self.onnx_path_var.get())))
        self._add_slider(parent, "onnx_confidence")
        ttk.Separator(parent).pack(fill="x", pady=14)
        ttk.Label(parent, text="Train your own gestures (e.g. a peace sign = right click):").pack(anchor="w")
        buttons = ttk.Frame(parent)
        buttons.pack(fill="x", pady=8)
        ttk.Button(buttons, text="Record Gesture...", command=self.on_record).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Train Model...", command=self.on_train).pack(side="left")
        ttk.Label(
            parent,
            text="1. Record 100-300 samples per gesture, plus a 'none' class of relaxed poses.\n2. Train: a model + labels file are written.\n"
            "3. It is selected here automatically. A label called right_click triggers a right click.",
            foreground="#666666", justify="left",
        ).pack(anchor="w", pady=(8, 0))

    def on_browse_onnx(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(title="Choose a gesture model", filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")])
        if path:
            self.onnx_path_var.set(path)

    def on_record(self) -> None:
        reason = self.model.camera_busy_reason()
        if reason:
            self._say(reason)
            return
        _SubprocessDialog(self, "Record Gesture", fields=[("Label", "right_click"), ("Samples", "150"), ("Save to", "data/gestures.csv")],
                          build=lambda v: self.model.record_command(v[0], int(v[1] or 0), v[2]), run_label="Start recording")

    def on_train(self) -> None:
        _SubprocessDialog(self, "Train Model", fields=[("Recording", "data/gestures.csv"), ("Model output", "models/gestures.onnx")],
                          build=lambda v: self.model.train_command(v[0], v[1]), run_label="Train",
                          on_success=self._adopt_model)

    def _adopt_model(self, values: list[str]) -> None:
        self.model.adopt_trained_model(values[1])
        self.refresh_from_model()

    # ============================================================== tab: LLM
    def _build_llm_tab(self, parent: Any) -> None:
        tk, ttk = self.tk, self.ttk
        self._add_flag(parent, "llm_enabled")
        grid = ttk.Frame(parent)
        grid.pack(fill="x", pady=(10, 0))
        grid.columnconfigure(1, weight=1)

        ttk.Label(grid, text="Provider").grid(row=0, column=0, sticky="w", pady=4)
        self.provider_var = tk.StringVar()
        self.provider_box = ttk.Combobox(grid, textvariable=self.provider_var, values=[p.label for p in PROVIDERS], state="readonly")
        self.provider_box.grid(row=0, column=1, sticky="ew", padx=8)
        self.provider_box.bind("<<ComboboxSelected>>", lambda _e: self.on_provider())

        ttk.Label(grid, text="Base URL").grid(row=1, column=0, sticky="w", pady=4)
        self.url_var = tk.StringVar()
        ttk.Entry(grid, textvariable=self.url_var).grid(row=1, column=1, sticky="ew", padx=8)
        self.url_var.trace_add("write", lambda *_: None if self._busy_setting else self._say(self.model.set_text("base_url", self.url_var.get())))

        ttk.Label(grid, text="Model name").grid(row=2, column=0, sticky="w", pady=4)
        self.model_var = tk.StringVar()
        ttk.Entry(grid, textvariable=self.model_var).grid(row=2, column=1, sticky="ew", padx=8)
        self.model_var.trace_add("write", lambda *_: None if self._busy_setting else self._say(self.model.set_text("model", self.model_var.get())))

        ttk.Label(grid, text="API key").grid(row=3, column=0, sticky="w", pady=4)
        self.key_var = tk.StringVar()
        self.key_entry = ttk.Entry(grid, textvariable=self.key_var, show="•")
        self.key_entry.grid(row=3, column=1, sticky="ew", padx=8)
        self.key_var.trace_add("write", lambda *_: None if self._busy_setting else self._say(self.model.set_text("api_key", self.key_var.get())))
        ttk.Label(
            grid, text="Optional for local servers. Or set OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY.",
            foreground="#666666", wraplength=400, justify="left",
        ).grid(row=4, column=1, sticky="w", padx=8)
        self.save_key_var = tk.BooleanVar()
        ttk.Checkbutton(grid, text="Remember the key in config.yaml (stored as plain text)", variable=self.save_key_var,
                        command=lambda: setattr(self.model, "save_api_key", self.save_key_var.get())).grid(row=5, column=1, sticky="w", padx=8, pady=(4, 0))

        test = ttk.Frame(parent)
        test.pack(fill="x", pady=(14, 0))
        self.test_btn = ttk.Button(test, text="Test Connection", command=self.on_test_connection)
        self.test_btn.pack(side="left")
        self.badge = tk.Label(test, text=" not tested ", bg=BADGE_COLORS["idle"], fg="white", padx=8, pady=2)
        self.badge.pack(side="left", padx=10)
        self.badge_msg = tk.StringVar()
        ttk.Label(parent, textvariable=self.badge_msg, wraplength=500, justify="left", foreground="#444444").pack(anchor="w", pady=(6, 0))

        ttk.Separator(parent).pack(fill="x", pady=12)
        ttk.Label(parent, text="Try a command (engine must be running):").pack(anchor="w")
        cmd = ttk.Frame(parent)
        cmd.pack(fill="x", pady=4)
        self.command_var = tk.StringVar()
        entry = ttk.Entry(cmd, textvariable=self.command_var)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _e: self.on_send())
        ttk.Button(cmd, text="Send", command=self.on_send).pack(side="left", padx=(8, 0))

    def on_provider(self) -> None:
        url, model = self.model.set_provider(self.provider_var.get())
        self._busy_setting = True
        try:
            self.url_var.set(url)
            self.model_var.set(model)
        finally:
            self._busy_setting = False
        self._set_badge("idle", "not tested", "")

    def on_test_connection(self) -> None:
        if self._test_running:
            return
        self._test_running = True
        self._set_badge("busy", "testing...", "")
        self.test_btn.state(["disabled"])

        def work() -> None:
            self._test_result = self.model.test_connection()

        threading.Thread(target=work, name="open-ai-hci-gui-test", daemon=True).start()

    def on_send(self) -> None:
        problem = self.model.send_command(self.command_var.get())
        self._say(problem or "Sent. Watch the HUD / console for the result.")
        if not problem:
            self.command_var.set("")

    def _set_badge(self, kind: str, text: str, message: str) -> None:
        self.badge.configure(text=f" {text} ", bg=BADGE_COLORS[kind])
        self.badge_msg.set(message)

    # ================================================================== footer
    def _build_footer(self, parent: Any) -> None:
        tk, ttk = self.tk, self.ttk
        foot = ttk.Frame(parent)
        foot.pack(fill="x", pady=(10, 0))
        self.save_btn = ttk.Button(foot, text="Save Profile", command=self.on_save)
        self.save_btn.pack(side="left")
        self.message_var = tk.StringVar(value=self.model.load_error)
        ttk.Label(foot, textvariable=self.message_var, wraplength=420, justify="left").pack(side="left", padx=12)

    def on_save(self) -> None:
        try:
            path = self.model.save_profile()
        except (OSError, ValueError) as exc:
            self._say(f"Could not save: {exc}")
            return
        note = " (API key stored in plain text)" if self.model.save_api_key and self.model.api_key else ""
        self._say(f"Saved to {path.resolve()}{note}")

    def _say(self, text: str) -> None:
        self.message_var.set(text)

    # ================================================================ engine
    def on_start(self) -> None:
        self._say(self.model.start())

    def on_stop(self) -> None:
        self.model.stop()

    def on_pause(self) -> None:
        self.model.runner.toggle_pause()

    # ================================================================ syncing
    def refresh_from_model(self) -> None:
        """Copy every setting from the model into the widgets (used at start-up and after loading / training)."""
        m = self.model
        self._busy_setting = True
        try:
            for name, var in self.flag_vars.items():
                var.set(m.get_flag(name))
            for name, var in self.slider_vars.items():
                var.set(m.get_value(name))
                self._on_slider(name, m.get_value(name))
            self.onnx_path_var.set(m.cfg.onnx.model_path)
            self.provider_var.set(m.provider.label)
            self.url_var.set(m.cfg.llm.base_url or m.provider.base_url)
            self.model_var.set(m.cfg.llm.model or m.provider.default_model)
            self.key_var.set(m.api_key)
            self.save_key_var.set(m.save_api_key)
            self.camera_var.set(str(m.cfg.camera.source))
        finally:
            self._busy_setting = False
        # the Base URL / model shown are defaults when the config left them empty: make that explicit
        m.cfg.llm.base_url = self.url_var.get().strip()
        m.cfg.llm.model = self.model_var.get().strip()

    def _tick(self) -> None:
        status = self.model.status
        self.status_var.set(status)
        self.status_dot.itemconfigure(self._dot, fill=STATUS_COLORS.get(status, STATUS_COLORS["Stopped"]))
        alive = self.model.runner.alive
        self.start_btn.state(["disabled"] if alive else ["!disabled"])
        self.stop_btn.state(["!disabled"] if alive else ["disabled"])
        self.pause_btn.state(["!disabled"] if status in ("Running", "Paused") else ["disabled"])
        self.fps_var.set(f"{self.model.runner.fps:.0f} FPS" if status in ("Running", "Paused") else "")
        self.error_var.set(self.model.runner.error)
        self.restart_var.set("Some changes (HUD, ONNX, LLM, camera) take effect after Stop, then Start." if self.model.restart_required else "")
        if self._test_result is not None:
            result, self._test_result = self._test_result, None
            self._test_running = False
            self.test_btn.state(["!disabled"])
            self._set_badge("ok" if result.ok else "fail", "connected" if result.ok else "failed", result.message)
        self._after_id = self.root.after(self.poll_ms, self._tick)

    def close(self) -> None:
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:  # pragma: no cover - already gone
                pass
        self.model.stop()
        self.root.destroy()


class _SubprocessDialog:
    """A small window that collects a few fields, runs ``open-ai-hci record|train`` and shows its output."""

    def __init__(self, panel: ControlPanel, title: str, *, fields: list[tuple[str, str]], build: Any, run_label: str, on_success: Any = None) -> None:
        tk, ttk = panel.tk, panel.ttk
        self.panel, self.build, self.on_success = panel, build, on_success
        self.proc: Any = None
        self.win = tk.Toplevel(panel.root)
        self.win.title(title)
        self.win.transient(panel.root)
        frame = ttk.Frame(self.win, padding=12)
        frame.pack(fill="both", expand=True)
        self.vars = []
        for i, (label, default) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=default)
            ttk.Entry(frame, textvariable=var, width=40).grid(row=i, column=1, sticky="ew", padx=8)
            self.vars.append(var)
        frame.columnconfigure(1, weight=1)
        self.run_btn = ttk.Button(frame, text=run_label, command=self.run)
        self.run_btn.grid(row=len(fields), column=1, sticky="w", padx=8, pady=8)
        self.log = tk.Text(frame, height=12, width=64, state="disabled", wrap="word")
        self.log.grid(row=len(fields) + 1, column=0, columnspan=2, sticky="nsew")
        frame.rowconfigure(len(fields) + 1, weight=1)
        self.win.protocol("WM_DELETE_WINDOW", self.close)

    def _write(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def run(self) -> None:
        values = [v.get().strip() for v in self.vars]
        try:
            argv = self.build(values)
        except ValueError as exc:
            self._write(f"{exc}\n")
            return
        try:
            self.proc = self.panel.model.launch(argv)
        except OSError as exc:
            self._write(f"Could not start: {exc}\n")
            return
        self.run_btn.state(["disabled"])
        self._write("$ " + " ".join(argv[2:]) + "\n")
        self._lines: list[str] = []
        threading.Thread(target=self._pump, daemon=True).start()
        self._poll(values)

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self._lines.append(line)

    def _poll(self, values: list[str]) -> None:
        if self._lines:
            chunk, self._lines[:] = "".join(self._lines), []
            self._write(chunk)
        code = self.proc.poll()
        if code is None:
            self.win.after(200, lambda: self._poll(values))
            return
        self.win.after(300, lambda: self._finish(code, values))

    def _finish(self, code: int, values: list[str]) -> None:
        if self._lines:
            self._write("".join(self._lines))
            self._lines.clear()
        self.run_btn.state(["!disabled"])
        self._write(f"\nFinished (exit code {code}).\n")
        if code == 0 and self.on_success is not None:
            self.on_success(values)
            self._write("The model is now selected in the panel.\n")

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.win.destroy()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``open-ai-hci gui``."""
    parser = argparse.ArgumentParser(prog="open-ai-hci gui", description="Desktop control panel for open-ai-hci.")
    parser.add_argument("--config", default="config.yaml", help="profile to load and save (default: config.yaml)")
    parser.add_argument("--autostart", action="store_true", help="start the engine immediately")
    args = parser.parse_args(argv)
    if not tkinter_available():
        print(
            "error: tkinter is not available in this Python.\n"
            "  Windows / macOS (python.org installer): re-run the installer and tick 'tcl/tk and IDLE'.\n"
            "  Debian / Ubuntu: sudo apt install python3-tk\n"
            "  Fedora: sudo dnf install python3-tkinter",
            file=sys.stderr,
        )
        return 1
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"error: cannot open a window ({exc}). Is a display available?", file=sys.stderr)
        return 1
    panel = ControlPanel(root, ControlPanelModel(config_path=Path(args.config)))
    if args.autostart:
        panel.on_start()
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
