"""Main window: start page, shared menu / status bar, models and job runner, and the two workspaces.

One document is open at a time: a photo (or .stecci project) in the photo workspace, or a 3D scan (or
.stecci3d project) in the 3D workspace. Opening a file of the other kind closes the current document first
(asking to save it). Both workspaces share one SAM model, so it is loaded only once.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

from ..config import (APP_NAME, APP_VERSION, IMAGE_EXTS, PROJECT_EXT_2D, PROJECT_EXT_3D, SCAN_EXTS, Settings,
                      file_kind)
from ..models import ModelManager
from .busy_overlay import BusyOverlay
from .settings_dialog import SettingsDialog
from . import theme
from .widgets import arrow_icon, set_role
from .worker import Runner, Task
from .workspace import Workspace
from .workspace2d import PhotoWorkspace
from .workspace3d import ScanWorkspace


def _filter(name: str, exts) -> str:
    return f"{name} (" + " ".join(f"*{e}" for e in exts) + ")"


PHOTO_FILTER = _filter("Photos", IMAGE_EXTS)
SCAN_FILTER = _filter("3D scans", SCAN_EXTS)
PROJECT_FILTER = _filter("Stecci projects", (PROJECT_EXT_2D, PROJECT_EXT_3D))
ALL_FILTER = _filter("All supported files", IMAGE_EXTS + SCAN_EXTS + (PROJECT_EXT_2D, PROJECT_EXT_3D))

COMMON_HELP = """
<h3>Editing</h3>
Ctrl+Z / Ctrl+Y = undo / redo · Esc = clear selection · Ctrl+S = save project · Ctrl+E = export ·
Ctrl+W = close the document
<h3>Files</h3>
Photos (.jpg .png .tif …) and photo projects (<i>.stecci</i>) open in the photo workspace;
3D scans (.glb .gltf .obj .ply) and 3D projects (<i>.stecci3d</i>) in the 3D workspace.
Ctrl+O opens any of them, or drop a file onto the window.
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1550, 980)
        self.qsettings = QSettings("Stecci", "StecciMappingTool_v1")
        try:
            self.settings = Settings.from_dict(json.loads(self.qsettings.value("settings", "{}")))
        except Exception:
            self.settings = Settings()
        theme.apply(self.settings.theme)        # before any widget is built
        self.models = ModelManager(self.settings)
        self.runner = Runner(self)
        self._job_id = 0

        self._build_actions()
        self._build_menus()
        self._build_status()

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_start_page())
        self.ws2d = PhotoWorkspace(self)
        self.ws3d = ScanWorkspace(self)
        self.stack.addWidget(self.ws2d)
        self.stack.addWidget(self.ws3d)
        self.setCentralWidget(self.stack)
        self.overlay = BusyOverlay(self.stack)
        self.setAcceptDrops(True)

        self.runner.busy_changed.connect(self._on_busy)
        self.runner.progress.connect(self._on_progress)
        self.show_start()

        # load + warm up SAM in the background while the user picks a file
        self._preload_pool = QThreadPool(self)
        self._preload_pool.setMaxThreadCount(1)
        self._preload_task = None
        self._pending_reset = False
        QTimer.singleShot(300, self._start_preload)

    # ════════════════════════════════════════════════════════════════════════
    # UI construction
    # ════════════════════════════════════════════════════════════════════════
    def _act(self, text, slot, shortcut=None, tip=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setToolTip(tip)
        a.triggered.connect(lambda _=False: slot())
        self.addAction(a)      # shortcuts work even when the menus are hidden
        return a

    def _build_actions(self):
        A = self._act
        self.a_open = A("Open…", self.open_dialog, "Ctrl+O")
        self.a_open_photo = A("Open Photo…", lambda: self.open_dialog("photo"))
        self.a_open_scan = A("Open 3D Scan…", lambda: self.open_dialog("scan"))
        self.a_open_proj = A("Open Project…", lambda: self.open_dialog("project"), "Ctrl+Shift+O")
        self.a_save = A("Save Project", lambda: self._cur("save_project"), "Ctrl+S")
        self.a_save_as = A("Save Project As…", lambda: self._cur("save_project", ask=True), "Ctrl+Shift+S")
        self.a_export = A("Export…", lambda: self._cur("export"), "Ctrl+E")
        self.a_close = A("Close", self.close_document, "Ctrl+W")
        self.a_settings = A("Settings", self.open_settings, "Ctrl+,")
        self.a_quit = A("Quit", self.close, "Ctrl+Q")
        self.undo_act = A("", lambda: self._cur("undo"), "Ctrl+Z", "Undo (Ctrl+Z)")
        self.redo_act = A("", lambda: self._cur("redo"), "Ctrl+Y", "Redo (Ctrl+Y)")
        self._update_icons()
        theme.manager().changed.connect(self._update_icons)
        self.a_help = A("Workflow && Shortcuts", self.show_help, "F1")
        self.a_about = A("About", self.show_about)

    def _update_icons(self):
        self.undo_act.setIcon(arrow_icon(False))
        self.redo_act.setIcon(arrow_icon(True))

    def _build_menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        for a in (self.a_open, None, self.a_open_photo, self.a_open_scan, self.a_open_proj, None,
                  self.a_save, self.a_save_as, self.a_export, None, self.a_close, None, self.a_quit):
            f.addSeparator() if a is None else f.addAction(a)
        mb.addAction(self.undo_act)
        mb.addAction(self.redo_act)
        mb.addAction(self.a_settings)
        h = mb.addMenu("&Help")
        h.addAction(self.a_help)
        h.addAction(self.a_about)

    def _build_status(self):
        sb = self.statusBar()
        self.status_msg = QLabel("Open a photo or a 3D scan to start (Ctrl+O).")
        self.status_msg.setContentsMargins(8, 0, 0, 0)        # keep the text off the window edges
        self.device_label = QLabel(self.models.describe())
        self.device_label.setObjectName("Muted")
        self.device_label.setContentsMargins(0, 0, 14, 0)
        sb.addWidget(self.status_msg, 1)
        sb.addPermanentWidget(self.device_label)

    def _build_start_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.addStretch(1)
        title = QLabel(APP_NAME)
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("StartTitle")
        v.addWidget(title)
        sub = QLabel("Isolate the stećak in a photo or a 3D scan, segment it, label weathering classes and "
                     "draw cracks.")
        sub.setAlignment(Qt.AlignCenter)
        sub.setObjectName("StartSub")
        v.addWidget(sub)
        v.addSpacing(30)

        cards = QHBoxLayout()
        cards.addStretch(1)
        for label, caption, slot, style in (
                ("Open Photo", "2D  ·  .jpg .png .tif …\nareas in pixels of the photo",
                 lambda: self.open_dialog("photo"), "primary"),
                ("Open 3D Scan", "3D  ·  .glb .gltf .obj .ply\nareas measured on the 3D surface",
                 lambda: self.open_dialog("scan"), "primary")):
            col = QVBoxLayout()
            btn = QPushButton(label)
            btn.setFixedSize(240, 64)
            btn.setObjectName("StartButton")
            set_role(btn, style)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, s=slot: s())
            col.addWidget(btn, 0, Qt.AlignHCenter)
            cap = QLabel(caption)
            cap.setAlignment(Qt.AlignCenter)
            cap.setObjectName("StartCaption")
            col.addWidget(cap)
            cards.addLayout(col)
            cards.addSpacing(24)
        cards.addStretch(1)
        v.addLayout(cards)
        v.addSpacing(16)

        links = QHBoxLayout()
        links.addStretch(1)
        for text, slot in [("Open project…", lambda: self.open_dialog("project")), ("Settings", self.open_settings),
                           ("Help", self.show_help)]:
            b = QPushButton(text)
            b.setFlat(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setObjectName("StartLink")
            b.clicked.connect(lambda _=False, s=slot: s())
            links.addWidget(b)
        links.addStretch(1)
        v.addLayout(links)
        hint = QLabel("or drop a photo, a 3D scan or a project here")
        hint.setAlignment(Qt.AlignCenter)
        hint.setObjectName("Muted")
        v.addWidget(hint)
        v.addStretch(1)
        return page

    # ════════════════════════════════════════════════════════════════════════
    # workspaces
    # ════════════════════════════════════════════════════════════════════════
    def current_workspace(self):
        w = self.stack.currentWidget()
        return w if isinstance(w, Workspace) else None

    def _cur(self, method: str, **kw):
        ws = self.current_workspace()
        if ws is not None and ws.has_document():
            return getattr(ws, method)(**kw)
        return None

    def show_workspace(self, ws):
        self.stack.setCurrentWidget(ws)
        self.statusBar().setVisible(True)
        self.update_file_actions()

    def show_start(self):
        self.stack.setCurrentIndex(0)
        self.statusBar().setVisible(False)
        self.setWindowTitle(APP_NAME)
        self.update_file_actions()

    def set_document_title(self, name: str):
        ws = self.current_workspace()
        kind = "3D Scan" if ws is self.ws3d else "Photo"
        self.setWindowTitle(f"{APP_NAME} – {name}  [{kind}]")

    def update_file_actions(self):
        if not hasattr(self, "stack"):
            return
        ws = self.current_workspace()
        has = ws is not None and ws.has_document()
        idle = not self.runner.busy
        for a in (self.a_save, self.a_save_as, self.a_close):
            a.setEnabled(has and idle)
        self.a_export.setEnabled(has and idle and ws.can_export())
        undo, redo = ws.undo_state() if has else (False, False)
        self.undo_act.setVisible(has)
        self.redo_act.setVisible(has)
        self.undo_act.setEnabled(undo and idle)
        self.redo_act.setEnabled(redo and idle)

    # ── opening files ────────────────────────────────────────────────────────
    def last_dir(self) -> str:
        return self.qsettings.value("last_dir", str(Path.home()))

    def open_dialog(self, which: str = "any"):
        if self.runner.busy:
            return
        title, flt = {"photo": ("Open stećak photo", PHOTO_FILTER),
                      "scan": ("Open 3D scan", SCAN_FILTER),
                      "project": ("Open project", PROJECT_FILTER),
                      "any": ("Open", ";;".join((ALL_FILTER, PHOTO_FILTER, SCAN_FILTER, PROJECT_FILTER)))}[which]
        path, _ = QFileDialog.getOpenFileName(self, title, self.last_dir(), flt)
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        """Open a photo, scan or project in the matching workspace."""
        if self.runner.busy:
            return
        kind = file_kind(path)
        if kind is None:
            self.show_error(f"Cannot open {Path(path).name}: unsupported file type.\n\n"
                            f"Photos: {' '.join(IMAGE_EXTS)}\n3D scans: {' '.join(SCAN_EXTS)}\n"
                            f"Projects: {PROJECT_EXT_2D} {PROJECT_EXT_3D}")
            return
        if not Path(path).is_file():
            self.show_error(f"File not found: {path}")
            return
        target = self.ws2d if kind in ("photo", "project2d") else self.ws3d
        cur = self.current_workspace()
        if cur is not None and cur is not target and cur.has_document():
            if not cur.confirm_discard():
                return
            cur.close_document()
            self.show_start()
        self.qsettings.setValue("last_dir", str(Path(path).parent))
        if kind == "photo":
            self.ws2d.open_image(path)
        elif kind == "project2d":
            self.ws2d.load_project(path)
        elif kind == "scan":
            self.ws3d.open_scan(path)
        else:
            self.ws3d.load_project(path)

    def close_document(self):
        ws = self.current_workspace()
        if ws is None or self.runner.busy or not ws.confirm_discard():
            return
        ws.close_document()
        self.show_start()
        self.status_msg.setText("Open a photo or a 3D scan to start (Ctrl+O).")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.open_path(urls[0])

    # ════════════════════════════════════════════════════════════════════════
    # model jobs
    # ════════════════════════════════════════════════════════════════════════
    def _start_preload(self):
        if self._preload_task is not None or self.models.ready:
            return
        t = Task(self.models.preload)
        t.signals.finished.connect(self._preload_done, Qt.QueuedConnection)
        t.signals.failed.connect(self._preload_failed, Qt.QueuedConnection)
        self._preload_task = t
        self._preload_pool.start(t)

    def _preload_done(self, _=None):
        self._preload_task = None
        self.device_label.setText(self.models.describe())
        self._apply_pending_reset()

    def _preload_failed(self, msg: str):
        # not fatal: the same error is reported (with a dialog) when a model is first needed
        self._preload_task = None
        self.device_label.setText("models not loaded – see Settings")
        self._apply_pending_reset()

    def _apply_pending_reset(self):
        """Model settings changed while models were busy: reload once everything is idle."""
        if not self._pending_reset or self.runner.busy or self._preload_task is not None:
            return
        self._pending_reset = False
        self.models.reset(self.settings)
        self.device_label.setText(self.models.describe())
        self._start_preload()

    def run_job(self, title: str, job, done, overlay_delay_ms: int = 250):
        """Start a background job; show the busy overlay if it takes longer than a moment."""
        self._job_id += 1
        jid = self._job_id
        self.overlay.set_message("Starting …")
        if self.runner.start(job, done, self.show_error):
            QTimer.singleShot(overlay_delay_ms,
                              lambda: self.runner.busy and self._job_id == jid and self.overlay.start(title))

    def _on_progress(self, msg: str):
        self.status_msg.setText(msg)
        self.overlay.set_message(msg)

    def _on_busy(self, busy: bool):
        if not busy:
            self.overlay.stop()
            self.device_label.setText(self.models.describe())
            QTimer.singleShot(0, self._apply_pending_reset)
        for ws in (self.ws2d, self.ws3d):
            ws.on_busy(busy)
        self.update_file_actions()

    def show_error(self, msg: str):
        self.status_msg.setText("Error, see dialog.")
        box = QMessageBox(QMessageBox.Critical, "Error", msg.split("\n\n")[0], parent=self)
        box.setDetailedText(msg)
        box.exec()

    # ════════════════════════════════════════════════════════════════════════
    # settings / help / close
    # ════════════════════════════════════════════════════════════════════════
    def save_settings(self):
        self.qsettings.setValue("settings", json.dumps(self.settings.to_dict()))

    def open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if not dlg.exec():
            return
        old, new = self.settings, dlg.result_settings()
        model_keys = ("sam_model_type", "sam_checkpoint", "gdino_model_id", "use_dino")
        reload = any(getattr(new, k) != getattr(old, k) for k in model_keys)
        self.settings = new
        self.models.settings = new
        self.save_settings()
        for ws in (self.ws2d, self.ws3d):
            ws.settings_changed(old, new)
        if reload:
            self._pending_reset = True
            self._apply_pending_reset()          # now, or as soon as the running job / preload ends

    def show_help(self):
        ws = self.current_workspace()
        if ws is None:
            text = self.ws2d.HELP_HTML + self.ws3d.HELP_HTML + COMMON_HELP
        else:
            text = ws.HELP_HTML + COMMON_HELP
        box = QMessageBox(QMessageBox.Information, "Help", text, parent=self)
        box.setTextFormat(Qt.RichText)
        box.exec()

    def show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<h3>{APP_NAME} {APP_VERSION}</h3>"
                          "<p>Weathering mapping of stećci on photos and textured 3D scans: isolate the stone, "
                          "segment it with SAM, assign classes, draw cracks and export maps and measurements.</p>"
                          f"<p style='color: gray'>{self.models.describe()}</p>")

    def closeEvent(self, e):
        self.save_settings()
        if self.runner.busy:
            QMessageBox.information(self, "Busy", "Please wait until the running job has finished.")
            e.ignore()
            return
        ws = self.current_workspace()
        if ws is None or ws.confirm_discard():
            # let a running background model load finish, so Python does not shut down mid-import
            self.status_msg.setText("Closing …")
            self._preload_pool.waitForDone(30000)
            e.accept()
        else:
            e.ignore()
