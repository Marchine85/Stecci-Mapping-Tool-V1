"""Base class of the photo (2D) and 3D-scan workspaces.

A workspace is the view (photo canvas or 3D viewer) plus the four-step panel on its right. Everything the two
workflows share lives here: panel building blocks, the class list and class editing, click / drag assigning
and Ctrl-selection, display preferences, undo plumbing. The main window owns the menu bar, the status bar,
the models, the job runner and the busy overlay; workspaces reach them through `self.app`.

Actions and shortcuts are added to the workspace widget itself, so they only fire while that workspace is
the one on screen (Qt ignores shortcuts of hidden widgets): both workspaces can use the same keys.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel,
                               QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton, QScrollArea, QSlider,
                               QVBoxLayout, QWidget, QWidgetAction)

from .widgets import Section, capitalize_words, color_icon, set_role

ASSIGN_HINT = ("Click / Drag: Assign · Right-Click: Unassign · Ctrl+Click / Drag: Select · M: Merge\n"
               "Shift+Drag: Brush Class · Shift+Right-Drag: Brush Stone · Shift+Wheel: Brush Size")


class Workspace(QWidget):
    KIND = ""                 # "2d" | "3d"
    MODE_LABEL = ""           # shown at the top of the panel
    HELP_HTML = ""
    PANEL_MIN_WIDTH = 330

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.project = None
        self.project_path: Path | None = None
        self.selected: set[int] = set()
        self._assigning = False
        self._selecting = False      # Ctrl+drag selection in progress
        self._select_add = True      # ...adding (True) or removing (False) segments
        self._last_assigned = None
        self.class_list: QListWidget | None = None

    # ════════════════════════════════════════════════════════════════════════
    # shared state of the main window
    # ════════════════════════════════════════════════════════════════════════
    @property
    def settings(self):
        return self.app.settings

    @property
    def models(self):
        return self.app.models

    @property
    def runner(self):
        return self.app.runner

    @property
    def qsettings(self):
        return self.app.qsettings

    @property
    def status_msg(self) -> QLabel:
        return self.app.status_msg

    @property
    def device_label(self) -> QLabel:
        return self.app.device_label

    def _run(self, title: str, job, done, overlay_delay_ms: int = 250):
        self.app.run_job(title, job, done, overlay_delay_ms)

    def _error(self, msg: str):
        self.app.show_error(msg)

    def _save_settings(self):
        self.app.save_settings()

    def _last_dir(self) -> str:
        return self.app.last_dir()

    def is_current(self) -> bool:
        return self.app.current_workspace() is self

    # ════════════════════════════════════════════════════════════════════════
    # interface used by the main window (overridden by the workspaces)
    # ════════════════════════════════════════════════════════════════════════
    def has_document(self) -> bool:
        raise NotImplementedError

    def can_export(self) -> bool:
        return self.has_document()

    def undo_state(self) -> tuple[bool, bool]:
        p = self.project
        return (bool(p._undo), bool(p._redo)) if p is not None and self.has_document() else (False, False)

    def close_document(self):
        raise NotImplementedError

    def save_project(self, ask: bool = False) -> bool:
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

    def undo(self):
        raise NotImplementedError

    def redo(self):
        raise NotImplementedError

    def on_busy(self, busy: bool):
        self.panel.setEnabled(not busy)
        self._update_enabled()

    def settings_changed(self, old, new):
        pass

    def confirm_discard(self) -> bool:
        """True when the current document may be replaced (saved, discarded or not modified)."""
        if self.project is None or not self.has_document() or not self.project.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save the current project first?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save_project()
        return r == QMessageBox.Discard

    _confirm_discard = confirm_discard

    # ════════════════════════════════════════════════════════════════════════
    # building blocks
    # ════════════════════════════════════════════════════════════════════════
    def _act(self, text, slot=None, shortcut=None, checkable=False, checked=False):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if checkable:
            a.setCheckable(True)
            a.setChecked(checked)
        if slot is not None:
            a.triggered.connect(lambda _=False: slot())
        self.addAction(a)      # shortcut active only while this workspace is on screen
        return a

    def _class_shortcuts(self):
        for i in range(10):
            QShortcut(QKeySequence(str(i)), self, activated=lambda i=i: self._select_class_row(i))

    def _slider_action(self, text, slider, value_label=None) -> QWidgetAction:
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(28, 4, 12, 4)
        hl.addWidget(QLabel(text))
        slider.setMinimumWidth(130)
        hl.addWidget(slider)
        if value_label is not None:
            value_label.setFixedWidth(40)
            hl.addWidget(value_label)
        wa = QWidgetAction(self)
        wa.setDefaultWidget(w)
        return wa

    def _big_button(self, text, slot, tip=""):
        b = QPushButton(text)
        b.setMinimumHeight(46)
        set_role(b, "secondary")
        b.setCursor(Qt.PointingHandCursor)
        b.setToolTip(tip)
        b.clicked.connect(lambda _=False: slot())
        return b

    def _link_toggle(self, text, tool, tip=""):
        b = QPushButton(text)
        b.setObjectName("Link")
        b.setCheckable(True)
        b.setCursor(Qt.PointingHandCursor)
        b.setToolTip(tip)
        b.toggled.connect(lambda on: self.set_tool(tool if on else "assign"))
        return b

    def _link_button(self, text, slot, tip=""):
        b = QPushButton(text)
        b.setObjectName("Link")
        b.setCursor(Qt.PointingHandCursor)
        b.setToolTip(tip)
        b.clicked.connect(lambda _=False: slot())
        return b

    def _hint(self, text):
        lab = QLabel(text)
        lab.setObjectName("Hint")
        lab.setWordWrap(True)
        return lab

    def _mode_header(self) -> QWidget:
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(2, 0, 2, 0)
        tag = QLabel(self.MODE_LABEL)
        tag.setObjectName("ModeTag")
        hl.addWidget(tag)
        hl.addStretch(1)
        return w

    def _build_classes_section(self, menu: QMenu, brush_limits: tuple[int, int], brush_tip: str) -> Section:
        """3 · Assign Classes: class list, + Add Class, class brush size, hint."""
        sec = Section(3, "Assign Classes", menu)
        self.class_list = QListWidget()
        self.class_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.class_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.class_list.customContextMenuRequested.connect(self._class_context_menu)
        self.class_list.itemDoubleClicked.connect(lambda _: self.rename_class())
        self.class_list.itemClicked.connect(lambda _: self.current_tool() != "assign" and self.set_tool("assign"))
        self.class_list.currentItemChanged.connect(lambda *_: self._on_active_class_changed())
        sec.body.addWidget(self.class_list)
        sec.body.addWidget(self._link_button("+  Add Class", self.add_class))
        row_w = QWidget()
        row = QHBoxLayout(row_w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Brush Size"))
        row_w.setToolTip(brush_tip)
        self.paint_slider = QSlider(Qt.Horizontal)
        self.paint_slider.setRange(*brush_limits)
        self.paint_slider.valueChanged.connect(lambda v: self.set_brush(v, "paint"))
        row.addWidget(self.paint_slider, 1)
        self.paint_label = QLabel()
        self.paint_label.setFixedWidth(46)
        self.paint_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.paint_label)
        sec.body.addWidget(row_w)
        self.sel_label = self._hint(ASSIGN_HINT)
        sec.body.addWidget(self.sel_label)
        return sec

    def _annotation_grid(self, rows) -> tuple[QGridLayout, dict]:
        """One row per layer: checkbox bound to an action, optional slider + "% " label on the right.

        rows: (action, text, slider | None, tooltip | None). Returns the grid and {text: value label}.
        """
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        grid.setColumnStretch(1, 1)
        labels = {}
        for i, (act, text, slider, tip) in enumerate(rows):
            cb = QCheckBox(text)
            cb.setChecked(act.isChecked())
            cb.setCursor(Qt.PointingHandCursor)
            cb.setToolTip(tip or text)
            cb.toggled.connect(act.setChecked)
            act.toggled.connect(cb.setChecked)
            grid.addWidget(cb, i, 0)
            if slider is not None:
                label = QLabel()
                slider.setMinimumWidth(80)
                grid.addWidget(slider, i, 1)
                label.setFixedWidth(42)
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                grid.addWidget(label, i, 2)
                slider.valueChanged.connect(lambda val, lab=label: lab.setText(f"{val} %"))
                label.setText(f"{slider.value()} %")
                # the slider only matters while its layer is switched on
                slider.setEnabled(act.isChecked())
                label.setEnabled(act.isChecked())
                act.toggled.connect(slider.setEnabled)
                act.toggled.connect(label.setEnabled)
                labels[text] = label
        return grid, labels

    def _install(self, view: QWidget, cards: list[QWidget], bottom: QWidget):
        """Lay out view | panel. The step cards scroll; the bottom card stays pinned to the corner."""
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(10)
        v.addWidget(self._mode_header())
        for c in cards:
            v.addWidget(c)
        v.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(body)
        scroll.setMinimumWidth(self.PANEL_MIN_WIDTH)
        panel = QWidget()
        panel.setObjectName("Panel")          # themed: see theme.style_sheet
        panel.setMaximumWidth(self.PANEL_MIN_WIDTH + 90)
        ov = QVBoxLayout(panel)
        ov.setContentsMargins(0, 0, 0, 0)
        ov.setSpacing(0)
        ov.addWidget(scroll, 1)
        vw = QVBoxLayout()
        vw.setContentsMargins(10, 0, 10, 10)
        vw.addWidget(bottom)
        ov.addLayout(vw)
        self.panel = panel
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)
        hl.addWidget(view, 1)
        hl.addWidget(panel, 0)

    # ── display preferences (remembered between sessions, per workspace) ─────
    def _load_display(self, key: str, defaults: dict, reset_keys: dict | None = None) -> dict:
        display = dict(defaults)
        try:
            stored = json.loads(self.qsettings.value(key, "{}"))
            old = int(stored.pop("version", 1))
            for ver, keys in (reset_keys or {}).items():        # changed defaults replace stored values
                if old < ver:
                    stored = {k: v for k, v in stored.items() if k not in keys}
            display.update(stored)
        except Exception:
            pass
        self._display_key = key
        return display

    def _save_display(self):
        self.qsettings.setValue(self._display_key, json.dumps(self.display))

    # ════════════════════════════════════════════════════════════════════════
    # hooks (overridden by the workspaces)
    # ════════════════════════════════════════════════════════════════════════
    def current_tool(self) -> str:
        raise NotImplementedError

    def set_tool(self, tool: str):
        raise NotImplementedError

    def set_brush(self, size: int, which: str | None = None):
        raise NotImplementedError

    def _segments(self):
        """Current segment ids (label image or per-face array) or None before segmentation."""
        raise NotImplementedError

    def _class_area_stats(self) -> tuple[dict, float]:
        """({class id: area}, stone area) for the percentages in the class list."""
        return {}, 0

    def _refresh_class_colours(self):
        """Segment colours changed (assignment / class colour / class deleted)."""

    def _refresh_selection_view(self):
        """The Ctrl-selection changed."""

    def _refresh_legend(self):
        pass

    def _on_active_class_changed(self):
        pass

    def _update_enabled(self):
        pass

    # ════════════════════════════════════════════════════════════════════════
    # classes
    # ════════════════════════════════════════════════════════════════════════
    def active_class(self) -> int:
        it = self.class_list.currentItem() if self.class_list is not None else None
        return it.data(Qt.UserRole) if it else 0

    def _class_context_menu(self, pos):
        it = self.class_list.itemAt(pos)
        if it is None:
            return
        self.class_list.setCurrentItem(it)
        m = QMenu(self)
        for a in (self.a_rename_class, self.a_color_class, self.a_delete_class, None, self.a_assign_sel):
            m.addSeparator() if a is None else m.addAction(a)
        m.exec(self.class_list.mapToGlobal(pos))

    def _refresh_classes(self):
        cur = self.active_class() if self.class_list.count() else 1
        areas, total = self._class_area_stats()
        self.class_list.blockSignals(True)
        self.class_list.clear()
        for i, (cid, c) in enumerate(sorted(self.project.classes.items())):
            pct = f"   {100 * areas.get(cid, 0) / total:.1f} %" if total and areas else ""
            key = f"{i}  " if i < 10 else "   "
            it = QListWidgetItem(color_icon(c["color"]), f"{key}{capitalize_words(c['name'])}{pct}")
            it.setData(Qt.UserRole, cid)
            self.class_list.addItem(it)
            if cid == cur:
                self.class_list.setCurrentItem(it)
        if self.class_list.currentItem() is None and self.class_list.count():
            self.class_list.setCurrentRow(min(1, self.class_list.count() - 1))
        self.class_list.blockSignals(False)
        self._on_active_class_changed()
        # height follows the number of classes (max 8 rows, then scroll)
        row_h = max(self.class_list.sizeHintForRow(0), 22) if self.class_list.count() else 26
        self.class_list.setFixedHeight(min(self.class_list.count(), 8) * row_h + 12)
        self._refresh_legend()
        self.app.update_file_actions()

    def _select_class_row(self, i: int):
        if i < self.class_list.count() and self._segments() is not None:
            self.class_list.setCurrentRow(i)
            if self.current_tool() != "assign":
                self.set_tool("assign")
            self._on_active_class_changed()

    def add_class(self):
        name, ok = QInputDialog.getText(self, "New class", "Name:")
        if not ok or not name.strip():
            return
        col = QColorDialog.getColor(QColor(*np.random.randint(40, 230, 3).tolist()), self, "Class colour")
        if not col.isValid():
            return
        self.project.push_undo()
        cid = self.project.next_class_id()
        self.project.classes[cid] = {"name": name.strip(), "color": [col.red(), col.green(), col.blue()]}
        self._refresh_classes()
        for i in range(self.class_list.count()):
            if self.class_list.item(i).data(Qt.UserRole) == cid:
                self.class_list.setCurrentRow(i)

    def rename_class(self):
        cid = self.active_class()
        if cid == 0:
            return
        c = self.project.classes[cid]
        name, ok = QInputDialog.getText(self, "Rename class", "Name:", text=c["name"])
        if ok and name.strip():
            self.project.push_undo()
            c["name"] = name.strip()
            self._refresh_classes()

    def recolor_class(self):
        cid = self.active_class()
        if cid == 0:
            return
        c = self.project.classes[cid]
        col = QColorDialog.getColor(QColor(*c["color"]), self, "Class colour")
        if col.isValid():
            self.project.push_undo()
            c["color"] = [col.red(), col.green(), col.blue()]
            self._refresh_classes()
            self._refresh_class_colours()

    def delete_class(self):
        cid = self.active_class()
        if cid == 0:
            return
        name = self.project.classes[cid]["name"]
        if QMessageBox.question(self, "Delete class",
                                f"Delete class '{name}'? Its segments become unassigned.") != QMessageBox.Yes:
            return
        p = self.project
        p.push_undo()
        p.seg_class = np.where(p.seg_class == cid, 0, p.seg_class).astype(np.int16)
        del p.classes[cid]
        self._refresh_classes()
        self._refresh_class_colours()

    # ════════════════════════════════════════════════════════════════════════
    # assigning / selecting
    # ════════════════════════════════════════════════════════════════════════
    def _require_segments(self) -> bool:
        if self._segments() is None:
            self.status_msg.setText("Press Segment first.")
            return False
        return True

    def _assign_press(self, sid: int, left: bool, ctrl: bool):
        """Mouse press on segment `sid` (0 = none) with the assign tool."""
        if not self._require_segments():
            return
        if ctrl:
            # Ctrl+drag continues what the first click did: add (or remove) every segment passed over
            self._select_add = sid not in self.selected
            self._selecting = True
            self._last_assigned = sid
            if sid:
                self.selected ^= {sid}
                self._refresh_selection_view()
            return
        if self.selected and sid not in self.selected:
            self.clear_selection()   # a plain click elsewhere drops the Ctrl+click selection
        if not sid:
            return
        cls = self.active_class() if left else 0
        self.project.push_undo()
        self._assigning = True
        if sid in self.selected:
            self._assign(self.selected, cls)
            self.clear_selection()   # otherwise the highlight hides the new colour
        else:
            self._assign({sid}, cls)
        self._last_assigned = sid

    def _assign_drag(self, sid: int, left: bool):
        if self._selecting:
            if sid and sid != self._last_assigned:
                self._last_assigned = sid
                if (sid not in self.selected) == self._select_add:
                    self.selected ^= {sid}
                    self._refresh_selection_view()
            return
        if not self._assigning:
            return
        if sid and sid != self._last_assigned:
            self._last_assigned = sid
            self._assign({sid}, self.active_class() if left else 0)

    def _assign_release(self):
        self._selecting = False
        if self._assigning:
            self._assigning = False
            self._refresh_classes()
            self._update_enabled()
        self._last_assigned = None

    def _assign(self, ids, cls):
        p = self.project
        p.ensure_class_capacity()
        ids = list(ids)
        if np.all(p.seg_class[ids] == cls):
            return
        p.seg_class = p.seg_class.copy()        # state arrays are replaced, never changed in place
        p.seg_class[ids] = cls
        self._refresh_class_colours()

    def assign_selection(self):
        if not self.selected:
            self.status_msg.setText("Ctrl+click segments to select them.")
            return
        self.project.push_undo()
        self._assign(self.selected, self.active_class())
        self.clear_selection()
        self._refresh_classes()

    def clear_assignments(self):
        p = self.project
        if self._segments() is None or not p.seg_class.any():
            self.status_msg.setText("No class assignments to clear.")
            return
        if QMessageBox.question(self, "Clear all assignments",
                                "Set every segment back to Unassigned/Stone? (Undo with Ctrl+Z)") != QMessageBox.Yes:
            return
        p.push_undo()
        p.seg_class = np.zeros_like(p.seg_class)
        self._refresh_class_colours()
        self._refresh_classes()
        self.status_msg.setText("All class assignments cleared.")

    def clear_selection(self):
        self.selected.clear()
        self._refresh_selection_view()
