"""Export dialog: stećak name, site information and target folder (photo and 3D-scan exports)."""
from __future__ import annotations

from PySide6.QtCore import QDate
from PySide6.QtWidgets import (QComboBox, QDateEdit, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QWidget)

SIDES_PHOTO = ["", "Front", "Back", "Left", "Right", "Top", "Bottom", "Whole stone", "Detail"]
SIDES_SCAN = ["Whole stone", "Front", "Back", "Left", "Right", "Top", "Detail", ""]
HINT_PHOTO = "Writes <name>.png, _white, _surroundings, _cracks, _stone, _stone_outlines and <name>.json"
HINT_SCAN = ("Writes <name>.png (current view), <name>_views.png (six views), <name>.glb (3D model with the "
             "mapping) and <name>.json (areas measured on the 3D surface)")
BAD_CHARS = '<>:"/\\|?*'


def clean_name(text: str) -> str:
    """Usable as a file name on Windows."""
    return "".join(ch for ch in text.strip() if ch not in BAD_CHARS).strip(" .")


class ExportDialog(QDialog):
    def __init__(self, name: str, site: dict, folder: str, parent=None, *, scan: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Export 3D Scan" if scan else "Export Photo")
        self.setMinimumWidth(460)
        f = QFormLayout(self)

        self.name = QLineEdit(name)
        self.name.setToolTip("Base name of every exported file, e.g. 'Kalinovik 12'")
        f.addRow("Stećak name", self.name)

        self.site = QLineEdit(site.get("site", ""))
        self.site.setPlaceholderText("necropolis / village / location")
        f.addRow("Site", self.site)

        self.side = QComboBox()
        self.side.setEditable(True)
        self.side.addItems(SIDES_SCAN if scan else SIDES_PHOTO)
        self.side.setCurrentText(site.get("side", ""))
        self.side.setToolTip("Which part of the stone the scan covers" if scan
                             else "Which side of the stone the photo shows")
        f.addRow("Side of stone", self.side)

        self.date = QDateEdit()
        self.date.setCalendarPopup(True)
        self.date.setDisplayFormat("yyyy-MM-dd")
        d = QDate.fromString(site.get("survey_date", ""), "yyyy-MM-dd")
        self.date.setDate(d if d.isValid() else QDate.currentDate())
        f.addRow("Survey date", self.date)

        self.surveyor = QLineEdit(site.get("surveyor", ""))
        f.addRow("Surveyor", self.surveyor)

        self.notes = QPlainTextEdit(site.get("notes", ""))
        self.notes.setFixedHeight(70)
        f.addRow("Notes", self.notes)

        self.folder = QLineEdit(folder)
        b = QPushButton("…")
        b.setFixedWidth(30)
        b.clicked.connect(self._browse)
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.folder)
        hl.addWidget(b)
        f.addRow("Folder", row)

        hint = QLabel(HINT_SCAN if scan else HINT_PHOTO)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        f.addRow(hint)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Export")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Export to folder", self.folder.text())
        if d:
            self.folder.setText(d)

    def _accept(self):
        if not clean_name(self.name.text()):
            self.name.setFocus()
            self.name.setPlaceholderText("please enter a name")
            return
        if not self.folder.text().strip():
            self._browse()
            if not self.folder.text().strip():
                return
        self.accept()

    def values(self) -> tuple[str, dict, str]:
        site = {"site": self.site.text().strip(), "side": self.side.currentText().strip(),
                "survey_date": self.date.date().toString("yyyy-MM-dd"),
                "surveyor": self.surveyor.text().strip(), "notes": self.notes.toPlainText().strip()}
        return clean_name(self.name.text()), site, self.folder.text().strip()
