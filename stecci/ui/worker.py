"""Run model calls off the GUI thread."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot


class _Signals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)


class Task(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _Signals()
        self.setAutoDelete(False)

    def run(self):
        try:
            res = self.fn(*self.args, progress=self.signals.progress.emit, **self.kwargs)
        except Exception as e:  # noqa: BLE001
            self.signals.failed.emit(f"{e}\n\n{traceback.format_exc()}")
        else:
            self.signals.finished.emit(res)


class Runner(QObject):
    """Single-slot task runner: one model job at a time."""
    busy_changed = Signal(bool)
    progress = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._task: Task | None = None

    @property
    def busy(self) -> bool:
        return self._task is not None

    def start(self, fn, on_done, on_error, *args, **kwargs) -> bool:
        if self._task is not None:
            return False
        t = Task(fn, *args, **kwargs)
        t.signals.progress.connect(self._on_progress, Qt.QueuedConnection)
        t.signals.finished.connect(self._on_finished, Qt.QueuedConnection)
        t.signals.failed.connect(self._on_failed, Qt.QueuedConnection)
        self._task = t
        self._callbacks = (on_done, on_error)
        self.busy_changed.emit(True)
        self.pool.start(t)
        return True

    @Slot(str)
    def _on_progress(self, msg):
        self.progress.emit(msg)

    @Slot(object)
    def _on_finished(self, res):
        self._task = None
        self.busy_changed.emit(False)
        self._callbacks[0](res)

    @Slot(str)
    def _on_failed(self, msg):
        self._task = None
        self.busy_changed.emit(False)
        self._callbacks[1](msg)
