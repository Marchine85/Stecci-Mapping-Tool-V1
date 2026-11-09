import multiprocessing
import sys


def selftest() -> int:
    """`StecciMappingTool.exe --selftest`: load SAM, isolate + segment a synthetic photo, check the 3D modules
    without the GUI, and write the result to selftest.log next to the app (the packaged exe has no console)."""
    import time
    import traceback

    import numpy as np

    from .config import Settings, app_dir, granularity

    log = app_dir() / "selftest.log"
    lines = []
    try:
        from .models import ModelManager
        t = time.perf_counter()
        m = ModelManager(Settings())
        m.load_sam()
        lines.append(f"SAM loaded: {m.describe()} ({time.perf_counter() - t:.1f} s)")
        img = np.full((512, 512, 3), 230, np.uint8)
        img[128:384, 160:352] = (120, 110, 100)           # a "stone" in the centre
        img[180:260, 200:300] = (70, 140, 70)             # a "patch" on it
        t = time.perf_counter()
        box, score, mask = m.detect_rock_sam(img, key="selftest")
        lines.append(f"photo isolate: box={np.round(box).astype(int).tolist()} score={score:.2f} "
                     f"area={int(mask.sum())} ({time.perf_counter() - t:.1f} s)")
        t = time.perf_counter()
        masks = m.auto_masks(img[128:384, 160:352], granularity(3), min_area=50)
        lines.append(f"photo segment: {len(masks)} masks ({time.perf_counter() - t:.1f} s)")
        from .scan3d import export3d, isolate, mesh, reduce, surface, unwrap  # noqa: F401
        import OpenGL.GL  # noqa: F401
        lines.append("3D modules: OK (trimesh, OpenGL)")
        lines.append("OK")
        code = 0
    except Exception:
        lines.append(traceback.format_exc())
        lines.append("FAILED")
        code = 1
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return code


def main():
    multiprocessing.freeze_support()
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    from PySide6.QtWidgets import QApplication

    from .config import APP_NAME
    from .ui.viewer import set_default_format

    set_default_format()                     # before the QApplication: OpenGL 3.3 compatibility, MSAA (3D viewer)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    from pathlib import Path

    from PySide6.QtGui import QIcon
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "icon.svg")))   # window + taskbar
    app.setStyle("Fusion")
    # palette, font and style sheet come from the chosen theme (ui/theme.py, applied by MainWindow)
    from .ui.main_window import MainWindow
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.open_path(sys.argv[1])          # photo, 3D scan, .stecci or .stecci3d
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
