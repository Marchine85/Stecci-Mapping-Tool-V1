# PyInstaller spec for the Stecci Mapping Tool v1 (one-folder build, photos + 3D scans).
# Normally run through packaging/build.py, which sets STECCI_BUILD_NAME and copies the model weights.
import os
from pathlib import Path

import torchvision
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

NAME = os.environ.get("STECCI_BUILD_NAME", "StecciMappingTool")
ROOT = Path(SPECPATH).parent

# torchvision loads its native ops (_C_stable.pyd, image_stable.pyd + DLLs) with torch.ops.load_library from its
# own folder, which PyInstaller's import analysis cannot see: copy them there explicitly
TV = Path(torchvision.__file__).parent
tv_binaries = [(str(f), "torchvision") for f in TV.iterdir() if f.suffix.lower() in (".pyd", ".dll")]

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    binaries=tv_binaries,
    # PyOpenGL picks its platform module at run time; trimesh loads exchange formats lazily
    hiddenimports=collect_submodules("OpenGL.platform") + collect_submodules("OpenGL.arrays")
                  + collect_submodules("trimesh.exchange") + ["fast_simplification"],
    # UI fonts of the themes (ui/theme.py loads them from stecci/assets/fonts)
    datas=collect_data_files("trimesh") + [(str(ROOT / "stecci" / "assets" / "fonts"), "stecci/assets/fonts")],
    excludes=[
        # Grounding DINO is optional and left out of the packaged app (SAM-only detection)
        "transformers", "tokenizers", "huggingface_hub",
        # not used, but pulled in by optional imports of torch / scipy / PIL / trimesh
        "matplotlib", "tkinter", "IPython", "notebook", "jupyter", "pytest", "tensorboard",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StecciMappingTool",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)
