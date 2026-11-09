"""Build the Stecci Mapping Tool v1 executables (photos + 3D scans).

    python packaging/build.py [gpu|cpu|all]      (default: all)

gpu  uses .venv      (CUDA torch)  ->  dist/StecciMappingTool_GPU/
cpu  uses .venv-cpu  (CPU torch)   ->  dist/StecciMappingTool_CPU/

Each output folder is self-contained and runs offline: StecciMappingTool.exe plus a models/ folder
with the SAM ViT-B checkpoint. Zip the folder to hand it on.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "stecci.spec"
VARIANTS = {"gpu": ".venv", "cpu": ".venv-cpu"}
SAM_FILE = "sam_vit_b_01ec64.pth"                  # the default model (Settings -> SAM model)


def find_sam() -> Path:
    p = ROOT / "models" / SAM_FILE
    if p.is_file():
        return p
    sys.exit(f"{SAM_FILE} not found: put it into {ROOT / 'models'}")


def build(variant: str, sam: Path):
    py = ROOT / VARIANTS[variant] / "Scripts" / "python.exe"
    if not py.is_file():
        sys.exit(f"{py} not found (create the {VARIANTS[variant]} venv first, see README -> Building the exe)")
    name = f"StecciMappingTool_{variant.upper()}"
    print(f"\n=== {name} ({py}) ===", flush=True)
    env = dict(os.environ, STECCI_BUILD_NAME=name)
    subprocess.run([str(py), "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build" / variant), str(SPEC)],
                   cwd=ROOT, env=env, check=True)
    out = ROOT / "dist" / name
    (out / "models").mkdir(exist_ok=True)
    shutil.copy2(sam, out / "models" / SAM_FILE)
    shutil.copy2(ROOT / "README.md", out / "README.md")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2**30
    print(f"-> {out}  ({size:.2f} GB)", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    variants = list(VARIANTS) if which == "all" else [which]
    if any(v not in VARIANTS for v in variants):
        sys.exit(__doc__)
    sam = find_sam()
    for v in variants:
        build(v, sam)


if __name__ == "__main__":
    main()
