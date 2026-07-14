"""检查课程所需的核心 Python 包。"""

from __future__ import annotations

import importlib
import platform
import sys


PACKAGES = ("numpy", "pandas", "matplotlib", "sklearn", "torch")


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"System: {platform.system()} {platform.release()}")
    missing: list[str] = []

    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {package}: {version}")
        except ImportError:
            missing.append(package)
            print(f"[MISSING] {package}")

    if missing:
        print("\n请运行: pip install -r requirements.txt")
        return 1

    torch = importlib.import_module("torch")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("\n环境检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
