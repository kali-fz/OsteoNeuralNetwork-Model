"""Gate 1: prove the compute environment actually works.

Run this before anything else. On an AMD RX 7900 XT the whole ROCm-on-Windows
stack either works or it does not, and finding out here costs thirty seconds
instead of surfacing as a cryptic HIP error forty minutes into a training run.

    python scripts/verify_env.py

Exit code 0 means training can proceed on the GPU. Exit code 1 with a GPU
failure means: update the AMD driver to 26.2.2+, or switch to the Kaggle
fallback and stop debugging the local stack.
"""

from __future__ import annotations

import platform
import sys

import _bootstrap  # noqa: F401  (path side effect)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"


def _section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def check_python() -> bool:
    _section("Python")
    version = sys.version_info
    print(f"  version : {platform.python_version()}")
    print(f"  exe     : {sys.executable}")
    print(f"  platform: {platform.platform()}")

    ok = version >= (3, 10)
    print(PASS if ok else FAIL, "Python >= 3.10")
    if version[:2] != (3, 12) and platform.system() == "Windows":
        print(WARN, "AMD's ROCm wheels are cp312-only; local GPU needs exactly Python 3.12")
    if "WindowsApps" in sys.executable:
        print(
            WARN,
            "Microsoft Store Python detected. Its redirected AppData occasionally breaks "
            "native extensions -- if the ROCm install misbehaves, reinstall from python.org.",
        )
    return ok


def check_torch() -> tuple[bool, object | None]:
    _section("PyTorch")
    try:
        import torch
    except ImportError as exc:
        print(FAIL, f"torch not importable: {exc}")
        print("        -> pip install -r requirements-rocm.txt")
        return False, None

    print(f"  torch   : {torch.__version__}")
    print(f"  hip     : {getattr(torch.version, 'hip', None)}")
    print(f"  cuda    : {getattr(torch.version, 'cuda', None)}")
    return True, torch


def check_gpu(torch) -> bool:
    _section("GPU")
    if not torch.cuda.is_available():
        print(FAIL, "no GPU visible to torch")
        print("        ROCm reports through the CUDA API, so cuda_available=False on a")
        print("        machine with a 7900 XT means the ROCm stack is not wired up.")
        print("        -> confirm AMD driver >= 26.2.2, then reinstall requirements-rocm.txt")
        print("        -> or use the Kaggle fallback: notebooks/kaggle_train.ipynb")
        return False

    props = torch.cuda.get_device_properties(0)
    backend = "ROCm/HIP" if torch.version.hip else "CUDA"
    print(f"  device  : {props.name}")
    print(f"  backend : {backend}")
    print(f"  memory  : {props.total_memory / 1024 ** 3:.1f} GB")
    print(f"  count   : {torch.cuda.device_count()}")

    # A real allocation + matmul. `cuda.is_available()` can return True on a
    # stack that faults the moment it is asked to do arithmetic.
    try:
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        c = (a @ b).sum().item()
        torch.cuda.synchronize()
        assert c == c, "matmul produced NaN"  # noqa: S101
        print(PASS, "fp32 matmul on GPU")
    except Exception as exc:  # noqa: BLE001
        print(FAIL, f"GPU matmul failed: {exc}")
        return False

    # bf16 is the configured AMP dtype: same exponent range as fp32, so it needs
    # no GradScaler and cannot silently underflow gradients the way fp16 does.
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            d = a @ b
        torch.cuda.synchronize()
        print(PASS, f"bfloat16 autocast (out dtype={d.dtype})")
    except Exception as exc:  # noqa: BLE001
        print(WARN, f"bfloat16 autocast unavailable ({exc}); set train.amp_dtype: float32")

    del a, b
    torch.cuda.empty_cache()
    return True


def check_libraries() -> bool:
    _section("Libraries")
    ok = True
    for module, hint in [
        ("monai", "pip install monai==1.5.2"),
        ("pydicom", "pip install pydicom"),
        ("numpy", "pip install numpy"),
        ("pandas", "pip install pandas"),
        ("sklearn", "pip install scikit-learn"),
        ("PIL", "pip install pillow"),
        ("cv2", "pip install opencv-python-headless"),
        ("matplotlib", "pip install matplotlib"),
        ("yaml", "pip install pyyaml"),
    ]:
        try:
            mod = __import__(module)
            print(f"  {module:<12} {getattr(mod, '__version__', 'ok')}")
        except ImportError:
            print(FAIL, f"{module} missing -> {hint}")
            ok = False
    return ok


def check_project() -> bool:
    _section("Project")
    try:
        from onnm import CLASS_NAMES, __version__
        from onnm.config import load_config

        cfg = load_config("configs/base.yaml")
        print(f"  onnm       : {__version__}")
        print(f"  classes    : {list(CLASS_NAMES)}")
        print(f"  image_size : {cfg.data.image_size}")
        print(f"  data_root  : {cfg.resolve_path('paths.data_root')}")
        print(PASS, "config loads")
        return True
    except Exception as exc:  # noqa: BLE001
        print(FAIL, f"project import/config failed: {exc}")
        return False


def main() -> int:
    print("=" * 68)
    print("ONNM environment verification")
    print("=" * 68)

    py_ok = check_python()
    torch_ok, torch = check_torch()
    gpu_ok = check_gpu(torch) if torch_ok else False
    libs_ok = check_libraries()
    proj_ok = check_project()

    _section("Summary")
    for name, ok in [
        ("python", py_ok),
        ("torch", torch_ok),
        ("gpu", gpu_ok),
        ("libraries", libs_ok),
        ("project", proj_ok),
    ]:
        print(f"  {name:<12}{'PASS' if ok else 'FAIL'}")

    if py_ok and torch_ok and libs_ok and proj_ok and gpu_ok:
        print("\nGPU training is ready. Next: python scripts/download_btxrd.py")
        return 0
    if py_ok and torch_ok and libs_ok and proj_ok:
        print("\nCPU-only. The pipeline and tests will run; real training will not be")
        print("practical. Fix the driver or use notebooks/kaggle_train.ipynb.")
        return 1
    print("\nEnvironment incomplete -- see FAIL lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
