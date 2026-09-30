"""Device policy for Apple Silicon. Imported by every later phase.

Rules (task Phase 0.2):
  * prefer MPS, then CUDA, fall back to CPU with the reason PRINTED; HADES_DEVICE=cuda|cpu|mps overrides
  * PYTORCH_ENABLE_MPS_FALLBACK=1 -- several ops have no MPS kernel and hard-fail otherwise
  * float32 everywhere; MPS has no float64
  * DataLoader num_workers=0 by default
  * seed torch / numpy / random, and note that MPS is not bit-deterministic
"""
import os
# must be set BEFORE torch is imported, or the fallback is not registered
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import random
import numpy as np
import torch

DTYPE = torch.float32
NUM_WORKERS = 0


def get_device(verbose: bool = True) -> torch.device:
    forced = os.environ.get("HADES_DEVICE", "").strip().lower()
    if forced:
        # explicit override (cuda | cpu | mps) -- e.g. a 4 GB CUDA card that cannot hold the full-panel forward
        dev = torch.device(forced)
        why = f"FORCED by HADES_DEVICE={forced}"
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        why = "MPS available and built"
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
        why = f"CUDA available ({torch.cuda.get_device_name(0)})"
    elif not torch.backends.mps.is_built():
        dev = torch.device("cpu")
        why = "FALLBACK TO CPU: this torch build has no MPS support and no CUDA device"
    else:
        dev = torch.device("cpu")
        why = "FALLBACK TO CPU: MPS is built but not available (no Apple GPU / unsupported macOS)"
    if verbose:
        print(f"[device] {dev.type.upper()} -- {why}")
        print(f"[device] torch {torch.__version__} | dtype {DTYPE} | "
              f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')} | "
              f"num_workers={NUM_WORKERS}")
    return dev


def seed_everything(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_t(a, device=None, dtype=DTYPE) -> torch.Tensor:
    """numpy/pandas -> tensor, forced to float32.

    pandas hands back float64; passing that to MPS either falls back to CPU
    silently or raises. Never let a float64 reach the device.
    """
    if isinstance(a, torch.Tensor):
        t = a
    else:
        t = torch.from_numpy(np.ascontiguousarray(np.asarray(a)))
    if t.is_floating_point():
        t = t.to(dtype)
    return t.to(device) if device is not None else t


def peak_rss_gb() -> float:
    import sys
    if sys.platform == "win32":
        # `resource` is Unix-only; the Windows equivalent is the process's peak working set
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        c = PMC(); c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        gpmi = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        gpmi.restype = wintypes.BOOL
        ok = gpmi(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / (1 << 30) if ok else float("nan")
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 30) if sys.platform == "darwin" else r / (1 << 20)
