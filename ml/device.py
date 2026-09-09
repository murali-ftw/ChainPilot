"""Device policy for Apple Silicon. Imported by every later phase.

Rules (task Phase 0.2):
  * prefer MPS, fall back to CPU with the reason PRINTED
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
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        why = "MPS available and built"
    elif not torch.backends.mps.is_built():
        dev = torch.device("cpu")
        why = "FALLBACK TO CPU: this torch build has no MPS support"
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
    import resource, sys
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 30) if sys.platform == "darwin" else r / (1 << 20)
