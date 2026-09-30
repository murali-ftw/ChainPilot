"""Phase 16 Stage 1 -- G1 inertness. Run once per code state; compare the byte hashes.

    python ml/eval/phase16_g1.py --code <repo root to import ml/ from> --out <json>
    python ml/eval/phase16_g1.py --compare a.json b.json

With every Phase 16 feature disabled, on a fixed batch and seed:
  (a) TCN output                                   (arrival's TCN, seed-7 init, one window, all 16,072 channels)
  (b) SHARE-lite (arrival h4) and HeteroMP (capacity h4) HeadNet forward outputs
  (c) a 3-epoch seed-7 training run (arrival lite h4 and capacity mp h4): every weight of the restored-best model
must be BIT-IDENTICAL between the branch point (90a38ed) and the Phase 16 branch.

CPU, float32. MPS is not bit-repeatable even against itself (a SHARE-lite forward run twice in one process differs at
1e-8: index_put_with_accumulate), so a bit-level gate on MPS would fail for reasons that have nothing to do with
Phase 16. CPU is forced by reporting MPS unavailable before torch's device probe runs; no repo file is edited.

--perturb share : wraps SHARELayer.forward to add 1e-6 to its output -- the gate must FIRE on arrival's (b) and (c) and
                  stay silent on the TCN and on capacity.
"""
from __future__ import annotations
import os, sys, json, hashlib, argparse


def h(t):
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main_run(code, out, perturb=None):
    import torch
    orig = torch.backends.mps.is_available
    no_mps = lambda: False                                     # force CPU before device.get_device() is called
    no_mps.__wrapped__ = getattr(orig, "__wrapped__", orig)    # torch._dynamo reads this attribute
    torch.backends.mps.is_available = no_mps
    ml = os.path.join(code, "ml")
    sys.path[:0] = [ml] + [os.path.join(ml, d) for d in ("train", "data", "models", "eval")]
    torch.set_num_threads(os.cpu_count())
    import numpy as np, subprocess
    import phase5_heads as P5, loop as L, folds
    assert P5.DEV.type == "cpu", P5.DEV
    if perturb == "share":
        import share
        f0 = share.SHARELayer.forward
        share.SHARELayer.forward = lambda self, *a, **k: f0(self, *a, **k) + 1e-6
    commit = subprocess.check_output(["git", "-C", code, "rev-parse", "--short", "HEAD"], text=True).strip()
    R = dict(code=code, commit=commit, perturb=perturb, device=str(P5.DEV), hashes={})
    for task, arch in (("arrival_week", "lite"), ("capacity_strain", "mp")):
        lb = P5.labels("v8", task); tr, va, te = folds.fixed_split(lb.snapshot_date)
        D = P5.device_inputs("v8", np.sort(lb.snapshot_date[tr].unique()), False)
        t0 = P5.t0_of(D["W"], np.sort(lb.snapshot_date[tr].unique())[10])
        L.seed_all(7)
        m = P5.HeadNet(D["X"].shape[2], task, arch, 4).to(P5.DEV).eval()
        sl = slice(t0 - P5.WIN + 1, t0 + 1)
        with torch.no_grad():
            if task == "arrival_week":
                R["hashes"]["a_tcn"] = h(m.tcn(D["X"][:, sl]))
            R["hashes"][f"b_{task}_forward"] = h(P5.forward(m, D, t0, torch.arange(len(D["W"]["cidx"]))))
        cfg = dict(task=task, world="v8", seed=7, arch=arch, depth=4, lr=0.00025, gate=False, wsla=False,
                   fill_loss="rps", max_epochs=3, patience=8)
        model, *_ , log = L.train(cfg)
        sd = model.state_dict()
        R["hashes"][f"c_{task}_weights"] = hashlib.sha256(b"".join(
            k.encode() + sd[k].detach().cpu().contiguous().numpy().tobytes() for k in sorted(sd))).hexdigest()
        R[f"c_{task}_losses"] = log["losses"]; R[f"c_{task}_vals"] = log["vals"]
        print(task, {k: v[:12] for k, v in R["hashes"].items()}, log["losses"], flush=True)
    json.dump(R, open(out, "w"), indent=1)


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    return {k: A["hashes"][k] == B["hashes"].get(k) for k in A["hashes"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--code"); ap.add_argument("--out"); ap.add_argument("--perturb", default=None)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        print(json.dumps(compare(*a.compare), indent=1))
    else:
        main_run(a.code, a.out, a.perturb)
