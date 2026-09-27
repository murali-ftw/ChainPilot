"""Phase 12 C3 -- the part-relation ablation must be shown REAL before any arm trains.

  1. assert_relation_dropped fires on four constructed wrong inputs and passes on the real ablation;
  2. gradients: after one backward pass on the ablated world, the part relation's weights receive EXACTLY zero
     gradient and every other relation's weights receive non-zero gradient (arrival SHARE-lite, capacity HeteroMP);
  3. the ablation bites: the shipped full-graph h4 model's outputs CHANGE when it is fed the ablated world.
"""
from __future__ import annotations
import phase12_common as C
import numpy as np, torch
import phase5_heads as P5, folds, loop as L
import temporal_share as TS
from graph_control import falsify_relation_drop, relation_dropped_world


def grads(task, arch, W):
    D = P5.device_inputs("v8", np.sort(P5.labels("v8", task).snapshot_date.unique())[:44], False)
    m = P5.HeadNet(D["X"].shape[2], task, arch, 4).to(P5.DEV)
    Dab = dict(D, W=relation_dropped_world(D["W"], "part", P5.DEV))
    t0 = P5.t0_of(Dab["W"], P5.labels("v8", task).snapshot_date.iloc[0])
    idx = torch.arange(512, device=P5.DEV)
    z = P5.forward(m, Dab, t0, idx)
    z.float().pow(2).mean().backward()
    out = {}
    if arch == "lite":
        for l, layer in enumerate(m.enc.layers):
            g = layer.W_r.grad.abs().sum((1, 2)).cpu().numpy()
            out[f"layer{l}"] = [float(x) for x in g]
        zero_part = all(v[1] == 0 and v[4] == 0 for v in out.values())
        others = all(all(v[r] > 0 for r in (0, 2, 3, 5)) for v in out.values())
    else:
        for r in range(m.enc.rounds):
            out[f"round{r}"] = [float(sum((p.grad.abs().sum() if p.grad is not None else torch.tensor(0.)).item()
                                          for p in list(m.enc.up[r][j].parameters()) + list(m.enc.down[r][j].parameters())))
                                for j in range(3)]
        zero_part = all(v[1] == 0 for v in out.values())
        others = all(v[0] > 0 and v[2] > 0 for v in out.values())
    return dict(per_relation_grad=out, part_grad_exactly_zero=zero_part, other_relations_nonzero=others)


def bites(task, bundle):
    B = L.load_bundle(bundle)
    lb = P5.labels("v8", task)
    tr, va, te = folds.fixed_split(lb.snapshot_date)
    D = P5.device_inputs("v8", np.sort(lb.snapshot_date[tr].unique()), False)
    m = L._materialise(B, D)
    Dab = dict(D, W=relation_dropped_world(D["W"], "part", P5.DEV))
    s = np.sort(lb.snapshot_date[te].unique())[0]
    t0 = P5.t0_of(D["W"], s); idx = torch.arange(len(D["W"]["cidx"]), device=P5.DEV)
    with torch.no_grad():
        a = P5.forward(m, D, t0, idx).float().cpu().numpy(); b = P5.forward(m, Dab, t0, idx).float().cpu().numpy()
    return dict(mean_abs_output_change=float(np.abs(a - b).mean()), frac_outputs_changed=float((np.abs(a - b) > 1e-6).mean()))


def main():
    st = C.require_clean()
    W = TS.load_world("v8")
    res = dict(stamp=st, falsification=falsify_relation_drop(W, "part"))
    for k, v in res["falsification"].items():
        print(k, "->", v)
    res["gradients"] = dict(arrival_lite=grads("arrival_week", "lite", W), capacity_mp=grads("capacity_strain", "mp", W))
    for k, v in res["gradients"].items():
        print(k, v["part_grad_exactly_zero"], v["other_relations_nonzero"])
    res["bites"] = dict(arrival=bites("arrival_week", f"{C.BUND}/arrival_week/v8_lite_h4_lr0.00025_s7"),
                        capacity=bites("capacity_strain", f"{C.BUND}/capacity_strain/v8_mp_h4_lr0.00025_s7"))
    print(res["bites"])
    ok = (all(v.startswith("FIRES") for k, v in res["falsification"].items() if k != "the real ablation")
          and all(g["part_grad_exactly_zero"] and g["other_relations_nonzero"] for g in res["gradients"].values())
          and all(b["frac_outputs_changed"] > 0.5 for b in res["bites"].values()))
    res["all_checks_pass"] = ok
    print("ALL CHECKS", "PASS" if ok else "FAIL")
    print(C.dump(res, "phase12_c3_checks.json"))
    assert ok


if __name__ == "__main__":
    main()
