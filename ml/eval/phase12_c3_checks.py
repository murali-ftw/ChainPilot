"""Phase 12 C3 -- the part-relation ablation must be shown REAL before any arm trains.

  1. assert_relation_dropped fires on four constructed wrong inputs and passes on the real ablation;
  2. gradients: after one backward pass the ablated arm's zero-gradient set is EXACTLY the full model's own
     zero-gradient set plus the part relation's weights (arrival SHARE-lite, capacity HeteroMP). A first version
     demanded non-zero gradient on every other weight and failed: the full model has dead weights of its own;
  3. the ablation bites: the shipped full-graph h4 model's outputs CHANGE when it is fed the ablated world.
"""
from __future__ import annotations
import phase12_common as C
import numpy as np, torch
import phase5_heads as P5, folds, loop as L
import temporal_share as TS
from graph_control import falsify_relation_drop, relation_dropped_world


def _grad_pattern(task, arch, D, world_override=None, seed=0):
    torch.manual_seed(seed)
    m = P5.HeadNet(D["X"].shape[2], task, arch, 4).to(P5.DEV)
    Dx = dict(D, W=world_override) if world_override is not None else D
    t0 = P5.t0_of(Dx["W"], P5.labels("v8", task).snapshot_date.iloc[0])
    z = P5.forward(m, Dx, t0, torch.arange(512, device=P5.DEV))
    z.float().pow(2).mean().backward()
    if arch == "lite":
        return {f"layer{l}": [float(x) for x in layer.W_r.grad.abs().sum((1, 2)).cpu().numpy()]
                for l, layer in enumerate(m.enc.layers)}
    return {f"round{r}": [float(sum((p.grad.abs().sum().item() if p.grad is not None else 0.0)
                                    for p in list(m.enc.up[r][j].parameters()) + list(m.enc.down[r][j].parameters())))
                          for j in range(3)] for r in range(m.enc.rounds)}


def grads(task, arch, W):
    """The ablated arm's zero-gradient set must be EXACTLY the full model's own zero-gradient set plus the part
    relation's weights. (The full model has structurally dead weights of its own: in SHARE-lite's last layer the
    channel->entity messages never reach the channel readout.)"""
    D = P5.device_inputs("v8", np.sort(P5.labels("v8", task).snapshot_date.unique())[:44], False)
    full = _grad_pattern(task, arch, D)
    abl = _grad_pattern(task, arch, D, relation_dropped_world(D["W"], "part", P5.DEV))
    part_ids = (1, 4) if arch == "lite" else (1,)
    zf = {(k, r) for k, v in full.items() for r, x in enumerate(v) if x == 0}
    za = {(k, r) for k, v in abl.items() for r, x in enumerate(v) if x == 0}
    expected = zf | {(k, r) for k in abl for r in part_ids}
    per = 16384 if arch == "lite" else 2 * (64 * 64 + 64)
    return dict(full_zero_grad=sorted(map(list, zf)), ablated_zero_grad=sorted(map(list, za)),
                ablated_zero_set_equals_full_plus_part=(za == expected),
                part_grad_exactly_zero=all((k, r) in za for k in abl for r in part_ids),
                params_dead_in_full_model=len(zf) * per,
                params_newly_dead_by_ablation=len(za - zf) * per)


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
        print(k, {x: v[x] for x in ("ablated_zero_set_equals_full_plus_part", "part_grad_exactly_zero",
                                    "params_dead_in_full_model", "params_newly_dead_by_ablation", "full_zero_grad")})
    res["bites"] = dict(arrival=bites("arrival_week", f"{C.BUND}/arrival_week/v8_lite_h4_lr0.00025_s7"),
                        capacity=bites("capacity_strain", f"{C.BUND}/capacity_strain/v8_mp_h4_lr0.00025_s7"))
    print(res["bites"])
    ok = (all(v.startswith("FIRES") for k, v in res["falsification"].items() if k != "the real ablation")
          and all(g["part_grad_exactly_zero"] and g["ablated_zero_set_equals_full_plus_part"] for g in res["gradients"].values())
          and all(b["frac_outputs_changed"] > 0.5 for b in res["bites"].values()))
    res["all_checks_pass"] = ok
    print("ALL CHECKS", "PASS" if ok else "FAIL")
    print(C.dump(res, "phase12_c3_checks.json"))
    assert ok


if __name__ == "__main__":
    main()
