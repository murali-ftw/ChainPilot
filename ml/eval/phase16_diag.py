"""Phase 16 diagnostics -- reported regardless of the metric outcome. Forward passes on the TEST split, 5 seeds each,
reaching paths only (Stage 0.0's map: layer-4 messages into entity nodes are excluded).

Candidate A (arrival):
  A-1 the learned attention weights on the six Delta dimensions (att_traj), per layer, per depth, per seed;
  A-2 attention entropy by node type, A1 vs A0 (same statistic and thresholds as Stage 0.1);
  A-3 WHAT vs SCALE: in each trained A1 model, zero Delta at inference and count the channel paths whose top-attended
      entity changes (a change of WHAT attention does), and the mean |alpha_A1 - alpha_A1(Delta=0)| (its size).
Candidate B:
  B-1 per node type, the mean norm of the incoming aggregate, B1 vs B0 (arrival: per layer; capacity: the channel ->
      entity aggregate per entity type and round, and the aggregate each channel receives);
  B-2 PLANT nodes: the share of the projected multi-aggregate's variance carried by each aggregator (mean / max / min /
      std, scalers pooled). Variance over plant nodes x test snapshots (x layers or rounds), summed over dimensions.
      If mean carries the majority, averaging was not destroying information: P4 is WRONG;
  B-3 the selected k (phase16_k_selection.json).
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np, torch, torch.nn.functional as F
import phase5_heads as P5, loop as L, folds
import phase16_heads as P16
from phase16_encoders import edge_softmax, seed_entities, AGGS
from phase16_stage0 import TYPES, instrumented as inc_instrumented, node_types, NEAR_ZERO, NEAR_UNIFORM, stats

SEEDS = (7, 17, 27, 37, 47)
ARR = {"A0": "v8_lite_h4_lr0.00025_s{s}", "A1": "v8_lite_h4_lr0.00025_s{s}_encshare_traj",
       "B1": "v8_lite_h4_lr0.00025_s{s}_encshare_pna_pnammms"}
CAP = {"B0": "v8_mp_h4_lr0.00025_s{s}", "B1": "v8_mp_h4_lr0.00025_s{s}_encheteromp_pna_pnammms"}


def load(task, pat, s, D):
    d = os.path.join(C.BUND, task, pat.format(s=s))
    B = L.load_bundle(d)
    assert B["cfg"]["seed"] == s
    return L._materialise(B, D), B


def traj_instrumented(m, D, t0, zero_delta=False):
    W = D["W"]; src, dst, rel, n_nodes, offs = W["graph"]
    hc = m.tcn(D["X"][:, t0 - P5.WIN + 1:t0 + 1])            # fires the TCN hooks
    enc = m.enc
    h = seed_entities(enc.inp(hc), W["rel_t"], offs, n_nodes)
    delta = enc.delta(W["rel_t"], offs, n_nodes, src, dst)
    if zero_delta:
        delta = torch.zeros_like(delta)
    alphas = []
    for layer in enc.layers[:m.depth]:
        e, hs = layer.logits(h, src, dst, rel, delta)
        a = edge_softmax(e, dst, n_nodes)
        agg = torch.zeros(n_nodes, layer.dim, device=h.device, dtype=h.dtype).index_add_(0, dst, a.unsqueeze(-1) * hs)
        h = F.relu(layer.self_loop(h) + agg)
        alphas.append(a)
    return alphas


def entropy_of(alphas, dst, n_nodes, deg, types, live):
    out = {t: [] for t in TYPES}
    for l, a in enumerate(alphas, 1):
        ent = torch.zeros(n_nodes, device=a.device).index_add_(0, dst, -(a * torch.log(a.clamp(min=1e-30)))).cpu().numpy()
        for k, t in enumerate(TYPES):
            if (l, t) in live:
                sel = (types == k) & (deg > 1)
                out[t].append(ent[sel] / np.log(deg[sel]))
    return {t: np.concatenate(v) for t, v in out.items() if v}


def ent_summary(H):
    return {t: dict(pooled=stats(v), frac_near_zero=float(np.mean(v < NEAR_ZERO)),
                    frac_near_uniform=float(np.mean(v > NEAR_UNIFORM))) for t, v in H.items()}


def top_change(a1, a0, dst, NCH):
    """Fraction of channel destinations whose arg-max incoming edge differs between two alpha vectors."""
    d = dst.cpu().numpy(); ch = d < NCH
    order = np.argsort(d[ch], kind="stable")
    e1 = a1.cpu().numpy()[ch][order].reshape(-1, 3); e0 = a0.cpu().numpy()[ch][order].reshape(-1, 3)
    return float(np.mean(e1.argmax(1) != e0.argmax(1))), float(np.mean(np.abs(e1 - e0)))


def block_contrib(proj, wide, d):
    """proj: Linear(12d -> d); wide [n, 12d] laid out (aggregator, scaler, dim). -> {aggregator: [n, d] contribution}."""
    Wt = proj.weight.reshape(d, len(AGGS), 3, d)
    w = wide.reshape(-1, len(AGGS), 3, d)
    return {a: torch.einsum("osi,nsi->no", Wt[:, k], w[:, k]) for k, a in enumerate(AGGS)}


def var_share(contribs):
    v = {a: float(np.var(np.concatenate(x), 0).sum()) for a, x in contribs.items()}
    tot = sum(v.values())
    return {a: v[a] / tot for a in v}


def arrival(R):
    lb = P5.labels("v8", "arrival_week"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    D = P5.device_inputs("v8", np.sort(lb.snapshot_date[tr].unique()), False)
    W = D["W"]; src, dst, rel, n_nodes, offs = W["graph"]
    types, NCH = node_types(W)
    deg = np.bincount(dst.cpu().numpy(), minlength=n_nodes)
    live = {(l, t) for l in range(1, 5) for t in TYPES if not (l == 4 and t != "channel")}
    snaps = np.sort(lb.snapshot_date[te].unique())
    A = dict(att_traj={}, entropy={}, what_vs_scale={})
    for arm in ("A0", "A1"):
        Hs = {t: [] for t in TYPES}
        for s in SEEDS:
            m, B = load("arrival_week", ARR[arm], s, D)
            if arm == "A1":
                A["att_traj"][s] = {f"L{l+1}": [float(x) for x in lay.att_traj.detach().cpu()] for l, lay in
                                    enumerate(m.enc.layers)}
                chg, mad = [], []
            with torch.no_grad():
                for sn in snaps:
                    t0 = P5.t0_of(W, sn)
                    if arm == "A0":
                        _, recs, _ = inc_instrumented(m, D, t0); al = [r["alpha"] for r in recs]
                    else:
                        al = traj_instrumented(m, D, t0)
                        al0 = traj_instrumented(m, D, t0, zero_delta=True)
                        per = [top_change(a1, a0, dst, NCH) for a1, a0 in zip(al, al0)]
                        chg.append([p[0] for p in per]); mad.append([p[1] for p in per])
                    for t, v in entropy_of(al, dst, n_nodes, deg, types, live).items():
                        Hs[t].append(v)
            if arm == "A1":
                A["what_vs_scale"][s] = dict(frac_channel_top_entity_changed_by_layer=list(np.mean(chg, 0)),
                                             mean_abs_alpha_change_by_layer=list(np.mean(mad, 0)))
                P16.teardown(m)
            del m
        A["entropy"][arm] = ent_summary({t: np.concatenate(v) for t, v in Hs.items() if v})
    R["A"] = A
    print("A att_traj", json.dumps(A["att_traj"])[:800], flush=True)
    print("A entropy", {arm: {t: round(v["frac_near_zero"], 3) for t, v in A["entropy"][arm].items()} for arm in A["entropy"]})
    print("A what/scale", json.dumps(A["what_vs_scale"])[:800], flush=True)
    # ---- B, arrival
    Bn = {arm: {} for arm in ("A0", "B1")}
    plant = {a: [] for a in AGGS}
    for arm in ("A0", "B1"):
        for s in SEEDS:
            m, _ = load("arrival_week", ARR[arm], s, D)
            acc = {}
            with torch.no_grad():
                for sn in snaps:
                    t0 = P5.t0_of(W, sn)
                    hc = m.tcn(D["X"][:, t0 - P5.WIN + 1:t0 + 1])
                    h = seed_entities(m.enc.inp(hc), W["rel_t"], offs, n_nodes)
                    S = m.enc.slots(dst, rel, h.dtype) if arm == "B1" else None
                    for l, layer in enumerate(m.enc.layers[:m.depth], 1):
                        if arm == "B1":
                            h_new, parts = layer(h, src, dst, rel, n_nodes, S, return_parts=True)
                            agg = parts["agg"]
                            if l < 4:
                                cb = block_contrib(layer.proj, parts["wide"], layer.dim)
                                pl = (S.slot_dst >= offs[2]) & (S.slot_dst < offs[2] + W["rel_sizes"][2])
                                for a in AGGS:
                                    plant[a].append(cb[a][pl].cpu().numpy())
                        else:
                            Hr = torch.einsum("nd,rde->rne", h, layer.rel_weights())
                            hs, hd = Hr[rel, src], Hr[rel, dst]
                            e = F.leaky_relu((hd * layer.att_dst).sum(-1) + (hs * layer.att_src).sum(-1), layer.slope)
                            a_ = edge_softmax(e, dst, n_nodes)
                            agg = torch.zeros(n_nodes, layer.dim, device=h.device).index_add_(0, dst, a_.unsqueeze(-1) * hs)
                            h_new = F.relu(layer.self_loop(h) + agg)
                        nrm = agg.norm(dim=-1).cpu().numpy()
                        for k, t in enumerate(TYPES):
                            if (l, t) in live:
                                acc.setdefault(f"L{l}|{t}", []).append(float(nrm[types == k].mean()))
                        h = h_new
            Bn[arm][s] = {k: float(np.mean(v)) for k, v in acc.items()}
            del m
    keys = Bn["A0"][SEEDS[0]].keys()
    R["B_arrival"] = dict(
        agg_norm={k: dict(B0=[Bn["A0"][s][k] for s in SEEDS], B1=[Bn["B1"][s][k] for s in SEEDS],
                          ratio_B1_over_B0_mean=float(np.mean([Bn["B1"][s][k] for s in SEEDS]) /
                                                      np.mean([Bn["A0"][s][k] for s in SEEDS]))) for k in keys},
        plant_variance_share=var_share(plant))
    print("B arrival", {k: round(v["ratio_B1_over_B0_mean"], 3) for k, v in R["B_arrival"]["agg_norm"].items()},
          R["B_arrival"]["plant_variance_share"], flush=True)


def capacity(R):
    lb = P5.labels("v8", "capacity_strain"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    D = P5.device_inputs("v8", np.sort(lb.snapshot_date[tr].unique()), False)
    W = D["W"]; snaps = np.sort(lb.snapshot_date[te].unique())
    ent_names = ("supplier", "part", "plant")
    Bn = {"B0": {}, "B1": {}}; plant = {a: [] for a in AGGS}
    for arm in ("B0", "B1"):
        for s in SEEDS:
            m, _ = load("capacity_strain", CAP[arm], s, D); enc = m.enc
            acc = {}
            with torch.no_grad():
                for sn in snaps:
                    t0 = P5.t0_of(W, sn)
                    hc = m.tcn(D["X"][:, t0 - P5.WIN + 1:t0 + 1])
                    for r in range(enc.rounds):
                        msgs = []
                        for j, (idx, n) in enumerate(zip(W["rel_t"], W["rel_sizes"])):
                            mm = enc.up[r][j](hc)
                            if arm == "B1":
                                deg, sc, _ = enc.scalers(j, idx, n, hc.dtype)
                                from phase16_encoders import pna_aggregate
                                wide = pna_aggregate(mm, idx, n, deg, sc)
                                agg = enc.proj[r][j](wide)
                                if j == 2:
                                    cb = block_contrib(enc.proj[r][j], wide, hc.shape[1])
                                    for a in AGGS:
                                        plant[a].append(cb[a].cpu().numpy())
                            else:
                                agg = torch.zeros(n, mm.shape[1], device=hc.device).index_add_(0, idx, mm)
                                cnt = torch.zeros(n, 1, device=hc.device).index_add_(0, idx, torch.ones_like(mm[:, :1]))
                                agg = agg / cnt.clamp(min=1.0)
                            acc.setdefault(f"R{r+1}|{ent_names[j]}", []).append(float(agg.norm(dim=-1).mean()))
                            msgs.append(enc.down[r][j](agg)[idx])
                        nb = torch.stack(msgs, 0).mean(0)
                        acc.setdefault(f"R{r+1}|channel", []).append(float(nb.norm(dim=-1).mean()))
                        g = torch.sigmoid(enc.gate[r](torch.cat([hc, nb], -1)))
                        hc = hc + g * nb
            Bn[arm][s] = {k: float(np.mean(v)) for k, v in acc.items()}
            del m
    keys = Bn["B0"][SEEDS[0]].keys()
    R["B_capacity"] = dict(
        agg_norm={k: dict(B0=[Bn["B0"][s][k] for s in SEEDS], B1=[Bn["B1"][s][k] for s in SEEDS],
                          ratio_B1_over_B0_mean=float(np.mean([Bn["B1"][s][k] for s in SEEDS]) /
                                                      np.mean([Bn["B0"][s][k] for s in SEEDS]))) for k in keys},
        plant_variance_share=var_share(plant))
    print("B capacity", {k: round(v["ratio_B1_over_B0_mean"], 3) for k, v in R["B_capacity"]["agg_norm"].items()},
          R["B_capacity"]["plant_variance_share"], flush=True)


def main():
    st = C.require_clean()
    R = dict(stamp=st)
    p = os.path.join(C.ART, "phase16_k_selection.json")
    R["k_selection"] = json.load(open(p)) if os.path.exists(p) else None
    arrival(R)
    capacity(R)
    print(C.dump(R, "phase16_diag.json"))


if __name__ == "__main__":
    main()
