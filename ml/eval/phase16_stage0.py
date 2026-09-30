"""Phase 16 Stage 0 -- measurement on the shipped SHARE-lite arrival encoder. FORWARD PASSES ONLY (plus one gradient
probe for the reachability map). No training.

Bundles read, exact names, no globs: ml/artifacts/bundles/arrival_week/v8_lite_h4_lr0.00025_s{7,17,27,37,47}.

0.0 REACHABILITY. For every (layer l, relation r, target node type): gate the messages of that path by a scalar g = 1
    and take d(readout)/dg, readout = the arrival head's output on every channel. Non-zero -> the path reaches the
    readout. The analytic expectation is also written down, and the two must agree.
0.1 ATTENTION ENTROPY on reaching paths. H_i = -sum_j alpha_ij log alpha_ij over the softmax the model ACTUALLY
    computes: SHARE-lite's softmax is relation-blind, over i's whole incoming neighbourhood (share.py). Normalised by
    log|N(i)| where |N(i)| > 1. Also the per-relation view the brief specifies: |N_r(i)| per node, and the fraction with
    |N_r(i)| = 1. DECLARED BEFORE RUNNING: "near zero" means H_norm < 0.10; "near uniform" means H_norm > 0.90.
    Gate 0.1 (P1): channels near zero on the MAJORITY (> 50%) of reaching (node, layer) paths, suppliers not.
0.2 EFFECTIVE MESSAGE MAGNITUDE s_r^l = mean_i || sum_{j in N_r(i)} alpha_ij W_r h_j ||_2, normalised within (layer,
    node type) over the relations that reach it. 5-seed [min, max]. Gate 0.2: part's share lowest at channel nodes
    (the only node type with more than one incoming relation) at every reaching layer <-> agrees with C3.
0.3 DEGREE SPECTRUM from the graph itself.
Test split, every test snapshot, 5 seeds.
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np, torch, torch.nn.functional as F
import phase5_heads as P5, loop as L, folds
from phase16_encoders import edge_softmax, seed_entities

SEEDS = (7, 17, 27, 37, 47)
BUNDLES = {s: os.path.join(C.BUND, "arrival_week", f"v8_lite_h4_lr0.00025_s{s}") for s in SEEDS}
TYPES = ("channel", "supplier", "part", "plant")
REL_NAMES = {0: "channel->supplier", 1: "channel->part", 2: "channel->plant",
             3: "supplier->channel", 4: "part->channel", 5: "plant->channel"}
TARGET = {0: "supplier", 1: "part", 2: "plant", 3: "channel", 4: "channel", 5: "channel"}
NEAR_ZERO, NEAR_UNIFORM = 0.10, 0.90
FIG = os.path.join(C.REPO, "reports", "part2", "figures")


def node_types(W):
    src, dst, rel, n_nodes, offs = W["graph"]
    NCH = len(W["cidx"]) if isinstance(W["cidx"], (list, np.ndarray)) else W["rel_t"][0].shape[0]
    t = np.zeros(n_nodes, int)
    for k, (o, n) in enumerate(zip(offs, W["rel_sizes"])):
        t[o:o + n] = k + 1
    return t, NCH


def instrumented(m, D, t0, gates=None):
    """SHARE-lite's forward, op for op, recording alpha and the per-edge messages. gates [L, 6] scales messages."""
    W = D["W"]; src, dst, rel, n_nodes, offs = W["graph"]
    sl = slice(t0 - P5.WIN + 1, t0 + 1)
    hc = m.tcn(D["X"][:, sl])
    enc = m.enc; NCH = hc.shape[0]
    h = seed_entities(enc.inp(hc), W["rel_t"], offs, n_nodes)
    recs = []
    for l, layer in enumerate(enc.layers[:m.depth]):
        Hr = torch.einsum("nd,rde->rne", h, layer.rel_weights())
        hs, hd = Hr[rel, src], Hr[rel, dst]
        e = F.leaky_relu((hd * layer.att_dst).sum(-1) + (hs * layer.att_src).sum(-1), layer.slope)
        alpha = edge_softmax(e, dst, n_nodes)
        msg = alpha.unsqueeze(-1) * hs
        if gates is not None:
            msg = msg * gates[l][rel].unsqueeze(-1)
        agg = torch.zeros(n_nodes, layer.dim, device=h.device, dtype=h.dtype).index_add_(0, dst, msg)
        h = F.relu(layer.self_loop(h) + agg)
        recs.append(dict(alpha=alpha, msg=msg))
    return h[:NCH], recs, hc


def reachability(m, D, t0, types):
    W = D["W"]; src, dst, rel, n_nodes, offs = W["graph"]
    g = torch.ones(m.depth, 6, device=P5.DEV, requires_grad=True)
    out, _, _ = instrumented(m, D, t0, g)
    z = m.head(out)
    torch.manual_seed(0)
    (z.float() * torch.randn_like(z.float())).sum().backward()
    G = g.grad.abs().cpu().numpy()
    rows = []
    for l in range(m.depth):
        for r in range(6):
            analytic = not (l == m.depth - 1 and r in (0, 1, 2))   # last-layer messages into entities: no later hop
            rows.append(dict(layer=l + 1, relation=REL_NAMES[r], target=TARGET[r], reaches=bool(G[l, r] > 0),
                             analytic=analytic, grad=float(G[l, r])))
    return rows


def stats(x):
    x = np.asarray(x, float)
    if not len(x):
        return None
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), p10=float(np.quantile(x, .1)),
                p90=float(np.quantile(x, .9)))


def degree_spectrum(D, types, NCH):
    W = D["W"]; src, dst, rel, n_nodes, offs = W["graph"]
    d, r_ = dst.cpu().numpy(), rel.cpu().numpy()
    out = {}
    for r in range(6):
        cnt = np.bincount(d[r_ == r], minlength=n_nodes)
        tgt = TYPES.index(TARGET[r])
        v = cnt[types == tgt]
        out[REL_NAMES[r]] = dict(target=TARGET[r], n_nodes=int(len(v)), min=int(v.min()),
                                 p10=float(np.quantile(v, .1)), median=float(np.median(v)),
                                 p90=float(np.quantile(v, .9)), max=int(v.max()), n_zero=int((v == 0).sum()),
                                 n_over_1000=int((v > 1000).sum()))
    tot = np.bincount(d, minlength=n_nodes)
    out["whole_neighbourhood"] = {t: dict(min=int(tot[types == k].min()), median=float(np.median(tot[types == k])),
                                          max=int(tot[types == k].max()), n_over_1000=int((tot[types == k] > 1000).sum()))
                                  for k, t in enumerate(TYPES)}
    # B-2 degeneracy check: which per-relation neighbourhoods have |N_r(i)| <= k
    slot = d * 6 + r_
    per_slot = np.bincount(slot, minlength=n_nodes * 6)
    edge_deg = per_slot[slot]
    out["bypass_edges"] = {f"k{k}": dict(n_edges=int((edge_deg <= k).sum()),
                                         target_types=sorted({TYPES[types[i]] for i in np.unique(d[edge_deg <= k])}))
                           for k in (1, 2)}
    out["bypass_k1_equals_k2"] = bool(np.array_equal(edge_deg <= 1, edge_deg <= 2))
    return out


def per_seed(seed, D, lb, te, types, NCH):
    B = L.load_bundle(BUNDLES[seed])
    assert B["cfg"]["arch"] == "lite" and B["cfg"]["depth"] == 4 and B["cfg"]["seed"] == seed
    assert np.allclose(B["norm"]["mu"], D["norm_mu"]) and np.allclose(B["norm"]["sd"], D["norm_sd"])
    m = L._materialise(B, D)
    W = D["W"]; src, dst, rel, n_nodes, offs = W["graph"]
    dst_np, rel_np = dst.cpu().numpy(), rel.cpu().numpy()
    deg = np.bincount(dst_np, minlength=n_nodes)
    snaps = np.sort(lb.snapshot_date[te].unique())
    reach = reachability(m, D, P5.t0_of(W, snaps[0]), types)
    live = {(x["layer"], x["target"]) for x in reach if x["reaches"]}
    has_r = {r: np.bincount(dst_np[rel_np == r], minlength=n_nodes) > 0 for r in range(6)}
    H = {(l, t): [] for l in range(1, 5) for t in TYPES}
    S = {(l, t, r): [] for l in range(1, 5) for t in TYPES for r in range(6)}
    max_err = 0.0
    with torch.no_grad():
        for s in snaps:
            t0 = P5.t0_of(W, s)
            out, recs, _ = instrumented(m, D, t0)
            ref = m.enc(m.tcn(D["X"][:, t0 - P5.WIN + 1:t0 + 1]), W["rel_t"], offs, n_nodes, src, dst, rel, depth=4)
            max_err = max(max_err, float((out - ref).abs().max()))
            for l, rc in enumerate(recs, 1):
                a = rc["alpha"]
                ent = torch.zeros(n_nodes, device=a.device).index_add_(0, dst, -(a * torch.log(a.clamp(min=1e-30))))
                ent = ent.cpu().numpy()
                for k, t in enumerate(TYPES):
                    if (l, t) not in live:
                        continue
                    sel = (types == k) & (deg > 1)
                    H[(l, t)].append(ent[sel] / np.log(deg[sel]))
                    for r in range(6):
                        if TARGET[r] != t:
                            continue
                        e_r = rel == r
                        v = torch.zeros(n_nodes, rc["msg"].shape[1], device=a.device).index_add_(
                            0, dst[e_r], rc["msg"][e_r]).norm(dim=-1).cpu().numpy()
                        S[(l, t, r)].append(float(v[(types == k) & has_r[r]].mean()))
    Hs = {f"L{l}|{t}": np.concatenate(v) for (l, t), v in H.items() if v}
    Sm = {f"L{l}|{t}|{REL_NAMES[r]}": float(np.mean(v)) for (l, t, r), v in S.items() if v}
    del m
    return reach, Hs, Sm, max_err, len(snaps)


def svg_figure(Hpool, path):
    """ONE figure: normalised attention entropy by node type (pooled over reaching layers and 5 seeds)."""
    bins = np.linspace(0, 1, 21)
    W_, H_ = 760, 420; pad = 60; cw = (W_ - 2 * pad) / 4
    colors = dict(channel="#1f77b4", supplier="#d62728", part="#2ca02c", plant="#9467bd")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{H_}" font-family="sans-serif" font-size="11">',
           f'<rect width="{W_}" height="{H_}" fill="white"/>',
           f'<text x="{W_/2}" y="20" text-anchor="middle" font-size="14">Phase 16 Stage 0.1 — normalised attention entropy '
           f'H/log|N(i)| by node type</text>',
           f'<text x="{W_/2}" y="36" text-anchor="middle" fill="#555">shipped SHARE-lite arrival h4, 5 seeds, every test '
           f'snapshot, reaching (node, layer) paths only; dashed lines at 0.10 and 0.90</text>']
    for k, t in enumerate(TYPES):
        x0 = pad + k * cw; y0, y1 = 60, H_ - 50
        v = Hpool[t]
        c, _ = np.histogram(v, bins); f = c / max(c.sum(), 1)
        out.append(f'<text x="{x0 + cw/2}" y="{y0 - 6}" text-anchor="middle" font-weight="bold">{t} (n={len(v):,})</text>')
        out.append(f'<line x1="{x0+10}" y1="{y1}" x2="{x0+cw-10}" y2="{y1}" stroke="#333"/>')
        bw = (cw - 20) / 20
        for i, fi in enumerate(f):
            hh = fi * (y1 - y0)
            out.append(f'<rect x="{x0+10+i*bw:.1f}" y="{y1-hh:.1f}" width="{bw-1:.1f}" height="{hh:.1f}" fill="{colors[t]}"/>')
        for q in (0.1, 0.9):
            xq = x0 + 10 + q * (cw - 20)
            out.append(f'<line x1="{xq:.1f}" y1="{y0}" x2="{xq:.1f}" y2="{y1}" stroke="#888" stroke-dasharray="4,3"/>')
        for q in (0, 0.5, 1):
            out.append(f'<text x="{x0+10+q*(cw-20):.1f}" y="{y1+14}" text-anchor="middle">{q:g}</text>')
        out.append(f'<text x="{x0 + cw/2}" y="{y1+30}" text-anchor="middle" fill="#555">median {np.median(v):.3f}, '
                   f'&gt;0.9: {np.mean(v > 0.9):.0%}</text>')
    out.append(f'<text x="14" y="{H_/2}" transform="rotate(-90 14 {H_/2})" text-anchor="middle">fraction of paths</text>')
    out.append("</svg>")
    open(path, "w").write("\n".join(out))


def main():
    st = C.require_clean()
    lb = P5.labels("v8", "arrival_week"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    D = P5.device_inputs("v8", np.sort(lb.snapshot_date[tr].unique()), False)
    types, NCH = node_types(D["W"])
    R = dict(stamp=st, bundles=BUNDLES, thresholds=dict(near_zero=NEAR_ZERO, near_uniform=NEAR_UNIFORM))
    R["degree_spectrum"] = degree_spectrum(D, types, NCH)
    print(json.dumps(R["degree_spectrum"], indent=1), flush=True)
    perseed = {}
    for s in SEEDS:
        reach, Hs, Sm, err, n_snap = per_seed(s, D, lb, te, types, NCH)
        perseed[s] = (reach, Hs, Sm)
        print(f"seed {s}: {n_snap} test snapshots, instrumented-vs-incumbent max |diff| {err:.2e}", flush=True)
        R.setdefault("instrumented_max_abs_diff", {})[s] = err
        R["n_test_snapshots"] = n_snap
    # 0.0
    reach7 = perseed[SEEDS[0]][0]
    agree = all(all(a["reaches"] == b["reaches"] for a, b in zip(reach7, perseed[s][0])) for s in SEEDS)
    R["reachability"] = dict(map=reach7, same_on_all_seeds=agree,
                             analytic_equals_empirical=all(x["reaches"] == x["analytic"] for x in reach7),
                             n_paths=len(reach7), n_excluded=sum(not x["reaches"] for x in reach7),
                             excluded=[x for x in reach7 if not x["reaches"]],
                             excluded_node_paths_per_snapshot=int(sum((types == TYPES.index(x["target"])).sum()
                                                                      for x in reach7 if not x["reaches"])),
                             excluded_params=int(sum(not x["reaches"] for x in reach7)) * 128 * 128)
    json.dump(dict(stamp=st, map=reach7), open(os.path.join(C.ART, "phase16_reachability.json"), "w"), indent=1)
    # 0.1
    ent = {}
    Hpool = {t: [] for t in TYPES}
    for key in perseed[SEEDS[0]][1]:
        per = [perseed[s][1][key] for s in SEEDS]
        allv = np.concatenate(per)
        l, t = key.split("|")
        Hpool[t].append(allv)
        ent[key] = dict(pooled=stats(allv), frac_near_zero=float(np.mean(allv < NEAR_ZERO)),
                        frac_near_uniform=float(np.mean(allv > NEAR_UNIFORM)),
                        seed_band_median=[float(min(np.median(p) for p in per)), float(max(np.median(p) for p in per))],
                        seed_band_frac_near_uniform=[float(min(np.mean(p > NEAR_UNIFORM) for p in per)),
                                                     float(max(np.mean(p > NEAR_UNIFORM) for p in per))])
    Hpool = {t: np.concatenate(v) for t, v in Hpool.items()}
    ds = R["degree_spectrum"]
    R["entropy"] = dict(by_layer_type=ent,
                        by_type={t: dict(pooled=stats(v), frac_near_zero=float(np.mean(v < NEAR_ZERO)),
                                         frac_near_uniform=float(np.mean(v > NEAR_UNIFORM))) for t, v in Hpool.items()},
                        frac_whole_neighbourhood_eq_1={t: 0.0 for t in TYPES},
                        frac_per_relation_eq_1={"channel": 1.0, "supplier": 0.0, "part": 0.0, "plant": 0.0})
    assert all(ds[r]["min"] > 1 for r in ("channel->supplier", "channel->part", "channel->plant"))
    assert all(ds[r]["max"] == 1 and ds[r]["min"] == 1 for r in ("supplier->channel", "part->channel", "plant->channel"))
    ch = R["entropy"]["by_type"]["channel"]["frac_near_zero"]; su = R["entropy"]["by_type"]["supplier"]["frac_near_zero"]
    R["gate_0_1"] = dict(channel_frac_near_zero=ch, supplier_frac_near_zero=su,
                         P1_holds=bool(ch > 0.5 and su <= 0.5),
                         verdict="P1 HOLDS" if (ch > 0.5 and su <= 0.5) else "P1 WRONG")
    os.makedirs(FIG, exist_ok=True)
    svg_figure(Hpool, os.path.join(FIG, "phase16_entropy.svg"))
    # 0.2
    mag = {}
    for key in perseed[SEEDS[0]][2]:
        l, t, r = key.split("|")
        mag.setdefault(f"{l}|{t}", {})[r] = [perseed[s][2][key] for s in SEEDS]
    share = {}
    for lt, rels in mag.items():
        tot = np.sum([v for v in rels.values()], 0)
        share[lt] = {r: dict(raw_band=[float(min(v)), float(max(v))],
                             share_band=[float(min(np.array(v) / tot)), float(max(np.array(v) / tot))],
                             share_mean=float(np.mean(np.array(v) / tot))) for r, v in rels.items()}
    R["message_magnitude"] = share
    rank = {}
    for l in range(1, 5):
        sh = share[f"L{l}|channel"]
        order = sorted(sh, key=lambda r: sh[r]["share_mean"])
        lowest = order[0]
        # disjoint-lowest: part's max share below every other relation's min share
        disj = all(sh["part->channel"]["share_band"][1] < sh[r]["share_band"][0] for r in sh if r != "part->channel")
        rank[f"L{l}"] = dict(order_low_to_high=order, part_lowest=lowest == "part->channel", part_lowest_disjoint=disj)
    R["gate_0_2"] = dict(per_layer=rank, agrees_with_C3=all(v["part_lowest"] for v in rank.values()),
                         agrees_disjointly=all(v["part_lowest_disjoint"] for v in rank.values()))
    print(json.dumps({k: R[k] for k in ("gate_0_1", "gate_0_2")}, indent=1))
    print(json.dumps(R["entropy"]["by_type"], indent=1))
    print(json.dumps({k: R["reachability"][k] for k in ("same_on_all_seeds", "analytic_equals_empirical", "n_paths",
                                                        "n_excluded", "excluded_node_paths_per_snapshot",
                                                        "excluded_params")}, indent=1))
    print(C.dump(R, "phase16_stage0.json"))


if __name__ == "__main__":
    main()
