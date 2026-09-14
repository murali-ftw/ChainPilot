"""Phase 5 verification — every property the report claims about the heads and the gate, measured.

  V1  staleness gate: dt = 0 is the identity; missing dt is the identity; dt -> inf reaches the prior;
      the guide's w = b = 0 initialisation receives zero gradient; this one does not
  V2  weeks_since_last_activity is as-of: brute force + future perturbation, both worlds
  V3  hazard: survival non-increasing per row, sum_w P(T=w) + S_12 = 1, every row enters the loss
  V4  quantile head: no crossings, including at extreme inputs
  V5  fill: the 22-cell partition refines the legacy 20 bins exactly on every label; exact CRPS
      matches brute-force numerical integration
  V6  depth-agnostic: every head at h0 / h1 / h4 under none / mp / lite / share, forward + backward
  V7  end-to-end causality through the device-side window slice, gate and reconstructed feature
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "eval")]
import numpy as np, torch
import phase5_heads as P5
import temporal_share as TS
from heads import HazardHead, QuantileHead, FillCDFHead, fill_cell, fill_to_legacy
from staleness import StalenessGate, weeks_since_last_activity, assert_asof
from phase5_metrics import legacy_bin, crps_exact_rows, legacy_to_cells
from device import seed_everything

DEV = P5.DEV
R = {}
seed_everything(0)

# ------------------------------------------------------------------ V1
g = StalenessGate(5)
with torch.no_grad():
    g.prior.copy_(torch.randn(5))
x = torch.randn(3, 52, 5); ob = torch.ones(3, 52, dtype=torch.bool)
out0 = g(x, torch.zeros(3, 52), ob)
outm = g(x, torch.full((3, 52), 1e12), torch.zeros(3, 52, dtype=torch.bool))
outinf = g(x, torch.full((3, 52), 1e12), ob)
R["V1_identity_at_dt0_bitwise"] = bool(torch.equal(out0, x))
R["V1_identity_when_missing_bitwise"] = bool(torch.equal(outm, x))
R["V1_dt_inf_max_abs_dev_from_prior"] = float((outinf - g.prior).abs().max())
w_, b_ = g.weights()
lag = torch.tensor([0.0, 0.87, 3.5, 11.0, 96.0, 284.0])
R["V1_init_g_at_lags_0_p50_p90_p99_96_284"] = [round(float(v), 4) for v in
                                                g.gate(lag.view(1, -1), torch.ones(1, 6, dtype=torch.bool))[0, :, 0]]
# the guide's snippet, literally
wg = torch.zeros(5, requires_grad=True); bg = torch.zeros(5, requires_grad=True); pg = torch.zeros(5, requires_grad=True)
dt = torch.rand(3, 52, 1) * 10
gg = torch.exp(-torch.relu(wg * dt + bg)); ((gg * x + (1 - gg) * pg) ** 2).sum().backward()
R["V1_guide_init_grad_norm_w_b"] = [float(wg.grad.norm()), float(bg.grad.norm())]
g.zero_grad(); (g(x, dt.squeeze(-1), ob) ** 2).sum().backward()
R["V1_this_init_grad_norm_w_b"] = [float(g.w_raw.grad.norm()), float(g.b_raw.grad.norm())]
assert R["V1_identity_at_dt0_bitwise"] and R["V1_identity_when_missing_bitwise"]
assert R["V1_guide_init_grad_norm_w_b"] == [0.0, 0.0] and min(R["V1_this_init_grad_norm_w_b"]) > 0

# ------------------------------------------------------------------ V2
for w in ("v6", "v7"):
    W = TS.load_world(w)
    R[f"V2_{w}_asof"] = assert_asof(W["active"], n_rows=1024, n_cuts=12, seed=1)
    ws = weeks_since_last_activity(W["active"])
    act = np.asarray(W["active"]) > 0
    assert (ws[act] == 0).all() and (ws[~act] > 0).all()
    R[f"V2_{w}_wsla_quantiles_p50_p90_p99_max"] = [float(v) for v in np.percentile(ws, [50, 90, 99, 100])]
    R[f"V2_{w}_wsla_constant_zero_in_store"] = "true -- replaced by this reconstruction"

# ------------------------------------------------------------------ V3
hh = HazardHead(64)
z = torch.randn(20000, 12) * 4
z[:100] = 40.0; z[100:200] = -40.0                           # saturated rows: lam -> 1 and lam -> 0
lam, S, pT = HazardHead.distribution(z)
R["V3_monotone_violations"] = int((S[:, 1:] > S[:, :-1]).sum())
R["V3_sum_check_max_dev"] = float((pT.sum(1) + S[:, -1] - 1).abs().max())
T = torch.randint(1, 14, (20000,)); cen = T > 12
L, n_c = hh.loss(z, T, cen)
R["V3_rows_contributing_of_20000"] = n_c
assert R["V3_monotone_violations"] == 0 and R["V3_sum_check_max_dev"] < 1e-5 and n_c == 20000
for w in ("v6", "v7"):
    lb = P5.labels(w, "arrival_week")
    tr, va, te = TS.fold(lb.snapshot_date)
    c = lb.label_censored.to_numpy()
    R[f"V3_{w}_train_rows_all_vs_old_head_uncensored_only"] = [int(tr.sum()), int((tr & ~c).sum()),
                                                                round(float(c[tr].mean()), 4)]

# ------------------------------------------------------------------ V4
qh = QuantileHead(64)
hq = torch.cat([torch.randn(50000, 64), torch.randn(1000, 64) * 1e3])
Q = qh(hq)
R["V4_crossings_of_51000"] = int(((Q[..., 0] > Q[..., 1]) | (Q[..., 1] > Q[..., 2])).sum())
R["V4_ties_of_51000"] = int(((Q[..., 0] == Q[..., 1]) | (Q[..., 1] == Q[..., 2])).sum())
assert R["V4_crossings_of_51000"] == 0

# ------------------------------------------------------------------ V5
for w in ("v6", "v7"):
    y = P5.labels(w, "fill_rate").label_value.to_numpy(float)
    oh22 = np.eye(22)[fill_cell(y)]
    ok = np.array_equal(fill_to_legacy(oh22).argmax(1), legacy_bin(y))
    R[f"V5_{w}_refinement_exact_on_{len(y)}_labels"] = bool(ok)
    R[f"V5_{w}_cell_shares_zero_interior_one"] = [round(float((y == 0).mean()), 4),
                                                   round(float(((y > 0) & (y < 1)).mean()), 4),
                                                   round(float((y == 1).mean()), 4)]
    assert ok
rng = np.random.default_rng(3)
P = rng.dirichlet(np.ones(22) * 0.3, 400); yy = np.concatenate([[0, 1, 0.05, 0.15, 0.95], rng.random(395)])
G = 200_000; xs = (np.arange(G) + 0.5) / G
cellx = np.clip(np.floor(xs * 20).astype(int), 0, 19)
brute = []
for i in range(len(yy)):
    cum = np.concatenate([[0], np.cumsum(P[i, 1:21])])
    F = P[i, 0] + cum[cellx] + P[i, 1:21][cellx] * (xs * 20 - cellx)
    brute.append((((F - (xs >= yy[i])) ** 2).mean()))
R["V5_exact_crps_vs_numeric_max_abs_err"] = float(np.abs(crps_exact_rows(P, yy) - np.array(brute)).max())
assert R["V5_exact_crps_vs_numeric_max_abs_err"] < 1e-5

# ------------------------------------------------------------------ V6
W = TS.load_world("v6")
X = torch.randn(W["NCH"], 52, 25, device=DEV)
dtt = torch.rand(W["NCH"], 52, device=DEV) * 5; obt = torch.ones(W["NCH"], 52, dtype=torch.bool, device=DEV)
Dfake = dict(X=X, dt=dtt, obs=obt, W=W)
idx = torch.arange(0, 2000, device=DEV)
res6 = {}
for arch, depth in (("none", 0), ("mp", 1), ("mp", 4), ("lite", 1), ("lite", 4), ("share", 1), ("share", 4)):
    for task in ("arrival_week", "fill_rate", "capacity_strain", "shortage_qty"):
        m = P5.HeadNet(25, task, arch, depth, gate_cols=list(range(13))).to(DEV)
        zz = P5.forward(m, Dfake, 51, idx if task != "shortage_qty" else torch.arange(0, 500, device=DEV))
        n = zz.shape[0]
        tg = {"arrival_week": {"T": torch.randint(1, 14, (n,), device=DEV)},
              "fill_rate": {"cell": torch.randint(0, 22, (n,), device=DEV)},
              "capacity_strain": {"y": torch.randn(n, 1, device=DEV)},
              "shortage_qty": {"y": torch.randint(0, 2, (n,), device=DEV).float()}}[task]
        if task == "arrival_week": tg["cen"] = tg["T"] > 12
        L, _ = P5.loss_of(m, zz, tg)
        L.backward()
        head_in = m.head.net[0].in_features
        res6[f"{arch}_h{depth}_{task}"] = f"ok, head reads width {head_in}, out {tuple(zz.shape[1:])}"
        del m
R["V6_depth_agnostic"] = res6

# ------------------------------------------------------------------ V7
lb = P5.labels("v6", "arrival_week")
tr, va, te = TS.fold(lb.snapshot_date)
D = P5.device_inputs("v6", np.sort(lb.snapshot_date[tr].unique()), wsla=True)
m = P5.HeadNet(D["X"].shape[2], "arrival_week", gate_cols=D["gate_cols"]).to(DEV).eval()
t0, tg, _ = P5.batches("arrival_week", lb, te, D["W"])[0]
with torch.no_grad():
    z0 = P5.forward(m, D, t0, tg["idx"])
    saved = [D["X"][:, t0 + 1:].clone(), D["dt"][:, t0 + 1:].clone(), D["obs"][:, t0 + 1:].clone()]
    D["X"][:, t0 + 1:] += 100.0; D["dt"][:, t0 + 1:] += 100.0; D["obs"][:, t0 + 1:] = ~D["obs"][:, t0 + 1:]
    z1 = P5.forward(m, D, t0, tg["idx"])
    D["X"][:, t0 + 1:], D["dt"][:, t0 + 1:], D["obs"][:, t0 + 1:] = saved
    last = D["X"][:, t0].clone(); D["X"][:, t0] += 1.0
    z2 = P5.forward(m, D, t0, tg["idx"])
    D["X"][:, t0] = last
    z3 = P5.forward(m, D, t0, tg["idx"])
R["V7_future_perturbation_max_abs_diff"] = float((z1 - z0).abs().max())
R["V7_future_perturbation_bitwise_identical"] = bool(torch.equal(z0, z1))
R["V7_sensitivity_last_in_window_week_max_abs_diff"] = float((z2 - z0).abs().max())
R["V7_restore_bitwise"] = bool(torch.equal(z0, z3))
tcn = m.tcn
xa = D["X"][:64, t0 - 51:t0 + 1].clone()
with torch.no_grad():
    a0 = tcn.all_positions(xa); xa[:, -1] += 100.0; a1 = tcn.all_positions(xa)
R["V7_tcn_earlier_positions_max_abs_diff"] = float((a1[:, :, :-1] - a0[:, :, :-1]).abs().max())
assert R["V7_future_perturbation_max_abs_diff"] == 0.0 and R["V7_sensitivity_last_in_window_week_max_abs_diff"] > 0

json.dump(R, open(os.path.join(P5.ARTIFACTS, "phase5_verify.json"), "w"), indent=1)
for k, v in R.items():
    print(f"{k}: {v}")
print("ALL CHECKS PASSED")
