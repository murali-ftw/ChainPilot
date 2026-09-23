"""Phase 5 — train one head, at h0 by default, optionally behind the 5.0 staleness gate.

What differs from temporal_share.py (Phases 2-4), and only these things:

  * the HEAD: hazard (arrival), point-mass CDF (fill), quantile (capacity) from ml/models/heads.py;
    shortage keeps its BCE head as a diagnostic
  * `capacity_strain` is a task -- the first time it has been put through any model
  * an optional StalenessGate on reporting_lag_days and an optional reconstructed
    weeks_since_last_activity feature (ml/models/staleness.py)
  * windows are sliced from a normalised panel held ON THE DEVICE rather than rebuilt in numpy and
    copied every step. The normaliser is fitted on the identical sample temporal_share uses (every
    4th training snapshot, every 8th channel, trailing 52 weeks), so the inputs are the same numbers.

Split, fit window, population (all 16,072 channels), TCN, one optimiser step per snapshot, early
stopping on validation with restore-best -- all imported from temporal_share, not copied.
"""
from __future__ import annotations
import os, sys, json, time, copy, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd, torch, torch.nn as nn
import temporal_share as TS
from config import WORLDS, FIT_WINDOW, ARTIFACTS
from device import seed_everything, DTYPE, peak_rss_gb
from loader import read_df
from sequences import Normaliser
from tcn import TCN, HeteroMP
from share import SHARE
from heads import HazardHead, FillCDFHead, QuantileHead, BinaryHead, fill_cell
from staleness import StalenessGate, weeks_since_last_activity
from metrics import cindex, pr_auc
import phase5_metrics as M

DEV = TS.DEV
WIN = 52
HORIZON_WEEKS = 12
PRED = os.path.join(ARTIFACTS, "phase5_preds")
GRID = os.path.join(ARTIFACTS, "phase5_grid.json")
LOG1P_COLS = ("qty_ordered", "qty_received", "lead_time_actual_days")
HEAD_OF = {"arrival_week": "hazard", "fill_rate": "cdf22", "capacity_strain": "quantile",
           "shortage_qty": "binary"}


# ---------------------------------------------------------------- labels
def labels(w, task, row_features: bool = False):
    """row_features is threaded through to TS.labels_for; see its docstring (deviation 59)."""
    if task != "capacity_strain":
        lb = TS.labels_for(w, task, row_features=row_features)
    else:
        lb = read_df(WORLDS[w], "training_labels",
                     usecols=["snapshot_date", "entity_id", "task", "label_value", "label_censored"])
        lb = lb[lb.task == task].copy()
        lb["snapshot_date"] = pd.to_datetime(lb.snapshot_date)
        lb = lb[(lb.snapshot_date >= pd.Timestamp(FIT_WINDOW[0])) &
                (lb.snapshot_date <= pd.Timestamp(FIT_WINDOW[1]))]
        # capacity's censoring flag is an independent 4% coin flip in both generators and the value
        # is present on every row; it is carried for the record and not used
        lb["label_censored"] = lb.label_censored.astype(str).str.lower().isin(["true", "1"])
        lb["key"] = lb.entity_id
    return lb.reset_index(drop=True)


def ordered(lb, mask):
    """Row positions of lb[mask] in emission order: snapshot ascending, label order within."""
    idx = np.flatnonzero(mask)
    return idx[np.argsort(lb.snapshot_date.values[idx], kind="stable")]


def t0_of(W, s):
    t0 = int((pd.Timestamp(s) - W["w0"]).days // 7)
    assert WIN - 1 <= t0 < W["T"], f"snapshot {s} -> week {t0} needs left padding; not expected in the fit window"
    return t0


# ---------------------------------------------------------------- inputs on device
_DEV_CACHE = {}


def device_inputs(w, snaps_train, wsla):
    key = (w, bool(wsla), tuple(pd.Timestamp(s) for s in snaps_train))
    if key in _DEV_CACHE:
        return _DEV_CACHE[key]
    _DEV_CACHE.clear()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    W = TS.load_world(w)
    cols, null = list(W["meta"]["cols"]), list(W["meta"]["nullable"])
    names = cols + ["obs:" + c for c in null]
    X = np.concatenate([W["panel"], W["miss"]], 2)
    if wsla:
        X = np.concatenate([X, weeks_since_last_activity(W["active"])[..., None]], 2)
        names.append("weeks_since_last_activity")
    log1p = [names.index(c) for c in LOG1P_COLS] + ([len(names) - 1] if wsla else [])
    # identical sample to temporal_share.fit_normaliser
    ts = [t0_of(W, s) for s in snaps_train[::4]]
    Xs = np.concatenate([X[::8, t - WIN + 1:t + 1] for t in ts])
    nz = Normaliser(log1p_idx=log1p).fit(Xs, np.ones(Xs.shape[:2], bool))
    del Xs
    mu = torch.from_numpy(nz.mu.astype(np.float32)).to(DEV)
    sd = torch.from_numpy(nz.sd.astype(np.float32)).to(DEV)
    li = torch.tensor(log1p, dtype=torch.long, device=DEV)
    Xt = torch.empty(X.shape, dtype=DTYPE, device=DEV)
    for i in range(0, X.shape[0], 2048):
        c = torch.from_numpy(np.ascontiguousarray(X[i:i + 2048])).to(DEV)
        c[..., li] = torch.log1p(c[..., li].clamp(min=0))
        Xt[i:i + 2048] = (c - mu) / sd
    del X
    lag = cols.index("reporting_lag_days")
    D = dict(X=Xt, names=names, W=W,
             # the normaliser travels with a trained model (Phase 6 bundle); additive, changes no number
             norm_mu=nz.mu.astype(np.float32), norm_sd=nz.sd.astype(np.float32), norm_log1p_idx=list(log1p),
             dt=torch.from_numpy(np.ascontiguousarray(W["panel"][..., lag])).to(DEV),
             obs=torch.from_numpy(np.ascontiguousarray(W["miss"][..., null.index("reporting_lag_days")] > 0)).to(DEV),
             gate_cols=[i for i, c in enumerate(cols) if c != "reporting_lag_days"])
    _DEV_CACHE[key] = D
    return D


# ---------------------------------------------------------------- model
class HeadNet(nn.Module):
    def __init__(self, d_in, task, arch="none", depth=0, gate_cols=None, fill_loss="rps", n_row_feats=0):
        super().__init__()
        h = TS.HP["tcn_hidden"]
        self.task, self.arch, self.depth, self.fill_loss = task, arch, depth, fill_loss
        self.n_row_feats = int(n_row_feats)        # per-LINE inputs concatenated to the channel encoding
        self.tcn = TCN(d_in, h)
        self.gate = StalenessGate(len(gate_cols)) if gate_cols is not None else None
        if gate_cols is not None:
            self.register_buffer("gate_idx", torch.tensor(gate_cols, dtype=torch.long))
        wide = h
        if depth > 0 and arch in ("share", "lite"):
            self.enc = SHARE(h, TS.HP["graph_hidden"], n_rel=6, n_layers=TS.HP["share_layers"],
                             n_bases=None if arch == "lite" else TS.HP["bases"])
            wide = TS.HP["graph_hidden"]
        elif depth > 0 and arch == "mp":
            self.enc = HeteroMP(h, 3, rounds=max(1, depth // 2))
        else:
            self.enc = None
        wide += self.n_row_feats
        self.head = {"hazard": lambda: HazardHead(wide, HORIZON_WEEKS),
                     "cdf22": lambda: FillCDFHead(wide),
                     "quantile": lambda: QuantileHead(wide),
                     "binary": lambda: BinaryHead(wide)}[HEAD_OF[task]]()

    def encode(self, X, dt, obs, W, return_gate=False):
        g = None
        if self.gate is not None:
            xg, g = self.gate(X.index_select(-1, self.gate_idx), dt, obs, return_gate=True)
            X = X.index_copy(-1, self.gate_idx, xg)
        h = self.tcn(X)
        if self.enc is not None:
            if self.arch in ("share", "lite"):
                src, dst, rel, n_nodes, offs = W["graph"]
                h = self.enc(h, W["rel_t"], offs, n_nodes, src, dst, rel, depth=self.depth)
            else:
                h = self.enc(h, W["rel_t"], W["rel_sizes"])
        return (h, g) if return_gate else h


def forward(model, D, t0, idx, return_gate=False, xrow=None):
    W = D["W"]
    sl = slice(t0 - WIN + 1, t0 + 1)
    h, g = model.encode(D["X"][:, sl], D["dt"][:, sl], D["obs"][:, sl], W, return_gate=True)
    if model.task == "shortage_qty":
        n_pp = len(W["pp_uniq"])
        agg = torch.zeros(n_pp, h.shape[1], device=DEV, dtype=h.dtype).index_add_(0, W["pp_of_chan"], h)
        cnt = torch.zeros(n_pp, 1, device=DEV, dtype=h.dtype).index_add_(
            0, W["pp_of_chan"], torch.ones_like(h[:, :1]))
        h = agg / cnt.clamp(min=1.0)
    hi = h[idx]
    if getattr(model, "n_row_feats", 0):
        assert xrow is not None and xrow.shape[1] == model.n_row_feats, \
            f"model expects {model.n_row_feats} per-row features, got {None if xrow is None else xrow.shape}"
        hi = torch.cat([hi, xrow.to(hi.dtype)], dim=-1)
    else:
        assert xrow is None, "per-row features supplied to a model that has none"
    z = model.head(hi)
    return (z, g) if return_gate else z


# ---------------------------------------------------------------- batches
def batches(task, lb, mask, W, ymu=0.0, ysd=1.0):
    out = []
    order = ordered(lb, mask)
    dates = lb.snapshot_date.values[order]
    for s in np.unique(dates):
        ii = order[dates == s]
        rows = lb.iloc[ii]
        keymap = W["pp_uniq"] if task == "shortage_qty" else W["cidx"]
        k = rows.key.map(keymap)
        assert k.notna().all(), f"{task}: {int(k.isna().sum())} label keys absent from the graph"
        tg = {"idx": torch.from_numpy(k.to_numpy(np.int64)).to(DEV)}
        if task == "arrival_week" and "line_age_weeks" in rows:
            # AS-OF: every row used must already be recorded at its own snapshot. Asserted, not assumed.
            assert (rows.line_recorded_ts <= rows.snapshot_date).all(), \
                "a po_line is not yet recorded at its own snapshot -- as-of violation"
            pw = rows.promise_week.to_numpy(float); ag = rows.line_age_weeks.to_numpy(float)
            assert np.isfinite(pw).all() and np.isfinite(ag).all(), "non-finite per-row feature"
            assert (ag >= 0).all(), "negative line age"
            tg["xrow"] = torch.from_numpy(np.stack([pw / 10.0, ag / 10.0], 1).astype(np.float32)).to(DEV)
        y = rows.label_value.to_numpy(float)
        cen = rows.label_censored.to_numpy(bool)
        if task == "arrival_week":
            assert (y[~cen] >= 1).all() and (y[~cen] <= HORIZON_WEEKS).all(), "observed arrival outside 1..12"
            assert (y[cen] > HORIZON_WEEKS).all(), "censored arrival inside the horizon"
            tg["T"] = torch.from_numpy(np.minimum(y, HORIZON_WEEKS + 1).astype(np.int64)).to(DEV)
            tg["cen"] = torch.from_numpy(cen).to(DEV)
        elif task == "fill_rate":
            tg["cell"] = torch.from_numpy(fill_cell(y)).to(DEV)
        elif task == "capacity_strain":
            tg["y"] = torch.from_numpy(((y - ymu) / ysd).astype(np.float32)).to(DEV).unsqueeze(1)
        else:
            tg["y"] = torch.from_numpy(y.astype(np.float32)).to(DEV)
        out.append((t0_of(W, s), tg, ii))
    return out


def loss_of(model, z, tg):
    hd = model.head
    if model.task == "arrival_week":
        return hd.loss(z, tg["T"], tg["cen"])
    if model.task == "fill_rate":
        return hd.loss(z, tg["cell"], model.fill_loss), len(tg["cell"])
    return hd.loss(z, tg["y"]), len(tg["y"])


@torch.no_grad()
def predict(model, D, lb, mask, ymu=0.0, ysd=1.0, gate_stats=False):
    """-> dict of arrays in emission order, plus Y / EV / AUX."""
    model.eval()
    W = D["W"]
    parts, rows_all, gs = {}, [], []
    for t0, tg, ii in batches(model.task, lb, mask, W, ymu, ysd):
        z, g = forward(model, D, t0, tg["idx"], return_gate=True,
                       xrow=tg.get("xrow") if getattr(model, "n_row_feats", 0) else None)
        if model.task == "arrival_week":
            lam, S, pT = HazardHead.distribution(z)
            got = {"P": HazardHead.expected_time(S), "S": S, "pT": pT}
        elif model.task == "fill_rate":
            got = {"P": FillCDFHead.probs(z)}
        elif model.task == "capacity_strain":
            got = {"P": z[:, 0, :] * ysd + ymu}
        else:
            got = {"P": torch.sigmoid(z)}
        for k, v in got.items():
            parts.setdefault(k, []).append(v.float().cpu().numpy())
        rows_all.append(ii)
        if gate_stats and g is not None:
            ob = D["obs"][:, t0 - WIN + 1:t0 + 1]
            gs.append((g[ob].float().cpu().numpy()))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    ii = np.concatenate(rows_all)
    out["Y"] = lb.label_value.to_numpy(float)[ii]
    out["EV"] = ~lb.label_censored.to_numpy(bool)[ii]
    out["AUX"] = lb.promise_week.to_numpy(float)[ii] if "promise_week" in lb else np.full(len(ii), np.nan)
    if gs:
        G = np.concatenate(gs)                                   # [positions, n_gated]
        out["_gate"] = dict(mean_g=float(G.mean()), frac_lt_099=float((G < 0.99).mean()),
                            frac_lt_090=float((G < 0.90).mean()),
                            mean_g_per_feature=[float(x) for x in G.mean(0)],
                            min_g_per_feature=[float(x) for x in G.min(0)])
    return out


def val_score(task, pr):
    if task == "arrival_week":
        return cindex(pr["P"], pr["Y"], pr["EV"])
    if task == "fill_rate":
        return float(M.crps_exact_rows(pr["P"], pr["Y"]).mean())
    if task == "capacity_strain":
        return float(np.mean([M.pinball_rows(pr["Y"], pr["P"][:, j], t).mean() for j, t in enumerate(M.QS)]))
    return pr_auc(pr["Y"], pr["P"])


HIGHER = {"arrival_week": True, "fill_rate": False, "capacity_strain": False, "shortage_qty": True}


def train_cell(world, task, lr, seed=7, arch="none", depth=0, gate=False, wsla=False,
               max_epochs=120, patience=8, verbose=True, epoch_limit=None, fill_loss="rps"):
    seed_everything(seed)
    lb = labels(world, task)
    tr, va, te = TS.fold(lb.snapshot_date)
    snaps = np.sort(lb.snapshot_date[tr].unique())
    D = device_inputs(world, snaps, wsla)
    W = D["W"]
    ymu, ysd = 0.0, 1.0
    if task == "capacity_strain":
        ymu = float(lb.label_value[tr].mean()); ysd = float(lb.label_value[tr].std())
    seed_everything(seed)                           # model init independent of the data path above
    model = HeadNet(D["X"].shape[2], task, arch, depth,
                    gate_cols=D["gate_cols"] if gate else None, fill_loss=fill_loss).to(DEV).to(DTYPE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=TS.HP["wd"])
    train_b = batches(task, lb, tr, W, ymu, ysd)
    n_train_rows = int(tr.sum())

    best, best_ep, best_state, bad, losses, vals, rows_in_loss = None, -1, None, 0, [], [], []
    t_start = time.time(); ep_times = []
    for ep in range(max_epochs if epoch_limit is None else epoch_limit):
        te0 = time.time()
        model.train(); ep_l, n_rows = [], 0
        for t0, tg, _ in train_b:
            z = forward(model, D, t0, tg["idx"],
                        xrow=tg.get("xrow") if getattr(model, "n_row_feats", 0) else None)
            L, n_c = loss_of(model, z, tg)
            opt.zero_grad(); L.backward(); opt.step()
            ep_l.append(float(L.detach())); n_rows += n_c
        rows_in_loss.append(n_rows)
        assert n_rows == n_train_rows, f"rows entering the loss {n_rows:,} != training population {n_train_rows:,}"
        losses.append(float(np.mean(ep_l)))
        v = val_score(task, predict(model, D, lb, va, ymu, ysd))
        vals.append(v)
        if best is None or (v > best if HIGHER[task] else v < best):
            best, best_ep, bad = v, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        ep_times.append(time.time() - te0)
        if verbose:
            print(f"      ep {ep:>3}  loss {losses[-1]:.5f}  val {v:.5f}{'  *' if bad == 0 else ''}"
                  f"  {ep_times[-1]:.1f}s", flush=True)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    stop = "patience" if bad >= patience else ("CAP (floor)" if epoch_limit is None else "timing")
    info = dict(epochs_run=len(losses), best_epoch=best_ep, best_val=best, stop=stop, losses=losses,
                vals=vals, seconds=time.time() - t_start, sec_per_epoch=float(np.mean(ep_times)),
                params=sum(p.numel() for p in model.parameters()), ymu=ymu, ysd=ysd,
                n_train_rows=n_train_rows, rows_in_loss_each_epoch=sorted(set(rows_in_loss)),
                n_features=int(D["X"].shape[2]))
    return model, D, lb, info


def run(world, task, lr, seed, arch, depth, gate, wsla, tag, grid, max_epochs, patience, fill_loss="rps",
        pred_dir=PRED):
    rows = json.load(open(grid)) if os.path.exists(grid) else []
    key = dict(world=world, task=task, head=HEAD_OF[task], arch=arch, depth=depth, seed=seed,
               lr=lr, gate=int(gate), wsla=int(wsla))
    if task == "fill_rate" and fill_loss != "rps":        # absent key == the specified RPS loss
        key["fill_loss"] = fill_loss
    # fill_loss must match too: a row without the key is the specified RPS loss
    if any(all(r.get(k) == v for k, v in key.items()) and r.get("fill_loss") == key.get("fill_loss")
           for r in rows):
        print(f"[skip] {key}", flush=True); return
    print(f"\n=== {tag} {key} ===", flush=True)
    t0 = time.time()
    model, D, lb, info = train_cell(world, task, lr, seed, arch, depth, gate, wsla,
                                    max_epochs=max_epochs, patience=patience, verbose=False,
                                    fill_loss=fill_loss)
    tr, va, te = TS.fold(lb.snapshot_date)
    pr = predict(model, D, lb, te, info["ymu"], info["ysd"], gate_stats=gate)
    if gate:
        w_, b_ = model.gate.weights()
        # NOT info["gate"]: the grid row already carries gate=0/1 from the key, and dict(**key, **info)
        # raises on the duplicate -- which is exactly how the first gate-on cell died after training
        info["gate_diag"] = dict(test=pr.pop("_gate"), w=[float(x) for x in w_], b=[float(x) for x in b_],
                            prior=[float(x) for x in model.gate.prior],
                            features=[D["names"][i] for i in D["gate_cols"]])
    os.makedirs(pred_dir, exist_ok=True)
    f = os.path.join(pred_dir, f"{world}_{task}_{HEAD_OF[task]}_{arch}_h{depth}_s{seed}_lr{lr:g}"
                           f"_g{int(gate)}_w{int(wsla)}{'' if fill_loss == 'rps' else '_' + fill_loss.replace('+', '')}.npz")
    np.savez_compressed(f, **{k: v for k, v in pr.items() if not k.startswith("_")})
    # validation predictions and the restored best weights, so post-hoc steps fitted on the validation
    # fold (recalibration, Phase 5 §6.6) never need test data and never need a re-train
    pv = predict(model, D, lb, va, info["ymu"], info["ysd"])
    np.savez_compressed(f.replace(".npz", "_val.npz"), **{k: v for k, v in pv.items() if not k.startswith("_")})
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, f.replace(".npz", ".pt"))
    rows = json.load(open(grid)) if os.path.exists(grid) else []
    rows.append(dict(**key, tag=tag, preds=f, wall=time.time() - t0, rss=peak_rss_gb(),
                     **{k: v for k, v in info.items() if k != "losses"}))
    json.dump(rows, open(grid, "w"), indent=1)
    print(f"  [{info['stop']}] best_ep {info['best_epoch']} of {info['epochs_run']}  val {info['best_val']:.5f}"
          f"  {time.time()-t0:.0f}s  {info['sec_per_epoch']:.1f}s/ep  RSS {peak_rss_gb():.2f} GB", flush=True)
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--worlds", default="v6")
    ap.add_argument("--lrs", required=True)
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--arch", default="none")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--wsla", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--grid", default=GRID)
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--time-epochs", type=int, default=0, help="train this many epochs, print timing, exit")
    ap.add_argument("--fill-loss", default="rps", choices=["rps", "ce", "rps+ce"])
    ap.add_argument("--pred-dir", default=PRED, help="a separate directory keeps re-trains from overwriting earlier predictions")
    a = ap.parse_args()
    if a.time_epochs:
        for task in a.tasks.split(","):
            for w in a.worlds.split(","):
                _, _, _, info = train_cell(w, task, float(a.lrs.split(",")[0]), 7, a.arch, a.depth,
                                           bool(a.gate), bool(a.wsla), epoch_limit=a.time_epochs)
                print(f"TIMING {w} {task} gate={a.gate} wsla={a.wsla}: {info['sec_per_epoch']:.2f} s/epoch "
                      f"(first epoch includes warm-up), rows in loss {info['rows_in_loss_each_epoch']} "
                      f"of {info['n_train_rows']}, params {info['params']:,}, RSS {peak_rss_gb():.2f} GB",
                      flush=True)
        sys.exit(0)
    for seed in [int(s) for s in a.seeds.split(",")]:
        for w in a.worlds.split(","):
            for task in a.tasks.split(","):
                for lr in [float(x) for x in a.lrs.split(",")]:
                    run(w, task, lr, seed, a.arch, a.depth, bool(a.gate), bool(a.wsla), a.tag,
                        a.grid, a.max_epochs, a.patience, a.fill_loss, a.pred_dir)
    print("DONE", flush=True)
