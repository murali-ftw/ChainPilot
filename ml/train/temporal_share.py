"""Run 8 — Temporal-SHARE vs HeteroMP, full channel population, trained to convergence.

What differs from `cross_world.py` (Phase 3), and only these things:

  * **All 16,072 channels.** So did Phase 3 -- `snapshot_windows(Wd, t0)` passed `chan=None`,
    which is the whole panel. The `X (4096, 52, 24)` in reports/phase-2.md is a demo build
    and reports/phase-0.md's `[4096, 38, 52]` is a throughput benchmark; neither is a
    training population. See §2 of the report.
  * **SHARE** (ml/models/share.py) as an alternative to HeteroMP. Same TCN front-end.
  * **Early stopping** on the validation fold instead of a fixed 6 epochs.
  * Panels held in RAM rather than memmapped -- a 50x speedup on the window build and
    numerically identical.

Hyperparameters are tuned ONCE on train-v7 / test-v7 with SHARE and then frozen for every
cell and both architectures.
"""
from __future__ import annotations
import os, sys, json, time, copy, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models")]
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from config import WORLDS, CACHE, SPLIT, FIT_WINDOW
from device import get_device, seed_everything, DTYPE, peak_rss_gb
import cache as C
from loader import read_df
from sequences import build_sequences, Normaliser
from tcn import TCN, HeteroMP
from share import SHARE, build_graph

DEV = get_device()
NBIN = 20
EDGES = np.linspace(0, 1, NBIN + 1)

# FROZEN per task. Each rate was chosen by the SAME five-point protocol on v6 / SHARE / h4,
# selected on the VALIDATION fold, extended when the optimum landed on a boundary.
#
# arrival_week -- run 9, cap 40 / patience 5. NOT retuned and NOT re-run in the closeout:
#   2e-3 val 0.6637 | 1e-3 0.6643 | 5e-4 0.6654 | 2.5e-4 0.6686 <- | 1.25e-4 0.6686
#
# fill_rate and shortage_qty -- retuned in the closeout, because run 9 froze arrival's rate
# across all three tasks and 14 of 34 cells hit the 40-epoch cap on those two.
LR_BY_TASK = {"arrival_week": 2.5e-4, "fill_rate": None, "shortage_qty": None}

HP = dict(tcn_hidden=64, graph_hidden=128, lr=2.5e-4, wd=1e-4,
          max_epochs=40, patience=5, bases=10, share_layers=4)

_WORLD_CACHE = {}


# ---------------------------------------------------------------- data
def load_world(w):
    if w in _WORLD_CACHE:
        return _WORLD_CACHE[w]
    panel, miss, active, meta = C.load_panel(os.path.join(CACHE, w))
    panel = np.array(panel); miss = np.array(miss); active = np.array(active)
    d = read_df(WORLDS[w], "sourcing_channels",
                usecols=["channel_id", "supplier_id", "part_id", "plant_id"])
    cidx = {v: i for i, v in enumerate(d.channel_id)}
    rel, sizes = [], []
    for col in ("supplier_id", "part_id", "plant_id"):
        u = {v: i for i, v in enumerate(sorted(d[col].unique()))}
        rel.append(d[col].map(u).to_numpy(np.int64)); sizes.append(len(u))
    W = dict(panel=panel, miss=miss, active=active, meta=meta, cidx=cidx,
             rel=rel, rel_sizes=sizes, NCH=meta["n_channels"], T=meta["T"],
             w0=pd.Timestamp(meta["week0"]), dcol=d)
    W["rel_t"] = [torch.from_numpy(i).to(DEV) for i in rel]
    src, dst, rl, n_nodes, offs = build_graph(W["rel_t"], sizes, DEV)
    W["graph"] = (src, dst, rl, n_nodes, offs)
    pk = (d.part_id + "|" + d.plant_id).to_numpy()
    uniq = {k: i for i, k in enumerate(sorted(set(pk)))}
    W["pp_uniq"] = uniq
    W["pp_of_chan"] = torch.from_numpy(np.array([uniq[k] for k in pk], np.int64)).to(DEV)
    _WORLD_CACHE[w] = W
    return W


def labels_for(w, task):
    D = WORLDS[w]
    lb = read_df(D, "training_labels",
                 usecols=["snapshot_date", "entity_id", "task", "label_value",
                          "label_censored"])
    lb = lb[lb.task == task].copy()
    lb["snapshot_date"] = pd.to_datetime(lb.snapshot_date)
    # §16 fit window, carried forward from Phase 2. cross_world.py (Phase 3) imported
    # SPLIT but never FIT_WINDOW, so Phase 3 trained on 66 snapshots where the baselines
    # it was compared against used 44. Applied here; the difference is recorded in the report.
    lb = lb[(lb.snapshot_date >= pd.Timestamp(FIT_WINDOW[0])) &
            (lb.snapshot_date <= pd.Timestamp(FIT_WINDOW[1]))]
    lb["label_censored"] = lb.label_censored.astype(str).str.lower().isin(["true", "1"])
    if task in ("fill_rate", "arrival_week"):
        cols = ["po_line_id", "channel_id"]
        if task == "arrival_week":
            cols.append("original_promise_date")     # for the binarised late/on-time ROC-AUC
        pol = read_df(D, "po_lines", usecols=cols)
        lb = lb.merge(pol, left_on="entity_id", right_on="po_line_id", how="inner")
        lb["key"] = lb.channel_id
        if task == "arrival_week":
            # promise expressed in the SAME units as the label: weeks after the snapshot
            prom = pd.to_datetime(lb.original_promise_date, errors="coerce")
            lb["promise_week"] = ((prom - lb.snapshot_date).dt.days / 7.0)
    else:
        lb[["part_id", "plant_id"]] = lb.entity_id.str.split("|", expand=True)
        lb["key"] = lb.part_id + "|" + lb.plant_id
        lb["label_value"] = (lb.label_value > 0).astype(float)
    return lb


def fold(dates):
    tr = dates <= pd.Timestamp(SPLIT["train_end"])
    va = (dates > pd.Timestamp(SPLIT["train_end"])) & (dates <= pd.Timestamp(SPLIT["val_end"]))
    te = dates > pd.Timestamp(SPLIT["val_end"])
    return tr.to_numpy(), va.to_numpy(), te.to_numpy()


def window(W, t0):
    d = W["meta"]["d"]
    lo = max(0, t0 + 1 - 52); hi = t0 + 1
    pad = max(0, 52 - (hi - lo))
    a = W["panel"][:, lo:hi, :]; b = W["miss"][:, lo:hi, :]
    X = np.concatenate([a, b], axis=2)
    if pad:
        X = np.concatenate([np.zeros((X.shape[0], pad, X.shape[2]), np.float32), X], 1)
    return X


# ---------------------------------------------------------------- metrics
def crps(cdf, y):
    step = float(np.diff(EDGES)[0])
    ind = (y[:, None] <= EDGES[None, 1:]).astype(float)
    return float((((cdf - ind) ** 2) * step).sum(1).mean())


def cindex(pred, t, e, n=1_000_000, seed=7):
    rng = np.random.default_rng(seed); N = len(t)
    i = rng.integers(0, N, n); j = rng.integers(0, N, n)
    ok = ((t[i] < t[j]) & e[i]) | ((t[j] < t[i]) & e[j])
    if ok.sum() == 0: return float("nan")
    i, j = i[ok], j[ok]
    ear = t[i] < t[j]
    conc = np.where(ear, pred[i] < pred[j], pred[j] < pred[i])
    ties = pred[i] == pred[j]
    return float((conc.sum() + 0.5 * ties.sum()) / len(i))


def pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


# ---------------------------------------------------------------- model
class Net(nn.Module):
    """arch in {"none", "mp", "share"}. At depth 0 NO graph module is constructed at all,
    so h0 is bit-identically the same model whichever architecture asked for it."""

    def __init__(self, d_in, task, arch, depth):
        super().__init__()
        h = HP["tcn_hidden"]
        self.tcn = TCN(d_in, h)
        self.arch, self.depth = arch, depth
        out = NBIN if task == "fill_rate" else 1
        wide = h
        if depth > 0 and arch in ("share", "lite"):
            self.enc = SHARE(h, HP["graph_hidden"], n_rel=6,
                             n_layers=HP["share_layers"],
                             n_bases=None if arch == "lite" else HP["bases"])
            wide = HP["graph_hidden"]
        elif depth > 0 and arch == "mp":
            self.enc = HeteroMP(h, 3, rounds=max(1, depth // 2))
        else:
            self.enc = None
        self.head = nn.Sequential(nn.Linear(wide, wide), nn.ReLU(), nn.Linear(wide, out))

    def encode(self, X, W):
        h = self.tcn(X)
        if self.enc is None:
            return h
        if self.arch in ("share", "lite"):
            src, dst, rel, n_nodes, offs = W["graph"]
            return self.enc(h, W["rel_t"], offs, n_nodes, src, dst, rel, depth=self.depth)
        return self.enc(h, W["rel_t"], W["rel_sizes"])


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- train / eval
def _forward(model, W, t0, norm, task, rows):
    X = torch.from_numpy(np.ascontiguousarray(window(W, t0))).to(DEV)
    X = norm.apply_t(X)
    h = model.encode(X, W)
    if task == "shortage_qty":
        n_pp = len(W["pp_uniq"])
        agg = torch.zeros(n_pp, h.shape[1], device=DEV, dtype=h.dtype)
        agg.index_add_(0, W["pp_of_chan"], h)
        cnt = torch.zeros(n_pp, 1, device=DEV, dtype=h.dtype)
        cnt.index_add_(0, W["pp_of_chan"], torch.ones_like(h[:, :1]))
        h = agg / cnt.clamp(min=1.0)
        idx = rows.key.map(W["pp_uniq"]).fillna(0).to_numpy(np.int64)
    else:
        idx = rows.key.map(W["cidx"]).fillna(0).to_numpy(np.int64)
    return model.head(h[torch.from_numpy(idx).to(DEV)])


def _loss(z, rows, task, ymu, ysd):
    if task == "shortage_qty":
        y = torch.from_numpy(rows.label_value.to_numpy(np.float32)).to(DEV)
        return F.binary_cross_entropy_with_logits(z.squeeze(-1), y)
    if task == "fill_rate":
        yb = np.clip(np.digitize(rows.label_value.to_numpy(float), EDGES[1:-1]), 0, NBIN - 1)
        return F.cross_entropy(z, torch.from_numpy(yb).to(DEV))
    obs = torch.from_numpy(~rows.label_censored.to_numpy()).to(DEV)
    if obs.sum() == 0: return None
    yv = (rows.label_value.to_numpy(np.float32) - ymu) / ysd
    y = torch.from_numpy(yv.astype(np.float32)).to(DEV)
    return F.mse_loss(z.squeeze(-1)[obs], y[obs])


class TNorm:
    """The Phase 2 normaliser, applied on-device. Same arithmetic, no numpy round-trip."""
    def __init__(self, norm, device):
        self.mu = torch.from_numpy(norm.mu.astype(np.float32)).to(device)
        self.sd = torch.from_numpy(norm.sd.astype(np.float32)).to(device)
        self.idx = torch.tensor(list(norm.log1p_idx), dtype=torch.long, device=device)

    def apply_t(self, X):
        if len(self.idx):
            X = X.clone()
            X[..., self.idx] = torch.log1p(X[..., self.idx].clamp(min=0))
        return (X - self.mu) / self.sd


def fit_normaliser(W, snaps):
    sample = [int((pd.Timestamp(s) - W["w0"]).days // 7) for s in snaps[::4]]
    Xs = np.concatenate([window(W, t)[::8] for t in sample])
    nz = Normaliser(log1p_idx=[W["meta"]["cols"].index(c) for c in
                               ("qty_ordered", "qty_received", "lead_time_actual_days")])
    nz.fit(Xs, np.ones(Xs.shape[:2], bool))
    del Xs
    return TNorm(nz, DEV)


@torch.no_grad()
def predict(model, W, norm, lb, mask, task):
    """-> P, Y, EV, AUX. AUX carries promise_week for arrival, NaN otherwise."""
    model.eval()
    rows_all = lb[mask]
    P, Y, EV, AUX = [], [], [], []
    for s in np.sort(rows_all.snapshot_date.unique()):
        t0 = int((pd.Timestamp(s) - W["w0"]).days // 7)
        if t0 < 0 or t0 >= W["T"]: continue
        rows = rows_all[rows_all.snapshot_date == s]
        if not len(rows): continue
        z = _forward(model, W, t0, norm, task, rows)
        z = torch.softmax(z, -1) if task == "fill_rate" else z.squeeze(-1)
        if task == "shortage_qty": z = torch.sigmoid(z)
        P.append(z.float().cpu().numpy()); Y.append(rows.label_value.to_numpy(float))
        EV.append(~rows.label_censored.to_numpy())
        AUX.append(rows.promise_week.to_numpy(float) if "promise_week" in rows
                   else np.full(len(rows), np.nan))
    return (np.concatenate(P), np.concatenate(Y), np.concatenate(EV),
            np.concatenate(AUX))


def score(P, Y, EV, task, aux=None):
    if task == "fill_rate":   return crps(np.cumsum(P, 1), Y)
    if task == "arrival_week": return cindex(P, Y, EV)
    return pr_auc(Y, P)


BETTER = {"fill_rate": lambda a, b: a < b, "arrival_week": lambda a, b: a > b,
          "shortage_qty": lambda a, b: a > b}


def train_cell(train_w, task, arch, depth, seed=7, verbose=True):
    seed_everything(seed)
    W = load_world(train_w)
    lb = labels_for(train_w, task)
    tr, va, te = fold(lb.snapshot_date)
    d_in = W["meta"]["d"] + len(W["meta"]["nullable"])
    snaps = np.sort(lb.snapshot_date[tr].unique())
    norm = fit_normaliser(W, snaps)
    ymu, ysd = 0.0, 1.0
    if task == "arrival_week":
        o = tr & (~lb.label_censored.to_numpy())
        ymu = float(lb.label_value[o].mean()); ysd = float(lb.label_value[o].std()) or 1.0
    model = Net(d_in, task, arch, depth).to(DEV).to(DTYPE)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])

    best, best_ep, best_state, bad, losses, vals = None, -1, None, 0, [], []
    t_start = time.time()
    for ep in range(HP["max_epochs"]):
        model.train(); ep_l = []
        for s in snaps:
            t0 = int((pd.Timestamp(s) - W["w0"]).days // 7)
            if t0 < 0 or t0 >= W["T"]: continue
            rows = lb[tr & (lb.snapshot_date == s)]
            if not len(rows): continue
            z = _forward(model, W, t0, norm, task, rows)
            L = _loss(z, rows, task, ymu, ysd)
            if L is None: continue
            opt.zero_grad(); L.backward(); opt.step(); ep_l.append(float(L.detach()))
        losses.append(float(np.mean(ep_l)) if ep_l else float("nan"))
        v = score(*predict(model, W, norm, lb, va, task)[:3], task)
        vals.append(v)
        if best is None or BETTER[task](v, best):
            best, best_ep, bad = v, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if verbose:
            print(f"      ep {ep:>2}  loss {losses[-1]:.4f}  val {v:.4f}"
                  f"{'  *' if bad == 0 else ''}", flush=True)
        if bad >= HP["patience"]:
            break
    model.load_state_dict(best_state)
    stop = "patience" if bad >= HP["patience"] else "CAP (did not converge)"
    return model, norm, dict(epochs_run=len(losses), best_epoch=best_ep, best_val=best,
                             stop=stop, losses=losses, vals=vals,
                             seconds=time.time() - t_start, params=n_params(model),
                             ymu=ymu, ysd=ysd)


def eval_on(model, norm, test_w, task):
    W = load_world(test_w)
    lb = labels_for(test_w, task)
    tr, va, te = fold(lb.snapshot_date)
    return score(*predict(model, W, norm, lb, te, task)[:3], task)


def predict_test(model, norm, test_w, task):
    """Raw test-fold predictions, for offline scoring with bootstrap intervals."""
    W = load_world(test_w)
    lb = labels_for(test_w, task)
    tr, va, te = fold(lb.snapshot_date)
    return predict(model, W, norm, lb, te, task)
