"""Phase 3 — cross-world 2x2 and the h0/h1/h4 ablation with a LEARNED encoder.

Same features, same hyperparameters, tuned once on train-v7/test-v7 and then FROZEN.
Baselines are recomputed against the world each cell is TESTED on.

The masters are byte-identical across the two worlds (Stage A A1.4), so the channel /
supplier / part / plant index maps coincide and a model trained on v6 can be evaluated on
v7 without any re-indexing. Nothing in the model is keyed on identity in any case.
"""
from __future__ import annotations
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models")]
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from config import WORLDS, CACHE, SPLIT
from device import get_device, seed_everything, DTYPE
import cache as C
from loader import read_df
from sequences import build_sequences, Normaliser
from tcn import TCN, HeteroMP

DEV = get_device()
W_WINDOW = 52
NBIN = 20
EDGES = np.linspace(0, 1, NBIN + 1)
HP = dict(hidden=64, lr=2e-3, epochs=12, wd=1e-4)     # tuned once on v7->v7, then frozen


# ---------------------------------------------------------------- data prep
def load_world(w):
    od = os.path.join(CACHE, w)
    panel, miss, active, meta = C.load_panel(od)
    d = read_df(WORLDS[w], "sourcing_channels", usecols=["channel_id", "supplier_id", "part_id", "plant_id"])
    cidx = {v: i for i, v in enumerate(d.channel_id)}
    rel = []
    for col in ("supplier_id", "part_id", "plant_id"):
        u = {v: i for i, v in enumerate(sorted(d[col].unique()))}
        rel.append((d[col].map(u).to_numpy(np.int64), len(u)))
    return dict(panel=panel, miss=miss, active=active, meta=meta, cidx=cidx, rel=rel,
                NCH=meta["n_channels"], T=meta["T"],
                w0=pd.Timestamp(meta["week0"]), dcol=d)


def labels_for(w, task):
    D = WORLDS[w]
    lb = read_df(D, "training_labels",
                 usecols=["snapshot_date", "entity_id", "task", "label_value", "label_censored"])
    lb = lb[lb.task == task].copy()
    lb["snapshot_date"] = pd.to_datetime(lb.snapshot_date)
    lb["label_censored"] = lb.label_censored.astype(str).str.lower().isin(["true", "1"])
    if task in ("fill_rate", "arrival_week"):
        pol = read_df(D, "po_lines", usecols=["po_line_id", "channel_id"])
        lb = lb.merge(pol, left_on="entity_id", right_on="po_line_id", how="inner")
        lb["key"] = lb.channel_id
    else:                                             # shortage_qty at part x plant
        lb[["part_id", "plant_id"]] = lb.entity_id.str.split("|", expand=True)
        lb["key"] = lb.part_id + "|" + lb.plant_id
        lb["label_value"] = (lb.label_value > 0).astype(float)
    return lb


def fold(dates):
    tr = dates <= pd.Timestamp(SPLIT["train_end"])
    va = (dates > pd.Timestamp(SPLIT["train_end"])) & (dates <= pd.Timestamp(SPLIT["val_end"]))
    te = dates > pd.Timestamp(SPLIT["val_end"])
    return tr.to_numpy(), va.to_numpy(), te.to_numpy()


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
    def __init__(self, d_in, task, rounds, hidden=64):
        super().__init__()
        self.tcn = TCN(d_in, hidden)
        self.mp = HeteroMP(hidden, 3, rounds) if rounds > 0 else None
        out = NBIN if task == "fill_rate" else 1
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def encode(self, X, rel_idx, rel_size):
        h = self.tcn(X)
        if self.mp is not None:
            h = self.mp(h, rel_idx, rel_size)
        return h


def snapshot_windows(W, t0_week, chan=None):
    X, M, A, P = build_sequences(W["panel"], W["miss"], W["active"],
                                 np.full(W["NCH"] if chan is None else len(chan), t0_week), chan)
    return X


def run_cell(train_w, task, rounds, norm=None, model=None, epochs=None, seed=7):
    """Train on `train_w`. Returns model, normaliser."""
    seed_everything(seed)
    Wd = load_world(train_w)
    lb = labels_for(train_w, task)
    tr, va, te = fold(lb.snapshot_date)
    d_in = Wd["meta"]["d"] + len(Wd["meta"]["nullable"])
    rel_idx = [torch.from_numpy(i).to(DEV) for i, _ in Wd["rel"]]
    rel_size = [n for _, n in Wd["rel"]]

    snaps = np.sort(lb.snapshot_date[tr].unique())
    # normaliser fitted on the TRAIN fold only
    if norm is None:
        sample_t = [(pd.Timestamp(s) - Wd["w0"]).days // 7 for s in snaps[::4]]
        Xs = np.concatenate([snapshot_windows(Wd, t, np.arange(0, Wd["NCH"], 8)) for t in sample_t])
        norm = Normaliser(log1p_idx=[Wd["meta"]["cols"].index(c)
                                     for c in ("qty_ordered", "qty_received", "lead_time_actual_days")])
        norm.fit(Xs, np.ones(Xs.shape[:2], bool))
        del Xs
    ymu, ysd = 0.0, 1.0
    if task == "arrival_week":
        obs_tr = tr & (~lb.label_censored.to_numpy())
        ymu = float(lb.label_value[obs_tr].mean()); ysd = float(lb.label_value[obs_tr].std()) or 1.0
    if model is None:
        model = Net(d_in, task, rounds, HP["hidden"]).to(DEV).to(DTYPE)
        model.ymu, model.ysd = ymu, ysd
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])

    key2row = {}
    if task == "shortage_qty":
        dd = Wd["dcol"]; pk = (dd.part_id + "|" + dd.plant_id).to_numpy()
        uniq = {k: i for i, k in enumerate(sorted(set(pk)))}
        pp_of_chan = torch.from_numpy(np.array([uniq[k] for k in pk], np.int64)).to(DEV)
        n_pp = len(uniq)

    losses = []
    for ep in range(epochs or HP["epochs"]):
        model.train(); ep_l = []
        for s in snaps:
            t0 = int((pd.Timestamp(s) - Wd["w0"]).days // 7)
            if t0 < 0 or t0 >= Wd["T"]: continue
            rows = lb[tr & (lb.snapshot_date == s)]
            if len(rows) == 0: continue
            X = torch.from_numpy(norm.transform(snapshot_windows(Wd, t0))).to(DEV)
            h = model.encode(X, rel_idx, rel_size)
            if task == "shortage_qty":
                agg = torch.zeros(n_pp, h.shape[1], device=DEV, dtype=h.dtype)
                agg.index_add_(0, pp_of_chan, h)
                cnt = torch.zeros(n_pp, 1, device=DEV, dtype=h.dtype)
                cnt.index_add_(0, pp_of_chan, torch.ones_like(h[:, :1]))
                hh = agg / cnt.clamp(min=1.0)
                idx = torch.from_numpy(rows.key.map(uniq).fillna(0).to_numpy(np.int64)).to(DEV)
                z = model.head(hh[idx]).squeeze(-1)
                y = torch.from_numpy(rows.label_value.to_numpy(np.float32)).to(DEV)
                loss = F.binary_cross_entropy_with_logits(z, y)
            else:
                idx = torch.from_numpy(rows.key.map(Wd["cidx"]).fillna(0).to_numpy(np.int64)).to(DEV)
                z = model.head(h[idx])
                if task == "fill_rate":
                    yb = np.clip(np.digitize(rows.label_value.to_numpy(float), EDGES[1:-1]), 0, NBIN - 1)
                    loss = F.cross_entropy(z, torch.from_numpy(yb).to(DEV))
                else:
                    yv = (rows.label_value.to_numpy(np.float32) - ymu) / ysd
                    y = torch.from_numpy(yv.astype(np.float32)).to(DEV)
                    obs = torch.from_numpy((~rows.label_censored.to_numpy())).to(DEV)
                    if obs.sum() == 0: continue
                    loss = F.mse_loss(z.squeeze(-1)[obs], y[obs])
            opt.zero_grad(); loss.backward(); opt.step(); ep_l.append(float(loss.detach()))
        losses.append(float(np.mean(ep_l)) if ep_l else float('nan'))
    model.losses = losses
    return model, norm


@torch.no_grad()
def eval_cell(model, norm, test_w, task):
    model.eval()
    Wd = load_world(test_w)
    lb = labels_for(test_w, task)
    tr, va, te = fold(lb.snapshot_date)
    rows_all = lb[te]
    rel_idx = [torch.from_numpy(i).to(DEV) for i, _ in Wd["rel"]]
    rel_size = [n for _, n in Wd["rel"]]
    if task == "shortage_qty":
        dd = Wd["dcol"]; pk = (dd.part_id + "|" + dd.plant_id).to_numpy()
        uniq = {k: i for i, k in enumerate(sorted(set(pk)))}
        pp_of_chan = torch.from_numpy(np.array([uniq[k] for k in pk], np.int64)).to(DEV)
        n_pp = len(uniq)
    P, Y, EV = [], [], []
    for s in np.sort(rows_all.snapshot_date.unique()):
        t0 = int((pd.Timestamp(s) - Wd["w0"]).days // 7)
        if t0 < 0 or t0 >= Wd["T"]: continue
        rows = rows_all[rows_all.snapshot_date == s]
        X = torch.from_numpy(norm.transform(snapshot_windows(Wd, t0))).to(DEV)
        h = model.encode(X, rel_idx, rel_size)
        if task == "shortage_qty":
            agg = torch.zeros(n_pp, h.shape[1], device=DEV, dtype=h.dtype)
            agg.index_add_(0, pp_of_chan, h)
            cnt = torch.zeros(n_pp, 1, device=DEV, dtype=h.dtype); cnt.index_add_(0, pp_of_chan, torch.ones_like(h[:, :1]))
            hh = agg / cnt.clamp(min=1.0)
            idx = torch.from_numpy(rows.key.map(uniq).fillna(0).to_numpy(np.int64)).to(DEV)
            z = torch.sigmoid(model.head(hh[idx]).squeeze(-1))
        else:
            idx = torch.from_numpy(rows.key.map(Wd["cidx"]).fillna(0).to_numpy(np.int64)).to(DEV)
            z = model.head(h[idx])
            z = torch.softmax(z, -1) if task == "fill_rate" else z.squeeze(-1)
        P.append(z.float().cpu().numpy()); Y.append(rows.label_value.to_numpy(float))
        EV.append(~rows.label_censored.to_numpy())
    P = np.concatenate(P); Y = np.concatenate(Y); EV = np.concatenate(EV)
    if task == "fill_rate":  return {"crps": crps(np.cumsum(P, 1), Y)}
    if task == "arrival_week": return {"cindex": cindex(P, Y, EV)}
    return {"pr_auc": pr_auc(Y, P)}
