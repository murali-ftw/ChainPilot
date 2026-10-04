"""Serving path for the Phase 20 shipping decision (ml/configs/shipped.json -> "phase20_shipped"). No training.

  load_member   identity guard first (ml/serve/guard.py), then the stored neural bundle is rebuilt with the incumbent's
                own machinery (loop.load_bundle / loop._net). A Phase 19 row-family bundle is rebuilt at its recorded
                width (cfg["row_family_width"]) and fed the as-of features of ml/serve/features.py.
  predict       per requested test snapshot, exactly what phase5_heads.predict stores: arrival P (expected week), S, pT;
                fill P (22 cells); capacity P (quantiles, de-standardised).
  serve_item    every member of a configured ensemble, averaged the way the decision record states (arrival S / pT means;
                fill P means; capacity quantile means); the arrival interval item adds the stored conformal offsets.
                A "declared, not servable" item RAISES NotServable -- it is never silently replaced.
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "models"),
                os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd, torch
import loop as L, phase5_heads as P5, folds
from heads import HazardHead
import guard as G
import features as FE
from config import ARTIFACTS, REPO

SHIPPED = os.path.join(REPO, "ml", "configs", "shipped.json")
ROOTS = {"bundles": os.path.join(ARTIFACTS, "bundles"), "phase19": os.path.join(ARTIFACTS, "phase19", "bundles")}


def decision(path=SHIPPED):
    return json.load(open(path))["phase20_shipped"]


def bundle_dir(member):
    return os.path.join(ROOTS[member["root"]], member["task"], member["name"])


class Server:
    def __init__(self, world="v8"):
        self.world = world
        self._D, self._lb, self._feat = {}, {}, None

    def labels(self, task):
        if task not in self._lb:
            lb = P5.labels(self.world, task); tr, va, te = folds.fixed_split(lb.snapshot_date)
            self._lb[task] = (lb, tr, va, te)
        return self._lb[task]

    def inputs(self, task, cfg):
        key = (task, cfg.get("drop_relation"))
        if key not in self._D:
            lb, tr, _, _ = self.labels(task)
            self._D[key] = P5.device_inputs(self.world, np.sort(lb.snapshot_date[tr].unique()), cfg["wsla"],
                                            graph_shuffle=cfg.get("graph_shuffle"), drop_relation=cfg.get("drop_relation"))
        return self._D[key]

    def load_member(self, member):
        bdir = bundle_dir(member)
        cfg = G.check(bdir, member)                      # raises IdentityMismatch before anything is loaded
        B = L.load_bundle(bdir); D = self.inputs(cfg["task"], cfg)
        width = int(cfg["row_family_width"]) if cfg.get("row_family") else L.n_row_feats_of(cfg)
        m = L._net(cfg)(D["X"].shape[2], cfg["task"], cfg["arch"], cfg["depth"], fill_loss=cfg["fill_loss"],
                        n_row_feats=width, fill_head=cfg.get("fill_head", "cells22")).to(L.DEV)
        m.load_state_dict(torch.load(os.path.join(bdir, "checkpoint.pt"), map_location="cpu"))
        tl = json.load(open(os.path.join(bdir, "train_log.json")))
        return dict(model=m.eval(), cfg=cfg, D=D, tl=tl, B=B, dir=bdir)

    def predict(self, M, snapshots):
        cfg, D, task = M["cfg"], M["D"], M["cfg"]["task"]
        lb, tr, va, te = self.labels(task)
        ymu, ysd = (M["tl"]["ymu"], M["tl"]["ysd"]) if task == "capacity_strain" else (0.0, 1.0)
        want = set(pd.to_datetime(list(snapshots)))
        out, rows_all = {}, []
        if cfg.get("row_family") and self._feat is None:
            self._feat = FE.AsOfFeatures(self.world)
        with torch.no_grad():
            for t0, tg, ii in P5.batches(task, lb, te, D["W"], ymu, ysd):
                s = pd.Timestamp(lb.snapshot_date.values[ii[0]])
                if s not in want:
                    continue
                xr = None
                if cfg.get("row_family"):
                    sc = M["tl"]["row_family_scaler"]
                    assert sc["cols"] == sum((FE.FAMILY_COLS[p] for p in cfg["row_family"].split("+")), []), "scaler columns differ"
                    xr = torch.from_numpy(FE.transform(self._feat.raw(lb.iloc[ii], cfg["row_family"]), sc)).to(L.DEV)
                z = P5.forward(M["model"], D, t0, tg["idx"], xrow=xr)
                if task == "arrival_week":
                    lam, S, pT = HazardHead.distribution(z)
                    got = {"P": HazardHead.expected_time(S), "S": S, "pT": pT}
                elif task == "fill_rate":
                    got = {"P": type(M["model"].head).probs(z)}
                else:
                    got = {"P": z[:, 0, :] * ysd + ymu}
                for k, v in got.items():
                    out.setdefault(k, []).append(v.float().cpu().numpy())
                rows_all.append(ii)
        res = {k: np.concatenate(v) for k, v in out.items()}
        res["rows"] = np.concatenate(rows_all)
        return res

    def serve_item(self, name, snapshots, path=SHIPPED):
        item = decision(path)["items"][name]
        if item["status"] != "servable":
            raise G.NotServable(f"{name}: {item['status']} -- {item.get('why_not_servable', '')}")
        preds = [self.predict(self.load_member(m), snapshots) for m in item["members"]]
        rows = preds[0]["rows"]
        assert all(np.array_equal(p["rows"], rows) for p in preds), "members disagree on rows"
        agg = {k: np.mean([p[k] for p in preds], 0) for k in preds[0] if k != "rows"}
        if item.get("interval"):
            S, pT = agg["S"].astype(float), agg["pT"].astype(float)
            ew = (pT * np.arange(1, 13)).sum(1) + 13 * S[:, -1]
            agg["expected_week"] = ew
            agg["interval_days"] = np.stack([7 * ew + item["interval"]["q10_days"], 7 * ew + item["interval"]["q90_days"]], 1)
        agg["rows"] = rows
        agg["members"] = [m["name"] for m in item["members"]]
        return agg
