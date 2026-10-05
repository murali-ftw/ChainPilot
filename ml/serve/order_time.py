"""Phase 22 Stage 2e -- the ORDER-TIME arrival product as a loadable, identity-guarded service (new file; nothing existing edited).

What it says for a PO line on the day it is raised (pre-registration D6; Stage 2 choices, all on VALIDATION):
  expected arrival   the channel's shrunk Kaplan-Meier median lead (ml/data/grpstats.py, k = 10, L5 chain with the true
                     order month) + the validation offset a, in weeks from creation
  80% interval       split-conformal on validation signed-day residuals of that date, per creation month where validation
                     showed > 5 points of spread across months (Stage 2)
  late-risk score    the STRICT flag: the 5-seed mean P(late vs contract) of LightGBM on BASE_clean read at the panel row
                     BEFORE the creation week + the L4 group block (flag_lag1). Stage 2 measured it WATCHLIST (no alert bar):
                     it is a ranked watchlist score, and `watch` marks the top 5% by the validation threshold.
Inputs are as-of the Monday tau on or before creation:
  * panel features from the CLEAN cache (ml/data/clean_panel.py; asserted meta["clean_of"] is set) at the row whose week ends
    by tau (asserted);
  * group statistics from grpstats.Builder at tau (its own recorded_ts <= tau assertions).
Never read: PRIVILEGED paths, inventory_position_weekly, part_demand_weekly, a raw (unclean) panel.

  build_bundle()                          -> ml/artifacts/phase22/serve/order_time_v8clean/product.json
  OrderTimeService(bundle_dir, expected)  raises guard.IdentityMismatch if the bundle is not the configured product
"""
from __future__ import annotations
import os, sys, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "baselines"),
                os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import config
from guard import IdentityMismatch

ART = config.ARTIFACTS
MODELS = os.path.join(ART, "phase22", "models")
BUNDLE = os.path.join(ART, "phase22", "serve", "order_time_v8clean")
NAME = "order_time|world=v8clean|date=km_k10|flag=flag_lag1_k10|conformal=month"
SEEDS = (7, 17, 27, 37, 47)


def sha1(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def build_bundle(order_time_json=os.path.join(ART, "phase22", "order_time_v8clean.json"), world="v8clean"):
    """Freeze the Stage 2 choices (read from the scorer's output) and the persisted flag models into one product file."""
    ot = json.load(open(order_time_json))
    iv = ot["interval"]
    flags = []
    for s in SEEDS:
        ident = json.load(open(os.path.join(MODELS, f"{world}_arrival_place_p22_flag_lag1_k10_s{s}.identity.json")))
        flags.append(dict(seed=s, file=ident["models"][0], sha1=ident["sha1"][0], features=ident["features"]))
    assert all(f["features"] == flags[0]["features"] for f in flags)
    # the watch threshold: validation 5%-coverage cut of the 5-seed mean flag
    v = np.mean([np.load(os.path.join(ART, "phase22", "preds", f"{world}_arrival_place_p22_flag_lag1_k10_s{s}_val.npz"))["P"] for s in SEEDS], 0)
    thr = float(np.quantile(v, 0.95))
    prod = dict(name=NAME, world=world, parent=world.replace("clean", ""), k=10, km_offset_a_weeks=ot["date_estimators"]["km_offset_a_weeks"],
                date_estimator=ot["date_estimators"]["chosen_on_validation"],
                conformal=dict(q10_days=iv["q10_days"], q90_days=iv["q90_days"], month_specific=iv["month_specific_used"],
                               month_quantiles={str(k): v_ for k, v_ in iv["month_quantiles"].items()}),
                flag=dict(models=flags, aggregate="mean of 5 seeds", watch_threshold_val_top5pct=thr, panel_row="week before tau"),
                built_from=dict(order_time_json=os.path.basename(order_time_json), scorer_commit=ot["stamp"]["code_commit"]))
    os.makedirs(BUNDLE, exist_ok=True)
    for f in flags:
        dst = os.path.join(BUNDLE, f["file"])
        if not os.path.exists(dst):
            open(dst, "wb").write(open(os.path.join(MODELS, f["file"]), "rb").read())
    path = os.path.join(BUNDLE, "product.json")
    assert not os.path.exists(path) or json.load(open(path)) == prod, "refusing to overwrite a different product bundle"
    json.dump(prod, open(path, "w"), indent=1)
    return path


class OrderTimeService:
    def __init__(self, bundle_dir=BUNDLE, expected_name=NAME):
        import lightgbm as lgb
        p = os.path.join(bundle_dir, "product.json")
        if not os.path.exists(p):
            raise IdentityMismatch(f"no product bundle at {bundle_dir}")
        self.prod = json.load(open(p))
        if self.prod.get("name") != expected_name:
            raise IdentityMismatch(f"bundle is {self.prod.get('name')!r}, configured {expected_name!r}")
        self.models = []
        for f in self.prod["flag"]["models"]:
            fp = os.path.join(bundle_dir, f["file"])
            if not os.path.exists(fp) or sha1(fp) != f["sha1"]:
                raise IdentityMismatch(f"flag model {f['file']} missing or its SHA-1 differs from the bundle's record")
            self.models.append(lgb.Booster(model_file=fp))
        self.features = self.prod["flag"]["models"][0]["features"]
        for m in self.models:
            if list(m.feature_name()) != list(self.features):
                raise IdentityMismatch("a flag model's feature list differs from the product's")
        import phase21_paths as PP
        PP.register()
        w, pw = self.prod["world"], self.prod["parent"]
        config.WORLDS[w] = config.WORLDS[pw]; config.EXPECTED_PANEL_D[w] = 15
        meta = json.load(open(os.path.join(config.CACHE, w, "meta.json")))
        if not meta.get("clean_of"):
            raise IdentityMismatch(f"the panel cache for {w} is not a clean cache")
        import phase7_fit as P7, grpstats as GS
        self.P7, self.GS = P7, GS
        self.W = P7.World(w)
        self.src = GS.Source(pw)
        self.B = GS.Builder(self.src)

    def features_for(self, lines):
        """lines: DataFrame[channel_id, created_ts, qty_ordered]. -> (X for the flag, Z group rows, tau)."""
        GS = self.GS
        cr = pd.to_datetime(lines.created_ts)
        tau = (cr.dt.normalize() - pd.to_timedelta(cr.dt.weekday, unit="D")).to_numpy("datetime64[ns]")
        row_date = pd.to_datetime(tau) - pd.Timedelta(days=7)
        assert ((row_date + pd.Timedelta(days=7)) <= pd.to_datetime(tau)).all(), "panel row extends past tau"
        lb = pd.DataFrame({"snapshot_date": row_date, "key": lines.channel_id.to_numpy()})
        Xp = self.W.channel_features(lb, with_ids=False, flat=True)
        chan = lines.channel_id.map(self.src.cidx).to_numpy(np.int64)
        month = (cr.dt.month.to_numpy() - 1).astype(np.int64)
        Z, _ = GS.build_rows(self.B, tau, chan, month, None, want_fill=False)
        X = pd.concat([Xp, pd.DataFrame({"log1p_qty_ordered": np.log1p(np.maximum(lines.qty_ordered.to_numpy(float), 1.0)).astype(np.float32)}),
                       GS.assemble_arrival(Z, self.prod["k"], "L4")], axis=1).astype(np.float32)
        if list(X.columns) != list(self.features):
            raise IdentityMismatch("served feature columns differ from the trained flag's")
        return X, Z, tau

    def predict(self, lines):
        X, Z, tau = self.features_for(lines)
        weeks = self.GS.standalone_arrival_weeks(Z, self.prod["k"]) + self.prod["km_offset_a_weeks"]
        cq = self.prod["conformal"]
        mo = pd.to_datetime(lines.created_ts).dt.month.to_numpy()
        q = [cq["month_quantiles"].get(str(m), (cq["q10_days"], cq["q90_days"])) if cq["month_specific"] else (cq["q10_days"], cq["q90_days"]) for m in mo]
        lo = 7 * weeks + np.array([x[0] for x in q]); hi = 7 * weeks + np.array([x[1] for x in q])
        p = np.mean([m.predict(X) for m in self.models], 0)
        created = pd.to_datetime(lines.created_ts).dt.normalize()
        return pd.DataFrame({"expected_lead_weeks": weeks, "expected_arrival": created + pd.to_timedelta(np.round(7 * weeks), unit="D"),
                             "interval_lo_days": lo, "interval_hi_days": hi, "p_late": p,
                             "watch": p >= self.prod["flag"]["watch_threshold_val_top5pct"]})
