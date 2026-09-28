"""Phase 13 F2 -- as-of history-ratio features for fill, per candidate key, with a shrunk hierarchy.

CLOSED LINES AS-OF t0 (Stage 0.1): a PO line raised in (t0 - 104 wk, t0] is closed if
    closed-received : a final GRN line RECORDED <= t0          -> arrived = units received, recorded <= t0
    closed-zero     : a supplier rejection or a qty->0 revision RECORDED <= t0, and no GRN recorded <= t0 -> arrived 0
Every row that feeds a count carries the latest recorded_ts it depended on; `assert_asof_history` fails if any exceeds t0.

ESTIMATORS per cell c:  ratio_of_sums  sum(arrived_c) / sum(ordered_c)      mean_of_ratios  mean(arrived / ordered)

KEYS: ps = (part, supplier) | sp = (supplier, plant) | psp = (part, supplier, plant) | hier = the shrunk hierarchy
    psp -> ps -> supplier -> global,   r_hat_c = (kappa_t * mu_parent + sum arrived_c) / (kappa_t + sum ordered_c)
kappa_t is FITTED per tier by method of moments against the tier's observed dispersion around its parent (beta-
binomial variance with UNITS as trials), never chosen. Fill is not binomial -- atoms at 0 and 1, units bundled in
lines -- so this is a motivated SHRINKAGE DEVICE, not a correctly specified likelihood. It is validated empirically.

THE PRIOR IS AS-OF TOO (G2). mu_coarse and kappa are fitted only from the closed-line set as-of t0; the fitted prior
carries the latest recorded_ts of the data it saw, and the same assertion covers it. A kappa fitted on full history
admits future information into every cell at once, most invisibly into the cold-start cells.

Per row the feature is [ratio, has_own_history]: raw keys fall back to the as-of global ratio when the cell is empty.
For `hier`, `tier` records the finest tier with at least one closed line -- the tier that "fired" -- so every metric can
be broken out by it (a strong global fallback must not be mistaken for the ratio).
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np, pandas as pd
from config import WORLDS, ARTIFACTS

LOOKBACK_W = 104
KEYS = {"ps": ["part_id", "supplier_id"], "sp": ["supplier_id", "plant_id"],
        "psp": ["part_id", "supplier_id", "plant_id"], "supplier": ["supplier_id"]}
TIERS = ("psp", "ps", "supplier", "global")


class HistoryLeak(AssertionError):
    pass


def assert_asof_history(meta, t0):
    """Fires if the counts OR the fitted prior depended on anything recorded after t0, or if the history is empty."""
    t0 = pd.Timestamp(t0)
    if meta["n_closed"] <= 0:
        raise HistoryLeak(f"history at {t0.date()} is EMPTY -- every row would silently fall back to a constant")
    if pd.Timestamp(meta["counts_max_recorded"]) > t0:
        raise HistoryLeak(f"cell COUNTS at {t0.date()} use rows recorded {meta['counts_max_recorded']} (> t0)")
    if meta.get("key") == "hier":
        # the prior is a FITTED quantity and gets the same as-of treatment as the counts -- and an empty or
        # undated prior is a failure, not a pass (a first version passed G2 vacuously on exactly that)
        pmr = meta.get("prior_max_recorded")
        if meta.get("prior_n", 0) <= 0 or pmr is None or pd.isna(pmr):
            raise HistoryLeak(f"fitted PRIOR at {t0.date()} has no dated provenance (n={meta.get('prior_n')}) -- cannot be verified as-of")
        if pd.Timestamp(pmr) > t0:
            raise HistoryLeak(f"fitted PRIOR (mu, kappa) at {t0.date()} uses rows recorded {pmr} (> t0)")
    return True


class FillHistory:
    def __init__(self, world="v8"):
        D = WORLDS[world]
        pol = pd.read_csv(f"{D}/po_lines.csv", usecols=["po_line_id", "part_id", "channel_id", "qty_ordered",
                                                        "created_ts", "recorded_ts"])
        pol["created_ts"] = pd.to_datetime(pol.created_ts); pol["line_rec"] = pd.to_datetime(pol.recorded_ts)
        ch = pd.read_csv(f"{D}/sourcing_channels.csv", usecols=["channel_id", "part_id", "supplier_id", "plant_id"])
        self.channels = ch.set_index("channel_id")
        pol = pol.drop(columns="part_id").merge(ch, on="channel_id").drop(columns="recorded_ts")
        g = pd.read_csv(f"{D}/grn_lines.csv", usecols=["po_line_id", "qty_received", "is_final_receipt", "recorded_ts"])
        g["recorded_ts"] = pd.to_datetime(g.recorded_ts)
        g["final"] = g.is_final_receipt.astype(str).str.lower().isin(["true", "1"])
        self.grn = g
        ack = pd.read_csv(f"{D}/supplier_acknowledgements.csv", usecols=["po_line_id", "ack_status", "recorded_ts"])
        ack = ack[ack.ack_status == "rejected"]
        rev = pd.read_csv(f"{D}/po_line_revisions.csv", usecols=["po_line_id", "field_changed", "new_value", "recorded_ts"])
        rev = rev[(rev.field_changed == "qty") & (pd.to_numeric(rev.new_value, errors="coerce") == 0)]
        z = pd.concat([ack[["po_line_id", "recorded_ts"]], rev[["po_line_id", "recorded_ts"]]])
        z["recorded_ts"] = pd.to_datetime(z.recorded_ts)
        self.zero_sig = z.groupby("po_line_id").recorded_ts.min()
        self.pol = pol

    # ---------------------------------------------------------------- closed lines
    def closed(self, t0, leak_counts=False):
        """leak_counts=True is the G1 falsification: every filter on recorded_ts is dropped."""
        t0 = pd.Timestamp(t0)
        horizon = pd.Timestamp("2100-01-01") if leak_counts else t0
        p = self.pol[(self.pol.created_ts > t0 - pd.Timedelta(weeks=LOOKBACK_W)) & (self.pol.created_ts <= t0)]
        p = p[p.line_rec <= horizon]
        g = self.grn[self.grn.recorded_ts <= horizon]
        rec = g.groupby("po_line_id").qty_received.sum()
        g_last = g.groupby("po_line_id").recorded_ts.max()
        fin = g[g.final].groupby("po_line_id").recorded_ts.max()
        zs = self.zero_sig[self.zero_sig <= horizon]
        p = p.assign(fin_rec=p.po_line_id.map(fin), grn_rec=p.po_line_id.map(g_last), z_rec=p.po_line_id.map(zs))
        received = p.fin_rec.notna()
        zero = p.z_rec.notna() & p.grn_rec.isna()
        p = p[received | zero].copy()
        p["arrived"] = np.where(p.fin_rec.notna(), p.po_line_id.map(rec).fillna(0.0), 0.0)
        p["ordered"] = p.qty_ordered.clip(lower=1).astype(float)
        p["arrived"] = np.minimum(p.arrived, p.ordered)
        p["dep_rec"] = p[["line_rec", "grn_rec", "z_rec"]].max(axis=1)
        return p

    def closed_full(self):
        """G2 ONLY: the closed set over the whole dataset with no as-of filter at all."""
        end = self.pol.created_ts.max()
        saved = LOOKBACK_W
        try:
            globals()["LOOKBACK_W"] = 10_000
            return self.closed(end, leak_counts=True)
        finally:
            globals()["LOOKBACK_W"] = saved

    # ---------------------------------------------------------------- shrinkage prior
    @staticmethod
    def fit_kappa(A, N, mu_parent):
        """Method of moments: sum_c N_c [(r_c - mu_p)^2 - mu_p(1-mu_p)(kappa+N_c)/(N_c(kappa+1))] = 0 in kappa.
        Returns +inf (full pooling) when the cells are no more dispersed than binomial noise."""
        r = A / N
        mu = np.clip(mu_parent, 1e-6, 1 - 1e-6)
        obs = N * (r - mu) ** 2
        # beta-binomial: N Var(r) = mu(1-mu)(N + kappa)/(1 + kappa): mu(1-mu)N at kappa -> 0 (no pooling),
        # mu(1-mu) at kappa -> inf (binomial noise only, full pooling). The model term FALLS as kappa rises, so
        # f(kappa) = sum(obs - model) RISES in kappa. (A first version bisected with the sign reversed.)

        def f(lk):
            k = np.exp(lk)
            return float(np.sum(obs - mu * (1 - mu) * (N + k) / (1 + k)))
        lo, hi = np.log(1e-3), np.log(1e9)
        if f(lo) >= 0:
            return 1e-3            # over-dispersed beyond even kappa -> 0: cells stand on their own
        if f(hi) <= 0:
            return np.inf          # no more dispersed than binomial noise: pool fully
        for _ in range(100):
            m = 0.5 * (lo + hi)
            if f(m) < 0:
                lo = m
            else:
                hi = m
        return float(np.exp(0.5 * (lo + hi)))

    def prior(self, cl):
        """-> per-tier (mu per cell, kappa) fitted on the closed set `cl` only."""
        g_mu = cl.arrived.sum() / cl.ordered.sum()
        out = dict(global_mu=float(g_mu), kappa={}, prior_max_recorded=cl.dep_rec.max(), prior_n=int(len(cl)))
        sup = cl.groupby("supplier_id")[["arrived", "ordered"]].sum()
        out["kappa"]["supplier"] = self.fit_kappa(sup.arrived.values, sup.ordered.values, g_mu)
        k = out["kappa"]["supplier"]
        sup_mu = (k * g_mu + sup.arrived) / (k + sup.ordered) if np.isfinite(k) else pd.Series(g_mu, index=sup.index)
        ps = cl.groupby(["part_id", "supplier_id"])[["arrived", "ordered"]].sum()
        par = ps.index.get_level_values("supplier_id").map(sup_mu).to_numpy(float)
        out["kappa"]["ps"] = self.fit_kappa(ps.arrived.values, ps.ordered.values, par)
        k = out["kappa"]["ps"]
        ps_mu = (k * par + ps.arrived) / (k + ps.ordered) if np.isfinite(k) else pd.Series(par, index=ps.index)
        psp = cl.groupby(["part_id", "supplier_id", "plant_id"])[["arrived", "ordered"]].sum()
        par2 = pd.Series(ps_mu.values, index=ps.index).reindex(psp.index.droplevel("plant_id")).to_numpy(float)
        out["kappa"]["psp"] = self.fit_kappa(psp.arrived.values, psp.ordered.values, par2)
        out["sup_mu"], out["ps_mu"] = sup_mu, pd.Series(np.asarray(ps_mu, float), index=ps.index)
        return out

    # ---------------------------------------------------------------- features
    def features(self, t0, channels, key, est="ros", leak_counts=False, leak_prior=False):
        """-> DataFrame indexed like `channels` with ratio_x0 (ratio), ratio_x1 (has own history), tier; and meta."""
        t0 = pd.Timestamp(t0)
        cl = self.closed(t0, leak_counts=leak_counts)
        meta = dict(n_closed=int(len(cl)), counts_max_recorded=cl.dep_rec.max() if len(cl) else pd.NaT,
                    prior_max_recorded=None, key=key, est=est)
        chan = self.channels.reindex(channels)
        g_mu = cl.arrived.sum() / cl.ordered.sum() if len(cl) else np.nan
        if key == "hier":
            # G2 falsification: mu and kappa fitted on the FULL history -- every line ever raised, every record
            pcl = self.closed_full() if leak_prior else cl
            pr = self.prior(pcl)
            meta["prior_max_recorded"] = pr["prior_max_recorded"]; meta["prior_n"] = pr["prior_n"]
            meta["kappa"] = {k: (float(v) if np.isfinite(v) else "inf") for k, v in pr["kappa"].items()}
            psp = cl.groupby(["part_id", "supplier_id", "plant_id"])[["arrived", "ordered"]].sum()
            kp = pr["kappa"]["psp"]
            part = chan.part_id
            ps_mu = pr["ps_mu"].reindex(pd.MultiIndex.from_arrays([part, chan.supplier_id])).to_numpy(float)
            sup_mu = chan.supplier_id.map(pr["sup_mu"]).to_numpy(float)
            parent = np.where(np.isfinite(ps_mu), ps_mu, np.where(np.isfinite(sup_mu), sup_mu, pr["global_mu"]))
            ti = pd.MultiIndex.from_arrays([part, chan.supplier_id, chan.plant_id])
            A = psp.arrived.reindex(ti).fillna(0).to_numpy(float); N = psp.ordered.reindex(ti).fillna(0).to_numpy(float)
            r = (kp * parent + A) / (kp + N) if np.isfinite(kp) else parent
            r = np.where(N > 0, r, parent)
            tier = np.where(N > 0, "psp", np.where(np.isfinite(ps_mu), "ps", np.where(np.isfinite(sup_mu), "supplier", "global")))
            out = pd.DataFrame(dict(ratio_x0=r, ratio_x1=(N > 0).astype(float), tier=tier), index=chan.index)
        else:
            cols = KEYS[key]
            lk = chan
            if est == "ros":
                agg = cl.groupby(cols)[["arrived", "ordered"]].sum()
                val = agg.arrived / agg.ordered
            else:
                val = (cl.arrived / cl.ordered).groupby([cl[c] for c in cols]).mean()
            ki = pd.MultiIndex.from_arrays([lk[c] for c in cols]) if len(cols) > 1 else pd.Index(lk[cols[0]])
            v = val.reindex(ki).to_numpy(float)
            out = pd.DataFrame(dict(ratio_x0=np.where(np.isfinite(v), v, g_mu), ratio_x1=np.isfinite(v).astype(float),
                                    tier=np.where(np.isfinite(v), key, "global")), index=chan.index)
        assert_asof_history(meta, t0)
        return out, meta


def attach(lb, key, est="ros", world="v8", cache=True):
    """Add ratio_x0 / ratio_x1 / ratio_tier to a fill labels frame, snapshot by snapshot, as-of each snapshot."""
    tag = f"phase13_ratio_{world}_{key}_{est}"
    path = os.path.join(ARTIFACTS, "cache", tag + ".npz")
    snaps = np.sort(lb.snapshot_date.unique())
    if cache and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        f = pd.DataFrame(dict(snapshot_date=pd.to_datetime(z["snap"]), key=z["chan"], ratio_x0=z["x0"],
                              ratio_x1=z["x1"], ratio_tier=z["tier"]))
    else:
        FH = FillHistory(world)
        parts, metas = [], []
        for s in snaps:
            ch = lb.key[lb.snapshot_date == s].unique()
            o, m = FH.features(s, ch, key, est)
            parts.append(pd.DataFrame(dict(snapshot_date=pd.Timestamp(s), key=o.index, ratio_x0=o.ratio_x0.values,
                                           ratio_x1=o.ratio_x1.values, ratio_tier=o.tier.values)))
            metas.append(m)
        f = pd.concat(parts, ignore_index=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, snap=f.snapshot_date.astype(str).values, chan=f.key.values, x0=f.ratio_x0.values,
                 x1=f.ratio_x1.values, tier=f.ratio_tier.values)
    out = lb.merge(f, on=["snapshot_date", "key"], how="left", validate="many_to_one")
    assert out.ratio_x0.notna().all(), "a scoring row has no history-ratio feature"
    assert len(out) == len(lb)
    return out
