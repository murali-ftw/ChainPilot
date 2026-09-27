"""THE order policy -- the one place that decides WHEN an order is placed and HOW an order arrives.

Phase 12 A5/B2. Imported by ml/sim/montecarlo.py (the stock roll-forward) and ml/opt/schedule_lp.py (the
delivery schedule). Neither may carry trigger, lead-time or pipeline logic of its own:
ml/tests/test_single_order_policy.py fails if they do.

THE RULE (reports/part2/phase-12.md section 2.5, corrected against the brief with measurements):
  review    every week, per part-plant
  trigger   form "rop" (primary):  IP_w <= reorder_point_qty,   IP_w = on hand + everything ordered and not yet
                                   arrived (INCLUDING arrivals that land beyond the horizon)
            form "ss"  (sensitivity): on hand + pending - E[consumption over the P90 lead] <= safety_stock_qty
            The brief's `stock - E[consumption over lead] <= reorder_point_qty` is NOT offered: on v8 (ROP - SS) is
            1.9x planning-lead demand, so ROP already carries lead-time demand and subtracting it again double-counts.
  quantity  order-up-to:  Q = max(MOQ, lot * ceil((target - IP) / lot)),  target = ROP + cover * weekly requirement,
            cover estimated AS-OF from the part-plant's recorded line quantities (trailing 26 weeks). MOQ and lot size
            are part_plant's. Recorded lines are 2.8x MOQ, so MOQ-and-lot sizing alone would under-order.
  lead      the FULL lead-from-ORDER distribution, as-of: weeks from po_line created_ts to first receipt, over lines
            whose receipt was RECORDED <= t0, pooled per part-plant (trailing 104 weeks), shrunk to the global pmf
            below 20 receipts. The hazard head's T is NOT used as a lead: it is weeks from the SNAPSHOT for lines raised
            after t0, i.e. order-raise wait + lead. `lead_gate` measures that, and must fire on raw T.
  pipeline  the as-of store total `inventory_position_weekly.open_po_qty` (the store B1 reconciled at 100%). A
            line-level reconstruction cannot reproduce it: ~2% of lines close with ZERO delivery and write no GRN, so
            an unreceived old line is indistinguishable from a pending one. Timing comes from the part-plant's
            recent unreceived lines, each drawn from the lead pmf CONDITIONAL ON ITS AGE, P(L = a + k | L > a).
  carry     arrivals landing beyond the horizon stay in IP (so they are not re-ordered) and are reported.

WHAT THIS IS NOT. A reorder-point rule gives a FEASIBLE schedule, not an OPTIMAL one. It answers "when must I order
to avoid a stockout", not "when should I order to minimise cost"; those coincide only when holding cost is zero.
It sidesteps the missing cost parameters (holding, ordering, freight); it does not solve them.

DATA CAVEAT. part_plant.reorder_point_qty is the generator's END-OF-RUN reorder point (generator_v8.py:1675) stamped
effective_from 2016-01-01. For 2025 t0s it is close to the value then in force; for early t0s it is a later value
presented as an old one. Validation here is on 2025 only.

CONSUMPTION is the plan (part_demand_weekly p50) x N(1, 0.1265) -- the MEASURED dispersion of actual/planned
production on v8. It still represents PLAN ERROR, not demand uncertainty: no demand head exists.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
from config import WORLDS

LMAX = 52                    # lead support, weeks
PLAN_SD = 0.1265             # measured actual/planned sd on v8 (Phase 11C D.4); plan error, NOT demand uncertainty
MIN_RECEIPTS = 20
LEAD_LOOKBACK_W = 104
QTY_LOOKBACK_W = 26
PIPE_MAX_AGE_W = 26          # unreceived lines older than this are treated as zero-delivery closures


def _asof(df, col, t0):
    return df[pd.to_datetime(df[col]) <= pd.Timestamp(t0)]


# ------------------------------------------------------------------ inputs, as-of t0
class Tables:
    """Read once per world; every accessor filters as-of t0."""

    def __init__(self, world):
        D = WORLDS[world]
        self.world = world
        self.pp = pd.read_csv(f"{D}/part_plant.csv",
                              usecols=["part_id", "plant_id", "safety_stock_qty", "reorder_point_qty",
                                       "min_order_qty", "lot_size", "planning_lead_time_days"])
        ch = pd.read_csv(f"{D}/sourcing_channels.csv", usecols=["channel_id", "plant_id"])
        pol = pd.read_csv(f"{D}/po_lines.csv",
                          usecols=["po_line_id", "part_id", "channel_id", "qty_ordered", "created_ts", "recorded_ts"])
        pol = pol.merge(ch, on="channel_id")
        pol["created_ts"] = pd.to_datetime(pol.created_ts); pol["recorded_ts"] = pd.to_datetime(pol.recorded_ts)
        g = pd.read_csv(f"{D}/grn_lines.csv", usecols=["po_line_id", "event_ts", "recorded_ts"])
        g["event_ts"] = pd.to_datetime(g.event_ts); g["recorded_ts"] = pd.to_datetime(g.recorded_ts)
        first = g.sort_values("event_ts").groupby("po_line_id").head(1)
        self.pol = pol.merge(first.rename(columns={"event_ts": "rcv_ts", "recorded_ts": "rcv_rec"}),
                             on="po_line_id", how="left")
        self.pol["lead_w"] = np.clip(np.round((self.pol.rcv_ts - self.pol.created_ts).dt.days / 7.0), 1, LMAX)
        ipw = pd.read_csv(f"{D}/inventory_position_weekly.csv",
                          usecols=["part_id", "plant_id", "week_start", "open_po_qty", "recorded_ts"])
        ipw["recorded_ts"] = pd.to_datetime(ipw.recorded_ts)
        self.ipw = ipw


def params(T: Tables, t0, keys, weekly_req):
    """-> DataFrame aligned to keys: ss, rop, moq, lot, cover_w (as-of line-quantity estimate)."""
    x = T.pp.set_index(["part_id", "plant_id"]).reindex(keys)
    lo = pd.Timestamp(t0) - pd.Timedelta(weeks=QTY_LOOKBACK_W)
    h = T.pol[(T.pol.recorded_ts <= pd.Timestamp(t0)) & (T.pol.recorded_ts > lo)]
    assert len(h) and (h.recorded_ts <= pd.Timestamp(t0)).all()
    q = h.groupby(["part_id", "plant_id"]).qty_ordered.median().reindex(keys)
    wk = np.maximum(np.asarray(weekly_req, float), 1e-9)
    cover = (q.to_numpy(float) / wk)
    cover = np.where(np.isfinite(cover), cover, np.nanmedian(cover[np.isfinite(cover)]))
    return pd.DataFrame(dict(ss=x.safety_stock_qty.fillna(0).to_numpy(float),
                             rop=x.reorder_point_qty.fillna(0).to_numpy(float),
                             moq=x.min_order_qty.fillna(1).clip(lower=1).to_numpy(float),
                             lot=x.lot_size.fillna(1).clip(lower=1).to_numpy(float),
                             cover_w=np.clip(cover, 0, 52)))


def lead_pmf(T: Tables, t0, keys):
    """As-of lead-from-order pmf per part-plant, [P, LMAX+1] over weeks 0..LMAX (week 0 unused)."""
    t0 = pd.Timestamp(t0)
    h = T.pol[(T.pol.rcv_rec <= t0) & (T.pol.created_ts > t0 - pd.Timedelta(weeks=LEAD_LOOKBACK_W))]
    assert len(h) and (h.rcv_rec <= t0).all(), "as-of violation or empty lead history"
    glob = np.bincount(h.lead_w.astype(int), minlength=LMAX + 1).astype(float)
    glob /= glob.sum()
    cnt = h.groupby(["part_id", "plant_id", "lead_w"]).size()
    out = np.tile(glob, (len(keys), 1))
    n = np.zeros(len(keys))
    idx = {k: i for i, k in enumerate(keys)}
    for (p, pl, lw), c in cnt.items():
        i = idx.get((p, pl))
        if i is not None:
            if n[i] == 0:
                out[i] = 0.0
            out[i, int(lw)] += c; n[i] += c
    thin = n < MIN_RECEIPTS
    lam = np.where(thin, n / MIN_RECEIPTS, 1.0)[:, None]          # shrink thin part-plants to the global pmf
    emp = out / np.maximum(out.sum(1, keepdims=True), 1e-12)
    pmf = lam * emp + (1 - lam) * glob[None, :]
    return pmf / pmf.sum(1, keepdims=True), dict(receipts=int(len(h)), thin_part_plants=int(thin.sum()))


def pmf_median(pmf):
    return (np.cumsum(pmf, 1) < 0.5).sum(1)


def pmf_quantile(pmf, q):
    return (np.cumsum(pmf, 1) < q).sum(1)


def lead_gate(candidate_median, empirical_median, tol=1.0, need=0.80):
    """A lead distribution is admissible only if its median matches the as-of empirical lead median within tol weeks
    for at least `need` of part-plants. Raw hazard T (weeks from the SNAPSHOT) must FAIL this."""
    ok = np.abs(np.asarray(candidate_median, float) - np.asarray(empirical_median, float)) <= tol
    return dict(frac_within=float(ok.mean()), passes=bool(ok.mean() >= need), tol_weeks=tol, need=need)


def pipeline(T: Tables, t0, keys):
    """-> (store total per part-plant [P], list per part-plant of (age_weeks array, weight array)).
    Total from the as-of store; timing candidates from unreceived lines no older than PIPE_MAX_AGE_W."""
    t0 = pd.Timestamp(t0)
    iv = T.ipw[T.ipw.recorded_ts <= t0]
    assert len(iv) and (iv.recorded_ts <= t0).all()
    iv = iv.sort_values("week_start").groupby(["part_id", "plant_id"]).tail(1).set_index(["part_id", "plant_id"])
    total = iv.open_po_qty.reindex(keys).fillna(0).to_numpy(float)
    p = T.pol[(T.pol.recorded_ts <= t0) & (T.pol.created_ts <= t0)
              & ~((T.pol.rcv_rec <= t0) & T.pol.rcv_rec.notna())]
    age = ((t0 - p.created_ts).dt.days / 7.0).to_numpy()
    p = p.assign(age=np.floor(age)).query("age <= @PIPE_MAX_AGE_W")
    grp = p.groupby(["part_id", "plant_id"])
    cand = {k: (g.age.to_numpy(int), g.qty_ordered.to_numpy(float)) for k, g in grp}
    lines = [cand.get(k, (np.zeros(0, int), np.zeros(0))) for k in keys]
    recon = np.array([w.sum() for _, w in lines])
    return total, lines, dict(store_total=float(total.sum()), line_recon_total=float(recon.sum()),
                              part_plants_exact=float((recon == total).mean()),
                              part_plants_with_store_pipeline_but_no_lines=int(((total > 0) & (recon == 0)).sum()))


def assert_pipeline_identity(sim_total, store_total):
    """The simulated pipeline at t0 must equal the as-of store per part-plant. Fires on one part-plant dropped."""
    d = np.abs(np.asarray(sim_total, float) - np.asarray(store_total, float))
    assert (d < 0.5).all(), f"pipeline identity: {int((d >= 0.5).sum())} part-plants differ from open_po_qty"
    return True


# ------------------------------------------------------------------ sampling helpers
def _sample_cdf(cdf, u):
    """cdf [P, K], u [P, N] -> index [P, N]."""
    out = np.empty(u.shape, np.int64)
    for i in range(cdf.shape[0]):
        out[i] = np.searchsorted(cdf[i], u[i], side="right")
    return np.minimum(out, cdf.shape[1] - 1)


def sample_fill(pf, N, rng, fill_mid):
    cdf = np.cumsum(pf, 1); cdf[:, -1] = 1.0
    return fill_mid[_sample_cdf(cdf, rng.random((pf.shape[0], N)))]


def pipeline_arrivals(total, lines, pmf, N, H, rng):
    """Schedule the as-of pipeline: each unreceived recent line draws its arrival from P(L = a+k | L > a); the
    part-plant's draws are scaled so ordered units sum to the store total. Part-plants with a store pipeline but no
    candidate line draw from the unconditional pmf. -> ordered units scheduled [P, H, N]."""
    P = len(total)
    sched = np.zeros((P, H, N), np.float32)
    for i in range(P):
        if total[i] <= 0:
            continue
        ages, w = lines[i]
        if len(ages) == 0:
            ages, w = np.zeros(1, int), np.ones(1)
        w = w / w.sum() * total[i]
        for a, q in zip(ages, w):
            tail = pmf[i, a + 1:].copy() if a + 1 <= LMAX else np.zeros(0)
            if tail.sum() <= 0:                                  # older than any observed lead: arrives within 4 weeks
                k = rng.integers(1, 5, N)
            else:
                cdf = np.cumsum(tail / tail.sum())
                k = 1 + np.minimum(np.searchsorted(cdf, rng.random(N), side="right"), len(tail) - 1)
            k = np.minimum(k, H - 1)                             # week index from t0 (1 = next week)
            np.add.at(sched[i], (k - 1, np.arange(N)), q)
    return sched


# ------------------------------------------------------------------ the policy loop
def simulate(I0, cons, par, pmf, pfill, pipe_sched, W, rng, form="rop", fill_mid=None):
    """Roll stock forward under the policy. Arrivals are ENDOGENOUS (they depend on the stock path), so the loop owns
    the roll-forward; montecarlo.roll_forward on the returned arrivals must reproduce `pos` exactly (asserted by the
    caller). cons [P, W, N]; pipe_sched [P, H, N] ordered units; returns arrivals [P, W, N], positions [P, W, N] and
    bookkeeping."""
    P, _, N = cons.shape
    H = pipe_sched.shape[1]
    ordered = pipe_sched.astype(np.float32).copy()                  # ordered units by arrival week (for IP)
    f_pipe = sample_fill(pfill, N, rng, fill_mid)                    # one fill draw per part-plant-path for the pipeline
    arriving = ordered * f_pipe[:, None, :]                          # delivered units by arrival week
    cdf = np.cumsum(pmf, 1); cdf[:, -1] = 1.0
    L90 = pmf_quantile(pmf, 0.90).astype(float)
    wk = np.maximum(cons.mean((1, 2)), 1e-9)
    target = par.rop.to_numpy() + par.cover_w.to_numpy() * wk
    rop, ss = par.rop.to_numpy()[:, None], par.ss.to_numpy()[:, None]
    moq, lot = par.moq.to_numpy()[:, None], par.lot.to_numpy()[:, None]
    I = np.repeat(np.asarray(I0, float)[:, None], N, 1)
    pos = np.empty((P, W, N), np.float32)
    arr_out = np.zeros((P, W, N), np.float32)
    n_orders = np.zeros((P, N))
    units_ordered = np.zeros((P, N))
    for w in range(W):
        arr_out[:, w] = arriving[:, w]
        I = I + arriving[:, w] - cons[:, w]
        pos[:, w] = I
        pending = ordered[:, w + 1:].sum(1)
        IP = I + pending
        if form == "rop":
            trig = IP <= rop
        elif form == "ss":
            trig = (I + pending - (wk * L90)[:, None]) <= ss
        else:
            raise ValueError(form)
        if trig.any():
            need = np.maximum(target[:, None] - IP, 0.0)
            Q = np.maximum(moq, lot * np.ceil(need / lot))
            Q = np.where(trig, Q, 0.0)
            L = _sample_cdf(cdf, rng.random((P, N)))                 # weeks from order, >= 1
            L = np.maximum(L, 1)
            f = sample_fill(pfill, N, rng, fill_mid)
            k = np.minimum(w + L, H - 1)                             # beyond H-1 is lumped at the last carried week
            ii, nn = np.nonzero(trig)
            np.add.at(ordered, (ii, k[ii, nn], nn), Q[ii, nn])
            np.add.at(arriving, (ii, k[ii, nn], nn), Q[ii, nn] * f[ii, nn])
            n_orders += trig; units_ordered += Q
    post = arriving[:, W:].sum(1)
    return arr_out, pos, dict(orders_per_pp=float(n_orders.mean()), units_ordered_per_pp=float(units_ordered.mean()),
                              post_horizon_per_pp=float(post.mean()),
                              pipeline_units_per_pp=float(pipe_sched.sum(1).mean()),
                              L90_median=float(np.median(L90)))


# ------------------------------------------------------------------ the simulation arm
def make_draw(form="rop", plan_sd=PLAN_SD, carry_weeks=LMAX + 13, use_pipeline=True, block=800):
    """-> a draw(world, t0, sub, W, N, rng, op) for ml/sim/phase12_b2_validate.py and montecarlo's grid.
    Part-plants are processed in blocks so the full grid at N=1000 fits in memory."""
    cache = {}

    def draw(world, t0, sub, W, N, rng, op):
        import montecarlo as MC
        if world not in cache:
            cache[world] = Tables(world)
        T = cache[world]
        keys = list(zip(sub.part_id, sub.plant_id))
        heads = MC.read_heads(world, t0)
        plan = MC.forward_requirement(world, t0, sub, W)             # [P, W]
        cons = (plan[:, :, None] * np.clip(rng.normal(1.0, plan_sd, (len(sub), W, N)), 0, None)).astype(np.float32)
        par = params(T, t0, keys, plan.mean(1))
        pmf, lmeta = lead_pmf(T, t0, keys)
        pfill, fmeta = MC.fill_by_part_plant(heads, sub)
        total, lines, pmeta = pipeline(T, t0, keys)
        if not use_pipeline:
            total = np.zeros_like(total)
        I0 = op["qty_on_hand"].fillna(0).to_numpy(float)
        H = W + carry_weeks
        arr = np.zeros_like(cons)
        agg = {}
        for b0 in range(0, len(sub), block):
            sl = slice(b0, b0 + block)
            sched = pipeline_arrivals(total[sl], lines[sl], pmf[sl], N, H, rng)
            assert_pipeline_identity(sched.sum(1).mean(1), total[sl])
            a, pos, m = simulate(I0[sl], cons[sl], par.iloc[sl], pmf[sl], pfill[sl], sched, W, rng,
                                 form=form, fill_mid=MC.FILL_MID)
            pos2, _ = MC.roll_forward(I0[sl], a, cons[sl], np.ones(len(a)))
            assert np.allclose(pos2, pos, atol=1e-2, rtol=1e-5), "policy loop and roll_forward disagree -- two ledgers"
            arr[sl] = a
            n = len(a)
            for k, v in m.items():
                agg.setdefault(k, []).append((v, n))
        meta = {k: float(sum(v * n for v, n in xs) / sum(n for _, n in xs)) for k, xs in agg.items()}
        meta.update(lead=lmeta, fill=fmeta, pipeline=pmeta, form=form, plan_sd=plan_sd, use_pipeline=use_pipeline,
                    consumption_basis="plan x N(1, 0.1265): PLAN error, not demand uncertainty -- no demand head")
        return arr, cons, meta

    return draw


SIM_ARMS = {"policy_rop": make_draw("rop"), "policy_ss": make_draw("ss"),
            "policy_rop_nopipe": make_draw("rop", use_pipeline=False)}


# ------------------------------------------------------------------ the deterministic plan, for schedule_lp
def receipt_plan(world, t0, keys, plan, T: Tables = None, form="rop"):
    """The policy's TIMING on the expected path, for ml/opt/schedule_lp.py's mode="policy".

    Expected consumption = the plan; the pipeline lands on its EXPECTED week (age-conditional mean, rounded); each
    triggered order lands at placement + the MEDIAN lead. Returns per part-plant: opening level (as-of store),
    expected pipeline receipts per horizon week, the weeks in which a receipt may land (allowed[w]), and the policy's
    own order-up-to quantities (for comparison with the MILP's batching). Timing only: the MILP owns quantity."""
    T = T or Tables(world)
    t0 = pd.Timestamp(t0)
    P, W = plan.shape
    pmf, _ = lead_pmf(T, t0, keys)
    L50 = np.maximum(pmf_median(pmf), 1)
    L90 = pmf_quantile(pmf, 0.90)
    par = params(T, t0, keys, plan.mean(1))
    total, lines, _ = pipeline(T, t0, keys)
    iv = T.ipw[T.ipw.recorded_ts <= t0]
    D = WORLDS[world]
    oh = pd.read_csv(f"{D}/inventory_position_weekly.csv", usecols=["part_id", "plant_id", "week_start",
                                                                    "qty_on_hand", "recorded_ts"])
    oh = oh[pd.to_datetime(oh.recorded_ts) <= t0].sort_values("week_start").groupby(["part_id", "plant_id"]).tail(1)
    I0 = oh.set_index(["part_id", "plant_id"]).qty_on_hand.reindex(keys).fillna(0).to_numpy(float)
    out = []
    for i in range(P):
        pipe = np.zeros(W + LMAX + 1)
        ages, w = lines[i]
        if total[i] > 0:
            if len(ages) == 0:
                ages, w = np.zeros(1, int), np.ones(1)
            w = w / w.sum() * total[i]
            for a, q in zip(ages, w):
                tail = pmf[i, a + 1:]
                k = 1 + int(round(float((np.arange(len(tail)) * tail).sum() / tail.sum()))) if tail.sum() > 0 else 2
                pipe[min(k, len(pipe)) - 1] += q
        allowed = np.zeros(W, bool)
        own_q = np.zeros(W)
        sched = pipe.copy()
        I = I0[i]
        tgt = par.rop[i] + par.cover_w[i] * max(plan[i].mean(), 1e-9)
        for wk in range(W):
            I = I + sched[wk] - plan[i, wk]
            IP = I + sched[wk + 1:].sum()
            trig = IP <= par.rop[i] if form == "rop" else (IP - plan[i].mean() * L90[i]) <= par.ss[i]
            if trig:
                Q = max(par.moq[i], par.lot[i] * np.ceil(max(tgt - IP, 0) / par.lot[i]))
                k = wk + int(L50[i])
                if k < len(sched):
                    sched[k] += Q
                if k < W:
                    allowed[k] = True; own_q[k] += Q
        out.append(dict(I0=float(I0[i]), pipeline=pipe[:W].tolist(), allowed=allowed.tolist(),
                        policy_qty=own_q.tolist(), L50=int(L50[i]), L90=int(L90[i]),
                        ss=float(par.ss[i]), rop=float(par.rop[i])))
    return out
