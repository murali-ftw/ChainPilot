"""Joint-structure probes for run 3. Read-only. Mirrors validator.py population rules."""
import pandas as pd, numpy as np, sys, json
D = "/Users/muralik/Documents/Programs/HADES_v4/chainpilot_rane_15yr_synthetic_dataset_v3_FINAL/"
def L(n, **kw): return pd.read_csv(D+n+".csv", **kw)
out = {}
def spear(x,y):
    x=pd.Series(x).rank(); y=pd.Series(y).rank()
    return float(x.corr(y))

# ---------- shared PO-line population (mirrors fill_population) ----------
po = L("po_lines", parse_dates=["original_promise_date","created_ts"])
grn = L("grn_lines", parse_dates=["event_ts"])
qi  = L("quality_inspections")
rev = L("po_line_revisions")
cal = L("calendar", parse_dates=["date"])

acc_map = qi.dropna(subset=["qty_accepted"]).set_index("grn_line_id")["qty_accepted"]
grn["acc"] = grn["grn_line_id"].map(acc_map).fillna(grn["qty_received"])
agg = grn.groupby("po_line_id").agg(acc=("acc","sum"), rec=("qty_received","sum"),
                                    first=("event_ts","min"))
po = po.join(agg, on="po_line_id")
po["acc"] = po["acc"].fillna(0)

# regime per date
reg = (cal.drop_duplicates(subset=["date"]).set_index("date")["regime_flag"] if "regime_flag" in cal.columns else None)
po["regime"] = po["original_promise_date"].map(reg).fillna("normal") if reg is not None else "normal"

cancelled = set(rev.loc[rev["reason_code"]=="buyer_cancellation","po_line_id"])
span_end = max(po["created_ts"].max(), grn["event_ts"].max())
censor_cut = span_end - pd.Timedelta(days=90)

m = (po["qty_ordered"]>0) & (~po["po_line_id"].isin(cancelled)) \
    & (po["original_promise_date"] <= censor_cut)
pop = po[m].copy()
pop["fill"] = np.minimum(1.0, pop["acc"]/pop["qty_ordered"])
pop["short"] = (pop["fill"] < 0.9999).astype(int)
norm = pop[pop["regime"]=="normal"] if (pop["regime"]=="normal").any() else pop

# ---------- 1. late & short co-occur ----------
arr = norm.dropna(subset=["first","original_promise_date"]).copy()
arr["late"] = (arr["first"].dt.normalize() > arr["original_promise_date"]).astype(int)
a = arr["late"].values; b = arr["short"].values
phi = np.corrcoef(a,b)[0,1]
n11=int(((a==1)&(b==1)).sum()); n10=int(((a==1)&(b==0)).sum())
n01=int(((a==0)&(b==1)).sum()); n00=int(((a==0)&(b==0)).sum())
p_s_late = n11/max(1,(n11+n10)); p_s_on = n01/max(1,(n01+n00))
out["1_late_short"] = dict(n=len(arr), phi=float(phi), n11=n11,n10=n10,n01=n01,n00=n00,
    p_short_given_late=p_s_late, p_short_given_ontime=p_s_on,
    lift=(p_s_late/p_s_on if p_s_on else float("nan")))

# ---------- 2. constraint predicts shortfall ----------
ch = L("sourcing_channels")
ch_sup = ch.set_index("channel_id")["supplier_id"]
pop2 = pop.copy()
pop2["supplier_id"] = pop2["channel_id"].map(ch_sup)
pop2["month"] = pop2["original_promise_date"].values.astype("datetime64[M]")
g = pop2.dropna(subset=["supplier_id"]).groupby(["supplier_id","part_id","month"]).agg(
        acc=("acc","sum"), q=("qty_ordered","sum")).reset_index()
g = g[g["q"]>0]; g["fill"] = np.minimum(1.0, g["acc"]/g["q"])
rc = L("revealed_capacity_monthly")
rc["month"] = pd.to_datetime(rc["month"]).values.astype("datetime64[M]")
flag = rc["constrained_month_flag"].astype(str).str.lower().isin(["1","true","t","yes"])
rc = rc.assign(cflag=flag.astype(int))
j = g.merge(rc[["supplier_id","part_id","month","cflag","capacity_basis"]],
            on=["supplier_id","part_id","month"], how="inner")
c1 = j.loc[j.cflag==1,"fill"]; c0 = j.loc[j.cflag==0,"fill"]
out["2_constraint_fill"] = dict(n_matched=len(j), n_constrained=len(c1), n_unconstrained=len(c0),
    fill_constrained=float(c1.mean()) if len(c1) else None,
    fill_unconstrained=float(c0.mean()) if len(c0) else None,
    diff=(float(c0.mean()-c1.mean()) if len(c1) and len(c0) else None),
    corr=float(np.corrcoef(j["cflag"],j["fill"])[0,1]) if j["cflag"].nunique()>1 else None,
    share_constrained=float(j["cflag"].mean()) if len(j) else None)

# ---------- 3. shortages follow their cause ----------
se = L("shortage_events", parse_dates=["shortage_start_ts"])
pp_ch = ch.groupby(["part_id","plant_id"])["channel_id"].apply(set).to_dict()
ev = arr.copy()
ev["miss_ts"] = ev["first"]
misses = ev[(ev["late"]==1)|(ev["short"]==1)][["channel_id","miss_ts"]].dropna()
by_ch = {c: np.sort(gg["miss_ts"].values) for c,gg in misses.groupby("channel_id")}
def has_cause(part,plant,ts,weeks=8):
    chans = pp_ch.get((part,plant))
    if not chans: return None
    lo = np.datetime64(ts - pd.Timedelta(weeks=weeks)); hi = np.datetime64(ts)
    for c in chans:
        arr_ = by_ch.get(c)
        if arr_ is None: continue
        i = np.searchsorted(arr_, lo, "left"); jx = np.searchsorted(arr_, hi, "right")
        if jx > i: return True
    return False
res3 = [has_cause(r.part_id, r.plant_id, r.shortage_start_ts) for r in se.itertuples()]
known = [x for x in res3 if x is not None]
rng = np.random.default_rng(0)
shuf = se["shortage_start_ts"].sample(frac=1.0, random_state=1).values
res3p = [has_cause(r.part_id, r.plant_id, pd.Timestamp(t)) for r,t in zip(se.itertuples(), shuf)]
knownp = [x for x in res3p if x is not None]
out["3_shortage_cause"] = dict(n_events=len(se), n_with_channel=len(known),
    n_no_channel=int(sum(1 for x in res3 if x is None)),
    share_with_cause=float(np.mean(known)) if known else None,
    placebo_share=float(np.mean(knownp)) if knownp else None)

# ---------- 4. staleness vs lead-time volatility ----------
cw = L("channel_performance_weekly")
gg = cw.groupby("channel_id").agg(
        stale=("weeks_since_last_activity","mean"),
        lt_var=("lead_time_actual_days","var"),
        n_lt=("lead_time_actual_days","count")).dropna()
gg = gg[gg["n_lt"]>=5]
p4 = float(np.corrcoef(gg["stale"],gg["lt_var"])[0,1]) if len(gg)>2 else None
s4 = spear(gg["stale"],gg["lt_var"]) if len(gg)>2 else None
out["4_stale_vol"] = dict(n_channels=len(gg), pearson=p4, spearman=s4)

# ---------- 5. labels track their own features ----------
tl = L("training_labels", parse_dates=["snapshot_date"])
tl = tl[tl["label_censored"].astype(str).str.lower().isin(["0","false","f","no",""]) |
        tl["label_censored"].isna()]
po_ch = po.set_index("po_line_id")["channel_id"]
cw["week_start"] = pd.to_datetime(cw["week_start"])
sw = L("supplier_performance_weekly"); sw["week_start"]=pd.to_datetime(sw["week_start"])
pdw = L("plan_drift_features", parse_dates=["plan_date"])
ipw = L("inventory_position_weekly", parse_dates=["week_start"])

def asof_feat(keys, ts, store, keycols, feat):
    """latest store row per key with week<=ts"""
    s = store.dropna(subset=[feat]).sort_values(keycols[-1])
    return s
def corr_pair(x,y):
    d = pd.DataFrame({"x":x,"y":y}).dropna()
    if len(d)<10 or d["x"].nunique()<2 or d["y"].nunique()<2: return None,None,len(d)
    return float(d["x"].corr(d["y"])), spear(d["x"],d["y"]), len(d)

p5 = {}
# fill_rate & arrival_week: po_line -> channel
cwx = cw.sort_values("week_start")
for task, feat in [("fill_rate","fill_rate_last13"), ("arrival_week","otd_rate_last13")]:
    t = tl[tl["task"]==task].copy()
    t["channel_id"] = t["entity_id"].map(po_ch)
    t = t.dropna(subset=["channel_id"]).sort_values("snapshot_date")
    mm = pd.merge_asof(t, cwx[["channel_id","week_start",feat]].sort_values("week_start"),
                       left_on="snapshot_date", right_on="week_start",
                       left_by="channel_id", right_by="channel_id", direction="backward")
    r,s,n = corr_pair(pd.to_numeric(mm["label_value"],errors="coerce"), mm[feat])
    p5[task] = dict(feature=feat, pearson=r, spearman=s, n=n)
# capacity_strain: channel -> load_ratio
t = tl[tl["task"]=="capacity_strain"].copy().rename(columns={"entity_id":"channel_id"}).sort_values("snapshot_date")
mm = pd.merge_asof(t, cwx[["channel_id","week_start","load_ratio"]].sort_values("week_start"),
                   left_on="snapshot_date", right_on="week_start", by="channel_id", direction="backward")
r,s,n = corr_pair(pd.to_numeric(mm["label_value"],errors="coerce"), mm["load_ratio"])
p5["capacity_strain"] = dict(feature="load_ratio", pearson=r, spearman=s, n=n)
# demand_drift: product_plant -> plan_drift_features.drift_ratio
t = tl[tl["task"]=="demand_drift"].copy()
t[["product_id","plant_id"]] = t["entity_id"].str.split("|",expand=True)
t = t.sort_values("snapshot_date")
pdx = pdw.dropna(subset=["drift_ratio"]).sort_values("plan_date")
pdx["key"]=pdx["product_id"]+"|"+pdx["plant_id"]; t["key"]=t["entity_id"]
mm = pd.merge_asof(t, pdx[["key","plan_date","drift_ratio"]].sort_values("plan_date"),
                   left_on="snapshot_date", right_on="plan_date", by="key", direction="backward")
r,s,n = corr_pair(pd.to_numeric(mm["label_value"],errors="coerce"), mm["drift_ratio"])
p5["demand_drift"] = dict(feature="drift_ratio", pearson=r, spearman=s, n=n)
# shortage_qty: part_plant -> inventory days_of_supply
t = tl[tl["task"]=="shortage_qty"].copy()
t[["part_id","plant_id"]] = t["entity_id"].str.split("|",expand=True)
t["key"]=t["entity_id"]; t=t.sort_values("snapshot_date")
ix = ipw.dropna(subset=["days_of_supply"]).copy()
ix["key"]=ix["part_id"]+"|"+ix["plant_id"]; ix=ix.sort_values("week_start")
mm = pd.merge_asof(t, ix[["key","week_start","days_of_supply"]].sort_values("week_start"),
                   left_on="snapshot_date", right_on="week_start", by="key", direction="backward")
r,s,n = corr_pair(pd.to_numeric(mm["label_value"],errors="coerce"), mm["days_of_supply"])
p5["shortage_qty"] = dict(feature="days_of_supply (expect negative)", pearson=r, spearman=s, n=n)
out["5_label_feature"] = p5

print(json.dumps(out, indent=2, default=str))
