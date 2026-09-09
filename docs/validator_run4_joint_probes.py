"""Run-4 joint-structure probes. Read-only. Mirrors validator fill_population rules."""
import pandas as pd, numpy as np, json
D="/Users/muralik/Documents/Programs/HADES_v4/db/"
def spear(x,y): return float(pd.Series(x).rank().corr(pd.Series(y).rank()))
def cp(x,y):
    d=pd.DataFrame({'x':x,'y':y}).dropna()
    if len(d)<10 or d.x.nunique()<2 or d.y.nunique()<2: return None,None,len(d)
    return float(d.x.corr(d.y)), spear(d.x,d.y), len(d)
out={}

po=pd.read_csv(D+"po/po_lines.csv",parse_dates=["original_promise_date","created_ts"])
grn=pd.read_csv(D+"logistics/grn_lines.csv",parse_dates=["event_ts"])
qi=pd.read_csv(D+"logistics/quality_inspections.csv")
rev=pd.read_csv(D+"po/po_line_revisions.csv")
am=qi.dropna(subset=["qty_accepted"]).set_index("grn_line_id")["qty_accepted"]
grn["acc"]=grn["grn_line_id"].map(am).fillna(grn["qty_received"])
ag=grn.groupby("po_line_id").agg(acc=("acc","sum"),first=("event_ts","min"))
po=po.join(ag,on="po_line_id"); po["acc"]=po["acc"].fillna(0)
canc=set(rev.loc[rev.reason_code=="buyer_cancellation","po_line_id"])
end=max(po.created_ts.max(),grn.event_ts.max()); cut=end-pd.Timedelta(days=90)
pop=po[(po.qty_ordered>0)&(~po.po_line_id.isin(canc))&(po.original_promise_date<=cut)].copy()
pop["fill"]=np.minimum(1.0,pop.acc/pop.qty_ordered); pop["short"]=(pop.fill<0.9999).astype(int)
out["_population"]=dict(po_lines=len(po), cancelled_excluded=len(canc), after_filters=len(pop))

# 1 late <-> short
a=pop.dropna(subset=["first"]).copy()
a["late"]=(a["first"].dt.normalize()>a.original_promise_date).astype(int)
n11=int(((a.late==1)&(a.short==1)).sum()); n10=int(((a.late==1)&(a.short==0)).sum())
n01=int(((a.late==0)&(a.short==1)).sum()); n00=int(((a.late==0)&(a.short==0)).sum())
phi=float(np.corrcoef(a.late,a.short)[0,1])
a["late_days"]=(a["first"].dt.normalize()-a.original_promise_date).dt.days
r,s,n=cp(a.late_days,1-a.fill)
out["1_late_short"]=dict(n=len(a),phi=phi,n11=n11,n10=n10,n01=n01,n00=n00,
    p_short_given_late=n11/max(1,n11+n10), p_short_given_ontime=n01/max(1,n01+n00),
    continuous_pearson=r, continuous_spearman=s)

# 2 constrained vs unconstrained fill
ch=pd.read_csv(D+"master/sourcing_channels.csv",usecols=["channel_id","supplier_id","part_id"])
p2=pop.merge(ch[["channel_id","supplier_id"]],on="channel_id",how="left")
p2["month"]=p2.original_promise_date.values.astype("datetime64[M]")
g=p2.dropna(subset=["supplier_id"]).groupby(["supplier_id","part_id","month"]).agg(acc=("acc","sum"),q=("qty_ordered","sum")).reset_index()
g=g[g.q>0]; g["fill"]=np.minimum(1.0,g.acc/g.q)
rc=pd.read_csv(D+"model/revealed_capacity_monthly.csv",usecols=["supplier_id","part_id","month","constrained_month_flag","capacity_basis"])
rc["month"]=pd.to_datetime(rc.month).values.astype("datetime64[M]")
rc["cf"]=rc.constrained_month_flag.astype(str).str.lower().isin(["1","true","t","yes"]).astype(int)
j=g.merge(rc[["supplier_id","part_id","month","cf"]],on=["supplier_id","part_id","month"],how="inner")
c1=j.loc[j.cf==1,"fill"]; c0=j.loc[j.cf==0,"fill"]
out["2_constraint_fill"]=dict(n=len(j),n_constrained=int(len(c1)),n_unconstrained=int(len(c0)),
    fill_constrained=float(c1.mean()) if len(c1) else None,
    fill_unconstrained=float(c0.mean()) if len(c0) else None,
    diff=float(c0.mean()-c1.mean()) if len(c1) and len(c0) else None,
    corr=float(np.corrcoef(j.cf,j.fill)[0,1]) if j.cf.nunique()>1 else None,
    declared_constrained_share_full_table=float(rc.cf.mean()),
    capacity_basis_values=rc.capacity_basis.value_counts().to_dict())

# 3 shortage antecedents
se=pd.read_csv(D+"outcomes/shortage_events.csv",parse_dates=["shortage_start_ts"])
chf=pd.read_csv(D+"master/sourcing_channels.csv",usecols=["channel_id","part_id","plant_id"])
pp=chf.groupby(["part_id","plant_id"])["channel_id"].apply(set).to_dict()
miss=a[(a.late==1)|(a.short==1)][["channel_id","first"]].dropna()
by={c:np.sort(gg["first"].values) for c,gg in miss.groupby("channel_id")}
def cause(part,plant,tsv,weeks=8):
    cs=pp.get((part,plant))
    if not cs: return None
    lo=np.datetime64(tsv-pd.Timedelta(weeks=weeks)); hi=np.datetime64(tsv)
    for c in cs:
        arr=by.get(c)
        if arr is None: continue
        if np.searchsorted(arr,hi,"right")>np.searchsorted(arr,lo,"left"): return True
    return False
res=[cause(r.part_id,r.plant_id,r.shortage_start_ts) for r in se.itertuples()]
kn=[x for x in res if x is not None]
shuf=se.shortage_start_ts.sample(frac=1.0,random_state=1).values
resp=[cause(r.part_id,r.plant_id,pd.Timestamp(t)) for r,t in zip(se.itertuples(),shuf)]
knp=[x for x in resp if x is not None]
out["3_shortage_cause"]=dict(n_events=len(se),n_no_channel=int(sum(1 for x in res if x is None)),
    n_with_channel=len(kn), share_among_with_channel=float(np.mean(kn)) if kn else None,
    share_of_all=float(sum(1 for x in kn if x)/len(se)) if len(se) else None,
    placebo_share=float(np.mean(knp)) if knp else None)

# 4 staleness <-> lead-time variance
cw=pd.read_csv(D+"channel/channel_performance_weekly.csv",
               usecols=["channel_id","qty_ordered","lead_time_actual_days","weeks_since_last_activity"])
gg=cw.groupby("channel_id").agg(stale=("weeks_since_last_activity","mean"),
    var=("lead_time_actual_days","var"), n=("lead_time_actual_days","count")).dropna()
gg=gg[gg.n>=5]
out["4_stale_vol"]=dict(n_channels=len(gg),pearson=float(gg.stale.corr(gg["var"])),spearman=spear(gg.stale,gg["var"]))

# 5 labels vs most predictive feature
tl=pd.read_csv(D+"model/training_labels.csv",parse_dates=["snapshot_date"])
cwf=pd.read_csv(D+"channel/channel_performance_weekly.csv",
    usecols=["channel_id","week_start","fill_rate_last13","otd_rate_last13","load_ratio",
             "days_since_last_short","qty_ordered"])
cwf["week_start"]=pd.to_datetime(cwf.week_start)
cwf=cwf.drop_duplicates(subset=["channel_id","week_start"]).sort_values("week_start")
FEAT={"fill_rate":"fill_rate_last13","arrival_timing":"otd_rate_last13",
      "capacity_strain":"load_ratio","shortage_qty":"days_since_last_short",
      "demand_drift":"qty_ordered"}
p5={}
for task,feat in FEAT.items():
    t=tl[tl.task==task].copy().rename(columns={"entity_id":"channel_id"}).sort_values("snapshot_date")
    mm=pd.merge_asof(t,cwf[["channel_id","week_start",feat]].sort_values("week_start"),
        left_on="snapshot_date",right_on="week_start",by="channel_id",direction="backward")
    r,s,n=cp(pd.to_numeric(mm.label_value,errors="coerce"),mm[feat])
    p5[task]=dict(feature=feat,pearson=r,spearman=s,n=n)
out["5_label_feature"]=p5
print(json.dumps(out,indent=2,default=str))
