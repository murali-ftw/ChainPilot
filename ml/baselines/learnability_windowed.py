"""G2 / G3 / G4 learnability harness (synthetic_rules.md §16 Level 4).

Split is fixed and time-ordered for every task and every gate:
    train  snapshot_date <= 2023-12-31
    val    2024
    test   2025
Reported metrics come from the TEST fold only. Features are built as-of the snapshot:
the latest channel_performance_weekly row with week_start <= snapshot_date, so nothing
after t0 can enter a feature.
"""
import numpy as np, pandas as pd, sys, json, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score, brier_score_loss
import lightgbm as lgb

D = sys.argv[1].rstrip('/') + '/'
OUT = sys.argv[2] if len(sys.argv) > 2 else None
SEED = 7
TRAIN_END, VAL_END = pd.Timestamp('2023-12-31'), pd.Timestamp('2024-12-31')

CHF = ['fill_rate','fill_rate_last4','fill_rate_last13','fill_rate_last52',
       'lead_time_actual_days','lead_time_ratio','otd_rate_last13','ack_gap_ratio',
       'load_ratio','reporting_lag_days','qty_ordered','qty_received',
       'active_weeks_in_52','revision_count']

def log(*a): print(*a, flush=True)

# ------------------------------------------------------------------ load
log('loading', D)
lb = pd.read_csv(D+'training_labels.csv',
                 usecols=['snapshot_date','entity_type','entity_id','task','label_value','label_censored'])
lb['snapshot_date'] = pd.to_datetime(lb.snapshot_date)
# task constraint: fit on the 2019-2025 window, where the censored tail is 1.19x not 1.47x
FIT_LO, FIT_HI = pd.Timestamp('2019-01-01'), pd.Timestamp('2025-12-31')
_n0 = len(lb)
lb = lb[(lb.snapshot_date >= FIT_LO) & (lb.snapshot_date <= FIT_HI)]
log(f'  FIT WINDOW 2019-2025: {_n0:,} -> {len(lb):,} label rows ({100*len(lb)/_n0:.1f}%)')
lb['label_censored'] = lb.label_censored.astype(str).str.lower().isin(['true','1'])
ch = pd.read_csv(D+'sourcing_channels.csv',
                 usecols=['channel_id','supplier_id','part_id','plant_id',
                          'contracted_lead_time_days','transport_mode','transport_distance_km'])
pol = pd.read_csv(D+'po_lines.csv', usecols=['po_line_id','channel_id'])
cpw = pd.read_csv(D+'channel_performance_weekly.csv',
                  usecols=['channel_id','week_start']+CHF,
                  dtype={c: 'float32' for c in CHF})
cpw['week_start'] = pd.to_datetime(cpw.week_start)
log(f'  cpw {len(cpw):,}  labels {len(lb):,}  channels {len(ch):,}')

sup_i = {s: i for i, s in enumerate(sorted(ch.supplier_id.unique()))}
part_i = {s: i for i, s in enumerate(sorted(ch.part_id.unique()))}
plant_i = {s: i for i, s in enumerate(sorted(ch.plant_id.unique()))}
ch['si'] = ch.supplier_id.map(sup_i); ch['pi'] = ch.part_id.map(part_i); ch['li'] = ch.plant_id.map(plant_i)
ch['mode_i'] = ch.transport_mode.astype('category').cat.codes
ch_i = {c: i for i, c in enumerate(ch.channel_id)}
NCH = len(ch)

def asof_channel(pairs):
    """pairs: DataFrame[channel_id, snapshot_date] -> as-of channel features."""
    L = pairs.sort_values('snapshot_date')
    R = cpw.sort_values('week_start')
    j = pd.merge_asof(L, R, left_on='snapshot_date', right_on='week_start',
                      by='channel_id', direction='backward')
    return j

def add_static(df):
    m = df.merge(ch[['channel_id','si','pi','li','mode_i',
                     'contracted_lead_time_days','transport_distance_km']],
                 on='channel_id', how='left')
    return m

def split(df):
    tr = df.snapshot_date <= TRAIN_END
    va = (df.snapshot_date > TRAIN_END) & (df.snapshot_date <= VAL_END)
    te = df.snapshot_date > VAL_END
    return tr.to_numpy(), va.to_numpy(), te.to_numpy()

# ------------------------------------------------------------------ metrics
def crps_from_cdf(cdf, y, edges):
    """discrete CRPS: integral (F(x) - 1{y<=x})^2 dx over the bin grid"""
    step = np.diff(edges, prepend=edges[0])[1:]
    ind = (y[:, None] <= edges[None, 1:]).astype(float)
    return float((((cdf - ind) ** 2) * step[None, :]).sum(1).mean())

def cindex(pred, time, event, n_pairs=2_000_000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    n = len(time)
    i = rng.integers(0, n, n_pairs); j = rng.integers(0, n, n_pairs)
    ti, tj = time[i], time[j]; ei, ej = event[i], event[j]
    # comparable: the earlier one must be an observed event
    ok = ((ti < tj) & ei) | ((tj < ti) & ej)
    if ok.sum() == 0: return float('nan')
    i, j = i[ok], j[ok]; ti, tj = time[i], time[j]
    pi, pj = pred[i], pred[j]
    earlier_i = ti < tj
    conc = np.where(earlier_i, pi < pj, pj < pi)
    ties = pi == pj
    return float((conc.sum() + 0.5 * ties.sum()) / len(i))

def pinball(y, p, q):
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))

RES = {}
def rec(task, gate, name, **kw):
    RES.setdefault(task, {}).setdefault(gate, {})[name] = kw
    log(f'    {gate:<3} {name:<34} ' + '  '.join(f'{k}={v:.4f}' for k, v in kw.items()))

GBM = dict(objective=None, n_estimators=400, learning_rate=0.05, num_leaves=63,
           min_child_samples=50, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
           random_state=SEED, verbose=-1, n_jobs=-1)

# ------------------------------------------------------------------ graph propagation for G4
def propagate(X, si, pi, li, rounds):
    """One round = channels -> {supplier, part, plant} mean -> back to channels.
    A 4-layer heterogeneous encoder is two such rounds; h1 = one round."""
    outs = []
    cur = X.copy()
    for _ in range(rounds):
        agg = []
        for idx, n in ((si, si.max()+1), (pi, pi.max()+1), (li, li.max()+1)):
            s = np.zeros((n, cur.shape[1]), np.float64)
            np.add.at(s, idx, np.nan_to_num(cur))
            c = np.bincount(idx, minlength=n).astype(float)[:, None]
            agg.append((s / np.maximum(c, 1))[idx])
        cur = np.concatenate(agg, 1)
        outs.append(cur)
    return np.concatenate(outs, 1) if outs else None

# ==================================================================== FILL RATE (po_line)
log('\n=== FILL RATE (entity: po_line) ===')
f = lb[lb.task == 'fill_rate'].merge(pol, left_on='entity_id', right_on='po_line_id', how='inner')
f = asof_channel(f[['channel_id','snapshot_date','label_value','label_censored']])
f = add_static(f).dropna(subset=['fill_rate_last13'])
tr, va, te = split(f)
y = f.label_value.to_numpy(float)
FEAT = CHF + ['contracted_lead_time_days','transport_distance_km','mode_i','si','pi','li']
X = f[FEAT].to_numpy(np.float32)
log(f'  rows train {tr.sum():,} val {va.sum():,} test {te.sum():,}')

NB = 20; edges = np.linspace(0, 1, NB + 1)
ybin_all = np.clip(np.digitize(y, edges[1:-1]), 0, NB - 1)
def cdf_from_hist(h): return np.cumsum(h / max(h.sum(), 1e-9))
# G2 a: global empirical histogram
h = np.bincount(ybin_all[tr], minlength=NB)
c_glob = np.tile(cdf_from_hist(h), (te.sum(), 1))
rec('fill_rate','G2','global mean fill (empirical CDF)',
    crps=crps_from_cdf(c_glob, y[te], edges), mae=float(np.abs(y[te] - y[tr].mean()).mean()))
# G2 b: per-channel historical mean / empirical CDF
dtr = pd.DataFrame({'ch': f.channel_id[tr].to_numpy(), 'b': ybin_all[tr], 'y': y[tr]})
gm = dtr.groupby('ch').y.mean()
hist = (dtr.groupby(['ch','b']).size().unstack(fill_value=0)
          .reindex(columns=range(NB), fill_value=0))
hist_cdf = hist.div(hist.sum(1).replace(0, np.nan), axis=0).cumsum(1)
te_ch = f.channel_id[te].to_numpy()
cc = hist_cdf.reindex(te_ch).to_numpy()
cc = np.where(np.isnan(cc), np.tile(cdf_from_hist(h), (len(te_ch), 1)), cc)
pred_ch = gm.reindex(te_ch).fillna(y[tr].mean()).to_numpy()
rec('fill_rate','G2','per-channel historical mean',
    crps=crps_from_cdf(cc, y[te], edges), mae=float(np.abs(y[te] - pred_ch).mean()))
# G2 c: last observed value = the as-of fill_rate feature itself
lastv = np.nan_to_num(f.fill_rate.to_numpy(float), nan=float(y[tr].mean()))
lb_bin = np.clip(np.digitize(lastv, edges[1:-1]), 0, NB - 1)
c_last = (np.arange(NB)[None, :] >= lb_bin[te][:, None]).astype(float)
rec('fill_rate','G2','last observed value',
    crps=crps_from_cdf(c_last, y[te], edges), mae=float(np.abs(y[te] - lastv[te]).mean()))
# G3: LightGBM multiclass over the fill bins -> predictive CDF
m = lgb.LGBMClassifier(**{**GBM, 'objective': 'multiclass', 'num_class': NB})
m.fit(X[tr], ybin_all[tr], eval_set=[(X[va], ybin_all[va])],
      callbacks=[lgb.early_stopping(40, verbose=False)])
P = m.predict_proba(X[te]); cdf = np.cumsum(P, 1)
pt_ = (P * ((edges[:-1] + edges[1:]) / 2)[None, :]).sum(1)
rec('fill_rate','G3','LightGBM tabular (no graph)',
    crps=crps_from_cdf(cdf, y[te], edges), mae=float(np.abs(y[te] - pt_).mean()))
IMP_FILL = pd.Series(m.feature_importances_, index=FEAT).sort_values(ascending=False)
# G4: graph ablation
si = f.si.to_numpy(int); pi_ = f.pi.to_numpy(int); li = f.li.to_numpy(int)
Xn = np.nan_to_num(X)
for lbl, rounds in (('h0 features only', 0), ('h1 one hop', 1), ('h4 four layers', 2)):
    Xg = Xn if rounds == 0 else np.concatenate([Xn, propagate(Xn[:, :len(CHF)], si, pi_, li, rounds)], 1)
    mm = lgb.LGBMClassifier(**{**GBM, 'objective': 'multiclass', 'num_class': NB})
    mm.fit(Xg[tr], ybin_all[tr], eval_set=[(Xg[va], ybin_all[va])],
           callbacks=[lgb.early_stopping(40, verbose=False)])
    cdfg = np.cumsum(mm.predict_proba(Xg[te]), 1)
    rec('fill_rate','G4', lbl, crps=crps_from_cdf(cdfg, y[te], edges))

# ==================================================================== ARRIVAL (po_line)
log('\n=== ARRIVAL TIMING (entity: po_line) ===')
a = lb[lb.task == 'arrival_week'].merge(pol, left_on='entity_id', right_on='po_line_id', how='inner')
a = asof_channel(a[['channel_id','snapshot_date','label_value','label_censored']])
a = add_static(a).dropna(subset=['fill_rate_last13'])
tra, vaa, tea = split(a)
ya = a.label_value.to_numpy(float); ev = ~a.label_censored.to_numpy()
Xa = a[FEAT].to_numpy(np.float32); Xan = np.nan_to_num(Xa)
log(f'  rows train {tra.sum():,} val {vaa.sum():,} test {tea.sum():,}   censored {1-ev.mean():.1%}')
obs = tra & ev
lane = a.assign(y=ya).loc[obs].groupby(['si','li']).y.median()
gmed = float(np.median(ya[obs]))
p_lane = a.loc[tea].set_index(['si','li']).index.map(lane).to_numpy(float)
p_lane = np.where(np.isnan(p_lane), gmed, p_lane)
chmed = a.assign(y=ya).loc[obs].groupby('channel_id').y.median()
p_chan = chmed.reindex(a.channel_id[tea]).fillna(gmed).to_numpy()
for nm, p in (('per-lane median lead time', p_lane), ('per-channel median', p_chan),
              ('global median', np.full(tea.sum(), gmed))):
    rec('arrival_week','G2', nm, cindex=cindex(p, ya[tea], ev[tea]),
        mae_obs=float(np.abs(ya[tea][ev[tea]] - p[ev[tea]]).mean()))
ma = lgb.LGBMRegressor(**{**GBM, 'objective': 'l2'})
ma.fit(Xa[obs], ya[obs], eval_set=[(Xa[vaa & ev], ya[vaa & ev])],
       callbacks=[lgb.early_stopping(40, verbose=False)])
pa = ma.predict(Xa[tea])
rec('arrival_week','G3','LightGBM tabular (no graph)',
    cindex=cindex(pa, ya[tea], ev[tea]), mae_obs=float(np.abs(ya[tea][ev[tea]] - pa[ev[tea]]).mean()))
IMP_ARR = pd.Series(ma.feature_importances_, index=FEAT).sort_values(ascending=False)
sia, pia, lia = a.si.to_numpy(int), a.pi.to_numpy(int), a.li.to_numpy(int)
for lbl, rounds in (('h0 features only', 0), ('h1 one hop', 1), ('h4 four layers', 2)):
    Xg = Xan if rounds == 0 else np.concatenate([Xan, propagate(Xan[:, :len(CHF)], sia, pia, lia, rounds)], 1)
    mm = lgb.LGBMRegressor(**{**GBM, 'objective': 'l2'})
    mm.fit(Xg[obs], ya[obs], eval_set=[(Xg[vaa & ev], ya[vaa & ev])],
           callbacks=[lgb.early_stopping(40, verbose=False)])
    rec('arrival_week','G4', lbl, cindex=cindex(mm.predict(Xg[tea]), ya[tea], ev[tea]))

# ==================================================================== PART SHORTAGE (part_plant)
log('\n=== PART SHORTAGE (entity: part_plant) ===')
s = lb[lb.task == 'shortage_qty'].copy()
s[['part_id','plant_id']] = s.entity_id.str.split('|', expand=True)
POS_ONLY = bool((s.label_value > 0).all())
log(f'  label rows {len(s):,}; zero-valued rows {(s.label_value==0).sum():,}; '
    f'LABEL TABLE IS POSITIVE-ONLY: {POS_ONLY}')

cp = ch[['channel_id','part_id','plant_id','si','pi','li']]
# full part x plant x snapshot universe, so the task has negatives
snaps = np.sort(lb.snapshot_date.unique())
allpp = cp[['part_id','plant_id']].drop_duplicates()
uni = allpp.merge(pd.DataFrame({'snapshot_date': snaps}), how='cross')
import os as _os
_simf = D + '_sim.npz'
if _os.path.exists(_simf):
    _z = np.load(_simf)
    _T = len(_z['MONTHKEY']); _W = pd.date_range('2016-01-04', periods=_T, freq='W-MON')
    _st, _sch = _z['st'].astype(int), _z['sch'].astype(int)
    _NPL = len(plant_i)
    _cond = {}
    for _t, _pp in zip(_st, _sch): _cond.setdefault(int(_pp), []).append(int(_t))
    _cond = {k: np.array(sorted(v)) for k, v in _cond.items()}
    _pid = {i: s_ for s_, i in part_i.items()}; _lid = {i: s_ for s_, i in plant_i.items()}
    pos = set()
    for _d in np.sort(lb.snapshot_date.unique()):
        _d = pd.Timestamp(_d)
        _t0 = int(np.searchsorted(_W.values, np.datetime64(_d), 'right') - 1)
        _t1 = min(_T - 1, _t0 + 13)
        for _pp, _ts in _cond.items():
            if ((_ts > _t0) & (_ts <= _t1)).any():
                pos.add((_pid[_pp // _NPL], _lid[_pp % _NPL], _d))
    log(f'  shortage positives derived from simulation state: {len(pos):,} part-plant-snapshots')
else:
    sp_ = s[s.label_value > 0]
    pos = set(zip(sp_.part_id, sp_.plant_id, sp_.snapshot_date))
uni['y'] = [1 if k in pos else 0 for k in zip(uni.part_id, uni.plant_id, uni.snapshot_date)]
log(f'  reconstructed universe {len(uni):,} part-plant-snapshots, base rate {uni.y.mean():.4f}')

kc = uni[['part_id','plant_id','snapshot_date']].merge(cp, on=['part_id','plant_id'], how='inner')
kf = asof_channel(kc[['channel_id','snapshot_date','part_id','plant_id']])
agg = (kf.groupby(['part_id','plant_id','snapshot_date'])[CHF].agg(['mean','min']).reset_index())
agg.columns = ['part_id','plant_id','snapshot_date'] + [f'{a}_{b}' for a, b in agg.columns[3:]]
nsup = kc.groupby(['part_id','plant_id']).si.nunique().rename('n_suppliers').reset_index()
s2 = uni.merge(agg, on=['part_id','plant_id','snapshot_date'], how='left').merge(nsup, on=['part_id','plant_id'], how='left')
SF = [c for c in s2.columns if c.endswith(('_mean','_min'))] + ['n_suppliers']
s2 = s2.dropna(subset=['fill_rate_last13_mean'])
trs, vas, tes = split(s2)
ys = s2.y.to_numpy(int)
Xs = s2[SF].to_numpy(np.float32); Xsn = np.nan_to_num(Xs)
log(f'  rows train {trs.sum():,} val {vas.sum():,} test {tes.sum():,}   positive rate {ys[tes].mean():.4f}')
base = float(ys[trs].mean())
rec('shortage_qty','G2','marginal base rate',
    pr_auc=average_precision_score(ys[tes], np.full(tes.sum(), base)),
    brier=brier_score_loss(ys[tes], np.full(tes.sum(), base)))
pph = pd.DataFrame({'k': s2.part_id[trs] + '|' + s2.plant_id[trs], 'y': ys[trs]}).groupby('k').y.mean()
pk = (s2.part_id[tes] + '|' + s2.plant_id[tes]).map(pph).fillna(base).to_numpy()
rec('shortage_qty','G2','per-part-plant historical rate',
    pr_auc=average_precision_score(ys[tes], pk), brier=brier_score_loss(ys[tes], np.clip(pk, 0, 1)))
ms = lgb.LGBMClassifier(**{**GBM, 'objective': 'binary'})
ms.fit(Xs[trs], ys[trs], eval_set=[(Xs[vas], ys[vas])], callbacks=[lgb.early_stopping(40, verbose=False)])
ps = ms.predict_proba(Xs[tes])[:, 1]
rec('shortage_qty','G3','LightGBM tabular (no graph)',
    pr_auc=average_precision_score(ys[tes], ps), brier=brier_score_loss(ys[tes], ps))
IMP_SH = pd.Series(ms.feature_importances_, index=SF).sort_values(ascending=False)
gsi, gpi, gli = kc.si.to_numpy(int), kc.pi.to_numpy(int), kc.li.to_numpy(int)
for r_, lbl in ((0, 'h0 features only'), (1, 'h1 one hop'), (2, 'h4 four layers')):
    if r_ == 0:
        Xg = Xsn
    else:
        Xc = np.nan_to_num(kf[CHF].to_numpy(np.float32))
        prop = propagate(Xc, gsi, gpi, gli, r_)
        gcols = [f'g{i}' for i in range(prop.shape[1])]
        pdf = pd.DataFrame(prop, columns=gcols)
        pdf[['part_id','plant_id','snapshot_date']] = kf[['part_id','plant_id','snapshot_date']].values
        pa_ = pdf.groupby(['part_id','plant_id','snapshot_date'], as_index=False)[gcols].mean()
        mrg = s2[['part_id','plant_id','snapshot_date']].merge(pa_, on=['part_id','plant_id','snapshot_date'], how='left')
        Xg = np.concatenate([Xsn, np.nan_to_num(mrg[gcols].to_numpy(np.float32))], 1)
    mm = lgb.LGBMClassifier(**{**GBM, 'objective': 'binary'})
    mm.fit(Xg[trs], ys[trs], eval_set=[(Xg[vas], ys[vas])], callbacks=[lgb.early_stopping(40, verbose=False)])
    rec('shortage_qty','G4', lbl, pr_auc=average_precision_score(ys[tes], mm.predict_proba(Xg[tes])[:, 1]))
# and the regression the label table actually supports, among the positives it carries
sp = s.merge(agg, on=['part_id','plant_id','snapshot_date'], how='left').dropna(subset=['fill_rate_last13_mean'])
tp, vp, ep = split(sp)
yq = np.log1p(sp.label_value.to_numpy(float)); Xq = sp[[c for c in SF if c in sp.columns]].to_numpy(np.float32)
med = float(np.median(yq[tp]))
rec('shortage_qty','G2','median log-qty (positives only)',
    mae=float(np.abs(yq[ep] - med).mean()), pinball90=pinball(yq[ep], np.full(ep.sum(), med), 0.9))
mq = lgb.LGBMRegressor(**{**GBM, 'objective': 'l1'})
mq.fit(Xq[tp], yq[tp], eval_set=[(Xq[vp], yq[vp])], callbacks=[lgb.early_stopping(40, verbose=False)])
pq_ = mq.predict(Xq[ep])
rec('shortage_qty','G3','LightGBM log-qty (positives only)',
    mae=float(np.abs(yq[ep] - pq_).mean()), pinball90=pinball(yq[ep], pq_, 0.9))

# ==================================================================== DEMAND DRIFT (product_plant)
log('\n=== DEMAND DRIFT (entity: product_plant) ===')
d = lb[lb.task == 'demand_drift'].copy()
d['k'] = d.entity_id
d = d.sort_values('snapshot_date')
trd, vad, ted = split(d)
yd = d.label_value.to_numpy(float)
prev = d.groupby('k').label_value.shift(1)
d['last_value'] = prev
yr_ago = d.assign(yr=d.snapshot_date.dt.year).groupby(['k'])['label_value'].shift(8)
d['seasonal_naive'] = yr_ago
gmean = float(yd[trd].mean())
for nm, col in (('naive seasonal (same week last year)', 'seasonal_naive'), ('last value', 'last_value')):
    p = d[col].fillna(gmean).to_numpy()
    rec('demand_drift','G2', nm, mae=float(np.abs(yd[ted] - p[ted]).mean()),
        pinball90=pinball(yd[ted], p[ted], 0.9))
rec('demand_drift','G2','global mean', mae=float(np.abs(yd[ted] - gmean).mean()),
    pinball90=pinball(yd[ted], np.full(ted.sum(), gmean), 0.9))
d['month'] = d.snapshot_date.dt.month; d['woy'] = d.snapshot_date.dt.isocalendar().week.astype(int)
d['kc'] = d.k.astype('category').cat.codes
DF = ['month','woy','kc','last_value','seasonal_naive']
Xd = d[DF].to_numpy(np.float32)
md = lgb.LGBMRegressor(**{**GBM, 'objective': 'l1'})
md.fit(Xd[trd], yd[trd], eval_set=[(Xd[vad], yd[vad])], callbacks=[lgb.early_stopping(40, verbose=False)])
pdte = md.predict(Xd[ted])
rec('demand_drift','G3','LightGBM tabular (no graph)',
    mae=float(np.abs(yd[ted] - pdte).mean()), pinball90=pinball(yd[ted], pdte, 0.9))
IMP_DD = pd.Series(md.feature_importances_, index=DF).sort_values(ascending=False)

# ==================================================================== output
log('\n=== TOP FEATURE IMPORTANCES (leak check) ===')
IMPS = {'fill_rate': IMP_FILL, 'arrival_week': IMP_ARR, 'shortage_qty': IMP_SH, 'demand_drift': IMP_DD}
for k, v in IMPS.items():
    tot = v.sum()
    log(f'  {k}: ' + ', '.join(f'{i}={x/tot:.1%}' for i, x in v.head(5).items()))
if OUT:
    json.dump({'dataset': D, 'split': {'train': '<=2023', 'val': '2024', 'test': '2025'},
               'results': RES,
               'importances': {k: {i: float(x) for i, x in v.head(10).items()} for k, v in IMPS.items()}},
              open(OUT, 'w'), indent=1)
    log(f'\nwritten -> {OUT}')
