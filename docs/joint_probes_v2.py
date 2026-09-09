"""§16 Level-3 joint probes, v2.

Corrects two measurement bugs in docs/validator_run5_joint_probes.py (v1), which is kept
unchanged for continuity with runs 5-6:

  probe 3  v1 computes late = (event_ts > current_promise_date).fillna(True). For an
           undelivered line event_ts is NaT and `NaT > date` evaluates to False, not NaN,
           so .fillna(True) never fires and every open/undelivered line is scored as
           ON TIME -- the inverse of the truth. v2 handles NaT explicitly and reports the
           delivered-only population alongside the full one.

  probe 7  v1 averages supplier_performance_weekly.fill_rate, which is forward-filled across
           inactive weeks and unweighted. v2 measures the arc ordered-weighted on simulation
           state, and reports the within-month and next-month decompositions separately.

Everything else is the same statistic as v1.
"""
import numpy as np, pandas as pd, sys, math, os
D = sys.argv[1].rstrip('/') + '/'
def L(n, **kw): return pd.read_csv(D+n+'.csv', **kw)
def hdr(t): print(f'\n{"="*92}\n{t}\n{"="*92}', flush=True)
def spear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30: return float('nan'), int(m.sum())
    a, b = pd.Series(x[m]).rank().to_numpy(), pd.Series(y[m]).rank().to_numpy()
    if a.std() == 0 or b.std() == 0: return float('nan'), int(m.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(m.sum())

hdr('PROBE 1 (DECISIVE) -- each label vs its most predictive feature')
lb = L('training_labels'); lb['snapshot_date'] = pd.to_datetime(lb.snapshot_date)
cpw = L('channel_performance_weekly', usecols=['channel_id','week_start','fill_rate_last13',
                                               'otd_rate_last13','load_ratio','lead_time_ratio'])
cpw['week_start'] = pd.to_datetime(cpw.week_start)
pol = L('po_lines', usecols=['po_line_id','channel_id'])
feat = cpw.sort_values('week_start')
for task, col in [('fill_rate','fill_rate_last13'), ('arrival_week','otd_rate_last13'),
                  ('capacity_strain','load_ratio')]:
    s = lb[lb.task == task]
    s = s.merge(pol, left_on='entity_id', right_on='po_line_id', how='inner') if task != 'capacity_strain' \
        else s.assign(channel_id=s.entity_id)
    j = pd.merge_asof(s.sort_values('snapshot_date'),
                      feat[['week_start','channel_id',col]].sort_values('week_start'),
                      left_on='snapshot_date', right_on='week_start', by='channel_id', direction='backward')
    r_, n = spear(j.label_value.to_numpy(float), j[col].to_numpy(float))
    print(f'  {task:<18} vs {col:<20} spearman r = {r_:+.4f}   n={n:,}')

hdr('PROBE 2 -- capacity pressure UP -> fill DOWN, and -> lead time UP')
c2 = L('channel_performance_weekly', usecols=['load_ratio','fill_rate','lead_time_ratio','qty_ordered'])
c2 = c2[c2.qty_ordered > 0]
for a, b, want in (('load_ratio','fill_rate','NEGATIVE'), ('load_ratio','lead_time_ratio','POSITIVE')):
    r_, n = spear(c2[a].to_numpy(float), c2[b].to_numpy(float))
    print(f'  {a} vs {b:<18} r = {r_:+.4f}  n={n:,}   (expect {want})')

hdr('PROBE 3 -- late and short co-occur on the same PO lines   [v2: NaT handled]')
pl = L('po_lines', usecols=['po_line_id','qty_ordered','current_promise_date'])
gl = L('grn_lines', usecols=['po_line_id','qty_received','event_ts'])
m = pl.merge(gl, on='po_line_id', how='left'); m['qty_received'] = m.qty_received.fillna(0)
short = (m.qty_received < m.qty_ordered).to_numpy()
ev = pd.to_datetime(m.event_ts); pr = pd.to_datetime(m.current_promise_date)
delivered = ev.notna().to_numpy()
# an undelivered line is not on time -- it is unresolved, and past its promise it is late
late_full = np.where(delivered, (ev > pr).to_numpy(), True)
def phi(a, b):
    a, b = a.astype(float), b.astype(float)
    if a.std() == 0 or b.std() == 0: return float('nan')
    return float(np.corrcoef(a, b)[0, 1])
print(f'  full population    phi={phi(short, late_full):+.4f}  '
      f'P(late|short)={late_full[short].mean():.4f}  P(late|full)={late_full[~short].mean():.4f}  n={len(short):,}')
sd_, ld_ = short[delivered], (ev > pr).to_numpy()[delivered]
print(f'  delivered only     phi={phi(sd_, ld_):+.4f}  '
      f'P(late|short)={ld_[sd_].mean():.4f}  P(late|full)={ld_[~sd_].mean():.4f}  n={int(delivered.sum()):,}')
print(f'  undelivered lines scored on-time by v1: {int((~delivered).sum()):,}')

hdr('PROBE 4 -- fill in constrained vs unconstrained supplier-months')
rc = L('revealed_capacity_monthly', usecols=['supplier_id','month','constrained_month_flag'])
sw = L('supplier_performance_weekly', usecols=['supplier_id','week_start','fill_rate','qty_ordered'])
sw = sw[sw.qty_ordered > 0].copy()
sw['month'] = pd.to_datetime(sw.week_start).dt.to_period('M').dt.to_timestamp()
rc['month'] = pd.to_datetime(rc.month).dt.to_period('M').dt.to_timestamp()
j = sw.merge(rc, on=['supplier_id','month'], how='inner')
g = j.groupby('constrained_month_flag').fill_rate.mean()
print(f'  mean fill constrained={g.get(True, float("nan")):.4f}  unconstrained={g.get(False, float("nan")):.4f}'
      f'  gap={g.get(False,0)-g.get(True,0):+.4f}   n={len(j):,}')

hdr('PROBE 5 -- shortage events with an identifiable upstream cause')
se = L('shortage_events')
sup = set(L('suppliers', usecols=['supplier_id']).supplier_id)
has = se.root_cause.isin(['supplier_delay','supplier_short','quality_reject']) & \
      se.responsible_supplier_id.astype(str).str.strip().ne('') & \
      se.responsible_supplier_id.astype(str).isin(sup)
print(f'  events {len(se):,}   identifiable upstream cause {has.mean()*100:6.2f}%   target ~100%')
print('  root_cause mix:'); print(se.root_cause.value_counts(normalize=True).to_string())

hdr('PROBE 6 -- shortage -> expedite / revision / alternate-sourcing probability UP')
ex = L('expedite_events', usecols=['part_id','plant_id'])
pp_all = L('part_plant', usecols=['part_id','plant_id'])
shk = set(se.part_id + '|' + se.plant_id); exk = set(ex.part_id + '|' + ex.plant_id)
allk = set(pp_all.part_id + '|' + pp_all.plant_id)
p1 = len(exk & shk) / max(1, len(shk)); p0 = len(exk - shk) / max(1, len(allk - shk))
print(f'  P(expedite | had a shortage) = {p1:.4f}   P(expedite | none) = {p0:.4f}   lift = {p1/max(p0,1e-9):,.1f}x')
print('  resolution_action mix:'); print(se.resolution_action.value_counts(normalize=True).to_string())

hdr('PROBE 7 -- alt sourcing -> receiver load UP -> its other channels degrade  [v2: ordered-weighted]')
sim = D + '_sim.npz'
if os.path.exists(sim):
    z = np.load(sim)
    pt, pch, pq, pd_ = z['pt'], z['pch'], z['pq'], z['pd']
    CS, MK, util = z['CS'], z['MONTHKEY'], z['util']
    sup_i = CS[pch]; mon = MK[pt]; NS = util.shape[1]; NM = util.shape[0]
    num = np.bincount(sup_i*NM+mon, weights=pd_.astype(float), minlength=NS*NM).reshape(NS, NM)
    den = np.bincount(sup_i*NM+mon, weights=pq.astype(float),  minlength=NS*NM).reshape(NS, NM)
    fill = np.where(den > 0, num/np.maximum(den, 1e-9), np.nan); U = util.T
    for lbl, u, f, d in (('utilisation(m) -> fill(m)  ', U.ravel(), fill.ravel(), den.ravel()),
                         ('utilisation(m) -> fill(m+1)', U[:, :-1].ravel(), fill[:, 1:].ravel(), den[:, 1:].ravel())):
        m_ = np.isfinite(u) & np.isfinite(f) & (d > 0)
        r_, n = spear(u[m_], f[m_])
        print(f'  {lbl}  rho = {r_:+.4f}   n={n:,}   (expect NEGATIVE)')
    up = U[:, :-1].ravel(); un = U[:, 1:].ravel()
    r_, n = spear(up, un); print(f'  utilisation persistence m -> m+1   rho = {r_:+.4f}   n={n:,}')
else:
    print('  _sim.npz absent -- cannot measure ordered-weighted; run on a generated dataset')

hdr('PROBE 8 -- staleness UP -> lead-time variance UP across channels')
c3 = L('channel_performance_weekly', usecols=['channel_id','reporting_lag_days','lead_time_actual_days','qty_ordered'])
c3 = c3[c3.qty_ordered > 0]
g3 = c3.groupby('channel_id').agg(st=('reporting_lag_days','median'), lv=('lead_time_actual_days','std')).dropna()
r_, n = spear(g3.st.to_numpy(float), g3.lv.to_numpy(float))
print(f'  median staleness vs sd(lead time)  r = {r_:+.4f}  n={n:,}   (expect POSITIVE)')

hdr('PROBE 9 -- KS of each label against Uniform(0,1); must reject at p < 0.01')
def ks_uniform(x):
    x = np.sort(np.asarray(x, float)); x = x[np.isfinite(x)]; n = len(x)
    if n < 10: return float('nan'), float('nan'), n
    cdf = np.clip(x, 0, 1)
    d = max(np.max(np.arange(1, n+1)/n - cdf), np.max(cdf - np.arange(0, n)/n))
    lam = (math.sqrt(n) + 0.12 + 0.11/math.sqrt(n)) * d
    p = 2*sum((-1)**(k-1)*math.exp(-2*k*k*lam*lam) for k in range(1, 101))
    return d, max(min(p, 1.0), 0.0), n
for task in lb.task.unique():
    v = lb[lb.task == task].label_value.to_numpy(float)
    d, p, n = ks_uniform(v)
    print(f'  {task:<18} D={d:.4f} p={p:.3e} n={n:,} mean={np.nanmean(v):.4f} sd={np.nanstd(v):.4f}  '
          f'{"REJECT (good)" if p < 0.01 else "CANNOT REJECT (bad)"}')
print('\n  censoring per task (§14: 1-60%):')
for task in lb.task.unique():
    c = lb[lb.task == task].label_censored.astype(str).str.lower().isin(['true','1'])
    print(f'  {task:<18} censored {c.mean()*100:6.2f}%')
