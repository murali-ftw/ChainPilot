"""synthetic_rules.md §16 Level-3 joint probes, run in the order the rules specify."""
import numpy as np, pandas as pd, sys, math
D = sys.argv[1] if len(sys.argv) > 1 else 'gen_v5/seed_1001'
def L(n, **kw): return pd.read_csv(f'{D}/{n}.csv', **kw)
def pear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30: return float('nan'), int(m.sum())
    x, y = x[m], y[m]
    if x.std() == 0 or y.std() == 0: return float('nan'), int(m.sum())
    return float(np.corrcoef(x, y)[0, 1]), int(m.sum())
def spear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30: return float('nan'), int(m.sum())
    return pear(pd.Series(x[m]).rank().to_numpy(), pd.Series(y[m]).rank().to_numpy())
def hdr(t): print(f'\n{"="*92}\n{t}\n{"="*92}')

hdr('PROBE 1 (DECISIVE) -- each label vs its most predictive feature')
lb = L('training_labels')
cpw = L('channel_performance_weekly', usecols=['channel_id','week_start','fill_rate_last13','otd_rate_last13','load_ratio','lead_time_ratio'])
cpw['week_start'] = pd.to_datetime(cpw.week_start)
pol = L('po_lines', usecols=['po_line_id','channel_id'])
lb['snapshot_date'] = pd.to_datetime(lb.snapshot_date)
feat = cpw.sort_values('week_start')
res = []
for task, col in [('fill_rate','fill_rate_last13'), ('arrival_week','otd_rate_last13'),
                  ('capacity_strain','load_ratio')]:
    s = lb[lb.task == task]
    if task in ('fill_rate','arrival_week'):
        s = s.merge(pol, left_on='entity_id', right_on='po_line_id', how='inner')
    else:
        s = s.assign(channel_id=s.entity_id)
    s = s.sort_values('snapshot_date')
    j = pd.merge_asof(s, feat[['week_start','channel_id',col]].sort_values('week_start'),
                      left_on='snapshot_date', right_on='week_start', by='channel_id', direction='backward')
    r_, n = spear(j.label_value.to_numpy(float), j[col].to_numpy(float))
    res.append((task, col, r_, n))
    print(f'  {task:<18} vs {col:<20} spearman r = {r_:+.4f}   n={n:,}')
print('\n  VERDICT: near-zero would mean labels are detached from features.')

hdr('PROBE 2 -- capacity pressure UP -> fill DOWN, and -> lead time UP')
c2 = L('channel_performance_weekly', usecols=['load_ratio','fill_rate','lead_time_ratio','qty_ordered'])
c2 = c2[c2.qty_ordered > 0]
r1, n1 = spear(c2.load_ratio.to_numpy(float), c2.fill_rate.to_numpy(float))
r2, n2 = spear(c2.load_ratio.to_numpy(float), c2.lead_time_ratio.to_numpy(float))
print(f'  load_ratio vs fill_rate        r = {r1:+.4f}  n={n1:,}   (expect NEGATIVE)')
print(f'  load_ratio vs lead_time_ratio  r = {r2:+.4f}  n={n2:,}   (expect POSITIVE)')

hdr('PROBE 3 -- late and short co-occur on the same PO lines')
pl = L('po_lines', usecols=['po_line_id','qty_ordered','current_promise_date'])
gl = L('grn_lines', usecols=['po_line_id','qty_received','event_ts'])
m = pl.merge(gl, on='po_line_id', how='left')
m['qty_received'] = m.qty_received.fillna(0)
short = (m.qty_received < m.qty_ordered).to_numpy()
late = (pd.to_datetime(m.event_ts) > pd.to_datetime(m.current_promise_date)).fillna(True).to_numpy()
r_, n = pear(short.astype(float), late.astype(float))
a = pd.crosstab(short, late)
print(f'  phi(short, late) = {r_:+.4f}  n={n:,}')
print(f'  P(late | short) = {late[short].mean():.4f}   P(late | full) = {late[~short].mean():.4f}')

hdr('PROBE 4 -- fill in constrained vs unconstrained supplier-months')
rc = L('revealed_capacity_monthly', usecols=['supplier_id','month','constrained_month_flag'])
sw = L('supplier_performance_weekly', usecols=['supplier_id','week_start','fill_rate','qty_ordered'])
sw = sw[sw.qty_ordered > 0].copy()
sw['month'] = pd.to_datetime(sw.week_start).dt.to_period('M').dt.to_timestamp()
rc['month'] = pd.to_datetime(rc.month).dt.to_period('M').dt.to_timestamp()
j = sw.merge(rc, on=['supplier_id','month'], how='inner')
g = j.groupby('constrained_month_flag').fill_rate.mean()
print(f'  mean fill, constrained   = {g.get(True, float("nan")):.4f}')
print(f'  mean fill, unconstrained = {g.get(False, float("nan")):.4f}')
print(f'  gap = {g.get(False,0)-g.get(True,0):+.4f}   (expect materially lower when constrained)')

hdr('PROBE 5 -- shortage events with an identifiable upstream cause')
se = L('shortage_events')
has = se.root_cause.isin(['supplier_delay','supplier_short','quality_reject']) & se.responsible_supplier_id.notna() & (se.responsible_supplier_id.astype(str).str.strip() != '')
sup = set(L('suppliers', usecols=['supplier_id']).supplier_id)
res_ok = se.responsible_supplier_id.astype(str).isin(sup)
print(f'  events = {len(se):,}')
print(f'  with an identifiable upstream cause      = {has.mean()*100:6.2f}%   target ~100%')
print(f'  responsible_supplier_id resolves to master = {res_ok.mean()*100:6.2f}%')

hdr('PROBE 6 -- shortage -> expedite / revision / alternate-sourcing probability UP')
ex = L('expedite_events', usecols=['part_id','plant_id','event_ts'])
se2 = se[['part_id','plant_id','shortage_start_ts']].copy()
se2['k'] = se2.part_id + '|' + se2.plant_id
ex['k'] = ex.part_id + '|' + ex.plant_id
shk = set(se2.k); allk = set(L('part_plant', usecols=['part_id','plant_id']).apply(lambda x: x.part_id+'|'+x.plant_id, axis=1))
p_short = len(set(ex.k) & shk) / max(1, len(shk))
p_none = len(set(ex.k) - shk) / max(1, len(allk - shk))
print(f'  P(expedite | part-plant had a shortage) = {p_short:.4f}')
print(f'  P(expedite | no shortage)               = {p_none:.4f}')
print(f'  lift = {p_short/max(p_none,1e-9):.1f}x')
print(f'  resolution_action distribution:'); print(se.resolution_action.value_counts(normalize=True).to_string())

hdr('PROBE 7 -- alternate sourcing -> receiving supplier load UP -> its other channels degrade')
rc2 = L('revealed_capacity_monthly', usecols=['supplier_id','month','max_delivered_12m','declared_capacity_qty'])
rc2['utilisation_pct'] = rc2.max_delivered_12m / rc2.declared_capacity_qty.replace(0, float('nan')) * 100
rc2['month'] = pd.to_datetime(rc2.month)
rc2 = rc2.sort_values(['supplier_id','month'])
rc2['util_next'] = rc2.groupby('supplier_id').utilisation_pct.shift(-1)
sw2 = L('supplier_performance_weekly', usecols=['supplier_id','week_start','fill_rate','lead_time_ratio','qty_ordered'])
sw2 = sw2[sw2.qty_ordered > 0].copy()
sw2['month'] = pd.to_datetime(sw2.week_start).dt.to_period('M').dt.to_timestamp()
sm = sw2.groupby(['supplier_id','month']).fill_rate.mean().reset_index().rename(columns={'fill_rate':'fill_next'})
sm['month'] = sm.month - pd.offsets.MonthBegin(1)
j2 = rc2.merge(sm, on=['supplier_id','month'], how='inner')
r_, n = spear(j2.utilisation_pct.to_numpy(float), j2.fill_next.to_numpy(float))
print(f'  supplier utilisation(m) vs its fill(m+1)  r = {r_:+.4f}  n={n:,}   (expect NEGATIVE)')
r3, n3 = spear(rc2.utilisation_pct.to_numpy(float), rc2.util_next.to_numpy(float))
print(f'  utilisation persistence m -> m+1          r = {r3:+.4f}  n={n3:,}')

hdr('PROBE 8 -- staleness UP -> lead-time variance UP across channels')
c3 = L('channel_performance_weekly', usecols=['channel_id','reporting_lag_days','lead_time_actual_days','qty_ordered'])
c3 = c3[c3.qty_ordered > 0]
g3 = c3.groupby('channel_id').agg(stale=('reporting_lag_days','median'), lv=('lead_time_actual_days','std')).dropna()
r_, n = spear(g3.stale.to_numpy(float), g3.lv.to_numpy(float))
print(f'  median staleness vs sd(lead time)  r = {r_:+.4f}  n={n:,}   (expect POSITIVE)')

hdr('PROBE 9 -- KS test of each label against Uniform(0,1); must reject at p < 0.01')
def ks_uniform(x):
    x = np.sort(np.asarray(x, float)); x = x[np.isfinite(x)]
    n = len(x)
    if n < 10: return float('nan'), float('nan'), n
    cdf = np.clip(x, 0, 1)
    d = max(np.max(np.arange(1, n+1)/n - cdf), np.max(cdf - np.arange(0, n)/n))
    lam = (math.sqrt(n) + 0.12 + 0.11/math.sqrt(n)) * d
    p = 2*sum((-1)**(k-1)*math.exp(-2*k*k*lam*lam) for k in range(1, 101))
    return d, max(min(p, 1.0), 0.0), n
for task in lb.task.unique():
    v = lb[lb.task == task].label_value.to_numpy(float)
    d, p, n = ks_uniform(v)
    print(f'  {task:<18} D={d:.4f}  p={p:.3e}  n={n:,}  mean={np.nanmean(v):.4f} sd={np.nanstd(v):.4f}  '
          f'{"REJECT (good)" if p < 0.01 else "CANNOT REJECT -- looks uniform (bad)"}')
print('\n  censoring per task (§14: must be 1-60%):')
for task in lb.task.unique():
    c = lb[lb.task == task].label_censored
    c = c.astype(str).str.lower().isin(['true','1'])
    print(f'  {task:<18} censored {c.mean()*100:6.2f}%')
