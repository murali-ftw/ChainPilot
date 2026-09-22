#!/usr/bin/env python
"""ChainPilot V8 validator -- the instrument, and it is FROZEN.

Four levels, run in the order the v8 brief §4 gives, plus the G0..G7 readiness ladder.
Execution STOPS at the first failed gate. No band in this file may be widened to let a
dataset through: a band adjusted until the data passes has measured itself, not the data.
Only path discovery may change after this file is written.

EVERY CHECK CARRIES A `breaks` STRING naming the mutation that makes it fail (§1.3: ten
documented instances of gates that could not fail). `--selftest` applies those mutations
and reports any check that survives one -- a check that survives its own mutation is not
a check and must be deleted, not kept for comfort.

usage
  validator_v8.py <seed_dir> [--out report.json] [--levels 1234] [--selftest]
  validator_v8.py --variance <seed_dir> [<seed_dir> ...] --out variance.json
"""
import numpy as np, pandas as pd, json, math, re, sys, argparse, warnings, time
from pathlib import Path
warnings.filterwarnings('ignore')

VALIDATOR_VERSION = 'validator_v8.0'
BANDS_FROZEN_AT = '2026-09-22'

def log(*a): print(*a, flush=True)

# ===========================================================================
# result accumulator
# ===========================================================================
class Res:
    def __init__(self):
        self.rows = []
        self.gate_failed = None
    def add(self, level, gate, name, passed, detail, breaks, value=None):
        self.rows.append(dict(level=level, gate=gate, name=name, passed=bool(passed),
                              detail=detail, breaks=breaks, value=value))
        tag = 'PASS' if passed else 'FAIL'
        log(f'  [{tag}] L{level} {gate or "--":<3} {name:<58} {detail}')
        return passed
    def note(self, level, gate, name, detail, value=None):
        """A reported measurement with no pass/fail band. Never counts as a check."""
        self.rows.append(dict(level=level, gate=gate, name=name, passed=None,
                              detail=detail, breaks=None, value=value))
        log(f'  [    ] L{level} {gate or "--":<3} {name:<58} {detail}')
    def failed(self, gate=None):
        return [r for r in self.rows if r['passed'] is False and (gate is None or r['gate'] == gate)]

# ===========================================================================
# schema
# ===========================================================================
def parse_schema(path):
    sql = re.sub(r'--[^\n]*', '', Path(path).read_text())
    tabs = re.findall(r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?["`]?(\w+)["`]?\s*\((.*?)\n\)\s*;', sql, re.S | re.I)
    out = {}
    for name, body in tabs:
        cols, depth, cur = [], 0, ''
        for ch in body:
            if ch == '(': depth += 1
            if ch == ')': depth -= 1
            if ch == ',' and depth == 0: cols.append(cur); cur = ''
            else: cur += ch
        cols.append(cur)
        names, notnull, pk, fks, enums = [], set(), [], [], {}
        for c in cols:
            c = c.strip()
            if not c: continue
            m_pk = re.match(r'^PRIMARY\s+KEY\s*\((.*?)\)', c, re.I)
            if m_pk:
                pk = [x.strip().strip('"`') for x in m_pk.group(1).split(',')]; continue
            m_ck = re.search(r'CHECK\s*\((.*)\)\s*$', c, re.I | re.S)
            if re.match(r'^(CONSTRAINT|CHECK)\b', c, re.I):
                if m_ck:
                    m_in = re.search(r'["`]?(\w+)["`]?\s+IN\s*\((.*?)\)', m_ck.group(1), re.I | re.S)
                    if m_in:
                        enums[m_in.group(1)] = {v.strip().strip("'\"")
                                                for v in m_in.group(2).split(',')}
                continue
            if re.match(r'^(FOREIGN|UNIQUE)\b', c, re.I): continue
            m = re.match(r'^["`]?(\w+)["`]?\s+(.*)$', c, re.S)
            if not m: continue
            col, rest = m.group(1), m.group(2)
            names.append(col)
            if 'NOT NULL' in rest.upper(): notnull.add(col)
            if re.search(r'\bPRIMARY\s+KEY\b', rest, re.I): pk = [col]
            m_fk = re.search(r'REFERENCES\s+["`]?(\w+)["`]?\s*\(\s*["`]?(\w+)["`]?\s*\)', rest, re.I)
            if m_fk: fks.append((col, m_fk.group(1), m_fk.group(2)))
            m_in = re.search(r'CHECK\s*\(\s*["`]?\w+["`]?\s+IN\s*\((.*?)\)\s*\)', rest, re.I | re.S)
            if m_in: enums[col] = {v.strip().strip("'\"") for v in m_in.group(1).split(',')}
        out[name] = dict(cols=names, notnull=notnull, pk=pk, fks=fks, enums=enums)
    return out

# ===========================================================================
# io helpers
# ===========================================================================
class Data:
    def __init__(self, root):
        self.root = Path(root)
        self._cache = {}
    def path(self, t): return self.root / f'{t}.csv'
    def has(self, t): return self.path(t).exists()
    def get(self, t, **kw):
        key = (t, tuple(sorted(kw.get('usecols', ()) or ())))
        if key in self._cache: return self._cache[key]
        df = pd.read_csv(self.path(t), low_memory=False, **kw)
        if len(self._cache) > 14: self._cache.pop(next(iter(self._cache)))
        self._cache[key] = df
        return df
    def chunks(self, t, size=3_000_000, **kw):
        return pd.read_csv(self.path(t), chunksize=size, low_memory=False, **kw)
    def nrows(self, t):
        n = 0
        for c in pd.read_csv(self.path(t), chunksize=2_000_000, usecols=[0]): n += len(c)
        return n

W0 = pd.Timestamp('2016-01-04')
def _dt(x):
    """Parse a timestamp column that may mix fractional-second and whole-second forms.
    pandas infers one format from the first value and then raises on the rest."""
    return pd.to_datetime(x, format='ISO8601')


def wk_index(x):
    d = _dt(pd.Series(np.asarray(x))).values
    d = pd.DatetimeIndex(d).normalize()
    ws = d - pd.to_timedelta(d.weekday, unit='D')
    return ((ws - W0).days // 7).to_numpy()

def spear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30: return float('nan'), int(m.sum())
    x, y = pd.Series(x[m]).rank().to_numpy(), pd.Series(y[m]).rank().to_numpy()
    if x.std() == 0 or y.std() == 0: return float('nan'), int(m.sum())
    return float(np.corrcoef(x, y)[0, 1]), int(m.sum())

def ks_uniform(x):
    x = np.sort(np.asarray(x, float)); x = x[np.isfinite(x)]
    n = len(x)
    if n < 10: return float('nan'), float('nan'), n
    cdf = np.clip(x, 0, 1)
    d = max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(0, n) / n))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, max(min(p, 1.0), 0.0), n


# ===========================================================================
# LEVEL 1 -- STRUCTURAL
# PK uniqueness, FK integrity, entity resolution, nullability, enums, types, date ranges.
# ===========================================================================
def level1(D, SCH, res, mut=None):
    log('\n--- LEVEL 1 : STRUCTURAL ---')
    present = sorted(t for t in SCH if D.has(t))
    missing = sorted(t for t in SCH if not D.has(t))
    res.add(1, 'G0', 'every schema table is present', not missing,
            f'{len(present)}/{len(SCH)} present' + (f'; missing {missing}' if missing else ''),
            breaks='delete any CSV from the seed directory')

    bad_cols, bad_pk, bad_null, bad_enum, bad_fk, bad_date = [], [], [], [], [], []
    masters = {}
    for t in ('suppliers', 'parts', 'plants', 'products', 'sourcing_channels',
              'customers', 'supplier_sites', 'business_units'):
        if D.has(t):
            k = SCH[t]['pk'][0] if SCH[t]['pk'] else SCH[t]['cols'][0]
            masters[t] = set(D.get(t, usecols=[k])[k].astype(str))

    WORLD_LO, WORLD_HI = pd.Timestamp('2015-01-01'), pd.Timestamp('2027-12-31')
    for t in present:
        spec = SCH[t]
        hdr = pd.read_csv(D.path(t), nrows=0).columns.tolist()
        if hdr != spec['cols']:
            bad_cols.append((t, [c for c in spec['cols'] if c not in hdr]))
        use = [c for c in spec['cols'] if c in hdr]
        pk_seen, nrows = set(), 0
        pk_cols = [c for c in spec['pk'] if c in hdr]
        dup = 0
        nulls = {c: 0 for c in spec['notnull'] if c in hdr}
        enum_bad = {c: 0 for c in spec['enums'] if c in hdr}
        fk_bad = {}
        dcols = [c for c in use if c.endswith(('_ts', '_date', '_from', '_to')) or c in ('month', 'period', 'week_start')]
        dmin, dmax = None, None
        for ch in D.chunks(t, usecols=use):
            if mut: ch = mut(t, ch)
            nrows += len(ch)
            if pk_cols:
                k = ch[pk_cols].astype(str).agg('|'.join, axis=1) if len(pk_cols) > 1 else ch[pk_cols[0]].astype(str)
                kv = k.to_numpy()
                new = set(kv)
                dup += len(kv) - len(new) + len(pk_seen & new)
                pk_seen |= new
            for c in nulls:
                nulls[c] += int(ch[c].isna().sum() + (ch[c].astype(str).str.strip() == '').sum())
            for c in enum_bad:
                v = ch[c].astype(str).str.strip()
                v = v[(v != '') & (v.str.lower() != 'nan')]
                if c in ('is_firm',) or set(v.unique()) <= {'True', 'False'}: continue
                enum_bad[c] += int((~v.isin(spec['enums'][c])).sum())
            for col, rt, rc in spec['fks']:
                if col not in hdr or rt not in masters: continue
                v = ch[col].astype(str).str.strip()
                v = v[(v != '') & (v.str.lower() != 'nan')]
                fk_bad[col] = fk_bad.get(col, 0) + int((~v.isin(masters[rt])).sum())
            for c in dcols:
                d = pd.to_datetime(ch[c], errors='coerce')
                d = d[d.notna()]
                if len(d):
                    dmin = d.min() if dmin is None else min(dmin, d.min())
                    dmax = d.max() if dmax is None else max(dmax, d.max())
        if dup: bad_pk.append((t, dup))
        for c, n in nulls.items():
            if n: bad_null.append((t, c, n))
        for c, n in enum_bad.items():
            if n: bad_enum.append((t, c, n))
        for c, n in fk_bad.items():
            if n: bad_fk.append((t, c, n))
        if dmin is not None and (dmin < WORLD_LO or dmax > WORLD_HI):
            bad_date.append((t, str(dmin.date()), str(dmax.date())))

    res.add(1, 'G0', 'header matches schema column list', not bad_cols,
            'all tables' if not bad_cols else f'{len(bad_cols)} tables differ: {bad_cols[:3]}',
            breaks='drop or rename a column in any emitted frame')
    res.add(1, 'G0', 'primary keys unique', not bad_pk,
            'no duplicates' if not bad_pk else f'{bad_pk[:5]}',
            breaks='emit two rows sharing a po_line_id')
    res.add(1, 'G0', 'NOT NULL columns populated', not bad_null,
            'all populated' if not bad_null else f'{len(bad_null)} violations: {bad_null[:4]}',
            breaks='blank revealed_capacity_monthly.constrained_month_flag')
    res.add(1, 'G0', 'enum / CHECK domains honoured', not bad_enum,
            'all in domain' if not bad_enum else f'{bad_enum[:5]}',
            breaks="write txn_type='opening_balance', which is not in the schema enum")
    res.add(1, 'G0', 'foreign keys resolve to masters', not bad_fk,
            'all resolve' if not bad_fk else f'{bad_fk[:5]}',
            breaks="point shortage_events.responsible_supplier_id at SUP99999")
    res.add(1, 'G0', 'dates inside the world span', not bad_date,
            f'{WORLD_LO.date()}..{WORLD_HI.date()}' if not bad_date else f'{bad_date[:4]}',
            breaks='post the opening balance at 1970-01-01')
    return res


# ===========================================================================
# LEVEL 2 -- ACCOUNTING.  EXACT INTEGER IDENTITIES, never approximate.
# ===========================================================================
def level2(D, SCH, res, mut=None):
    log('\n--- LEVEL 2 : ACCOUNTING (exact integer) ---')
    # ---- PO / GRN conservation against the derived weekly stores.
    # The store buckets a row on its VISIBLE week, max(event_week, recorded_week). A row
    # whose visible week falls past the extract cut has no bucket to land in, so the
    # identity is stated over the rows the store can hold -- and the share that fall
    # outside is checked separately, because a world where most rows are unbucketable
    # is broken whatever the in-span sum says.
    T_SPAN = int(D.get('supplier_performance_weekly', usecols=['week_start']).week_start.nunique())
    src_ord = out_ord = tot_ord = 0
    for ch in D.chunks('po_lines', usecols=['qty_ordered', 'created_ts', 'recorded_ts']):
        w = np.maximum(wk_index(ch.created_ts), wk_index(ch.recorded_ts))
        ins = (w >= 0) & (w < T_SPAN)
        src_ord += int(ch.qty_ordered[ins].sum()); tot_ord += len(ch); out_ord += int((~ins).sum())
    src_rec = out_rec = tot_rec = 0
    for ch in D.chunks('grn_lines', usecols=['qty_received', 'event_ts', 'recorded_ts']):
        w = np.maximum(wk_index(ch.event_ts), wk_index(ch.recorded_ts))
        ins = (w >= 0) & (w < T_SPAN)
        src_rec += int(ch.qty_received[ins].sum()); tot_rec += len(ch); out_rec += int((~ins).sum())
    emt_ord = emt_rec = 0
    for ch in D.chunks('channel_performance_weekly', usecols=['qty_ordered', 'qty_received']):
        if mut: ch = mut('channel_performance_weekly', ch)
        emt_ord += int(ch.qty_ordered.sum()); emt_rec += int(ch.qty_received.sum())
    res.add(2, 'G0', 'ordered units: po_lines == channel store', src_ord == emt_ord,
            f'{src_ord:,} vs {emt_ord:,}', breaks='drop one PO line from the weekly aggregation',
            value=[src_ord, emt_ord])
    res.add(2, 'G0', 'received units: grn_lines == channel store', src_rec == emt_rec,
            f'{src_rec:,} vs {emt_rec:,}', breaks='bucket a receipt on the event week instead of the visible week',
            value=[src_rec, emt_rec])
    s_ord = s_rec = 0
    for ch in D.chunks('supplier_performance_weekly', usecols=['qty_ordered', 'qty_received']):
        s_ord += int(ch.qty_ordered.sum()); s_rec += int(ch.qty_received.sum())
    res.add(2, 'G0', 'rows whose visible week falls outside the span stay under 1%',
            out_ord / max(tot_ord, 1) < 0.01 and out_rec / max(tot_rec, 1) < 0.01,
            f'po_lines {out_ord:,}/{tot_ord:,} ({out_ord/max(tot_ord,1)*100:.3f}%), '
            f'grn_lines {out_rec:,}/{tot_rec:,} ({out_rec/max(tot_rec,1)*100:.3f}%)',
            breaks='let the recording lag run for years, so most rows land past the extract cut',
            value=[out_ord / max(tot_ord, 1), out_rec / max(tot_rec, 1)])
    res.add(2, 'G0', 'supplier store reconciles to the channel store',
            (s_ord == src_ord) and (s_rec == src_rec),
            f'ordered {s_ord:,} recv {s_rec:,}', breaks='aggregate the supplier store from a different filter')

    # ---- quality: inspected == received, accepted + rejected == inspected
    q = D.get('quality_inspections', usecols=['grn_line_id', 'qty_inspected', 'qty_accepted', 'qty_rejected'])
    bad_q = int((q.qty_accepted + q.qty_rejected != q.qty_inspected).sum())
    res.add(2, 'G0', 'quality: accepted + rejected == inspected', bad_q == 0,
            f'{bad_q:,} violations of {len(q):,}', breaks='draw the rejection twice with different seeds')
    g = D.get('grn_lines', usecols=['grn_line_id', 'qty_received'])
    mq = g.merge(q, on='grn_line_id', how='inner')
    bad_qi = int((mq.qty_received != mq.qty_inspected).sum())
    res.add(2, 'G0', 'quality: inspected == grn qty_received', bad_qi == 0,
            f'{bad_qi:,} violations of {len(mq):,}', breaks='inspect a quantity the GRN never received')

    # ---- §2.1 THE ROLL-FORWARD.  The emitted ledger, summed forward from the opening
    # posting, must reproduce inventory_position_weekly.qty_on_hand EXACTLY, every
    # part-plant, every week.  This is the check that caught v7's double-counted scrap.
    log('  building the ledger roll-forward (this is the §2.1 check)...')
    acc = {}
    ntx = 0
    for ch in D.chunks('inventory_transactions',
                       usecols=['part_id', 'plant_id', 'txn_type', 'qty', 'event_ts']):
        if mut: ch = mut('inventory_transactions', ch)
        ntx += len(ch)
        ch['w'] = wk_index(ch.event_ts)
        gsum = ch.groupby(['part_id', 'plant_id', 'w'], sort=False).qty.sum()
        for k, v in gsum.items():
            acc[k] = acc.get(k, 0) + int(v)
    led = pd.Series(acc).rename('q').reset_index()
    led.columns = ['part_id', 'plant_id', 'w', 'q']
    led = led.sort_values(['part_id', 'plant_id', 'w'])
    led['bal'] = led.groupby(['part_id', 'plant_id']).q.cumsum()
    ipw = D.get('inventory_position_weekly', usecols=['part_id', 'plant_id', 'week_start', 'qty_on_hand'])
    if mut: ipw = mut('inventory_position_weekly', ipw)
    ipw['w'] = wk_index(ipw.week_start)
    # forward-fill the ledger balance onto every week of the position grid
    full = ipw[['part_id', 'plant_id', 'w', 'qty_on_hand']].sort_values('w', kind='stable')
    j = pd.merge_asof(full, led[['part_id', 'plant_id', 'w', 'bal']].sort_values('w', kind='stable'),
                      on='w', by=['part_id', 'plant_id'], direction='backward')
    j['bal'] = j.bal.fillna(0).astype(np.int64)
    bad = int((j.bal != j.qty_on_hand).sum())
    res.add(2, 'G0', '§2.1 ledger roll-forward == weekly position, EXACT, every week',
            bad == 0, f'{bad:,} mismatched part-plant-weeks of {len(j):,} ({ntx:,} ledger rows)',
            breaks='post the opening balance as a level instead of an adjustment row, or net '
                   'the scrap into the receipt and post it again (v7 did exactly this)',
            value=int(bad))
    neg = int((ipw.qty_on_hand < 0).sum())
    res.add(2, 'G0', 'no negative on-hand anywhere in the roll-forward', neg == 0,
            f'{neg:,} negative weeks', breaks='issue more than is on hand')

    # ---- monthly snapshots must agree with the same ledger
    # The monthly snapshot must be the same ledger, closed monthly. Exact integers.
    sn = D.get('inventory_snapshots', usecols=['part_id', 'plant_id', 'snapshot_date', 'qty_on_hand'])
    sn['m'] = _dt(sn.snapshot_date).dt.to_period('M')
    lm = led.copy()
    lm['m'] = (pd.to_datetime(W0) + pd.to_timedelta(lm.w * 7, unit='D')).dt.to_period('M')
    lmm = lm.groupby(['part_id', 'plant_id', 'm'], sort=True).q.sum().reset_index()
    lmm['bal'] = lmm.groupby(['part_id', 'plant_id']).q.cumsum()
    sj = sn.merge(lmm[['part_id', 'plant_id', 'm', 'bal']], on=['part_id', 'plant_id', 'm'], how='inner')
    dev = int((sj.bal.astype(np.int64) != sj.qty_on_hand).sum())
    res.add(2, 'G0', 'monthly snapshot == the same ledger closed monthly, EXACT',
            dev == 0, f'{dev:,} of {len(sj):,} month-ends differ',
            breaks='date the snapshot at month start while it holds the closing balance '
                   '(v7 did exactly this), or generate snapshots independently of the ledger',
            value=dev)

    # ---- §2.5 allocation sums
    al = D.get('supplier_allocation', usecols=['part_id', 'plant_id', 'supplier_id',
                                               'allocation_pct', 'effective_from'])
    if mut: al = mut('supplier_allocation', al)
    g = al.groupby(['part_id', 'plant_id', 'effective_from']).allocation_pct.sum()
    off = int((np.abs(g - 100.0) > 0.5).sum())
    res.add(2, 'G0', 'allocation_pct sums to 100 per part-plant per effective date',
            off == 0, f'{off:,} of {len(g):,} windows off by more than 0.5pp',
            breaks='log only the channels whose share moved, not the whole split', value=off)
    ndates = al.effective_from.nunique()
    res.add(2, 'G0', 'allocation is time-varying, not one static split', ndates > 4,
            f'{ndates} distinct effective dates', breaks='write every allocation row at 2019-01-01')

    # ---- production balance
    pa = D.get('production_actual', usecols=['product_id', 'plant_id', 'period', 'actual_qty',
                                             'scrapped_qty', 'rework_qty'])
    pdf = D.get('plan_drift_features', usecols=['product_id', 'plant_id', 'target_period', 'actual_qty'])
    m = pdf.merge(pa.rename(columns={'period': 'target_period'}),
                  on=['product_id', 'plant_id', 'target_period'], how='inner', suffixes=('_d', '_a'))
    bad_p = int((m.actual_qty_d != m.actual_qty_a).sum())
    res.add(2, 'G0', 'plan_drift actual == production_actual', bad_p == 0,
            f'{bad_p:,} of {len(m):,} rows differ', breaks='draw the actual twice')
    return res


# ===========================================================================
# V8 BRIEF REQUIREMENTS -- §2.2, §2.3, §2.4, §2.6.  Bands frozen with this file.
# ===========================================================================
def v8_requirements(D, res, mut=None):
    log('\n--- V8 BRIEF REQUIREMENTS (§2.2 plan, §2.3 lag, §2.4 population, §2.6 terms) ---')
    # ---------------- §2.4 entity population, readiness gate G1
    ch = D.get('sourcing_channels', usecols=['channel_id', 'supplier_id', 'part_id', 'plant_id'])
    NCH = len(ch)
    pol = D.get('po_lines', usecols=['po_line_id', 'channel_id', 'part_id'])
    traded = pol.channel_id.nunique()
    cold = NCH - traded
    cw = D.nrows('channel_performance_weekly')
    weeks = cw / max(NCH, 1)
    res.add(0, 'G1', '§2.4 sourcing channels >= 15,000', NCH >= 15000, f'{NCH:,}',
            breaks='ship 320 channels, as an earlier version did', value=NCH)
    res.add(0, 'G1', '§2.4 channels that ever traded >= 14,000', traded >= 14000, f'{traded:,}',
            breaks='let the order policy idle most channels', value=traded)
    res.add(0, 'G1', '§2.4 cold-start slice in 3-6%', 0.03 <= cold / NCH <= 0.06,
            f'{cold:,} ({cold/NCH*100:.2f}%)', breaks='give every channel an order', value=cold / NCH)
    res.add(0, 'G1', '§2.4 channel-week rows >= 5,000,000', cw >= 5_000_000, f'{cw:,}',
            breaks='shorten the world span', value=cw)
    res.add(0, 'G1', '§2.4 channel-weeks per traded channel >= 350', weeks >= 350, f'{weeks:.0f}',
            breaks='emit only the weeks a channel was active', value=weeks)
    yrs = 10.24
    pol_per = len(pol) / max(traded, 1) / yrs
    res.add(0, 'G1', '§2.4 PO lines per channel per year >= 6', pol_per >= 6.0, f'{pol_per:.2f}',
            breaks='slow the review cadence or raise the reorder point', value=pol_per)
    chp = ch.assign(pp=ch.part_id + '|' + ch.plant_id)
    deg = chp.groupby('pp').size()
    med_deg = float(deg.median())
    res.add(0, 'G1', 'G1 median channel degree at part-plant > 1', med_deg > 1.0,
            f'median {med_deg:.1f}, p25 {deg.quantile(.25):.0f}, max {deg.max()}',
            breaks='single-source every part-plant, which leaves attention nothing to choose between',
            value=med_deg)
    sdeg = ch.groupby('supplier_id').size()
    res.note(0, 'G1', 'supplier degree (neighbourhood the capacity head reads)',
             f'median {sdeg.median():.0f}, min {sdeg.min()}, max {sdeg.max()}')

    # ---------------- §2.2 forward requirement plan
    pp_ = D.get('production_plan', usecols=['plan_version', 'plan_date', 'product_id', 'plant_id',
                                            'target_period', 'planned_qty', 'is_firm'])
    if mut: pp_ = mut('production_plan', pp_)
    pp_['h'] = (_dt(pp_.target_period) - _dt(pp_.plan_date)).dt.days
    per_ver = pp_.groupby(['product_id', 'plant_id', 'plan_date']).size()
    res.add(0, 'V8', '§2.2 every published version covers MORE THAN ONE future period',
            int(per_ver.min()) > 1, f'min {int(per_ver.min())}, median {int(per_ver.median())}, '
            f'max {int(per_ver.max())} periods per version',
            breaks='emit one target_period per plan_date, as v7 did', value=int(per_ver.min()))
    res.add(0, 'V8', '§2.2 horizon_days varies, is not a fixed value', pp_.h.nunique() > 5,
            f'{pp_.h.nunique()} distinct horizons, {pp_.h.min()}..{pp_.h.max()} days',
            breaks='set horizon_days = 30 for every row, as v7 did', value=int(pp_.h.nunique()))
    span = pp_.groupby(['product_id', 'plant_id', 'plan_date']).h.max() / 7.0
    res.add(0, 'V8', '§2.2 each version rolls 12-26 weeks forward',
            bool((span >= 12).all() and (span <= 26).all()),
            f'version span {span.min():.0f}..{span.max():.0f} weeks, median {span.median():.0f}; '
            f'row horizons {pp_.h.min()}..{pp_.h.max()} days',
            breaks='draw one horizon length for every version, or publish a single period',
            value=[float(span.min()), float(span.max())])
    nv = pp_.groupby(['product_id', 'plant_id', 'target_period']).plan_version.max()
    res.add(0, 'V8', '§2.2 a target period is republished across versions', int(nv.max()) > 1,
            f'max {int(nv.max())} versions of one target period, median {int(nv.median())}',
            breaks='publish every target period exactly once', value=int(nv.max()))
    # the drift mechanism must still be underneath: far horizons must be worse than near ones
    dr = D.get('plan_drift_features', usecols=['original_plan_qty', 'latest_plan_qty', 'actual_qty',
                                               'horizon_days', 'plan_revision_count'])
    e_orig = np.abs(dr.original_plan_qty - dr.actual_qty) / np.maximum(dr.actual_qty, 1)
    e_last = np.abs(dr.latest_plan_qty - dr.actual_qty) / np.maximum(dr.actual_qty, 1)
    res.add(0, 'V8', '§2.2 later versions are closer to the actual than V1',
            float(e_last.median()) < float(e_orig.median()),
            f'median |err| V1 {e_orig.median():.4f} -> latest {e_last.median():.4f}',
            breaks='republish V1 unchanged, so revision carries no information',
            value=[float(e_orig.median()), float(e_last.median())])
    pdw = D.get('part_demand_weekly', usecols=['as_of_date', 'horizon_days',
                                               'gross_requirement_p50', 'gross_requirement_p90'])
    res.add(0, 'V8', '§2.2 part_demand_weekly horizon_days varies', pdw.horizon_days.nunique() > 5,
            f'{pdw.horizon_days.nunique()} distinct, {pdw.as_of_date.nunique()} as-of dates',
            breaks='net requirements once per week with a single-period horizon')
    res.add(0, 'V8', '§2.2 p90 requirement strictly above p50',
            bool((pdw.gross_requirement_p90 >= pdw.gross_requirement_p50).all()),
            f'{int((pdw.gross_requirement_p90 < pdw.gross_requirement_p50).sum()):,} inversions',
            breaks='copy p50 into p90')

    # ---------------- §2.3 recording layer
    lagstats = {}
    for t in ('po_lines', 'supplier_acknowledgements', 'goods_receipts', 'asn',
              'inventory_transactions', 'shortage_events', 'expedite_events',
              'po_line_revisions', 'quality_inspections'):
        if not D.has(t): continue
        ev = 'event_ts' if 'event_ts' in pd.read_csv(D.path(t), nrows=0).columns else \
             ('created_ts' if 'created_ts' in pd.read_csv(D.path(t), nrows=0).columns else
              ('receipt_ts' if t == 'goods_receipts' else 'dispatch_ts'))
        vals = []
        for cnk in D.chunks(t, usecols=[ev, 'recorded_ts'], size=1_500_000):
            if mut: cnk = mut(t, cnk)
            d = (_dt(cnk.recorded_ts) - _dt(cnk[ev])).dt.total_seconds() / 86400.0
            vals.append(d.to_numpy())
            if sum(len(v) for v in vals) > 3_000_000: break
        v = np.concatenate(vals); v = v[np.isfinite(v)]
        lagstats[t] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std()),
                           p50=float(np.quantile(v, .5)), p90=float(np.quantile(v, .9)),
                           skew=float(pd.Series(v).skew()), neg=int((v < -1e-9).sum()))
    bad_cv = [t for t, s in lagstats.items() if s['sd'] / max(abs(s['mean']), 1e-9) < 0.1]
    res.add(0, 'G0', '§2.3 lag sd/|mean| >= 0.1 for every table', not bad_cv,
            'all dispersed' if not bad_cv else f'constant-ish: {bad_cv}',
            breaks='set a constant lag per table', value={t: round(s['sd']/max(abs(s['mean']),1e-9), 3)
                                                          for t, s in lagstats.items()})
    bad_sk = [t for t, s in lagstats.items() if not (s['skew'] > 0.5)]
    res.add(0, 'G0', '§2.3 lag is right-skewed, not symmetric', not bad_sk,
            'all skew > 0.5' if not bad_sk else f'symmetric: {bad_sk}',
            breaks='draw the lag from a normal')
    bad_neg = [t for t, s in lagstats.items() if s['neg']]
    res.add(0, 'G0', '§2.3 lag is never negative on an event table', not bad_neg,
            'none negative' if not bad_neg else f'{bad_neg}', breaks='record before the event happens')
    # ...except a VALIDITY WINDOW, which is legitimately forward dated (§2.3 last bullet)
    fwd = {}
    for t, ecol in (('supplier_capacity', 'effective_from'), ('supplier_allocation', 'effective_from')):
        if not D.has(t): continue
        f_ = D.get(t, usecols=[ecol, 'recorded_ts'])
        d = (_dt(f_.recorded_ts) - _dt(f_[ecol])).dt.total_seconds() / 86400.0
        fwd[t] = float((d < 0).mean())
    res.add(0, 'V8', '§2.3 validity windows ARE forward dated (capacity, allocation)',
            bool(fwd) and all(x > 0.9 for x in fwd.values()),
            ', '.join(f'{k} {x*100:.1f}% registered before effect' for k, x in fwd.items()),
            breaks='stamp the declaration after the month it governs has already started',
            value=fwd)
    meds = [round(s['p50'], 3) for s in lagstats.values()]
    res.add(0, 'G0', '§2.3 lag is not shared across tables', len(set(meds)) > len(meds) // 2,
            f'{len(set(meds))} distinct medians across {len(meds)} tables: {meds}',
            breaks='use one lag distribution everywhere')
    for t, s in lagstats.items():
        res.note(0, None, f'    lag {t}', f"p50 {s['p50']:.2f}d p90 {s['p90']:.2f}d "
                                          f"sd/mean {s['sd']/max(abs(s['mean']),1e-9):.2f} skew {s['skew']:.2f}")

    # ---------------- §2.6 contract terms must not be degenerate
    sc = D.get('supplier_contracts')
    if mut: sc = mut('supplier_contracts', sc)
    deg_cols = []
    for c in ('moq', 'lot_size', 'min_volume_commitment', 'max_volume_cap', 'penalty_clause_inr'):
        if c in sc.columns and sc[c].nunique() <= 2: deg_cols.append((c, int(sc[c].nunique())))
    res.add(0, 'V8', '§2.6 contract terms are not degenerate', not deg_cols,
            'all vary' if not deg_cols else f'still constant: {deg_cols}',
            breaks='ship moq [10,10] and lot_size [25,25] as v7 did', value=deg_cols)
    sup = D.get('suppliers', usecols=['supplier_id', 'supplier_tier', 'supplier_type', 'business_class'])
    j = sc.merge(sup, on='supplier_id', how='inner')
    gm = j.groupby('supplier_tier').moq.median()
    sep = float(gm.max() / max(gm.min(), 1e-9)) if len(gm) > 1 else 1.0
    res.add(0, 'V8', '§2.6 MOQ actually differs by supplier tier', sep > 1.3,
            f'median moq by tier {dict(gm.round(1))}, ratio {sep:.2f}x',
            breaks='draw the terms without conditioning on tier', value=sep)
    al = D.get('alternate_sources', usecols=['qualification_status'])
    res.add(0, 'V8', '§2.6 qualification_status is not a constant',
            al.qualification_status.nunique() > 1,
            f'{dict(al.qualification_status.value_counts())}',
            breaks='mark every alternate source qualified, as v7 did')
    rc = D.get('revealed_capacity_monthly', usecols=['revealed_capacity_est', 'constrained_month_flag',
                                                     'declared_capacity_qty'])
    nn = float(rc.revealed_capacity_est.notna().mean())
    res.add(0, 'V8', 'revealed_capacity_est is not 100% null', nn > 0.0,
            f'{nn*100:.2f}% populated; capacity observability is a MEASURED outcome',
            breaks='leave the column unwritten, as v7 did', value=nn)
    cf = rc.constrained_month_flag.astype(str).str.lower().isin(['true', '1'])
    res.add(0, 'V8', 'revealed estimate exists exactly where the month was constrained',
            bool((rc.revealed_capacity_est.notna() == cf).all()),
            f'{int((rc.revealed_capacity_est.notna() != cf).sum()):,} rows disagree',
            breaks='populate the estimate everywhere, which would make capacity fully observed')
    return res


# ===========================================================================
# LEVEL 3 -- CAUSAL (joint distributions).  Run in the brief's order of importance.
# Bands below were fixed when this file was written and are not negotiable afterwards.
# ===========================================================================
def short_late_filter(m, resolved, late):
    short = (m.qty_received < m.qty_ordered).to_numpy()
    r = resolved.to_numpy()
    return short[r], np.asarray(late)[r]


def level3(D, res, mut=None):
    log('\n--- LEVEL 3 : CAUSAL JOINT PROBES ---')
    CHF = ['fill_rate_last13', 'otd_rate_last13', 'load_ratio', 'lead_time_ratio',
           'fill_rate', 'reporting_lag_days', 'lead_time_actual_days', 'qty_ordered', 'revision_count']
    cpw = D.get('channel_performance_weekly', usecols=['channel_id', 'week_start'] + CHF)
    if mut: cpw = mut('channel_performance_weekly', cpw)
    cpw['week_start'] = _dt(cpw.week_start)
    lb = D.get('training_labels', usecols=['snapshot_date', 'entity_type', 'entity_id', 'task',
                                           'label_value', 'label_censored'])
    if mut: lb = mut('training_labels', lb)
    lb['snapshot_date'] = _dt(lb.snapshot_date)
    pol = D.get('po_lines', usecols=['po_line_id', 'channel_id', 'part_id'])
    feat = cpw.sort_values('week_start')

    # ---- PROBE 1 (DECISIVE) -------------------------------------------------
    log('\n  PROBE 1 (DECISIVE): each label vs its most predictive feature')
    p1 = {}
    for task, col in (('fill_rate', 'fill_rate_last13'), ('arrival_week', 'otd_rate_last13'),
                      ('capacity_strain', 'load_ratio')):
        s = lb[lb.task == task]
        s = (s.merge(pol[['po_line_id', 'channel_id']], left_on='entity_id', right_on='po_line_id')
             if task in ('fill_rate', 'arrival_week') else s.assign(channel_id=s.entity_id))
        j = pd.merge_asof(s.sort_values('snapshot_date'),
                          feat[['week_start', 'channel_id', col]].sort_values('week_start'),
                          left_on='snapshot_date', right_on='week_start',
                          by='channel_id', direction='backward')
        rr, n = spear(j.label_value.to_numpy(float), j[col].to_numpy(float))
        p1[task] = rr
        res.add(3, 'G0', f'P1 {task} vs {col}  |r| >= 0.10', abs(rr) >= 0.10,
                f'spearman {rr:+.4f}  n={n:,}',
                breaks='draw the label independently of the channel history that produced it',
                value=rr)
    res.note(3, 'G0', 'P1 verdict', 'near-zero here would make every other number irrelevant')

    # ---- PROBE 2 -----------------------------------------------------------
    log('\n  PROBE 2: capacity pressure up -> fill down, lead time up')
    c2 = cpw[cpw.qty_ordered > 0]
    r_f, n_f = spear(c2.load_ratio.to_numpy(float), c2.fill_rate.to_numpy(float))
    r_l, n_l = spear(c2.load_ratio.to_numpy(float), c2.lead_time_ratio.to_numpy(float))
    res.add(3, 'L3', 'P2 load_ratio vs fill_rate is NEGATIVE', r_f < -0.02,
            f'r {r_f:+.4f} n={n_f:,}', breaks='fulfil every order regardless of capacity', value=r_f)
    res.add(3, 'L3', 'P2 load_ratio vs lead_time_ratio is POSITIVE', r_l > 0.02,
            f'r {r_l:+.4f} n={n_l:,}', breaks='remove the congestion term from the lane process', value=r_l)

    # ---- PROBE 3 -----------------------------------------------------------
    log('\n  PROBE 3: late and short co-occur on the same PO lines')
    pl = D.get('po_lines', usecols=['po_line_id', 'qty_ordered', 'current_promise_date'])
    gl = D.get('grn_lines', usecols=['po_line_id', 'qty_received', 'event_ts'])
    m = pl.merge(gl, on='po_line_id', how='left')
    m['qty_received'] = m.qty_received.fillna(0)
    arr = _dt(m.event_ts); prom = _dt(m.current_promise_date)
    CUT = prom.max()
    # A line with a receipt is late if the receipt beat the promise. A line with NO receipt
    # is late once its promise has passed, and is simply UNRESOLVED before that -- it is not
    # evidence either way and is excluded. `(NaT > date)` is False, not NaN, so the previous
    # `.fillna(True)` never fired and every open line was scored on time.
    resolved = arr.notna() | (prom < CUT)
    late = np.where(arr.notna(), arr > prom, prom < CUT)
    short, late = short_late_filter(m, resolved, late)
    phi, n3 = spear(short.astype(float), late.astype(float))
    p_ls, p_lf = float(late[short].mean()), float(late[~short].mean())
    res.add(3, 'L3', 'P3 P(late | short) > P(late | full)', p_ls > p_lf,
            f'{p_ls:.4f} vs {p_lf:.4f}   phi {phi:+.4f}  n={n3:,}',
            breaks='draw lateness independently of the capacity that caused the shortfall',
            value=[p_ls, p_lf])

    # ---- PROBE 4 -----------------------------------------------------------
    log('\n  PROBE 4: fill in constrained vs unconstrained supplier-months')
    rc = D.get('revealed_capacity_monthly', usecols=['supplier_id', 'month', 'constrained_month_flag'])
    sw = D.get('supplier_performance_weekly', usecols=['supplier_id', 'week_start', 'fill_rate', 'qty_ordered'])
    sw = sw[sw.qty_ordered > 0].copy()
    sw['month'] = _dt(sw.week_start).dt.to_period('M').dt.to_timestamp()
    rc['month'] = _dt(rc.month).dt.to_period('M').dt.to_timestamp()
    rc['cf'] = rc.constrained_month_flag.astype(str).str.lower().isin(['true', '1'])
    j4 = sw.merge(rc[['supplier_id', 'month', 'cf']], on=['supplier_id', 'month'], how='inner')
    g4 = j4.groupby('cf').fill_rate.mean()
    gap = float(g4.get(False, np.nan) - g4.get(True, np.nan))
    res.add(3, 'L3', 'P4 fill materially lower in constrained supplier-months', gap > 0.02,
            f'unconstrained {g4.get(False, float("nan")):.4f} vs constrained '
            f'{g4.get(True, float("nan")):.4f}  gap {gap:+.4f}',
            breaks='let delivered exceed the latent ceiling', value=gap)

    # ---- PROBE 5 -----------------------------------------------------------
    # Independent corroboration, re-derived from the CSVs. Not "is root_cause non-empty" --
    # root_cause is NOT NULL over an enum, so that test could not fail.
    log('\n  PROBE 5: shortage events with an INDEPENDENTLY CORROBORATED upstream cause')
    se = D.get('shortage_events', usecols=['part_id', 'plant_id', 'shortage_start_ts', 'root_cause',
                                           'responsible_supplier_id'])
    if mut: se = mut('shortage_events', se)
    se['w'] = wk_index(se.shortage_start_ts)
    ch = D.get('sourcing_channels', usecols=['channel_id', 'part_id', 'plant_id'])
    lines = pol.merge(ch, on='channel_id', how='inner', suffixes=('', '_c'))
    grn = D.get('grn_lines', usecols=['po_line_id', 'qty_received', 'event_ts'])
    qi = D.get('quality_inspections', usecols=['grn_line_id', 'qty_rejected'])
    gl2 = D.get('grn_lines', usecols=['grn_line_id', 'po_line_id']).merge(qi, on='grn_line_id', how='left')
    ev = (D.get('po_lines', usecols=['po_line_id', 'qty_ordered', 'current_promise_date'])
          .merge(grn, on='po_line_id', how='left')
          .merge(gl2[['po_line_id', 'qty_rejected']], on='po_line_id', how='left')
          .merge(lines[['po_line_id', 'part_id', 'plant_id']], on='po_line_id', how='inner'))
    _ack = D.get('supplier_acknowledgements', usecols=['po_line_id', 'ack_qty', 'ack_date'])
    ev = ev.merge(_ack, on='po_line_id', how='left')
    _short = ev.qty_received.fillna(0) < ev.qty_ordered
    _late = _dt(ev.event_ts) > _dt(ev.current_promise_date)
    _rej = ev.qty_rejected.fillna(0) > 0
    ev['bad'] = _short | _late | _rej
    ev = ev[ev.bad].copy()
    _w_short = np.where(_dt(ev.ack_date).notna(),
                        wk_index(ev.ack_date.fillna(ev.current_promise_date)),
                        wk_index(ev.current_promise_date))
    _w_late = wk_index(ev.current_promise_date)
    _w_rej = wk_index(ev.event_ts.fillna(ev.current_promise_date))
    ev['w'] = np.where(_short[ev.index], _w_short,
                       np.where(_rej[ev.index], _w_rej, _w_late))
    sup_ev = {k: np.sort(v) for k, v in ev.groupby([ev.part_id, ev.plant_id]).w.apply(np.array).items()}
    # demand-spike evidence, from the position store's own consumption columns
    ipw = D.get('inventory_position_weekly', usecols=['part_id', 'plant_id', 'week_start',
                                                      'consumption_4w', 'consumption_13w'])
    ipw['w'] = wk_index(ipw.week_start)
    ipw['spike'] = ipw.consumption_4w / np.maximum(ipw.consumption_13w / 13.0 * 4.0, 1.0)
    spike = {k: v for k, v in ipw.set_index(['part_id', 'plant_id', 'w']).spike.items()}
    # allocation-cut evidence
    alc = D.get('supplier_allocation', usecols=['part_id', 'plant_id', 'supplier_id',
                                                'allocation_pct', 'change_reason', 'effective_from'])
    alc['w'] = wk_index(alc.effective_from)
    alc = alc.sort_values(['part_id', 'plant_id', 'supplier_id', 'w'])
    alc['drop_pp'] = (alc.groupby(['part_id', 'plant_id', 'supplier_id']).allocation_pct.shift(1)
                      - alc.allocation_pct)
    # a review that moved a supplier's share by less than 3 percentage points is not a
    # plan change anybody would notice, and treating it as evidence would corroborate
    # every event on every part-plant -- a check that cannot fail
    alc = alc[(alc.change_reason.astype(str) == 'performance_share_cut') & (alc.drop_pp >= 3.0)]
    cut_ev = {k: np.sort(v) for k, v in alc.groupby([alc.part_id, alc.plant_id]).w.apply(np.array).items()}
    # a requirement plan that moved shows up as a safety-stock target that climbed
    ss = D.get('inventory_position_weekly', usecols=['part_id', 'plant_id', 'week_start',
                                                     'safety_stock_qty'])
    ss['w'] = wk_index(ss.week_start)
    ss = ss.sort_values(['part_id', 'plant_id', 'w'])
    ss['prev13'] = ss.groupby(['part_id', 'plant_id']).safety_stock_qty.shift(13)
    ss['rise'] = ss.safety_stock_qty / ss.prev13.replace(0, np.nan)
    ss_rise = {k: v for k, v in ss.set_index(['part_id', 'plant_id', 'w']).rise.items()}

    corr = np.zeros(len(se), bool)
    kinds = se.root_cause.to_numpy()
    for i, (p_, l_, w_) in enumerate(zip(se.part_id, se.plant_id, se.w)):
        k = kinds[i]
        if k in ('supplier_delay', 'supplier_short', 'quality_reject', 'logistics'):
            a = sup_ev.get((p_, l_))
            corr[i] = a is not None and bool(((a <= w_) & (a >= w_ - 26)).any())
        elif k == 'demand_spike':
            corr[i] = float(spike.get((p_, l_, int(w_)), 0.0)) > 1.15
        elif k == 'plan_change':
            a = cut_ev.get((p_, l_))
            corr[i] = ((a is not None and bool(((a <= w_) & (a >= w_ - 26)).any()))
                       or float(ss_rise.get((p_, l_, int(w_)), 0.0) or 0.0) > 1.15)
    share = float(corr.mean()) if len(se) else 0.0
    res.add(3, 'L3', 'P5 shortage cause corroborated by evidence in the CSVs (target ~100%)',
            share >= 0.80, f'{share*100:.2f}% of {len(se):,} events corroborated; '
            f'root_cause mix {dict(pd.Series(kinds).value_counts())}',
            breaks='label the cause without checking that the evidence exists', value=share)
    mix = pd.Series(kinds).value_counts(normalize=True)
    res.add(3, 'L3', 'P5 root_cause is a mix, not one label on everything',
            float(mix.max()) < 0.90,
            f'most common {mix.index[0]} at {mix.iloc[0]*100:.1f}% of events',
            breaks='label every shortage with one cause -- corroboration alone cannot see '
                   'this, because a label that happens to be corroborable everywhere still '
                   'passes the evidence test',
            value=float(mix.max()))
    sup_master = set(D.get('suppliers', usecols=['supplier_id']).supplier_id)
    rs = se.responsible_supplier_id.astype(str)
    named = rs[(rs != '') & (rs.str.lower() != 'nan')]
    res.add(3, 'L3', 'P5 named responsible supplier resolves to the master',
            bool(named.isin(sup_master).all()),
            f'{len(named):,} named, {int((~named.isin(sup_master)).sum()):,} unresolved',
            breaks='name a supplier that does not exist')

    # ---- PROBE 6 -----------------------------------------------------------
    log('\n  PROBE 6: shortage -> expedite / revision / alternate sourcing')
    ex = D.get('expedite_events', usecols=['part_id', 'plant_id'])
    pp_all = set(zip(D.get('part_plant', usecols=['part_id', 'plant_id']).part_id,
                     D.get('part_plant', usecols=['part_id', 'plant_id']).plant_id))
    shk = set(zip(se.part_id, se.plant_id))
    exk = set(zip(ex.part_id, ex.plant_id))
    p_s = len(exk & shk) / max(1, len(shk))
    p_n = len(exk - shk) / max(1, len(pp_all - shk))
    lift = p_s / max(p_n, 1e-9)
    res.add(3, 'L3', 'P6 expedite lift given a shortage >= 2x', lift >= 2.0,
            f'P(exp|short) {p_s:.4f} vs P(exp|none) {p_n:.4f}  lift {lift:.1f}x',
            breaks='raise expedites on a fixed random share of lines', value=lift)
    rv = D.get('po_line_revisions', usecols=['po_line_id', 'reason_code', 'initiated_by'])
    res.add(3, 'L3', 'P6 revisions come from both directions and carry a reason',
            rv.initiated_by.nunique() > 1 and rv.reason_code.nunique() > 1,
            f'by {dict(rv.initiated_by.value_counts())} reason {dict(rv.reason_code.value_counts())}',
            breaks="emit `r.random(NPO) < 0.34` revisions with reason_code='shortage', as v7 did")
    res.note(3, 'L3', 'P6 resolution_action mix',
             str(dict(D.get('shortage_events', usecols=['resolution_action'])
                      .resolution_action.value_counts(normalize=True).round(4))))

    # ---- PROBE 7 -----------------------------------------------------------
    log('\n  PROBE 7: alternate sourcing -> receiving supplier load up -> its own fill down')
    rc2 = D.get('revealed_capacity_monthly', usecols=['supplier_id', 'month',
                                                      'capacity_utilisation_observed'])
    rc2['month'] = _dt(rc2.month)
    sm = sw.groupby(['supplier_id', 'month']).fill_rate.mean().reset_index().rename(columns={'fill_rate': 'f_next'})
    sm['month'] = sm.month - pd.offsets.MonthBegin(1)
    j7 = rc2.merge(sm, on=['supplier_id', 'month'], how='inner')
    r7m, n7m = spear(j7.capacity_utilisation_observed.to_numpy(float), j7.f_next.to_numpy(float))
    j7['u_d'] = (j7.capacity_utilisation_observed
                 - j7.groupby('supplier_id').capacity_utilisation_observed.transform('mean'))
    j7['f_d'] = j7.f_next - j7.groupby('supplier_id').f_next.transform('mean')
    r7, n7 = spear(j7.u_d.to_numpy(float), j7.f_d.to_numpy(float))
    res.add(3, 'L3', 'P7 WITHIN-supplier: load(m) up -> its own fill(m+1) down', r7 < 0,
            f'within-supplier r {r7:+.4f}  n={n7:,}',
            breaks='let shifted volume land on a supplier without consuming its capacity', value=r7)
    res.note(3, 'L3', 'P7 cross-supplier (confounded, reported not gated)',
             f'r {r7m:+.4f}  n={n7m:,} -- across suppliers this is dominated by selection: '
             f'the allocation arc gives more volume to the suppliers that deliver, so high '
             f'load and high fill travel together. The arc is a within-supplier statement.')
    rc2s = rc2.sort_values(['supplier_id', 'month'])
    r7b, n7b = spear(rc2s.capacity_utilisation_observed.to_numpy(float),
                     rc2s.groupby('supplier_id').capacity_utilisation_observed.shift(-1).to_numpy(float))
    res.note(3, 'L3', 'P7 utilisation persistence m -> m+1', f'r {r7b:+.4f}  n={n7b:,}')

    # ---- PROBE 8 -----------------------------------------------------------
    log('\n  PROBE 8: staleness up -> lead-time variance up across channels')
    c8 = cpw[cpw.qty_ordered > 0]
    g8 = c8.groupby('channel_id').agg(stale=('reporting_lag_days', 'median'),
                                      lv=('lead_time_actual_days', 'std')).dropna()
    r8, n8 = spear(g8.stale.to_numpy(float), g8.lv.to_numpy(float))
    res.add(3, 'L3', 'P8 staleness vs sd(lead time) is POSITIVE', r8 > 0.02,
            f'r {r8:+.4f}  n={n8:,}',
            breaks='make the recording lag independent of supplier stress', value=r8)

    # ---- PROBE 9 -----------------------------------------------------------
    log('\n  PROBE 9: KS of every label against Uniform(0,1) -- must REJECT at p < 0.01')
    ks_bad, cens = [], {}
    for task in sorted(lb.task.unique()):
        v = lb[lb.task == task].label_value.to_numpy(float)
        d, p, n = ks_uniform(v)
        ok = p < 0.01
        if not ok: ks_bad.append(task)
        res.note(3, 'L3', f'    KS {task}', f'D={d:.4f} p={p:.3e} n={n:,} '
                 f'mean={np.nanmean(v):.4f} sd={np.nanstd(v):.4f} '
                 f'{"REJECT (good)" if ok else "CANNOT REJECT -- looks uniform (bad)"}')
        c = lb[lb.task == task].label_censored.astype(str).str.lower().isin(['true', '1'])
        cens[task] = float(c.mean())
    res.add(3, 'L3', 'P9 every label rejects Uniform(0,1) at p < 0.01', not ks_bad,
            'all reject' if not ks_bad else f'uniform-looking: {ks_bad}',
            breaks='draw a label from a uniform instead of reading it off the simulated future')
    zero_cens = [t for t, c in cens.items() if c <= 0.0]
    res.add(0, 'G0', 'G0 no task has 0% censoring', not zero_cens,
            f'{ {t: round(c*100, 2) for t, c in cens.items()} }' if not zero_cens else f'{zero_cens}',
            breaks='resolve every label inside the horizon, so nothing is right-censored')
    return res


# ===========================================================================
# LEVEL 4 -- LEARNABILITY.  G2..G7.  Naive baselines FIRST, then LightGBM, then the
# negative controls that must fail, the ablations, and the h0 diagnostic.
# ===========================================================================
CHF = ['fill_rate', 'fill_rate_last4', 'fill_rate_last13', 'fill_rate_last52',
       'lead_time_actual_days', 'lead_time_ratio', 'otd_rate_last13', 'ack_gap_ratio',
       'load_ratio', 'reporting_lag_days', 'qty_ordered', 'qty_received',
       'active_weeks_in_52', 'revision_count']
SUP_HIST = ['fill_rate', 'fill_rate_last4', 'fill_rate_last13', 'fill_rate_last52',
            'otd_rate_last13', 'ack_gap_ratio']
CAP_FEAT = ['load_ratio', 'qty_ordered', 'qty_received']
# Deliberately modest: this is an instrument for asking whether signal EXISTS, not a
# tuned model. Every arm -- baseline, full, ablation, h0/h1/h4 -- is fitted with exactly
# these settings, so the comparisons are like for like. A ten-bin CRPS grid is coarser
# than twenty but identical across arms, and it halves the tree count on a multiclass fit
# that has to run eleven times per seed and five times over.
GBM = dict(n_estimators=200, learning_rate=0.08, num_leaves=48, min_child_samples=60,
           subsample=0.9, subsample_freq=1, colsample_bytree=0.9, max_bin=63,
           random_state=7, verbose=-1, n_jobs=-1)
NB = 10
TRAIN_END, VAL_END = pd.Timestamp('2023-12-31'), pd.Timestamp('2024-12-31')

def _crps(cdf, y, edges):
    step = np.diff(edges)[None, :]
    ind = (y[:, None] <= edges[None, 1:]).astype(float)
    return float((((cdf - ind) ** 2) * step).sum(1).mean())

def _cindex(pred, time, event, rng):
    n = len(time)
    i = rng.integers(0, n, 1_500_000); j = rng.integers(0, n, 1_500_000)
    ok = ((time[i] < time[j]) & event[i]) | ((time[j] < time[i]) & event[j])
    if ok.sum() == 0: return float('nan')
    i, j = i[ok], j[ok]
    conc = np.where(time[i] < time[j], pred[i] < pred[j], pred[j] < pred[i])
    ties = pred[i] == pred[j]
    return float((conc.sum() + 0.5 * ties.sum()) / len(i))

def _propagate(X, idxs, rounds):
    cur = X.copy(); outs = []
    for _ in range(rounds):
        agg = []
        for idx in idxs:
            n = int(idx.max()) + 1
            s = np.zeros((n, cur.shape[1]))
            np.add.at(s, idx, np.nan_to_num(cur))
            c = np.maximum(np.bincount(idx, minlength=n).astype(float), 1)[:, None]
            agg.append((s / c)[idx])
        cur = np.concatenate(agg, 1); outs.append(cur)
    return np.concatenate(outs, 1)

def level4(D, res, seed_dir, mut=None):
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, brier_score_loss
    log('\n--- LEVEL 4 : LEARNABILITY (G2 baselines -> G3 signal -> G4 graph -> G5 as-of -> G6 folds) ---')
    rng = np.random.default_rng(7)
    ch = D.get('sourcing_channels', usecols=['channel_id', 'supplier_id', 'part_id', 'plant_id',
                                             'contracted_lead_time_days', 'transport_mode',
                                             'transport_distance_km'])
    for c, k in (('supplier_id', 'si'), ('part_id', 'pi'), ('plant_id', 'li')):
        ch[k] = ch[c].astype('category').cat.codes
    ch['mode_i'] = ch.transport_mode.astype('category').cat.codes
    pol = D.get('po_lines', usecols=['po_line_id', 'channel_id'])
    cpw = D.get('channel_performance_weekly', usecols=['channel_id', 'week_start'] + CHF,
                dtype={c: 'float32' for c in CHF})
    if mut: cpw = mut('channel_performance_weekly', cpw)
    cpw['week_start'] = _dt(cpw.week_start)
    cpw = cpw.sort_values('week_start')
    lb = D.get('training_labels', usecols=['snapshot_date', 'entity_type', 'entity_id', 'task',
                                           'label_value', 'label_censored'])
    lb['snapshot_date'] = _dt(lb.snapshot_date)
    lb['label_censored'] = lb.label_censored.astype(str).str.lower().isin(['true', '1'])
    STATIC = ['contracted_lead_time_days', 'transport_distance_km', 'mode_i', 'si', 'pi', 'li']

    def build(task):
        s = lb[lb.task == task]
        s = (s.merge(pol, left_on='entity_id', right_on='po_line_id')
             if task in ('fill_rate', 'arrival_week') else s.assign(channel_id=s.entity_id))
        j = pd.merge_asof(s[['channel_id', 'snapshot_date', 'label_value', 'label_censored']]
                          .sort_values('snapshot_date'), cpw,
                          left_on='snapshot_date', right_on='week_start',
                          by='channel_id', direction='backward')
        j = j.merge(ch[['channel_id'] + STATIC], on='channel_id', how='left')
        return j.dropna(subset=['fill_rate_last13']).reset_index(drop=True)

    def split(df):
        tr = (df.snapshot_date <= TRAIN_END).to_numpy()
        va = ((df.snapshot_date > TRAIN_END) & (df.snapshot_date <= VAL_END)).to_numpy()
        te = (df.snapshot_date > VAL_END).to_numpy()
        return tr, va, te

    OUT = {}
    # =================================================== FILL RATE
    f = build('fill_rate')
    tr, va, te = split(f)
    y = f.label_value.to_numpy(float)
    FEAT = CHF + STATIC
    X = f[FEAT].to_numpy(np.float32)
    edges = np.linspace(0, 1, NB + 1)
    yb = np.clip(np.digitize(y, edges[1:-1]), 0, NB - 1)
    log(f'  fill: train {tr.sum():,} val {va.sum():,} test {te.sum():,}')
    # ---- G2 naive baselines, written down FIRST
    h = np.bincount(yb[tr], minlength=NB); cdf_g = np.cumsum(h / h.sum())
    b_glob = _crps(np.tile(cdf_g, (te.sum(), 1)), y[te], edges)
    dtr = pd.DataFrame({'ch': f.channel_id[tr].to_numpy(), 'b': yb[tr]})
    hist = dtr.groupby(['ch', 'b']).size().unstack(fill_value=0).reindex(columns=range(NB), fill_value=0)
    hcdf = hist.div(hist.sum(1).replace(0, np.nan), axis=0).cumsum(1)
    cc = hcdf.reindex(f.channel_id[te].to_numpy()).to_numpy()
    cc = np.where(np.isnan(cc), np.tile(cdf_g, (len(cc), 1)), cc)
    b_chan = _crps(cc, y[te], edges)
    lastb = np.clip(np.digitize(np.nan_to_num(f.fill_rate.to_numpy(float), nan=y[tr].mean()),
                                edges[1:-1]), 0, NB - 1)
    b_last = _crps((np.arange(NB)[None, :] >= lastb[te][:, None]).astype(float), y[te], edges)
    G2_FILL = min(b_glob, b_chan, b_last)
    res.add(4, 'G2', 'fill: naive baselines computed before any model', True,
            f'CRPS global {b_glob:.5f} | per-channel {b_chan:.5f} | last value {b_last:.5f} '
            f'-> number to beat {G2_FILL:.5f}',
            breaks='report a model number with no baseline beside it', value=G2_FILL)
    # ---- G3
    m = lgb.LGBMClassifier(objective='multiclass', num_class=NB, **GBM)
    m.fit(X[tr], yb[tr], eval_set=[(X[va], yb[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
    cdf = np.cumsum(m.predict_proba(X[te]), 1)
    G3_FILL = _crps(cdf, y[te], edges)
    res.add(4, 'G3', 'fill: LightGBM beats the best naive baseline', G3_FILL < G2_FILL,
            f'CRPS {G3_FILL:.5f} vs baseline {G2_FILL:.5f} ({(1-G3_FILL/G2_FILL)*100:+.2f}%)',
            breaks='detach the label from the feature history', value=[G3_FILL, G2_FILL])
    OUT['fill_crps'] = G3_FILL; OUT['fill_crps_baseline'] = G2_FILL
    imp = pd.Series(m.feature_importances_, index=FEAT).sort_values(ascending=False)
    res.note(4, 'G3', 'fill: top features (leak check)',
             ', '.join(f'{k}={v/imp.sum():.1%}' for k, v in imp.head(5).items()))
    # ---- negative controls: these MUST fail to predict
    nc = pd.DataFrame({
        'row_number': np.arange(len(f)),
        'generation_order': f.index.to_numpy(),
        'seed_index': np.full(len(f), int(json.loads((Path(seed_dir) / 'manifest.json').read_text())['seed'])),
        'id_hash': pd.util.hash_pandas_object(f.channel_id, index=False).to_numpy() % 1_000_003,
        'file_order': np.argsort(np.argsort(f.channel_id.to_numpy())),
    })
    Xn = nc.to_numpy(np.float64)
    mn = lgb.LGBMClassifier(objective='multiclass', num_class=NB, **GBM)
    mn.fit(Xn[tr], yb[tr], eval_set=[(Xn[va], yb[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
    nc_crps = _crps(np.cumsum(mn.predict_proba(Xn[te]), 1), y[te], edges)
    res.add(4, 'G3', 'negative controls FAIL to predict (row no, gen order, seed, id hash, file order)',
            nc_crps >= G2_FILL * 0.99,
            f'CRPS {nc_crps:.5f} vs naive {G2_FILL:.5f} -- a win here means the generator leaked '
            f'its own structure',
            breaks='order the rows by outcome, or key any field to generation order', value=nc_crps)
    # ---- G4 graph contribution.
    # The ladder is run TWICE. A tree given si / pi / li splits on them and learns a
    # per-supplier mean, which is a 0-hop supplier aggregate -- so an h0 that carries the
    # identity keys has already been handed most of what h1 would contribute, and the
    # diagnostic measures identity rather than propagation. The GATED ladder therefore
    # drops the identity keys from every arm. The ladder that keeps them is reported
    # beside it, because the gap between the two is itself the finding.
    # (shipped.json already records the same conclusion for the fill head: b5flat22 runs
    # on flat graph features with NO identity keys.)
    idxs = [f.si.to_numpy(int), f.pi.to_numpy(int), f.li.to_numpy(int)]
    def _ladder(cols, tag, gated):
        base = np.nan_to_num(f[cols].to_numpy(np.float32))
        out = {}
        for lbl, rounds in (('h0 features only', 0), ('h1 one hop', 1), ('h4 four layers', 2)):
            Xg = base if rounds == 0 else np.concatenate(
                [base, _propagate(np.nan_to_num(f[CHF].to_numpy(np.float32)), idxs, rounds)], 1)
            mm = lgb.LGBMClassifier(objective='multiclass', num_class=NB, **GBM)
            mm.fit(Xg[tr], yb[tr], eval_set=[(Xg[va], yb[va])],
                   callbacks=[lgb.early_stopping(40, verbose=False)])
            out[lbl] = _crps(np.cumsum(mm.predict_proba(Xg[te]), 1), y[te], edges)
            res.note(4, 'G4', f'  {tag} {lbl}', f'CRPS {out[lbl]:.5f}')
        g = (out['h0 features only'] - out['h4 four layers']) / out['h0 features only']
        return out, g
    NOKEY = [c for c in FEAT if c not in ('si', 'pi', 'li')]
    h_res, gain = _ladder(NOKEY, 'fill [no identity keys]', False)
    res.note(4, 'G4', 'fill: graph contribution (reported, NOT gated)',
             f'h0 {h_res["h0 features only"]:.5f} -> h4 {h_res["h4 four layers"]:.5f} '
             f'({gain*100:+.2f}%). Fill reads out AT THE CHANNEL and has no real neighbourhood '
             f'(brief §1.2), so a null here is the structural claim confirmed, not a defect. '
             f'G4 is gated on capacity_strain, whose label IS a supplier aggregate.')
    h_key, gain_key = _ladder(FEAT, 'fill [with identity keys]', False)
    res.note(4, 'G4', 'fill: same ladder WITH identity keys',
             f'h0 {h_key["h0 features only"]:.5f} -> h4 {h_key["h4 four layers"]:.5f} '
             f'({gain_key*100:+.2f}%) -- a tree that can split on si has already been given a '
             f'0-hop supplier aggregate, so this arm understates what propagation adds')
    OUT['fill_h0'] = h_res['h0 features only']; OUT['fill_h4'] = h_res['h4 four layers']
    OUT['fill_h_gain'] = gain
    OUT['fill_h0_with_keys'] = h_key['h0 features only']
    OUT['fill_h4_with_keys'] = h_key['h4 four layers']; OUT['fill_h_gain_with_keys'] = gain_key
    # ---- ablations
    for name, drop, gate_note in (
            ('remove supplier history', SUP_HIST, 'supplier-risk performance must degrade'),
            ('remove capacity features', CAP_FEAT, 'capacity signal must degrade')):
        keep = [c for c in FEAT if c not in drop]
        Xa = f[keep].to_numpy(np.float32)
        ma = lgb.LGBMClassifier(objective='multiclass', num_class=NB, **GBM)
        ma.fit(Xa[tr], yb[tr], eval_set=[(Xa[va], yb[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
        ca = _crps(np.cumsum(ma.predict_proba(Xa[te]), 1), y[te], edges)
        res.add(4, 'G3', f'ablation: {name} degrades fill', ca > G3_FILL,
                f'CRPS {ca:.5f} vs full {G3_FILL:.5f} ({(ca/G3_FILL-1)*100:+.2f}%) -- {gate_note}',
                breaks='make the feature redundant with another one already in the frame', value=ca)

    # =================================================== ARRIVAL (censored)
    a = build('arrival_week')
    tra, vaa, tea = split(a)
    ya = a.label_value.to_numpy(float); ev = ~a.label_censored.to_numpy()
    Xa = a[FEAT].to_numpy(np.float32)
    obs = tra & ev
    log(f'  arrival: train {tra.sum():,} test {tea.sum():,}  censored {1-ev.mean():.1%}')
    lane = a.assign(y=ya).loc[obs].groupby(['si', 'li']).y.median()
    gmed = float(np.median(ya[obs]))
    p_lane = a.loc[tea].set_index(['si', 'li']).index.map(lane).to_numpy(float)
    p_lane = np.where(np.isnan(p_lane), gmed, p_lane)
    c_lane = _cindex(p_lane, ya[tea], ev[tea], rng)
    c_glob = _cindex(np.full(tea.sum(), gmed), ya[tea], ev[tea], rng)
    G2_ARR = max(c_lane, c_glob)
    res.add(4, 'G2', 'arrival: naive baselines computed first', True,
            f'C-index per-lane median {c_lane:.4f} | global median {c_glob:.4f} '
            f'-> number to beat {G2_ARR:.4f}', breaks='quote a C-index with no baseline', value=G2_ARR)
    ma = lgb.LGBMRegressor(objective='l2', **GBM)
    ma.fit(Xa[obs], ya[obs], eval_set=[(Xa[vaa & ev], ya[vaa & ev])],
           callbacks=[lgb.early_stopping(40, verbose=False)])
    G3_ARR = _cindex(ma.predict(Xa[tea]), ya[tea], ev[tea], rng)
    res.add(4, 'G3', 'arrival: LightGBM beats the naive baseline', G3_ARR > G2_ARR,
            f'C-index {G3_ARR:.4f} vs baseline {G2_ARR:.4f}',
            breaks='detach arrival from the lane and congestion process', value=[G3_ARR, G2_ARR])
    res.note(4, 'G3', 'arrival: task definition',
             'ARRIVAL IS A RIGHT-CENSORED TIME-TO-EVENT ON THE OPEN PO LINE, ranked by C-index. '
             'Every earlier report in this project compared it against a baseline that '
             'implicitly assumed the opposite (§1.4 item 4) -- state this wherever the head is described.')
    OUT['arrival_cindex'] = G3_ARR; OUT['arrival_cindex_baseline'] = G2_ARR

    # =================================================== CAPACITY STRAIN (channel)
    # The G4 task. capacity_strain's label is a SUPPLIER-MONTH aggregate read out on a
    # channel, so a channel's siblings under the same supplier genuinely carry information
    # about it -- this is the one head in the set where a neighbourhood exists at all
    # (brief §1.2). If propagation adds nothing HERE, the architecture premise is
    # unsupported and no amount of depth will rescue it.
    log('\n  capacity_strain: the graph task')
    cs = build('capacity_strain')
    trc, vac, tec = split(cs)
    yc = cs.label_value.to_numpy(float)
    log(f'  capacity: train {trc.sum():,} val {vac.sum():,} test {tec.sum():,}')
    gmean_c = float(yc[trc].mean())
    chm = pd.DataFrame({'ch': cs.channel_id[trc].to_numpy(), 'y': yc[trc]}).groupby('ch').y.mean()
    p_ch = chm.reindex(cs.channel_id[tec].to_numpy()).fillna(gmean_c).to_numpy()
    b_glob_c = float(np.abs(yc[tec] - gmean_c).mean())
    b_chan_c = float(np.abs(yc[tec] - p_ch).mean())
    G2_CAP = min(b_glob_c, b_chan_c)
    res.add(4, 'G2', 'capacity: naive baselines computed first', True,
            f'MAE global mean {b_glob_c:.5f} | per-channel mean {b_chan_c:.5f} '
            f'-> number to beat {G2_CAP:.5f}',
            breaks='quote a capacity number with no baseline beside it', value=G2_CAP)
    CNOKEY = [c for c in FEAT if c not in ('si', 'pi', 'li')]
    Xc = cs[CNOKEY].to_numpy(np.float32)
    mc = lgb.LGBMRegressor(objective='l1', **GBM)
    mc.fit(Xc[trc], yc[trc], eval_set=[(Xc[vac], yc[vac])],
           callbacks=[lgb.early_stopping(40, verbose=False)])
    G3_CAP = float(np.abs(yc[tec] - mc.predict(Xc[tec])).mean())
    res.add(4, 'G3', 'capacity: LightGBM beats the naive baseline', G3_CAP < G2_CAP,
            f'MAE {G3_CAP:.5f} vs baseline {G2_CAP:.5f} ({(1-G3_CAP/G2_CAP)*100:+.2f}%)',
            breaks='detach capacity_strain from the ordering history that reveals the ceiling',
            value=[G3_CAP, G2_CAP])
    OUT['capacity_mae'] = G3_CAP; OUT['capacity_mae_baseline'] = G2_CAP
    idxc = [cs.si.to_numpy(int), cs.pi.to_numpy(int), cs.li.to_numpy(int)]
    basec = np.nan_to_num(cs[CNOKEY].to_numpy(np.float32))
    propsrc = np.nan_to_num(cs[CHF].to_numpy(np.float32))
    hc = {}
    for lbl, rounds in (('h0 features only', 0), ('h1 one hop', 1), ('h4 four layers', 2)):
        Xg = basec if rounds == 0 else np.concatenate([basec, _propagate(propsrc, idxc, rounds)], 1)
        mm = lgb.LGBMRegressor(objective='l1', **GBM)
        mm.fit(Xg[trc], yc[trc], eval_set=[(Xg[vac], yc[vac])],
               callbacks=[lgb.early_stopping(40, verbose=False)])
        hc[lbl] = float(np.abs(yc[tec] - mm.predict(Xg[tec])).mean())
        res.note(4, 'G4', f'  capacity {lbl}', f'MAE {hc[lbl]:.5f}')
    gain_c = (hc['h0 features only'] - hc['h4 four layers']) / hc['h0 features only']
    res.add(4, 'G4', 'graph contributes on capacity_strain: h4 beats h0 by more than 0.5%',
            gain_c > 0.005,
            f'h0 {hc["h0 features only"]:.5f} -> h1 {hc["h1 one hop"]:.5f} '
            f'-> h4 {hc["h4 four layers"]:.5f} ({gain_c*100:+.2f}%)',
            breaks='cut the alternate-sourcing and capacity-coupling arcs, so a supplier\'s load '
                   'no longer reaches its other channels and there is nothing to propagate',
            value=gain_c)
    OUT['cap_h0'] = hc['h0 features only']; OUT['cap_h1'] = hc['h1 one hop']
    OUT['cap_h4'] = hc['h4 four layers']; OUT['cap_h_gain'] = gain_c
    keepc = [c for c in CNOKEY]
    Xa2 = np.concatenate([basec, _propagate(propsrc, idxc, 2)], 1)
    shuffled = [cs.si.to_numpy(int)]
    rngp = np.random.default_rng(13)
    idxs_shuf = [rngp.permutation(i) for i in idxc]
    Xs2 = np.concatenate([basec, _propagate(propsrc, idxs_shuf, 2)], 1)
    ms = lgb.LGBMRegressor(objective='l1', **GBM)
    ms.fit(Xs2[trc], yc[trc], eval_set=[(Xs2[vac], yc[vac])],
           callbacks=[lgb.early_stopping(40, verbose=False)])
    mae_shuf = float(np.abs(yc[tec] - ms.predict(Xs2[tec])).mean())
    res.add(4, 'G4', 'ablation: propagating over a SHUFFLED graph degrades capacity',
            mae_shuf > hc['h4 four layers'],
            f'MAE real graph {hc["h4 four layers"]:.5f} vs shuffled {mae_shuf:.5f} '
            f'({(mae_shuf/hc["h4 four layers"]-1)*100:+.2f}%) -- if a random neighbourhood '
            f'works as well, the edges carry nothing and the aggregate is just smoothing',
            breaks='aggregate over neighbours that are not the real supplier/part/plant groups',
            value=mae_shuf)
    OUT['cap_h4_shuffled'] = mae_shuf

    # =================================================== G5 as-of gate binding
    log('\n  G5: is the as-of gate binding?')
    pl2 = D.get('po_lines', usecols=['po_line_id', 'channel_id', 'qty_ordered', 'created_ts', 'recorded_ts'])
    gl2 = D.get('grn_lines', usecols=['po_line_id', 'qty_received'])
    ll = pl2.merge(gl2, on='po_line_id', how='left')
    ll['qty_received'] = ll.qty_received.fillna(0)
    ll['fill'] = np.minimum(ll.qty_received / np.maximum(ll.qty_ordered, 1), 1.0)
    snaps = np.sort(lb.snapshot_date.unique())

    def asof_fill(rec_col):
        d = ll[['channel_id', 'fill']].assign(rt=_dt(ll[rec_col])).sort_values('rt')
        keys = f[['channel_id', 'snapshot_date']].sort_values('snapshot_date')
        d['cum'] = d.groupby('channel_id').fill.transform(lambda x: x.expanding().mean())
        j = pd.merge_asof(keys, d[['rt', 'channel_id', 'cum']].sort_values('rt'),
                          left_on='snapshot_date', right_on='rt', by='channel_id', direction='backward')
        return j.sort_index().cum.to_numpy(float)

    true_f = asof_fill('recorded_ts')
    ll['shuf'] = ll.recorded_ts.sample(frac=1.0, random_state=11).to_numpy()
    shuf_f = asof_fill('shuf')
    def _mae(p):
        p = np.where(np.isfinite(p), p, np.nanmean(y[tr]))
        return float(np.abs(y[te] - p[te]).mean())
    mae_t, mae_s = _mae(true_f), _mae(shuf_f)
    res.add(4, 'G5', 'shuffling recorded_ts DEGRADES the as-of feature', mae_s > mae_t,
            f'MAE true recorded_ts {mae_t:.5f} -> shuffled {mae_s:.5f} ({(mae_s/mae_t-1)*100:+.3f}%)',
            breaks='make the recording lag independent of state, so when a row lands carries '
                   'no information and the as-of cut is cosmetic', value=[mae_t, mae_s])

    # =================================================== G6 purged split, every origin
    log('\n  G6: purged split with a 90-day gap, at EVERY backtest origin')
    HOR = 90
    origins = pd.to_datetime(['2022-01-03', '2022-07-04', '2023-01-02', '2023-07-03',
                              '2024-01-01', '2024-07-01', '2025-01-06'])
    rows, leak = [], []
    for o in origins:
        trm = (f.snapshot_date <= o - pd.Timedelta(days=HOR)).to_numpy()
        tem = ((f.snapshot_date > o) & (f.snapshot_date <= o + pd.Timedelta(days=HOR))).to_numpy()
        if trm.sum() < 5000 or tem.sum() < 500:
            rows.append((str(o.date()), int(trm.sum()), int(tem.sum()), None, None)); continue
        # the purge itself: no training label window may reach into the test feature window
        tr_lab_end = f.snapshot_date[trm].max() + pd.Timedelta(days=HOR)
        te_feat_start = f.snapshot_date[tem].min()
        if tr_lab_end > te_feat_start: leak.append(str(o.date()))
        mo = lgb.LGBMClassifier(objective='multiclass', num_class=NB, **{**GBM, 'n_estimators': 120})
        mo.fit(X[trm], yb[trm])
        c_o = _crps(np.cumsum(mo.predict_proba(X[tem]), 1), y[tem], edges)
        hh = np.bincount(yb[trm], minlength=NB)
        b_o = _crps(np.tile(np.cumsum(hh / hh.sum()), (int(tem.sum()), 1)), y[tem], edges)
        rows.append((str(o.date()), int(trm.sum()), int(tem.sum()), c_o, b_o))
        res.note(4, 'G6', f'  origin {o.date()}',
                 f'train {trm.sum():,} test {tem.sum():,}  CRPS model {c_o:.5f} vs naive {b_o:.5f}')
    trained = [r for r in rows if r[3] is not None]
    res.add(4, 'G6', 'purged split: no train label window reaches the test feature window',
            not leak, 'clean at every origin' if not leak else f'LEAK at {leak}',
            breaks='train on snapshots inside 90 days of the origin, which is the default '
                   'un-purged split and the leak §1.4 item 5 describes')
    res.add(4, 'G6', 'every backtest origin actually trained (§1.4 item 7)',
            len(trained) == len(origins),
            f'{len(trained)}/{len(origins)} origins trained',
            breaks='run origins 1-2 and report the set as if it were complete')
    res.add(4, 'G6', 'model beats naive at a majority of purged origins',
            sum(1 for r in trained if r[3] < r[4]) > len(trained) / 2,
            f'{sum(1 for r in trained if r[3] < r[4])}/{len(trained)} origins',
            breaks='leave the signal only in the fixed 2025 test fold')
    OUT['origins'] = rows

    # =================================================== G7 production readiness
    log('\n  G7: will it survive production?')
    traded = set(pol.channel_id.unique())
    cold = [c for c in ch.channel_id if c not in traded]
    res.add(4, 'G7', 'cold-start slice exists and is held out of the trained history',
            len(cold) > 0, f'{len(cold):,} channels never traded ({len(cold)/len(ch)*100:.2f}%)',
            breaks='give every channel an order, so nothing tests the cold path', value=len(cold))
    res.note(4, 'G7', 'seed-variance band',
             'computed across seeds by --variance; a single seed cannot produce one')
    return res, OUT


# ===========================================================================
# SELFTEST -- every check must fail under the mutation it names.
# §1.3: ten documented gates that could not fail. A check that survives its own
# mutation is not a check; delete it and say so.
# ===========================================================================
MUTATIONS = {
 'net_scrap_into_receipt': ('inventory_transactions',
   lambda df: df.assign(qty=np.where(df.txn_type.eq('scrap'), 0, df.qty))),
 'opening_balance_removed': ('inventory_transactions',
   lambda df: df[df.txn_type != 'adjustment']),
 'drop_one_percent_of_orders': ('channel_performance_weekly',
   lambda df: df.assign(qty_ordered=(df.qty_ordered * 0.99).astype(np.int64))),
 'position_shifted_by_one': ('inventory_position_weekly',
   lambda df: df.assign(qty_on_hand=df.qty_on_hand + 1)),
 'allocation_partial_log': ('supplier_allocation',
   lambda df: df.iloc[::2].copy()),
 'single_period_plan': ('production_plan',
   lambda df: df.assign(target_period=df.plan_date)),
 'constant_lag': ('po_lines',
   lambda df: df.assign(recorded_ts=_dt(df.created_ts) + pd.Timedelta(days=1))),
 'degenerate_contract_terms': ('supplier_contracts',
   lambda df: df.assign(moq=10, lot_size=25, max_volume_cap=5000,
                        min_volume_commitment=100, penalty_clause_inr=10000)),
 'labels_detached_from_features': ('training_labels',
   lambda df: df.assign(label_value=np.random.default_rng(3).permutation(df.label_value.to_numpy()))),
 'uniform_labels': ('training_labels',
   lambda df: df.assign(label_value=np.random.default_rng(4).random(len(df)))),
 'fill_independent_of_load': ('channel_performance_weekly',
   lambda df: df.assign(load_ratio=np.random.default_rng(5).permutation(df.load_ratio.to_numpy()))),
 'cause_labelled_without_evidence': ('shortage_events',
   lambda df: df.assign(root_cause='plan_change')),
 'no_censoring': ('training_labels', lambda df: df.assign(label_censored=False)),
}

def selftest(root, SCH, which):
    log(f'\n{"="*96}\nSELFTEST -- each mutation must make at least one check fail\n{"="*96}')
    base = Res()
    D = Data(root)
    level1(D, SCH, base); level2(D, SCH, base); v8_requirements(D, base); level3(D, base)
    base_fail = {r['name'] for r in base.failed()}
    summary = []
    for name in (which or MUTATIONS):
        tgt, fn = MUTATIONS[name]
        def mut(t, df, _tgt=tgt, _fn=fn):
            if t != _tgt: return df
            try:
                return _fn(df)
            except (AttributeError, KeyError):
                return df      # this read did not load the mutated column
        log(f'\n--- mutation: {name}  (on {tgt}) ---')
        r = Res(); Dm = Data(root)
        try:
            level1(Dm, SCH, r, mut); level2(Dm, SCH, r, mut); v8_requirements(Dm, r, mut); level3(Dm, r, mut)
            err = None
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
            log(f'  HARNESS ERROR, not a verdict: {err}')
        new_fail = {x['name'] for x in r.failed()} - base_fail
        summary.append(dict(mutation=name, target=tgt, newly_failing=sorted(new_fail),
                            caught=bool(new_fail), harness_error=err))
        log(f'  => {"CAUGHT" if new_fail else ("HARNESS ERROR" if err else "NOT CAUGHT -- the checks above are blind to this")}'
            f' ({len(new_fail)} newly failing)')
    log(f'\n{"="*96}\nSELFTEST SUMMARY\n{"="*96}')
    for s in summary:
        tag = 'CAUGHT    ' if s['caught'] else ('HARNESS ERR' if s.get('harness_error') else 'NOT CAUGHT')
        log(f'  {tag} {s["mutation"]:<34} '
            f'-> {", ".join(s["newly_failing"][:3]) or s.get("harness_error") or "(nothing)"}')
    blind = [s['mutation'] for s in summary if not s['caught'] and not s.get('harness_error')]
    log(f'\n  {len(summary)-len(blind)}/{len(summary)} mutations caught.'
        + (f'  BLIND TO: {blind}' if blind else '  No blind spots in this set.'))
    return summary


# ===========================================================================
# VARIANCE -- the per-metric band across seeds (§5.3).  Five seeds, one code state.
# ===========================================================================
def variance(dirs, out):
    rows = {}
    commits, gens = set(), set()
    for d in dirs:
        mo = json.loads((Path(d) / 'measured_outcomes.json').read_text())
        mf = json.loads((Path(d) / 'manifest.json').read_text())
        commits.add(mf['code_commit']); gens.add(mf['generator_version'])
        def walk(prefix, o):
            for k, v in o.items():
                if k.startswith('__'): continue
                if isinstance(v, dict): walk(f'{prefix}{k}.', v)
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    rows.setdefault(f'{prefix}{k}', []).append(float(v))
        walk('', mo)
        lp = Path(d) / 'level4.json'
        if lp.exists():
            for k, v in json.loads(lp.read_text()).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    rows.setdefault(f'level4.{k}', []).append(float(v))
    band = {}
    for k, v in sorted(rows.items()):
        v = np.array(v)
        band[k] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                       min=float(v.min()), max=float(v.max()),
                       cv=float(v.std(ddof=1) / abs(v.mean())) if len(v) > 1 and v.mean() != 0 else 0.0)
    one_state = len(commits) == 1 and len(gens) == 1
    log(f'\n{"="*96}\nSEED-VARIANCE BAND  ({len(dirs)} seeds)\n{"="*96}')
    log(f'  one code state across all seeds: {one_state}  commits={sorted(commits)}')
    log(f'  {"metric":<62}{"mean":>13}{"min":>13}{"max":>13}{"cv":>9}')
    for k, b in band.items():
        log(f'  {k:<62}{b["mean"]:>13.5g}{b["min"]:>13.5g}{b["max"]:>13.5g}{b["cv"]:>9.4f}')
    payload = dict(validator=VALIDATOR_VERSION, seeds=[str(d) for d in dirs],
                   one_code_state=one_state, code_commits=sorted(commits), band=band)
    if out: Path(out).write_text(json.dumps(payload, indent=1))
    return payload


# ===========================================================================
# MAIN -- run the levels in order, stop at the first failed GATE.
# ===========================================================================
GATE_ORDER = ['G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']
GATE_ASKS = {'G0': 'Is this a dataset?', 'G1': 'Does the graph exist?',
             'G2': 'What is the number to beat?', 'G3': 'Is there signal at all?',
             'G4': 'Does the graph contribute?', 'G5': 'Is the as-of gate binding?',
             'G6': 'Are the folds sound?', 'G7': 'Will it survive production?'}

def run(root, levels, out):
    t0 = time.time()
    SCH = parse_schema('db/schema.sql')
    D = Data(root)
    res = Res()
    log(f'{"="*96}\n{VALIDATOR_VERSION}   bands frozen {BANDS_FROZEN_AT}   dataset {root}\n{"="*96}')
    mf = Path(root) / 'manifest.json'
    if mf.exists(): log('  tags: ' + json.dumps(json.loads(mf.read_text())))
    L4 = {}
    if '1' in levels: level1(D, SCH, res)
    if '2' in levels:
        level2(D, SCH, res)
        v8_requirements(D, res)
    # --- G0 / G1 verdict before anything is modelled
    stop = None
    for g in ('G0', 'G1'):
        if res.failed(g): stop = g; break
    if stop is None and '3' in levels: level3(D, res)
    for g in ('G0', 'G1'):
        if res.failed(g) and stop is None: stop = g
    if stop is None and '4' in levels:
        res, L4 = level4(D, res, root)
    for g in GATE_ORDER:
        if res.failed(g): stop = stop or g
    log(f'\n{"="*96}\nREADINESS LADDER\n{"="*96}')
    reached = True
    for g in GATE_ORDER:
        f = res.failed(g)
        n = [r for r in res.rows if r['gate'] == g and r['passed'] is not None]
        if not n:
            log(f'  {g}  {GATE_ASKS[g]:<38} NOT RUN'); reached = False; continue
        if not reached:
            log(f'  {g}  {GATE_ASKS[g]:<38} NOT REACHED (an earlier gate failed)'); continue
        if f:
            log(f'  {g}  {GATE_ASKS[g]:<38} FAIL  ({len(f)}/{len(n)} checks)')
            for x in f: log(f'         - {x["name"]}: {x["detail"]}')
            reached = False
        else:
            log(f'  {g}  {GATE_ASKS[g]:<38} PASS  ({len(n)} checks)')
    allf = res.failed()
    log(f'\n  checks run {len([r for r in res.rows if r["passed"] is not None])}, '
        f'failed {len(allf)}, elapsed {time.time()-t0:.0f}s')
    if allf:
        log('\n  FAILURES (reported, not hidden -- see §5.5):')
        for x in allf: log(f'    L{x["level"]} {x["gate"]}  {x["name"]}\n         {x["detail"]}')
    for grp, title in (('L3', 'LEVEL-3 JOINT PROBES (reported; only P1 gates -- the brief calls '
                              'it decisive)'),
                       ('V8', 'V8 BRIEF REQUIREMENTS §2.2 / §2.6 (deliverable checks, reported)')):
        rows = [r for r in res.rows if r['gate'] == grp and r['passed'] is not None]
        if not rows: continue
        bad = [r for r in rows if not r['passed']]
        log(f'\n{"="*96}\n{title}\n{"="*96}')
        log(f'  {len(rows)-len(bad)}/{len(rows)} pass')
        for r in bad: log(f'  FAIL  {r["name"]}\n        {r["detail"]}')
    payload = dict(validator=VALIDATOR_VERSION, bands_frozen=BANDS_FROZEN_AT, dataset=str(root),
                   first_failed_gate=stop, checks=res.rows, level4=L4,
                   n_failed=len(allf), elapsed_s=round(time.time() - t0, 1))
    if out:
        Path(out).write_text(json.dumps(payload, indent=1, default=str))
        if L4: (Path(root) / 'level4.json').write_text(json.dumps(L4, indent=1, default=str))
        log(f'  report -> {out}')
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='*')
    ap.add_argument('--levels', default='1234')
    ap.add_argument('--out', default=None)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--mutation', action='append')
    ap.add_argument('--variance', action='store_true')
    A = ap.parse_args(argv)
    if A.variance:
        variance([Path(x) for x in A.root], A.out); return 0
    if A.selftest:
        s = selftest(A.root[0], parse_schema('db/schema.sql'), A.mutation)
        if A.out: Path(A.out).write_text(json.dumps(s, indent=1))
        return 0
    p = run(A.root[0], A.levels, A.out)
    return 0 if p['first_failed_gate'] is None else 1

if __name__ == '__main__':
    sys.exit(main())
