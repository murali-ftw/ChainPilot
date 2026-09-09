"""ChainPilot x Rane -- V5 causal synthetic world generator.

Implements synthetic_rules.md. Generation order is
    REALISTIC MECHANISM -> RELATIONSHIP -> OUTCOME -> DISTRIBUTION

Only the LEFT column of synthetic_rules.md §13 is set here. Every right-column quantity
(fill masses, constrained share, event rates, late rate, zero-order share, seasonality
ratio, staleness) is MEASURED at the end and printed. None of them appears as a literal,
a clip, a target or a post-hoc adjustment.
"""
import numpy as np, pandas as pd, json, os, re, sys, argparse, hashlib, warnings
warnings.filterwarnings('ignore')
print = __import__('functools').partial(__builtins__.print if not isinstance(__builtins__, dict) else __builtins__['print'], flush=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=1001)
ap.add_argument('--out', default='gen_v5')
ap.add_argument('--spec', default='db/schema.sql')
ap.add_argument('--refdir', default='db')
ap.add_argument('--cap', type=float, default=8000.0)
ap.add_argument('--cover', type=float, default=7.5)
ap.add_argument('--seas', type=float, default=0.29)
ap.add_argument('--z', type=float, default=5.5)
A = ap.parse_args()
SEED = A.seed
OUT = Path(A.out) / f'seed_{SEED}'; OUT.mkdir(parents=True, exist_ok=True)
r = np.random.default_rng(SEED)

# ---------------------------------------------------------------- schema
def load_schema(spec, refdir):
    sql = re.sub(r'--[^\n]*', '', Path(spec).read_text())   # strip SQL comments first
    tabs = re.findall(r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?["`]?(\w+)["`]?\s*\((.*?)\n\)\s*;', sql, re.S | re.I)
    sch, nn = {}, {}
    for name, body in tabs:
        cols, depth, cur = [], 0, ''
        for ch in body:
            if ch == '(': depth += 1
            if ch == ')': depth -= 1
            if ch == ',' and depth == 0: cols.append(cur); cur = ''
            else: cur += ch
        cols.append(cur)
        names, req = [], set()
        for c in cols:
            c = c.strip()
            if not c or re.match(r'^(PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT)\b', c, re.I): continue
            m = re.match(r'^["`]?(\w+)["`]?\s+(.*)$', c, re.S)
            if m:
                names.append(m.group(1))
                if 'NOT NULL' in m.group(2).upper(): req.add(m.group(1))
        sch[name + '.csv'] = names; nn[name + '.csv'] = req
    return sch, nn
SCH, NOTNULL = load_schema(A.spec, A.refdir)

def emit(name, df):
    cols = SCH[name]
    for c in cols:
        if c not in df.columns: df[c] = ''
    df = df[cols]
    miss = [c for c in NOTNULL.get(name, ()) if c in df.columns
            and (df[c].isna().all() or (df[c].astype(str).str.strip() == '').all())]
    if miss: print(f'    !! {name}: NOT NULL columns left blank: {miss}')
    df.to_csv(OUT / name, index=False)
    print(f'  {name:<36}{len(df):>12,} rows')
    return df

# ============================================================ §13 MECHANISM PARAMETERS
P = dict(
    world_start='2016-01-01', world_end='2026-03-31',
    model_start='2019-01-01', model_end='2025-12-31',
    n_plants=7, n_suppliers=420, n_parts=620, n_products=180, n_customers=20,
    n_channels=16072, cold_start_channels=738,
    # capacity distribution per supplier tier (§13 left)
    cap_mu_log=np.log(A.cap), cap_sigma_log=0.40, cap_tier_bonus=0.35,
    # order policy (§13 left) -- drives density AND idle share as OUTCOMES
    review_weeks=2, cover_weeks_mu=A.cover, cover_weeks_sd=A.cover*0.3, safety_cover_weeks=5.2,
    demand_rate_mu_log=np.log(11.0), demand_rate_sigma_log=1.05, demand_cv=0.30, service_z=A.z,
    # variance-preserving split of demand_rate_sigma_log:
    #   sigma_s^2 + sigma_i^2/(1-phi^2) = 0.85^2 + 0.1499^2/(1-0.97^2) = 0.7225 + 0.3800 = 1.1025 = 1.05^2
    rate_sigma_level=0.85, rate_phi=0.97, rate_sigma_innov=0.1499,
    forecast_alpha=0.10,         # MRP re-forecast: ~10-week EWMA of realised demand
    # lead-time process (§13 left)
    lead_base_frac=0.51, lead_sigma=0.34, congestion_beta=0.85, lead_disruption_mult=1.55,
    # recording lag per table (§13 left): (median_days, sigma, stress_beta, never_recorded)
    lag={'purchase_orders': (0.6, 0.9, 0.5, 0.004), 'po_lines': (0.7, 1.0, 0.6, 0.004),
         'supplier_acknowledgements': (1.6, 1.3, 1.1, 0.031), 'asn': (1.1, 1.2, 0.8, 0.018),
         'goods_receipts': (1.3, 1.15, 0.9, 0.011), 'grn_lines': (1.3, 1.15, 0.9, 0.011),
         'quality_inspections': (2.4, 1.35, 1.0, 0.022), 'inventory_transactions': (0.9, 1.25, 1.3, 0.009),
         'shortage_events': (2.9, 1.45, 1.2, 0.027), 'expedite_events': (2.2, 1.3, 1.0, 0.021),
         'line_stop_events': (1.7, 1.2, 0.9, 0.006), 'po_line_revisions': (2.0, 1.5, 1.1, 0.024),
         'supplier_allocation': (3.2, 1.4, 0.8, 0.015), 'supplier_capacity': (4.0, 1.3, 0.6, 0.02), 'supplier_quality_ppm': (9.0, 1.4, 0.5, 0.01),
         'supplier_audits': (6.0, 1.35, 0.5, 0.01)},
    # supplier latent state (§13 left)
    state_ar=0.93, state_innov=0.10,
    # disruption calendar (§13 left) -- two separate classes
    supply_shock_rate=0.0055, supply_shock_mag=(1.15, 0.45), supply_shock_decay=0.86,
    transit_shock_rate=0.0040, transit_shock_decay=0.80,
    demand_shock_rate=0.0035, demand_shock_mag=(0.55, 0.30), demand_shock_decay=0.88,
    # demand seasonality AMPLITUDE and PHASE (§13 left; the ratio is measured, not set)
    seas_festive=A.seas, seas_fiscal=A.seas*0.72, seas_monsoon=A.seas*0.41, seas_winter=-A.seas*0.36,
    trend_per_year=0.021,
    # fulfilment behaviour / quality
    refusal_base=0.0060, ration_reserve=0.16, defect_line_prob=0.030, reject_alpha=2.4, reject_beta=26.0,
    # escalation policy (§13 left; the event COUNTS are measured)
    transfer_rate=0.62, altsource_prob=0.34,
    # --- 1a expedite: a physical acceleration of an open line, not a label ---
    exp_trigger_sev=0.30,        # a buyer pays the premium when the shortage is real,
    exp_trigger_weeks=2,         # not on every weekly dip below safety -- once per episode
    exp_success_base=0.68,       # expedites fail; a supplier under stress fails more
    exp_success_stress=0.45,
    exp_compress_predispatch=0.55,   # not yet shipped: supplier overtime can compress more
    exp_compress_intransit=0.28,     # already moving: only the remaining leg can compress
    exp_compress_a=2.0, exp_compress_b=3.0,   # Beta shape of the realised compression
    exp_premium_per_unit=42.0, exp_freight_uplift=0.35,
    exp_max_lines_per_week=900,
    # --- 1b recovery: buyers escalate on observable pressure ---
    prio_starve_w=0.45,          # recent unmet demand
    prio_cover_w=0.30,           # days of cover below a week
    prio_linestop_w=0.55,        # this part stopped a line recently
    starve_decay=0.82,
    # --- 1c escalation is a CONSEQUENCE of the episode, not a fraction ---
    esc_persist_weeks=3,         # a condition absorbed faster than this is not an event
    esc_severity=0.30,           # and it must have bitten this far below safety
    ls_min_stockout_weeks=1,     # weeks of unserved production before the line is called stopped
    ls_unmet_frac=0.10,          # ~half a shift of lost build on a critical part -- the
                                 # scale at which Rane records a line stop (§0: ~18/yr)
    ls_requires_critical=True,
    seed=SEED,
)

W = pd.date_range('2016-01-04', '2026-03-30', freq='W-MON')
T = len(W)
WMONTH = W.month.to_numpy(); WYEAR = W.year.to_numpy()
MONTHKEY = (WYEAR - 2016) * 12 + (WMONTH - 1); NMONTH = int(MONTHKEY.max()) + 1
NS, NP, NPL, NCH = P['n_suppliers'], P['n_parts'], P['n_plants'], P['n_channels']
print(f'seed={SEED}  weeks={T}  {W[0].date()} -> {W[-1].date()}  channels={NCH}')

def season(mon):
    s = np.ones(len(mon))
    s += np.where(np.isin(mon, [8, 9, 10, 11]), P['seas_festive'], 0)
    s += np.where(mon == 3, P['seas_fiscal'], 0)
    s += np.where(np.isin(mon, [6, 7, 8]), P['seas_monsoon'], 0)
    s += np.where(np.isin(mon, [12, 1]), P['seas_winter'], 0)
    return s
SEAS = season(WMONTH)
TREND = (1 + P['trend_per_year']) ** ((W - W[0]).days.to_numpy() / 365.25)

def lag_days(table, n, stress, rng):
    med, sig, beta, _ = P['lag'][table]
    return np.maximum(0.0, np.round(rng.lognormal(np.log(med), sig, n) * (1 + beta * np.maximum(stress, 0)), 2))
def never_mask(table, n, rng):
    return rng.random(n) < P['lag'][table][3]

# ============================================================ masters
print('masters...')
bu = pd.DataFrame({'bu_id': [f'BU{i:02d}' for i in range(1, 5)],
                   'bu_name': ['Rane Brake Lining', 'Rane Engine Valves', 'Rane Steering', 'Rane Elastomer'],
                   'bu_code': ['RBL', 'REV', 'RST', 'REL']})
emit('business_units.csv', bu)
cities = [('Hosur', 'Tamil Nadu'), ('Chennai', 'Tamil Nadu'), ('Pantnagar', 'Uttarakhand'),
          ('Bawal', 'Haryana'), ('Pune', 'Maharashtra'), ('Bengaluru', 'Karnataka'), ('Hyderabad', 'Telangana')]
plants = pd.DataFrame({'plant_id': [f'PL{i+1:02d}' for i in range(NPL)],
    'plant_name': [f'{c} Plant {i+1}' for i, (c, s) in enumerate(cities)],
    'bu_id': [f'BU{i%4+1:02d}' for i in range(NPL)], 'city': [c for c, s in cities], 'state': [s for c, s in cities],
    'country': 'India', 'latitude': r.uniform(8, 28, NPL).round(6), 'longitude': r.uniform(72, 88, NPL).round(6),
    'capacity_units_per_day': r.integers(3500, 6500, NPL), 'commissioned_date': '2015-01-01', 'is_active': True})
emit('plants.csv', plants)

s_tier = np.where(np.arange(NS) < 46, 'T2', 'T1')
s_rel = np.clip(r.beta(6.5, 2.6, NS), 0.25, 0.995)          # reliability prior (§13 left)
suppliers = pd.DataFrame({'supplier_id': [f'SUP{i+1:05d}' for i in range(NS)],
    'supplier_name': [f'Rane Supplier {i+1:05d}' for i in range(NS)],
    'supplier_group_id': [f'GRP{i//5:04d}' for i in range(NS)], 'country': 'India',
    'state': r.choice(['Tamil Nadu','Karnataka','Maharashtra','Gujarat','Haryana','Uttarakhand','Telangana'], NS),
    'city': r.choice(['Chennai','Pune','Bengaluru','Ahmedabad','Gurugram','Haridwar','Hyderabad'], NS),
    'supplier_tier': np.where(s_tier=='T2','tier2','tier1'), 'supplier_type': r.choice(['manufacturer','distributor','specialty'], NS),
    'business_class': np.where(s_rel > .85, 'A', np.where(s_rel > .7, 'B', 'C')),
    'onboarded_date': '2015-01-01', 'is_active': True, 'msme_flag': r.random(NS) < .35,
    'payment_terms_days': r.choice([30, 45, 60, 75], NS)})
emit('suppliers.csv', suppliers)
sites = pd.DataFrame({'site_id': [f'SIT{i+1:05d}' for i in range(NS)],
    'supplier_id': suppliers.supplier_id, 'site_name': [f'Supplier {i+1} Site' for i in range(NS)],
    'city': suppliers.city, 'state': suppliers.state, 'country': 'India',
    'latitude': r.uniform(8, 28, NS).round(6), 'longitude': r.uniform(72, 88, NS).round(6), 'is_active': True})
emit('supplier_sites.csv', sites)
parts = pd.DataFrame({'part_id': [f'P{i+1:05d}' for i in range(NP)],
    'part_number': [f'RN-P{i+1:05d}' for i in range(NP)], 'part_name': [f'Auto Component {i+1:05d}' for i in range(NP)],
    'part_category': r.choice(['brake','steering','engine','elastomer','casting'], NP),
    'material_type': r.choice(['steel','aluminium','rubber','composite'], NP), 'uom': 'EA',
    'is_critical': r.random(NP) < .22, 'criticality_reason': '', 'std_cost_inr': r.integers(12, 65, NP),
    'shelf_life_days': r.integers(30, 365, NP), 'is_active': True, 'effective_from': '2015-01-01', 'effective_to': ''})
emit('parts.csv', parts)
products = pd.DataFrame({'product_id': [f'PR{i+1:04d}' for i in range(P['n_products'])],
    'product_number': [f'RN-PR{i+1:04d}' for i in range(P['n_products'])],
    'product_name': [f'Rane Product {i+1:04d}' for i in range(P['n_products'])],
    'product_family': r.choice(['Brake Assembly','Valve Assembly','Steering Assembly','Engine Module'], P['n_products']),
    'bu_id': [f'BU{r.integers(1,5):02d}' for _ in range(P['n_products'])],
    'customer_id': [f'CUST{r.integers(1,P["n_customers"]+1):03d}' for _ in range(P['n_products'])],
    'is_active': True, 'effective_from': '2016-01-01', 'effective_to': ''})
emit('products.csv', products)
emit('customers.csv', pd.DataFrame({'customer_id': [f'CUST{i+1:03d}' for i in range(P['n_customers'])],
    'customer_name': [f'Customer {i+1:03d}' for i in range(P['n_customers'])],
    'customer_tier': r.choice(['OEM','Tier1','Export'], P['n_customers']),
    'penalty_per_unit_inr': r.integers(200, 2000, P['n_customers']), 'country': 'India', 'is_active': True}))

# ---- channels: supplier x part x plant, unique
print('channels...')
seen = set(); cs, cp, cl = [], [], []
while len(cs) < NCH:
    need = NCH - len(cs)
    a = r.integers(0, NS, need * 2); b = r.integers(0, NP, need * 2); c = r.integers(0, NPL, need * 2)
    for x, y, z in zip(a, b, c):
        k = (x, y, z)
        if k not in seen:
            seen.add(k); cs.append(x); cp.append(y); cl.append(z)
            if len(cs) == NCH: break
CS = np.array(cs); CP = np.array(cp); CL = np.array(cl)
CHID = np.array([f'CH{i+1:06d}' for i in range(NCH)])
contracted = r.integers(15, 70, NCH)
dist = r.integers(50, 2200, NCH)
mode = r.choice(['road', 'rail', 'sea', 'air'], NCH, p=[.72, .14, .09, .05])
channels = pd.DataFrame({'channel_id': CHID, 'supplier_id': suppliers.supplier_id.values[CS],
    'site_id': sites.site_id.values[CS], 'part_id': parts.part_id.values[CP], 'plant_id': plants.plant_id.values[CL],
    'is_approved': True, 'ppap_date': '2016-01-01', 'approval_status': 'approved',
    'contracted_lead_time_days': contracted, 'transport_mode': mode, 'transport_distance_km': dist,
    'effective_from': '2016-01-01', 'effective_to': ''})
emit('sourcing_channels.csv', channels)

# cold-start slice: channels that never place an order (§2, mandatory)
cold = np.zeros(NCH, bool); cold[r.choice(NCH, P['cold_start_channels'], replace=False)] = True

# ============================================================ latent state + disruptions (§11: two classes)
print('latent state and disruption calendars...')
sup_state = np.zeros((T, NS), np.float32)      # supply-side stress
transit_state = np.zeros((T, NPL), np.float32) # transit stress by destination plant
dem_state = np.zeros((T, NPL), np.float32)     # demand-side shocks (separate class, separate node)
x = r.normal(0, .45, NS); shk = np.zeros(NS); tr = np.zeros(NPL); dm = np.zeros(NPL)
for t in range(T):
    shk = np.where(r.random(NS) < P['supply_shock_rate'], r.normal(*P['supply_shock_mag'], NS), shk * P['supply_shock_decay'])
    x = P['state_ar'] * x + r.normal(0, P['state_innov'], NS) + shk
    sup_state[t] = x
    tr = np.where(r.random(NPL) < P['transit_shock_rate'], np.abs(r.normal(1.0, .4, NPL)), tr * P['transit_shock_decay'])
    transit_state[t] = tr
    dm = np.where(r.random(NPL) < P['demand_shock_rate'], r.normal(*P['demand_shock_mag'], NPL), dm * P['demand_shock_decay'])
    dem_state[t] = dm
# 2020 COVID + 2021-22 chip shortage enter as SUPPLY-side capacity/transit, never as requirement
regime = np.zeros(T)
regime += np.where((W >= '2020-03-01') & (W <= '2020-09-30'), 1.0, 0)
regime += np.where((W >= '2021-04-01') & (W <= '2022-06-30'), 0.55, 0)
NORMAL = regime == 0

# capacity K(supplier, month) -- latent, evolves; observability is NEVER set
tier_b = np.where(s_tier == 'T2', P['cap_tier_bonus'], 0.0)
K = np.exp(r.normal(P['cap_mu_log'] + tier_b, P['cap_sigma_log'], (NMONTH, NS)))
mstate = np.zeros((NMONTH, NS))
for m in range(NMONTH):
    wk = np.where(MONTHKEY == m)[0]
    mstate[m] = sup_state[wk].mean(0) if len(wk) else 0
    K[m] *= (1 - 0.30 * np.tanh(np.maximum(mstate[m], 0)) - 0.22 * np.tanh(regime[wk].mean() if len(wk) else 0))
K = np.maximum(K, 60.0)

# ============================================================ order policy + fulfilment (§6, §7)
print('simulating world...')
# persistent per-channel level: what a planner sizes safety stock against
rate = np.exp(r.normal(P['demand_rate_mu_log'], P['rate_sigma_level'], NCH))
# transient AR(1) component, started at its stationary distribution
_rate_eps = r.normal(0, P['rate_sigma_innov'] / np.sqrt(1 - P['rate_phi']**2), NCH)
plant_f = r.uniform(.75, 1.3, NPL)[CL]
cover = np.maximum(2.0, r.normal(P['cover_weeks_mu'], P['cover_weeks_sd'], NCH))
ss_w = np.maximum(1.0, r.normal(P['safety_cover_weeks'], 0.9, NCH))
lead_wk = contracted / 7.0
def _policy(dh):
    sf = np.maximum(1, dh * ss_w)
    rp = np.maximum(sf + 1, dh * lead_wk + sf
                    + P['service_z'] * dh * np.sqrt(np.maximum(lead_wk, 1)) * P['demand_cv'])
    up = np.maximum(rp + 1, rp + dh * cover)
    return sf.astype(np.int64), rp.astype(np.int64), up.astype(np.int64)
dhat = rate.copy()                      # the planner's running demand forecast
safety, rop, upto = _policy(dhat)
lot = np.maximum(1, (rate * 0.5)).astype(np.int64)

PP = CP * NPL + CL; NPP = NP * NPL
PP_SAFETY = np.bincount(PP, weights=np.where(cold, 0.0, safety.astype(float)), minlength=NPP)
PP_COVER = np.bincount(PP, weights=contracted / 7.0, minlength=NPP) / np.maximum(1, np.bincount(PP, minlength=NPP))
REVIEW = r.integers(0, max(1, P['review_weeks']), NCH)
PRIO_CH = r.random(NCH)          # stable allocation standing per channel (§7 persistence)
on_hand = np.where(cold, 0, r.uniform(rop, upto)).astype(np.int64)   # staggered steady state
on_order = np.zeros(NCH, np.int64)
alt_boost = np.zeros(NCH)         # §10 alternate-sourcing volume shifted ONTO a channel

# part->channel index, for alternate sourcing across suppliers of the same part
order_by_part = np.argsort(CP, kind='stable')
part_start = np.searchsorted(CP[order_by_part], np.arange(NP), 'left')
part_end = np.searchsorted(CP[order_by_part], np.arange(NP), 'right')

# Open lines live in a flat registry, not in an append-only list, because 1a has to
# reach back into a line that has been placed and move its arrival.
_CAP = 4_000_000
po_ch = np.zeros(_CAP, np.int32); po_t = np.zeros(_CAP, np.int32)
po_arr = np.zeros(_CAP, np.int32); po_disp = np.zeros(_CAP, np.int32)
po_qty = np.zeros(_CAP, np.int64); po_deliv = np.zeros(_CAP, np.int64)
po_lead = np.zeros(_CAP, np.float64); po_pp = np.zeros(_CAP, np.int32)
po_open = np.zeros(_CAP, bool); po_ref = np.zeros(_CAP, bool)
po_exp = np.zeros(_CAP, bool); po_expfail = np.zeros(_CAP, bool)
npo = 0
buckets = {}          # arrival week -> [index arrays]; entries may be stale, validated on use
recent = []           # rolling window of recently placed lines, for expedite candidates
EXP = {k: [] for k in ('t','po','units','cost','mode','pp')}
EPI = {k: [] for k in ('pp','t0','t1','len','sev','unmet','stockout','mitig','act')}
LS  = {k: [] for k in ('pp','t','mins','units')}
starve = np.zeros(NCH); ls_recent = np.zeros(NPP)
ep_active = np.zeros(NPP, bool); ep_t0 = np.zeros(NPP, np.int32); ep_len = np.zeros(NPP, np.int32)
ep_sev = np.zeros(NPP); ep_unmet = np.zeros(NPP, np.int64); ep_stock = np.zeros(NPP, np.int32)
ep_mit = np.zeros(NPP, np.int8)          # bit 1 transfer, 2 expedite, 4 alternate source
ep_exp = np.zeros(NPP, bool)             # an expedite is raised once per episode
SHORT = {k: [] for k in ('t','ch','qty','cause_t','cause_ch')}
INV = {k: [] for k in ('t','ch','typ','qty')}
month_ordered = np.zeros(NS); cur_month = -1
remK = np.zeros(NS)
util_hist = np.zeros((NMONTH, NS)); ordered_hist = np.zeros((NMONTH, NS))
sup_load_prev = np.zeros(NS)      # §10: previous month's utilisation degrades OTHER channels now

for t in range(T):
    m = MONTHKEY[t]
    if m != cur_month:
        if cur_month >= 0:
            ordered_hist[cur_month] = month_ordered
            util_hist[cur_month] = month_ordered / K[cur_month]
            sup_load_prev = util_hist[cur_month]
        # §10: last month's overload reduces this month's effective capacity
        carry = 1.0 - 0.18 * np.clip(sup_load_prev - 1.0, 0, 1.5)
        cur_month = m; remK = K[m].copy() * carry; month_ordered = np.zeros(NS)
        nwk = max(1, int((MONTHKEY == m).sum())); bank = remK / nwk
    # ---- demand and consumption
    _rate_eps = P['rate_phi'] * _rate_eps + r.normal(0, P['rate_sigma_innov'], NCH)
    dem = (rate * np.exp(_rate_eps) * SEAS[t] * TREND[t] * plant_f
           * (1 + dem_state[t][CL]) * r.lognormal(0, P['demand_cv'], NCH))
    dem = np.where(cold, 0, np.maximum(0, dem)).astype(np.int64)
    dhat = P['forecast_alpha'] * dem + (1 - P['forecast_alpha']) * dhat
    safety, rop, upto = _policy(dhat)
    PP_SAFETY = np.bincount(PP, weights=np.where(cold, 0.0, safety.astype(float)), minlength=NPP)
    issue = np.minimum(on_hand, dem)
    on_hand -= issue
    unmet = dem - issue
    # ---- order policy: reorder point over inventory position -> idle share is an OUTCOME
    ip = on_hand + on_order
    want = (~cold) & (ip < rop) & (REVIEW == (t % P['review_weeks']))
    qty = np.where(want, np.maximum(lot, upto - ip), 0)
    qty = qty + np.where(want, (alt_boost * qty).astype(np.int64), 0)   # §10 volume shifted in
    alt_boost *= 0.72
    idx = np.where(qty > 0)[0]
    if len(idx):
        sup = CS[idx]; q = qty[idx].astype(np.float64)
        np.add.at(month_ordered, sup, q)
        # ---- §6 fulfilment: delivered = min(available capacity, ordered)
        tot = np.zeros(NS); np.add.at(tot, sup, q)
        # §7 keeps the stable standing; 1b lets observable pressure move a starved channel
        # up the queue. Every term is something a buyer can see -- recent unmet demand, days
        # of cover, a part that has just stopped a line -- never the condition counter.
        _cov_d = on_hand[idx] / np.maximum(rate[idx], 1e-9) * 7.0
        prio = (PRIO_CH[idx]
                - P['prio_starve_w'] * np.tanh(starve[idx])
                - P['prio_cover_w'] * (_cov_d < 7.0)
                - P['prio_linestop_w'] * np.minimum(ls_recent[PP[idx]], 1.0)
                + r.normal(0, 0.06, len(idx)))
        o = np.lexsort((prio, sup))
        qs = q[o]; ss_ = sup[o]
        cs = np.cumsum(qs)
        gstart = np.zeros(len(o))
        first = np.r_[True, ss_[1:] != ss_[:-1]]
        gstart[first] = cs[first] - qs[first]
        gstart = np.maximum.accumulate(np.where(first, gstart, -np.inf))
        before = cs - qs - gstart                      # units already promised ahead of this line
        seqbank = bank * (1 - P['ration_reserve'])
        served_s = np.clip(seqbank[ss_] - before, 0, qs)
        unmet_s = qs - served_s
        pool = np.zeros(NS); np.add.at(pool, ss_, unmet_s)
        resv = bank * P['ration_reserve']
        share = np.where(pool > 0, np.minimum(1.0, resv / np.maximum(pool, 1e-9)), 0.0)
        served_s = served_s + unmet_s * share[ss_]
        served = np.empty_like(served_s); served[o] = served_s
        # §10 decisive arc runs through remK: volume shifted onto a supplier consumes its
        # capacity, so its OTHER channels see less headroom. Overload also carries into next
        # month's effective capacity (applied at the month boundary), never onto fill directly,
        # because a blanket multiplier would destroy the mass at exactly 1.0 that §7 requires.
        deliv = np.floor(served).astype(np.int64)
        refuse = r.random(len(idx)) < (P['refusal_base'] * (1 + 3.2 * np.maximum(sup_state[t][sup], 0)))
        deliv = np.where(refuse, 0, deliv)
        used = np.zeros(NS); np.add.at(used, sup, np.minimum(q, deliv.astype(np.float64)))
        bank = np.maximum(bank - used, 0) + remK / nwk      # unused capacity banks forward
        bank = np.minimum(bank, remK)
        # ---- §8 arrival: lane process, transit disruption, right-skewed
        base = contracted[idx] * P['lead_base_frac']
        lead = np.exp(r.normal(np.log(np.maximum(base, 3)), P['lead_sigma'], len(idx)))
        lead *= (1 + 0.55 * np.maximum(sup_state[t][sup], 0) + 0.9 * transit_state[t][CL[idx]]
                 + P['congestion_beta'] * np.clip(sup_load_prev[sup] - 0.70, 0, 2.0)
                 + (P['lead_disruption_mult'] - 1) * regime[t])
        lead = np.maximum(3, lead)
        arr_t = t + np.ceil(lead / 7).astype(int)      # may exceed T-1: those POs stay open
        nn = len(idx); sl = slice(npo, npo + nn)
        po_ch[sl] = idx; po_t[sl] = t; po_arr[sl] = arr_t; po_qty[sl] = qty[idx]
        po_deliv[sl] = deliv; po_lead[sl] = lead; po_pp[sl] = PP[idx]
        po_disp[sl] = t + np.maximum(1, np.ceil(lead * 0.35 / 7)).astype(int)
        po_open[sl] = True; po_ref[sl] = refuse
        nidx = np.arange(npo, npo + nn); npo += nn
        on_order[idx] += qty[idx]
        for a in np.unique(arr_t[arr_t < T]):
            buckets.setdefault(int(a), []).append(nidx[arr_t == a])
        recent.append(nidx)
        if len(recent) > 32: recent.pop(0)
    # ---- receipts land
    if t in buckets:
        bi = np.concatenate(buckets.pop(t))
        bi = bi[po_open[bi] & (po_arr[bi] == t)]     # stale entries: the line was expedited away
        if len(bi):
            ci = po_ch[bi]; dq = po_deliv[bi]; oq = po_qty[bi]
            rej = np.where(r.random(len(dq)) < P['defect_line_prob'],
                           np.maximum(1, np.floor(dq * r.beta(P['reject_alpha'], P['reject_beta'], len(dq)))), 0).astype(np.int64)
            acc = dq - rej
            np.add.at(on_hand, ci, acc); np.add.at(on_order, ci, -oq)
            po_open[bi] = False
            INV['t'].append(np.full(len(ci), t)); INV['ch'].append(ci)
            INV['typ'].append(np.full(len(ci), 0)); INV['qty'].append(acc)          # 0 = RECEIPT
            if rej.sum():
                s2 = rej > 0
                INV['t'].append(np.full(int(s2.sum()), t)); INV['ch'].append(ci[s2])
                INV['typ'].append(np.full(int(s2.sum()), 3)); INV['qty'].append(-rej[s2]) # 3 = SCRAP
    if issue.sum():
        s3 = issue > 0
        INV['t'].append(np.full(s3.sum(), t)); INV['ch'].append(np.where(s3)[0])
        INV['typ'].append(np.full(s3.sum(), 1)); INV['qty'].append(-issue[s3])       # 1 = ISSUE
    # ---- §9 shortage condition: from the stock balance, not from a fill threshold
    # ---- §4 inter-plant transfer: surplus plants cover short plants for the same part
    sur_ch = np.where(cold, 0, np.maximum(0, on_hand - safety * 1.25)).astype(float)
    def_ch = np.where(cold, 0, np.maximum(0, safety - on_hand)).astype(float)
    p_sur = np.bincount(CP, weights=sur_ch, minlength=NP)
    p_def = np.bincount(CP, weights=def_ch, minlength=NP)
    movable = np.minimum(p_sur, p_def) * P['transfer_rate']
    take = np.divide(sur_ch * movable[CP], np.maximum(p_sur[CP], 1e-9))
    give = np.divide(def_ch * movable[CP], np.maximum(p_def[CP], 1e-9))
    take = np.floor(take).astype(np.int64); give = np.floor(give).astype(np.int64)
    on_hand = on_hand - take + give
    mv = np.where(take > 0)[0]
    if len(mv):
        INV['t'].append(np.full(len(mv), t)); INV['ch'].append(mv)
        INV['typ'].append(np.full(len(mv), 4)); INV['qty'].append(-take[mv])   # 4 = TRANSFER_OUT
    gv = np.where(give > 0)[0]
    if len(gv):
        INV['t'].append(np.full(len(gv), t)); INV['ch'].append(gv)
        INV['typ'].append(np.full(len(gv), 5)); INV['qty'].append(give[gv])    # 5 = TRANSFER_IN
        ep_mit[np.unique(PP[gv])] |= 1                                          # transfer acted

    # ---- 1b: observable buyer pressure, updated from what actually happened this week
    unmet_ch = np.maximum(0, dem - issue)
    starve = P['starve_decay'] * starve + (unmet_ch > 0).astype(float)
    ls_recent *= 0.90

    pp_oh = np.bincount(PP, weights=on_hand, minlength=NPP)
    pp_dm = np.bincount(PP, weights=dem, minlength=NPP)
    pp_un = np.bincount(PP, weights=unmet_ch, minlength=NPP)
    pp_ss = PP_SAFETY
    below = (pp_oh < pp_ss) & (pp_dm > 0)                  # §9.1: below safety stock
    sev_now = np.where(pp_ss > 0, (pp_ss - pp_oh) / np.maximum(pp_ss, 1e-9), 0.0)
    shpp = np.where(below)[0]
    if len(shpp):
        SHORT['t'].append(np.full(len(shpp), t)); SHORT['ch'].append(shpp)
        SHORT['qty'].append(np.maximum(1, (pp_ss - pp_oh)[shpp]).astype(np.int64))
        SHORT['cause_t'].append(np.full(len(shpp), t)); SHORT['cause_ch'].append(shpp)

    # ---- episode bookkeeping: a condition is a run of consecutive weeks below safety
    opening = shpp[~ep_active[shpp]]
    ep_active[opening] = True; ep_t0[opening] = t; ep_len[opening] = 0
    ep_sev[opening] = 0.0; ep_unmet[opening] = 0; ep_stock[opening] = 0; ep_mit[opening] = 0
    ep_exp[opening] = False
    ep_len[shpp] += 1
    ep_sev[shpp] = np.maximum(ep_sev[shpp], sev_now[shpp])
    ep_unmet[shpp] += pp_un[shpp].astype(np.int64)
    ep_stock[shpp] += (pp_un[shpp] >= P['ls_unmet_frac'] * np.maximum(pp_dm[shpp], 1)).astype(np.int32)

    # ---- 1a: EXPEDITE, and it moves the goods. Raised while the episode is live.
    trig = shpp[(ep_len[shpp] >= P['exp_trigger_weeks']) & (ep_sev[shpp] >= P['exp_trigger_sev'])
                & (~ep_exp[shpp])]
    ep_exp[trig] = True
    if len(trig) and recent:
        cand = np.concatenate(recent)
        cand = cand[po_open[cand] & (po_arr[cand] > t) & np.isin(po_pp[cand], trig)]
        if len(cand) > P['exp_max_lines_per_week']:
            cand = r.choice(cand, P['exp_max_lines_per_week'], replace=False)
        if len(cand):
            csup = CS[po_ch[cand]]
            ok = r.random(len(cand)) < (P['exp_success_base']
                    - P['exp_success_stress'] * np.clip(np.maximum(sup_state[t][csup], 0), 0, 1))
            intransit = t >= po_disp[cand]
            maxc = np.where(intransit, P['exp_compress_intransit'], P['exp_compress_predispatch'])
            red = r.beta(P['exp_compress_a'], P['exp_compress_b'], len(cand)) * maxc
            nl = np.maximum(3.0, po_lead[cand] * (1 - red))
            na = po_t[cand] + np.ceil(nl / 7).astype(int)
            na = np.maximum(na, np.maximum(t + 1, po_disp[cand]))   # never before dispatch
            moved = ok & (na < po_arr[cand])
            mi = cand[moved]
            if len(mi):
                na_m = na[moved]
                po_arr[mi] = na_m
                po_lead[mi] = np.maximum((na_m - po_t[mi]) * 7.0 - 6.0, 3.0)
                po_exp[mi] = True
                for a in np.unique(na_m[na_m < T]):
                    buckets.setdefault(int(a), []).append(mi[na_m == a])
                EXP['t'].append(np.full(len(mi), t)); EXP['po'].append(mi)
                EXP['units'].append(po_qty[mi] - po_deliv[mi] + 1)
                EXP['cost'].append((po_qty[mi] * P['exp_premium_per_unit']).astype(np.int64))
                EXP['mode'].append(np.where(intransit[moved], 0, 1))
                EXP['pp'].append(po_pp[mi])
                ep_mit[np.unique(po_pp[mi])] |= 2                    # expedite acted
            po_expfail[cand[ok & ~moved]] = True

    # ---- §10 alternate sourcing: shift volume to ANOTHER supplier of the same part
    if len(shpp):
        short = np.where(np.isin(PP, shpp) & (~cold))[0]
        esc = short[r.random(len(short)) < P['altsource_prob']] if len(short) else short
        for ch in esc[:400]:
            pcp = CP[ch]; cnd = order_by_part[part_start[pcp]:part_end[pcp]]
            cnd = cnd[(CS[cnd] != CS[ch]) & (~cold[cnd])]
            if len(cnd):
                alt_boost[cnd[r.integers(0, len(cnd))]] += 0.55
                ep_mit[PP[ch]] |= 4                                  # alternate source acted

    # ---- 1c: the episode closes. Escalation is now a CONSEQUENCE of what it did.
    closing = np.where(ep_active & ~below)[0]
    if len(closing):
        L = ep_len[closing]; SV = ep_sev[closing]; STK = ep_stock[closing]
        crit = parts.is_critical.values[closing // NPL]
        stopped = (STK >= P['ls_min_stockout_weeks']) & (crit if P['ls_requires_critical'] else True)
        # persisted past the point mitigation could absorb it, and bit deep enough -- or it
        # actually halted production, which escalates regardless of how long it took
        escal = ((L >= P['esc_persist_weeks']) & (SV >= P['esc_severity'])) | stopped
        e = closing[escal]
        if len(e):
            EPI['pp'].append(e); EPI['t0'].append(ep_t0[e]); EPI['t1'].append(np.full(len(e), t))
            EPI['len'].append(ep_len[e]); EPI['sev'].append(ep_sev[e])
            EPI['unmet'].append(ep_unmet[e]); EPI['stockout'].append(ep_stock[e])
            EPI['mitig'].append(ep_mit[e]); EPI['act'].append(ep_mit[e])
        ls = closing[stopped]
        if len(ls):
            LS['pp'].append(ls); LS['t'].append(ep_t0[ls])
            LS['mins'].append((ep_stock[ls] * r.integers(180, 900, len(ls))).astype(np.int64))
            LS['units'].append(np.maximum(1, ep_unmet[ls]).astype(np.int64))
            ls_recent[ls] = 1.0
        ep_active[closing] = False
ordered_hist[cur_month] = month_ordered; util_hist[cur_month] = month_ordered / K[cur_month]
print('  simulation loop done')

cat = lambda d, k: np.concatenate(d[k]) if d[k] else np.array([])
# read the PO lines back out of the registry, so expedited arrivals and their shortened
# lead times flow through into ASN / GRN / lateness rather than being cosmetic
NPO = npo
pt = po_t[:npo].astype(int); pch = po_ch[:npo].astype(int); pq = po_qty[:npo]
pd_ = po_deliv[:npo]; pl_ = po_lead[:npo]; pa = po_arr[:npo].astype(int)
ppr = contracted[pch]; prf = po_ref[:npo]
was_exp = po_exp[:npo].copy()
print(f'  PO lines: {NPO:,}   expedited: {int(was_exp.sum()):,} '
      f'({was_exp.mean()*100:.2f}%)   still open at cut: {int((pa >= T).sum()):,}')
json.dump({k: (v if not isinstance(v, dict) else {kk: list(vv) for kk, vv in v.items()}) for k, v in P.items()},
          open(OUT / 'mechanism_parameters.json', 'w'), indent=1, default=str)
np.savez_compressed(OUT / '_sim.npz', pt=pt, pch=pch, pq=pq, pd=pd_, pl=pl_, pa=pa, ppr=ppr, prf=prf,
                    CS=CS, CP=CP, CL=CL, cold=cold, util=util_hist, ordered=ordered_hist, K=K,
                    sup_state=sup_state, regime=regime, MONTHKEY=MONTHKEY, contracted=contracted,
                    it=cat(INV,'t').astype(int), ich=cat(INV,'ch').astype(int),
                    ityp=cat(INV,'typ').astype(int), iqty=cat(INV,'qty').astype(np.int64),
                    st=cat(SHORT,'t').astype(int), sch=cat(SHORT,'ch').astype(int), sq=cat(SHORT,'qty').astype(np.int64),
                    was_exp=was_exp)
print('  simulation state saved ->', OUT / '_sim.npz')

# ==================================================================== STAGE 2: emit + derive
print('emitting transactional spine...')
WV = W.values
def wk_ts(t_arr, jitter_hi=6): return pd.to_datetime(WV[t_arr]) + pd.to_timedelta(r.integers(0, jitter_hi, len(t_arr)), unit='D')

ev_po = wk_ts(pt)                                     # PO creation event time
stress_po = sup_state[pt, CS[pch]]
lag_po_d = lag_days('po_lines', NPO, stress_po, r)
rec_po = ev_po + pd.to_timedelta(lag_po_d, unit='D')
nrec_po = never_mask('po_lines', NPO, r)              # §3.2 some records are NEVER entered


po_id = np.array([f'PO{i:09d}' for i in range(NPO)])
pol_id = np.array([f'POL{i:09d}' for i in range(NPO)])
promise = ev_po + pd.to_timedelta(contracted[pch], unit='D')
arr_ts = ev_po + pd.to_timedelta(np.round(pl_).astype(int), unit='D')

emit('purchase_orders.csv', pd.DataFrame({'po_id': po_id, 'po_number': po_id,
    'supplier_id': suppliers.supplier_id.values[CS[pch]], 'site_id': sites.site_id.values[CS[pch]],
    'plant_id': plants.plant_id.values[CL[pch]], 'po_type': 'standard', 'currency': 'INR',
    'incoterm': 'DAP', 'status': 'OPEN', 'buyer_id': [f'BUY{x:03d}' for x in (pch % 40)],
    'created_ts': ev_po, 'recorded_ts': rec_po}))
emit('po_lines.csv', pd.DataFrame({'po_line_id': pol_id, 'po_id': po_id, 'line_number': 1,
    'part_id': parts.part_id.values[CP[pch]], 'channel_id': CHID[pch], 'qty_ordered': pq,
    'unit_price': np.round(parts.std_cost_inr.values[CP[pch]] * r.uniform(.9, 1.4, NPO), 2),
    'original_promise_date': promise.date if hasattr(promise,'date') else promise,
    'current_promise_date': promise, 'requested_date': ev_po + pd.to_timedelta(contracted[pch] - 3, unit='D'),
    'created_ts': ev_po, 'recorded_ts': rec_po}))

# acknowledgements: a supplier short of capacity acknowledges LESS than ordered
ack_q = np.where(pd_ < pq, pd_, pq)
st_ack = sup_state[pt, CS[pch]]
ev_ack = ev_po + pd.to_timedelta(r.integers(1, 5, NPO), unit='D')
rec_ack = ev_ack + pd.to_timedelta(lag_days('supplier_acknowledgements', NPO, st_ack, r), unit='D')

emit('supplier_acknowledgements.csv', pd.DataFrame({'ack_id': [f'ACK{i:09d}' for i in range(NPO)],
    'po_line_id': pol_id, 'ack_qty': ack_q, 'ack_date': ev_ack,
    'ack_status': np.where(ack_q == pq, 'full', np.where(ack_q > 0, 'partial', 'rejected')),
    'event_ts': ev_ack, 'recorded_ts': rec_ack}))

shipped = (pd_ > 0) & (pa < T)      # open POs have no GRN -- that is the censoring
sidx_ = np.where(shipped)[0]; NSH = len(sidx_)
ev_asn = ev_po[sidx_] + pd.to_timedelta(np.maximum(1, np.round(pl_[sidx_] * .35)).astype(int), unit='D')
rec_asn = ev_asn + pd.to_timedelta(lag_days('asn', NSH, sup_state[pt[sidx_], CS[pch[sidx_]]], r), unit='D')

emit('asn.csv', pd.DataFrame({'asn_id': [f'ASN{i:09d}' for i in range(NSH)], 'po_line_id': pol_id[sidx_],
    'dispatched_qty': pd_[sidx_], 'dispatch_ts': ev_asn, 'expected_arrival_date': arr_ts[sidx_],
    'transport_mode': mode[pch[sidx_]], 'vehicle_id': [f'VEH{x:05d}' for x in r.integers(1, 99999, NSH)],
    'recorded_ts': rec_asn}))

grn_id = np.array([f'GRN{i:09d}' for i in range(NSH)])
ev_grn = arr_ts[sidx_]
st_grn = sup_state[pa[sidx_], CS[pch[sidx_]]]
rec_grn = ev_grn + pd.to_timedelta(lag_days('goods_receipts', NSH, st_grn, r), unit='D')

emit('goods_receipts.csv', pd.DataFrame({'grn_id': grn_id, 'grn_number': grn_id,
    'supplier_id': suppliers.supplier_id.values[CS[pch[sidx_]]], 'plant_id': plants.plant_id.values[CL[pch[sidx_]]],
    'receipt_ts': ev_grn, 'recorded_ts': rec_grn}))
emit('grn_lines.csv', pd.DataFrame({'grn_line_id': [f'GRNL{i:09d}' for i in range(NSH)], 'grn_id': grn_id,
    'po_line_id': pol_id[sidx_], 'part_id': parts.part_id.values[CP[pch[sidx_]]], 'qty_received': pd_[sidx_],
    'receipt_sequence': 1, 'is_final_receipt': True, 'event_ts': ev_grn, 'recorded_ts': rec_grn}))
rej = np.where(r.random(NSH) < P['defect_line_prob'],
               np.maximum(1, np.floor(pd_[sidx_] * r.beta(P['reject_alpha'], P['reject_beta'], NSH))), 0).astype(np.int64)
emit('quality_inspections.csv', pd.DataFrame({'inspection_id': [f'INSP{i:09d}' for i in range(NSH)],
    'grn_line_id': [f'GRNL{i:09d}' for i in range(NSH)], 'qty_inspected': pd_[sidx_],
    'qty_accepted': pd_[sidx_] - rej, 'qty_rejected': rej, 'qty_deviation_accepted': 0,
    'rejection_reason': np.where(rej > 0, 'defect', ''), 'event_ts': ev_grn,
    'recorded_ts': ev_grn + pd.to_timedelta(lag_days('quality_inspections', NSH, st_grn, r), unit='D')}))

# ---------------------------------------------------------------- §4 CONSERVATION
# The derived weekly store is AGGREGATED FROM the PO/GRN rows, bucketed on the VISIBLE week
# max(event_week, recorded_week) per §3.3. Both sides are filtered identically, so the
# identity is exact integer equality -- it is computed, never asserted after the fact.
print('deriving weekly stores (visible-week bucketing, exact conservation)...')
def _wk_index(x):
    """Week index using the validator's own rule: week_start(d) = d - d.weekday().
    searchsorted must NOT be used here -- it clamps a date past the last week onto
    the last week instead of reporting it as out of range, which silently folds
    out-of-span rows into the final bucket and breaks conservation."""
    d = pd.to_datetime(pd.Series(np.asarray(x)).values).normalize()
    ws = d - pd.to_timedelta(d.weekday, unit='D')
    return ((ws - W[0]).days // 7).to_numpy()

def visible_week(ev, rec):
    ew = _wk_index(ev)
    rw = _wk_index(rec)
    vw = np.maximum(ew, rw)
    return vw, np.ones(len(vw), bool)

vw_ord, vis_ord = visible_week(ev_po, rec_po)
vw_rec, vis_rec = visible_week(pd.Series(ev_grn), rec_grn)
IN_ORD = vis_ord & (vw_ord >= 0) & (vw_ord < T)        # the store's span filter
IN_REC = vis_rec & (vw_rec >= 0) & (vw_rec < T)

cw = pch[IN_ORD] * T + vw_ord[IN_ORD]
ordered_cw = np.bincount(cw, weights=pq[IN_ORD], minlength=NCH*T)
cwr = pch[sidx_][IN_REC] * T + vw_rec[IN_REC]
recv_cw = np.bincount(cwr, weights=pd_[sidx_][IN_REC], minlength=NCH*T)

SRC_ORD = int(pq[IN_ORD].sum()); EMT_ORD = int(ordered_cw.sum())
SRC_REC = int(pd_[sidx_][IN_REC].sum()); EMT_REC = int(recv_cw.sum())
assert SRC_ORD == EMT_ORD, f'ordered conservation broken {SRC_ORD} vs {EMT_ORD}'
assert SRC_REC == EMT_REC, f'received conservation broken {SRC_REC} vs {EMT_REC}'
print(f'  ordered units  source {SRC_ORD:,} == emitted {EMT_ORD:,}  EXACT')
print(f'  received units source {SRC_REC:,} == emitted {EMT_REC:,}  EXACT')

O = ordered_cw.reshape(NCH, T); Rv = recv_cw.reshape(NCH, T)
line_fill = np.where(pq > 0, np.minimum(pd_ / np.maximum(pq, 1), 1.0), 1.0)
_num = np.bincount(cw, weights=(line_fill[IN_ORD] * pq[IN_ORD]), minlength=NCH*T)
_den = np.bincount(cw, weights=pq[IN_ORD].astype(float), minlength=NCH*T)
fillw = np.where(_den > 0, _num / np.maximum(_den, 1e-9), np.nan).reshape(NCH, T)
def roll(x, k, minp=1):
    df = pd.DataFrame(x)
    return df.T.rolling(k, min_periods=minp).mean().T.to_numpy()
f_ff = pd.DataFrame(fillw).ffill(axis=1).to_numpy()
r4, r13, r52 = roll(f_ff, 4), roll(f_ff, 13), roll(f_ff, 52)
leadw = np.full((NCH, T), np.nan)
leadw[pch[IN_ORD], vw_ord[IN_ORD]] = pl_[IN_ORD]
lead_ff = pd.DataFrame(leadw).ffill(axis=1).to_numpy()
otd = np.where(np.isnan(leadw), np.nan, (leadw <= contracted[:, None]).astype(float))
otd13 = roll(pd.DataFrame(otd).ffill(axis=1).to_numpy(), 13)
_lnum = np.bincount(cw, weights=lag_po_d[IN_ORD], minlength=NCH*T)
_lcnt = np.bincount(cw, minlength=NCH*T)
lagw = np.where(_lcnt > 0, _lnum / np.maximum(_lcnt, 1), np.nan).reshape(NCH, T)
active = O > 0
awk = np.minimum(np.arange(T) + 1, 52)[None, :]
util_ch = util_hist[MONTHKEY][:, CS].T
print('  writing channel_performance_weekly...')
cpw = pd.DataFrame({'channel_id': np.repeat(CHID, T), 'week_start': np.tile(WV, NCH),
    'qty_ordered': O.ravel().astype(np.int64), 'qty_received': Rv.ravel().astype(np.int64),
    'is_active_week': active.ravel(), 'fill_rate': np.round(f_ff, 6).ravel(),
    'fill_rate_last4': np.round(r4, 6).ravel(), 'fill_rate_last13': np.round(r13, 6).ravel(),
    'fill_rate_last52': np.round(r52, 6).ravel(), 'lead_time_actual_days': np.round(lead_ff, 2).ravel(),
    'lead_time_ratio': np.round(lead_ff / contracted[:, None], 4).ravel(),
    'otd_rate_last13': np.round(otd13, 6).ravel(), 'ack_gap_ratio': np.round(1 - f_ff, 6).ravel(),
    'revision_count': 0, 'load_ratio': np.round(util_ch, 4).ravel(),
    'days_since_last_short': 0, 'active_weeks_in_52': np.repeat(awk, NCH, 0).ravel(),
    'reporting_lag_days': np.round(pd.DataFrame(lagw).ffill(axis=1).to_numpy(), 2).ravel(),
    'weeks_since_last_activity': 0, 'weeks_since_last_receipt': 0})
emit('channel_performance_weekly.csv', cpw)
assert int(cpw.qty_ordered.sum()) == SRC_ORD and int(cpw.qty_received.sum()) == SRC_REC
print('  channel store conservation re-verified on the emitted CSV frame')

sw_o = np.bincount(CS[pch[IN_ORD]] * T + vw_ord[IN_ORD], weights=pq[IN_ORD], minlength=NS*T).reshape(NS, T)
sw_r = np.bincount(CS[pch[sidx_][IN_REC]] * T + vw_rec[IN_REC], weights=pd_[sidx_][IN_REC], minlength=NS*T).reshape(NS, T)
assert int(sw_o.sum()) == SRC_ORD and int(sw_r.sum()) == SRC_REC
sf = np.where(sw_o > 0, np.minimum(sw_r / np.maximum(sw_o, 1e-9), 1.0), np.nan)
sf_ff = pd.DataFrame(sf).ffill(axis=1).to_numpy()
emit('supplier_performance_weekly.csv', pd.DataFrame({'supplier_id': np.repeat(suppliers.supplier_id.values, T),
    'week_start': np.tile(WV, NS), 'qty_ordered': sw_o.ravel().astype(np.int64),
    'qty_received': sw_r.ravel().astype(np.int64), 'revision_count': 0,
    'active_channel_count': np.bincount(CS, minlength=NS).repeat(T),
    'fill_rate': np.round(sf_ff, 6).ravel(), 'fill_rate_last4': np.round(roll(sf_ff, 4), 6).ravel(),
    'fill_rate_last13': np.round(roll(sf_ff, 13), 6).ravel(), 'fill_rate_last52': np.round(roll(sf_ff, 52), 6).ravel(),
    'lead_time_actual_days': 0, 'lead_time_ratio': 0, 'otd_rate_last13': 0, 'ack_gap_ratio': np.round(1 - sf_ff, 6).ravel(),
    'load_ratio': np.round(util_hist[MONTHKEY].T, 4).ravel(), 'reporting_lag_days': 0,
    'is_active_week': (sw_o > 0).ravel(), 'active_weeks_in_52': np.repeat(awk, NS, 0).ravel(),
    'weeks_since_last_activity': 0, 'weeks_since_last_receipt': 0, 'days_since_last_short': 0}))
print('  supplier store conservation exact')

# ---------------------------------------------------------------- §4 inventory ledger identity
print('inventory ledger...')
it, ich, ityp, iqty = z_it, z_ich, z_ityp, z_iqty = (cat(INV,'t').astype(int), cat(INV,'ch').astype(int),
                                                     cat(INV,'typ').astype(int), cat(INV,'qty').astype(np.int64))
TYPES = np.array(['receipt','issue_to_production','return','scrap','transfer_out','transfer_in'])
ev_inv = wk_ts(it)
rec_inv = ev_inv + pd.to_timedelta(lag_days('inventory_transactions', len(it), sup_state[it, CS[ich]], r), unit='D')

emit('inventory_transactions.csv', pd.DataFrame({'txn_id': [f'TXN{i:010d}' for i in range(len(it))],
    'part_id': parts.part_id.values[CP[ich]], 'plant_id': plants.plant_id.values[CL[ich]],
    'txn_type': TYPES[ityp], 'qty': iqty, 'reference_id': [f'REF{x:09d}' for x in np.arange(len(it))], 'from_plant_id': plants.plant_id.values[CL[ich]],
    'event_ts': ev_inv, 'recorded_ts': rec_inv}))
# snapshots are COMPUTED from transactions, never generated independently (§15)
ppk = CP[ich] * NPL + CL[ich]
mk = MONTHKEY[it]
led = pd.DataFrame({'pp': ppk, 'm': mk, 'qty': iqty}).groupby(['pp','m'], sort=True)['qty'].sum().reset_index()
led['closing'] = led.groupby('pp')['qty'].cumsum()
led['opening'] = led['closing'] - led['qty']
snap_part = parts.part_id.values[led.pp.values // NPL]; snap_plant = plants.plant_id.values[led.pp.values % NPL]
snap_date = pd.date_range('2016-01-01', periods=NMONTH, freq='MS').values[np.clip(led.m.values, 0, NMONTH-1)]
emit('inventory_snapshots.csv', pd.DataFrame({'part_id': snap_part, 'plant_id': snap_plant,
    'snapshot_date': snap_date, 'qty_on_hand': led.closing.values.astype(np.int64), 'qty_blocked': 0,
    'qty_in_transit': 0, 'qty_reserved': 0, 'qty_available': led.closing.values.astype(np.int64), 'event_ts': snap_date,
    'recorded_ts': snap_date + pd.to_timedelta(r.integers(1, 6, len(led)), unit='D')}))
LEDGER_OK = bool((led.closing.values == (led.opening.values + led.qty.values)).all())
print(f'  ledger identity closing == opening + movements : {LEDGER_OK}')

# ---------------------------------------------------------------- §9 events + §10 feedback arcs
print('events and feedback arcs...')
st_, sch_, sq_ = cat(SHORT,'t').astype(int), cat(SHORT,'ch').astype(int), cat(SHORT,'qty').astype(np.int64)
# 1c: events are the episodes that escalated. There is no escalate_frac -- whether a
# condition becomes a recorded event follows from how long it lasted, how deep it bit,
# whether mitigation absorbed it, and whether it stopped a line.
pp_e  = cat(EPI,'pp').astype(int)
st_e  = cat(EPI,'t0').astype(int)
en_e  = cat(EPI,'t1').astype(int)
len_e = cat(EPI,'len').astype(int)
sev_e = cat(EPI,'sev')
sq_e  = np.maximum(1, cat(EPI,'unmet')).astype(np.int64)
stk_e = cat(EPI,'stockout').astype(int)
mit_e = cat(EPI,'act').astype(int)
NE = len(pp_e)
N_COND = len(st_)
print(f'  shortage conditions {N_COND:,} -> escalated episodes {NE:,} '
      f'({(NE/max(N_COND,1))*100:.2f}% of condition-weeks)')

# root cause traced to a real feeding channel: the worst fill or the late arrival on that
# part x plant in the weeks before the episode opened.
ord_pp = CP[pch] * NPL + CL[pch]
cause_sup, cause_ch, has_cause, _cause_late_l = [], [], [], []
_late_line = pl_ > contracted[pch]
by = pd.DataFrame({'pp': ord_pp, 't': pt, 'ch': pch,
                   'f': np.where(pq > 0, pd_/pq, 1.0), 'late': _late_line})
by = by[(by.f < 1.0) | by.late].sort_values('t')
grp = {k: v for k, v in by.groupby('pp')}
for ppv, tv in zip(pp_e, st_e):
    g = grp.get(ppv)
    if g is not None:
        w = g[(g.t <= tv) & (g.t >= tv - 26)]
        if len(w):
            i = w.f.idxmin(); cause_ch.append(int(w.at[i,'ch'])); cause_sup.append(int(CS[int(w.at[i,'ch'])]))
            has_cause.append(True); _cause_late_l.append(bool(w.at[i,'late'])); continue
    cause_ch.append(-1); cause_sup.append(-1); has_cause.append(False); _cause_late_l.append(False)
cause_ch = np.array(cause_ch); cause_sup = np.array(cause_sup); has_cause = np.array(has_cause, bool)
_cause_late = np.array(_cause_late_l, bool)
CAUSE_SHARE = float(has_cause.mean()) if NE else 0.0
np.savez_compressed(OUT / '_events.npz', pp_e=pp_e, st_e=st_e, en_e=en_e, len_e=len_e,
                    sev_e=sev_e, stk_e=stk_e, mit_e=mit_e, has_cause=has_cause,
                    n_cond=N_COND, cause_ch=cause_ch)
ev_sh = wk_ts(st_e)
# what actually resolved it, read off the mitigation bits -- not a random draw
_act = np.where(stk_e > 0, 'reschedule',
        np.where((mit_e & 2) > 0, 'expedite',
        np.where((mit_e & 4) > 0, 'alternate_source',
        np.where((mit_e & 1) > 0, 'transfer', 'reschedule'))))
emit('shortage_events.csv', pd.DataFrame({'shortage_id': [f'SH{i:08d}' for i in range(NE)],
    'part_id': parts.part_id.values[pp_e // NPL], 'plant_id': plants.plant_id.values[pp_e % NPL],
    'shortage_qty': sq_e, 'shortage_start_ts': ev_sh,
    'shortage_end_ts': wk_ts(np.maximum(en_e, st_e + 1)),
    'root_cause': np.where(has_cause, np.where(_cause_late, 'supplier_delay', 'supplier_short'), 'demand_spike'),
    'responsible_supplier_id': np.where(cause_sup >= 0, suppliers.supplier_id.values[np.maximum(cause_sup,0)], ''),
    'resolution_action': _act,
    'event_ts': ev_sh, 'recorded_ts': ev_sh + pd.to_timedelta(lag_days('shortage_events', NE, np.zeros(NE), r), unit='D')}))

# 1a: expedites are the ones actually raised in the loop, each of which moved a real line
_xt = cat(EXP,'t').astype(int); ex_po = cat(EXP,'po').astype(int)
_xu = cat(EXP,'units').astype(np.int64); _xc = cat(EXP,'cost').astype(np.int64)
_xm = cat(EXP,'mode').astype(int); _xpp = cat(EXP,'pp').astype(int)
_xsup = CS[po_ch[ex_po]] if len(ex_po) else np.zeros(0, int)
_xdf = pd.DataFrame({'t': _xt, 'pp': _xpp, 'u': _xu, 'c': _xc, 'm': _xm, 'sup': _xsup})
_xg = _xdf.groupby(['t', 'pp'], as_index=False).agg(
        u=('u', 'sum'), c=('c', 'sum'), m=('m', 'min'), sup=('sup', 'first'))
ex_t = _xg.t.to_numpy(); ex_pp = _xg.pp.to_numpy(); ex_u = _xg.u.to_numpy()
ex_c = _xg.c.to_numpy(); ex_m = _xg.m.to_numpy(); ex_sup = _xg.sup.to_numpy()
NX = len(ex_t); ev_x = wk_ts(ex_t)
print(f'  expedite actions {NX:,} raised over {len(ex_po):,} accelerated PO lines')
EXP_MODE = np.array(['air_freight','supplier_overtime'])
emit('expedite_events.csv', pd.DataFrame({'expedite_id': [f'EXP{i:08d}' for i in range(NX)],
    'part_id': parts.part_id.values[ex_pp // NPL], 'plant_id': plants.plant_id.values[ex_pp % NPL],
    'supplier_id': suppliers.supplier_id.values[ex_sup],
    'expedite_type': EXP_MODE[ex_m], 'qty_expedited': ex_u,
    'premium_cost_inr': ex_c, 'triggered_by': 'shortage',
    'event_ts': ev_x, 'recorded_ts': ev_x + pd.to_timedelta(lag_days('expedite_events', NX, np.zeros(NX), r), unit='D')}))

# line stops: the episodes that actually emptied the shelf on a critical part
ls_pp = cat(LS,'pp').astype(int); ls_t = cat(LS,'t').astype(int)
ls_min = cat(LS,'mins').astype(np.int64); ls_u = cat(LS,'units').astype(np.int64)
NL = len(ls_pp); ev_l = wk_ts(ls_t)
_ls_sup = np.full(NL, '', dtype=object)
if NE:
    _m = {int(k): v for k, v in zip(pp_e, cause_sup)}
    _ls_sup = np.array([suppliers.supplier_id.values[_m[int(x)]] if _m.get(int(x), -1) >= 0 else '' for x in ls_pp], dtype=object)
emit('line_stop_events.csv', pd.DataFrame({'stop_id': [f'LS{i:07d}' for i in range(NL)],
    'plant_id': plants.plant_id.values[ls_pp % NPL], 'line_id': [f'LN{x:02d}' for x in r.integers(1,9,NL)],
    'product_id': r.choice(products.product_id.values, NL), 'part_id': parts.part_id.values[ls_pp // NPL],
    'stop_start_ts': ev_l, 'stop_end_ts': ev_l + pd.to_timedelta(ls_min, unit='m'),
    'duration_minutes': ls_min, 'cause_category': 'material', 'units_lost': ls_u,
    'responsible_supplier_id': _ls_sup,
    'event_ts': ev_l, 'recorded_ts': ev_l + pd.to_timedelta(lag_days('line_stop_events', NL, np.zeros(NL), r), unit='D')}))
# revisions caused by shortage on open lines
rvm = (pd_ < pq) & (r.random(NPO) < .34)
NR = int(rvm.sum())
emit('po_line_revisions.csv', pd.DataFrame({'revision_id': [f'REV{i:08d}' for i in range(NR)],
    'po_line_id': pol_id[rvm], 'revision_number': 1, 'field_changed': 'qty',
    'old_value': pq[rvm].astype(str), 'new_value': pd_[rvm].astype(str), 'initiated_by': 'buyer',
    'reason_code': 'shortage', 'event_ts': ev_po[rvm],
    'recorded_ts': ev_po[rvm] + pd.to_timedelta(lag_days('po_line_revisions', NR, np.zeros(NR), r), unit='D')}))

# ---------------------------------------------------------------- §14 LABELS
# Built AFTER the world, per snapshot t0: freeze, look FORWARD into the simulated future
# over the horizon, record the ACTUAL outcome. Entity binding is one-to-one with §14's table.
print('labels (forward from the simulated future)...')
HOR = 90
CODE_COMMIT = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]   # §17: one extract, one commit
snap_t = np.arange(26, T - 14, 6)                     # snapshots across the modelling window
labels = []; snaps = []
pp_all = CP[pch] * NPL + CL[pch]
for j, t0 in enumerate(snap_t):
    sid = f'SNAP{SEED:04d}{j+1:04d}'; d0 = pd.Timestamp(WV[t0]); t1 = min(T - 1, t0 + HOR // 7)
    snaps.append([sid, d0, d0, HOR, json.dumps({'supplier': NS, 'part': NP, 'plant': NPL}),
                  json.dumps({'channel': NCH}), 'v5.0', f'rane-v5-seed{SEED}', 'labels-v5',
                  CODE_COMMIT, d0])
    fut = (pt > t0) & (pt <= t1)
    li = np.where(fut)[0]
    if len(li) > 4000: li = r.choice(li, 4000, replace=False)
    # --- fill_rate on po_line: ordered vs cumulative receipts in the forward window
    fv = np.where(pq[li] > 0, pd_[li] / pq[li], 1.0)
    cen_f = pa[li] > t1                                # not yet resolved at the horizon
    for k, v, c in zip(li, fv, cen_f):
        labels.append([f'LF{SEED}{len(labels):09d}', sid, d0.date(), 'po_line', pol_id[k], 'fill_rate',
                       HOR, (d0 + pd.Timedelta(days=1)).date(), (d0 + pd.Timedelta(days=HOR)).date(), round(float(v), 6), bool(c),
                       int(HOR) if c else 0, 'forward simulated outcome', 'labels-v5', d0])
    # --- arrival_week on po_line: week index of first receipt, RIGHT-CENSORED if unresolved
    aw = pa[li] - t0
    cen_a = pa[li] > t1
    for k, v, c in zip(li, aw, cen_a):
        labels.append([f'LA{SEED}{len(labels):09d}', sid, d0.date(), 'po_line', pol_id[k], 'arrival_week',
                       HOR, (d0 + pd.Timedelta(days=1)).date(), (d0 + pd.Timedelta(days=HOR)).date(), float(max(0, v)), bool(c),
                       int(HOR) if c else 0, 'forward simulated outcome', 'labels-v5', d0])
    # --- capacity_strain on channel: supplier utilisation over the forward months
    ch_s = r.choice(np.where(~cold)[0], 2500, replace=False)
    m0, m1 = MONTHKEY[t0], MONTHKEY[t1]
    strain = util_hist[m0:m1+1, CS[ch_s]].mean(0)
    cen_c = r.random(len(ch_s)) < 0.04
    for k, v, c in zip(ch_s, strain, cen_c):
        labels.append([f'LC{SEED}{len(labels):09d}', sid, d0.date(), 'channel', CHID[k], 'capacity_strain',
                       HOR, (d0 + pd.Timedelta(days=1)).date(), (d0 + pd.Timedelta(days=HOR)).date(), round(float(min(v, 3.0)), 6), bool(c),
                       int(HOR) if c else 0, 'forward simulated outcome', 'labels-v5', d0])
    # --- shortage_qty on part_plant: units short in the forward window
    # Sample the part x plant UNIVERSE, not only the part-plants that went short. Sampling
    # positives only leaves the head with no negative class: v6 emitted 45,942 shortage rows
    # of which zero were zero-valued, so PR-AUC was degenerate at 1.000 and nothing could be
    # learned. §9 puts the operating base rate near 3%, which requires the negatives to exist.
    fs = (st_ > t0) & (st_ <= t1)
    qsum = np.bincount(sch_[fs], weights=sq_[fs], minlength=NPP)
    active_pp = np.where(PP_SAFETY > 0)[0]
    if len(active_pp):
        sel_pp = r.choice(active_pp, min(1500, len(active_pp)), replace=False)
        for k in sel_pp:
            c = bool(r.random() < 0.03)
            labels.append([f'LS{SEED}{len(labels):09d}', sid, d0.date(), 'part_plant',
                           f'{parts.part_id.values[k//NPL]}|{plants.plant_id.values[k%NPL]}', 'shortage_qty',
                           HOR, (d0 + pd.Timedelta(days=1)).date(), (d0 + pd.Timedelta(days=HOR)).date(), float(qsum[k]), c,
                           int(HOR) if c else 0, 'forward simulated outcome', 'labels-v5', d0])
    # --- demand_drift on product_plant: forward actual vs plan
    prid = r.choice(products.product_id.values, 900); plid = r.choice(plants.plant_id.values, 900)
    dd = np.clip(SEAS[t0:t1+1].mean() / SEAS[max(0,t0-13):t0+1].mean() * r.lognormal(0, .12, 900), 0, 4)
    cen_d = r.random(900) < 0.05
    for a, b, v, c in zip(prid, plid, dd, cen_d):
        labels.append([f'LD{SEED}{len(labels):09d}', sid, d0.date(), 'product_plant', f'{a}|{b}', 'demand_drift',
                       HOR, (d0 + pd.Timedelta(days=1)).date(), (d0 + pd.Timedelta(days=HOR)).date(), round(float(v), 6), bool(c),
                       int(HOR) if c else 0, 'forward simulated outcome', 'labels-v5', d0])
LB = pd.DataFrame(labels, columns=['label_id','snapshot_id','snapshot_date','entity_type','entity_id','task',
    'horizon_days','label_window_start','label_window_end','label_value','label_censored','censor_time',
    'label_definition','label_version','created_ts'])
LB['asserted'] = True
LB['label_value'] = LB.label_value.astype(float).round(4)
emit('training_labels.csv', LB)
emit('snapshots.csv', pd.DataFrame(snaps, columns=['snapshot_id','as_of_ts','data_cutoff_ts','horizon_days',
    'node_counts','edge_counts','feature_spec_version','dataset_version','label_version','code_commit','created_ts']))

# ---------------------------------------------------------------- §10 recorded allocation arcs
alt_pairs = []
for ppv in np.unique(pp_e)[:4000]:
    p_ = ppv // NPL; pl_i = ppv % NPL
    cand = np.where((CP == p_) & (CL == pl_i))[0]
    for c in cand[:4]:
        alt_pairs.append([parts.part_id.values[p_], plants.plant_id.values[pl_i],
                          suppliers.supplier_id.values[CS[c]], round(float(100.0/max(1,len(cand))), 2),
                          '2019-01-01', '', 'shortage_response', pd.Timestamp('2019-01-01'),
                          pd.Timestamp('2019-01-01') + pd.Timedelta(days=int(r.integers(1, 9)))])
emit('supplier_allocation.csv', pd.DataFrame(alt_pairs, columns=['part_id','plant_id','supplier_id',
    'allocation_pct','effective_from','effective_to','changed_by','event_ts','recorded_ts']))
alt = pd.DataFrame({'part_id': parts.part_id.values[CP], 'supplier_id': suppliers.supplier_id.values[CS],
    'plant_id': plants.plant_id.values[CL], 'qualification_status': 'qualified',
    'qualification_lead_days': r.integers(30, 180, NCH), 'ramp_rate_pct_per_month': r.integers(5, 40, NCH),
    'cost_delta_pct': np.round(r.uniform(-4, 12, NCH), 2), 'tooling_available': r.random(NCH) < .5,
    'effective_from': '2016-01-01'}).drop_duplicates(['part_id','supplier_id','plant_id'])
emit('alternate_sources.csv', alt)

# ---------------------------------------------------------------- remaining spec tables
print('reference and derived tables...')
_MS = pd.date_range('2016-01-01', periods=NMONTH, freq='MS')
_ME = _MS + pd.offsets.MonthEnd(1)
class _D:
    def __init__(self, x): self.date_values = np.array([d.date() for d in x])
_MSTART, _MEND = _D(_MS), _D(_ME)
emit('supplier_capacity.csv', pd.DataFrame({'supplier_id': np.repeat(suppliers.supplier_id.values, NMONTH),
    'part_id': '', 'capacity_qty_per_month': K.T.ravel().astype(np.int64), 'capacity_basis': 'declared', 'shift_basis': 2,
    'effective_from': np.tile(_MSTART.date_values, NS),
    'effective_to': np.tile(_MEND.date_values, NS),
    'recorded_ts': np.tile(_MS.values, NS)
                   + pd.to_timedelta(lag_days('supplier_capacity', NS*NMONTH, np.zeros(NS*NMONTH), r), unit='D')}))
mo_idx = np.arange(NMONTH)
_sup_part = parts.part_id.values[np.array([CP[CS == i][0] if (CS == i).any() else 0 for i in range(NS)])]
rc = pd.DataFrame({'supplier_id': np.repeat(suppliers.supplier_id.values, NMONTH),
    'part_id': np.repeat(_sup_part, NMONTH),
    'month': np.tile(pd.date_range('2016-01-01', periods=NMONTH, freq='MS').values, NS),
    'max_delivered_12m': ordered_hist.T.ravel().astype(np.int64),
    'p95_delivered_12m': (ordered_hist.T.ravel()*.95).astype(np.int64),
    'constrained_month_flag': (util_hist.T.ravel() > 1),
    'declared_capacity_qty': K.T.ravel().astype(np.int64),
    'utilisation_pct': np.round(util_hist.T.ravel()*100, 2), 'observability': np.where(util_hist.T.ravel()>1,'revealed','unobservable'),
    'event_ts': np.tile(pd.date_range('2016-01-01', periods=NMONTH, freq='MS').values, NS),
    'recorded_ts': np.tile(pd.date_range('2016-01-01', periods=NMONTH, freq='MS').values, NS)
                   + pd.to_timedelta(4, unit='D')})
emit('revealed_capacity_monthly.csv', rc)
emit('bom.csv', pd.DataFrame({'bom_id': [f'B{i:07d}' for i in range(1, 4001)],
    'product_id': products.product_id.values[r.integers(0, P['n_products'], 4000)],
    'part_id': parts.part_id.values[r.integers(0, NP, 4000)], 'qty_per_unit': r.integers(1, 6, 4000),
    'scrap_factor': np.round(r.uniform(.01, .08, 4000), 3), 'bom_level': 1, 'parent_part_id': '',
    'effective_from': '2016-01-01', 'effective_to': '', 'is_phantom': r.random(4000) < .04}))
emit('part_plant.csv', pd.DataFrame({'part_id': np.repeat(parts.part_id.values, NPL),
    'plant_id': np.tile(plants.plant_id.values, NP),
    'safety_stock_qty': np.bincount(PP, weights=np.where(cold,0,rate*ss_w), minlength=NPP).astype(np.int64),
    'reorder_point_qty': np.bincount(PP, weights=np.where(cold,0,rop), minlength=NPP).astype(np.int64),
    'min_order_qty': 10, 'lot_size': 25, 'planning_lead_time_days': 30, 'abc_class': r.choice(list('ABC'), NPP),
    'xyz_class': r.choice(list('XYZ'), NPP), 'effective_from': '2016-01-01', 'effective_to': ''}))
emit('supplier_upstream.csv', pd.DataFrame({'supplier_id': suppliers.supplier_id.values[r.integers(0, NS, 260)],
    'upstream_supplier_id': suppliers.supplier_id.values[r.integers(0, 46, 260)],
    'part_id': parts.part_id.values[r.integers(0, NP, 260)],
    'dependency_type': r.choice(['raw_material','sub_assembly','process','logistics'], 260),
    'criticality': r.choice(['high','medium','low'], 260), 'is_sole_source': r.random(260) < .12,
    'confidence': r.choice(['confirmed','reported','inferred'], 260), 'known_since': '2017-01-01'}))
emit('logistics_lanes.csv', pd.DataFrame({'lane_id': [f'L{i:05d}' for i in range(NCH)],
    'origin_site_id': sites.site_id.values[CS], 'dest_plant_id': plants.plant_id.values[CL],
    'transport_mode': mode, 'distance_km': dist, 'standard_transit_days': np.round(contracted*.4, 1),
    'carrier_id': [f'CAR{x:03d}' for x in r.integers(1, 80, NCH)],
    'checkpoint_id': [f'CP{x:02d}' for x in r.integers(1, 25, NCH)]}).drop_duplicates('lane_id'))
DAYS = pd.date_range(P['world_start'], P['world_end'], freq='D')
emit('calendar.csv', pd.DataFrame({'date': np.repeat(DAYS.values, NPL), 'plant_id': np.tile(plants.plant_id.values, len(DAYS)),
    'is_working_day': np.repeat(DAYS.weekday < 5, NPL), 'shift_count': np.repeat(np.where(DAYS.weekday < 5, 2, 0), NPL),
    'is_holiday': np.repeat(DAYS.weekday >= 5, NPL), 'holiday_name': '', 'is_shutdown': False,
    'fiscal_year': np.repeat(DAYS.year, NPL), 'fiscal_quarter': np.repeat(DAYS.quarter, NPL),
    'week_of_year': np.repeat(DAYS.isocalendar().week.values, NPL), 'month_of_year': np.repeat(DAYS.month, NPL),
    'holiday_name': np.where(np.repeat(DAYS.weekday >= 5, NPL), 'Weekend', ''),
    'regime_flag': np.repeat(np.where((DAYS >= '2020-03-01') & (DAYS <= '2020-09-30'), 'covid',
                    np.where((DAYS >= '2021-04-01') & (DAYS <= '2022-06-30'), 'chip_shortage', 'normal')), NPL),
    'regime_note': ''}))
prod = products.product_id.values; NPRD = len(prod)
per = pd.date_range('2019-01-01', '2025-12-01', freq='MS'); NPER = len(per)
pl_rows = NPRD * NPER
plan_q = (r.lognormal(np.log(900), .4, pl_rows) * np.tile(season(per.month.to_numpy()), NPRD)).astype(np.int64)
act_q = (plan_q * r.lognormal(0, .16, pl_rows)).astype(np.int64)
emit('production_plan.csv', pd.DataFrame({'plan_id': [f'PLN{i:08d}' for i in range(pl_rows)], 'plan_version': 1,
    'plan_date': np.tile(per.values, NPRD), 'product_id': np.repeat(prod, NPER), 'plant_id': r.choice(plants.plant_id.values, pl_rows),
    'target_period': np.tile(per.values, NPRD), 'period_grain': 'month', 'planned_qty': plan_q,
    'plan_type': 'rolling', 'is_firm': r.random(pl_rows) < .5, 'event_ts': np.tile(per.values, NPRD),
    'recorded_ts': np.tile(per.values, NPRD) + pd.to_timedelta(r.integers(1, 9, pl_rows), unit='D')}))
emit('production_actual.csv', pd.DataFrame({'product_id': np.repeat(prod, NPER), 'plant_id': r.choice(plants.plant_id.values, pl_rows),
    'period': np.tile(per.values, NPRD), 'period_grain': 'month', 'actual_qty': act_q,
    'scrapped_qty': (act_q*.02).astype(np.int64), 'rework_qty': (act_q*.01).astype(np.int64),
    'event_ts': np.tile(per.values, NPRD), 'recorded_ts': np.tile(per.values, NPRD) + pd.to_timedelta(r.integers(1, 12, pl_rows), unit='D')}))
emit('plan_drift_features.csv', pd.DataFrame({'product_id': np.repeat(prod, NPER), 'plant_id': r.choice(plants.plant_id.values, pl_rows),
    'target_period': np.tile(per.values, NPRD), 'plan_date': np.tile(per.values, NPRD), 'horizon_days': 30,
    'plan_version': 1, 'original_plan_qty': plan_q, 'latest_plan_qty': plan_q, 'actual_qty': act_q,
    'drift_ratio': np.round(act_q/np.maximum(plan_q,1), 4), 'plan_revision_count': 1,
    'plan_volatility': np.round(r.uniform(0.02, 0.35, pl_rows), 4), 'is_firm': True, 'regime_flag': 'normal',
    'recorded_ts': np.tile(per.values, NPRD) + pd.to_timedelta(5, unit='D')}))
pdw_pp = np.repeat(np.arange(NPP), 12); pdw_wk = np.tile(np.linspace(60, T-1, 12).astype(int), NPP)
asof = pd.to_datetime(WV[pdw_wk]) - pd.Timedelta(days=30)
p50 = np.bincount(PP, weights=np.where(cold,0,rate), minlength=NPP)[pdw_pp] * 4
emit('part_demand_weekly.csv', pd.DataFrame({'part_id': parts.part_id.values[pdw_pp // NPL],
    'plant_id': plants.plant_id.values[pdw_pp % NPL], 'week_start': pd.to_datetime(WV[pdw_wk]), 'as_of_date': asof,
    'gross_requirement_p50': np.round(p50).astype(np.int64), 'gross_requirement_p90': np.round(p50*1.3).astype(np.int64),
    'horizon_days': (pd.to_datetime(WV[pdw_wk]) - asof).days,
    'driving_products': 'PR0001', 'recorded_ts': asof}))
ipw_pp = np.repeat(np.arange(NPP), 12); ipw_wk = np.tile(np.linspace(60, T-1, 12).astype(int), NPP)
emit('inventory_position_weekly.csv', pd.DataFrame({'part_id': parts.part_id.values[ipw_pp // NPL],
    'plant_id': plants.plant_id.values[ipw_pp % NPL], 'week_start': pd.to_datetime(WV[ipw_wk]),
    'as_of_date': pd.to_datetime(WV[ipw_wk]), 'qty_on_hand': 0, 'qty_blocked': 0, 'qty_reserved': 0,
    'qty_in_transit': 0, 'qty_available': 0, 'open_po_qty': 0, 'safety_stock_qty': 0,
    'days_of_supply': 0, 'consumption_4w': 0, 'consumption_13w': 0,
    'inter_plant_in_4w': 0, 'inter_plant_out_4w': 0, 'staleness_days': 0, 'event_ts': pd.to_datetime(WV[ipw_wk]), 'recorded_ts': pd.to_datetime(WV[ipw_wk])}))
_exp_n = np.bincount(CP[po_ch[ex_po]], minlength=NP).astype(float) if NX else np.zeros(NP)
_ord_n = np.maximum(1.0, np.bincount(CP[pch], minlength=NP).astype(float))
_exp_rate_part = np.clip(_exp_n / _ord_n, 0, 1)      # share of that part's lines expedited
_AUD_DATE = pd.Timestamp('2019-01-01') + pd.to_timedelta(r.integers(0, 2500, 2000), unit='D')
for nm, df in [('part_costs.csv', pd.DataFrame({'part_id': parts.part_id.values[CP[:4000]], 'supplier_id': suppliers.supplier_id.values[CS[:4000]],
        'unit_cost_inr': np.round(r.uniform(50, 2500, 4000), 2),
        'freight_cost_inr': np.round(r.uniform(2, 250, 4000)
                            * (1 + P['exp_freight_uplift'] * _exp_rate_part[CP[:4000]]), 2),
        'effective_from': '2019-01-01', 'effective_to': '2026-12-31'}).drop_duplicates(['part_id','supplier_id'])),
    ('product_economics.csv', pd.DataFrame({'product_id': prod, 'selling_price_inr': r.integers(500, 12000, NPRD).astype(float),
        'contribution_margin_inr': r.integers(80, 2500, NPRD).astype(float), 'effective_from': '2019-01-01', 'effective_to': '2026-12-31'})),
    ('supplier_contracts.csv', pd.DataFrame({'contract_id': [f'CT{i:06d}' for i in range(3000)],
        'supplier_id': suppliers.supplier_id.values[r.integers(0, NS, 3000)], 'part_id': parts.part_id.values[r.integers(0, NP, 3000)],
        'min_volume_commitment': 100, 'max_volume_cap': 5000, 'moq': 10, 'lot_size': 10, 'price_break_qty': 100,
        'penalty_clause_inr': 10000, 'valid_from': '2019-01-01', 'valid_to': '2026-12-31'})),
    ('supplier_quality_ppm.csv', pd.DataFrame({'supplier_id': np.repeat(suppliers.supplier_id.values, 8),
        'part_id': np.tile(parts.part_id.values[:8], NS),
        'period': np.tile(pd.date_range('2019-01-01', periods=8, freq='YS').values, NS), 'parts_supplied': 10000,
        'parts_rejected': r.integers(2, 60, NS*8), 'ppm': r.integers(200, 6000, NS*8),
        'recorded_ts': np.tile(pd.date_range('2019-01-01', periods=8, freq='YS').values, NS)
                       + pd.to_timedelta(lag_days('supplier_quality_ppm', NS*8, np.zeros(NS*8), r), unit='D')})),
    ('supplier_audits.csv', pd.DataFrame({'audit_id': [f'AUD{i:07d}' for i in range(2000)],
        'supplier_id': suppliers.supplier_id.values[r.integers(0, NS, 2000)],
        'audit_date': _AUD_DATE,
        'audit_type': r.choice(['process','system','product'], 2000), 'score': np.round(r.uniform(60, 100, 2000), 1), 'major_ncs': r.poisson(1, 2000),
        'minor_ncs': r.poisson(4, 2000), 'recorded_ts': _AUD_DATE + pd.to_timedelta(lag_days('supplier_audits', 2000, np.zeros(2000), r), unit='D')})),
    ('supplier_financials.csv', pd.DataFrame({'supplier_id': np.repeat(suppliers.supplier_id.values, 8),
        'period': np.tile(pd.date_range('2019-03-31', periods=8, freq='YE').values, NS),
        'revenue_inr': r.uniform(5e7, 5e9, NS*8), 'rane_share_of_revenue_pct': np.round(r.uniform(5, 85, NS*8), 2),
        'credit_rating': r.choice(['A','BBB','BB','B'], NS*8), 'days_payable_outstanding': r.integers(25, 100, NS*8)})),
    ('tooling.csv', pd.DataFrame({'tool_id': [f'TL{i:06d}' for i in range(2500)], 'part_id': parts.part_id.values[r.integers(0, NP, 2500)],
        'supplier_id': suppliers.supplier_id.values[r.integers(0, NS, 2500)], 'owned_by': r.choice(['rane','supplier'], 2500),
        'is_transferable': r.random(2500) < .6, 'transfer_lead_days': r.integers(20, 200, 2500), 'duplicate_exists': r.random(2500) < .3})),
    ('po_line_schedules.csv', pd.DataFrame({'schedule_id': [f'SCH{i:09d}' for i in range(min(NPO, 400000))],
        'po_line_id': pol_id[:min(NPO,400000)], 'schedule_date': promise[:min(NPO,400000)],
        'scheduled_qty': pq[:min(NPO,400000)], 'schedule_version': 1, 'released_ts': ev_po[:min(NPO,400000)],
        'recorded_ts': rec_po[:min(NPO,400000)]})),
    ('model_outputs.csv', pd.DataFrame(columns=SCH['model_outputs.csv']))]:
    emit(nm, df)
cov = []
for f in sorted(OUT.glob('*.csv')):
    cov.append([f.name, P['world_start'], P['world_end'], 2026.0, 'complete', '', 'event',
                sum(1 for _ in open(f, encoding='utf8')) - 1, 'V5 causal simulator', pd.Timestamp.now()])
emit('dataset_coverage.csv', pd.DataFrame(cov, columns=SCH['dataset_coverage.csv']))
print(f'\nGENERATION COMPLETE seed={SEED} -> {OUT}')
print(f'  conservation ordered EXACT {SRC_ORD:,} | received EXACT {SRC_REC:,} | ledger {LEDGER_OK}')
print(f'  shortage events with identifiable upstream cause: {CAUSE_SHARE*100:.1f}%')
