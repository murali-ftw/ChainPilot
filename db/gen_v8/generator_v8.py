"""ChainPilot x Rane -- V8 causal synthetic world generator.

Implements synthetic_rules.md. Generation order is
    REALISTIC MECHANISM -> RELATIONSHIP -> OUTCOME -> DISTRIBUTION

Only the LEFT column of synthetic_rules.md §13 is set here. Every right-column quantity
(fill masses, constrained share, event rates, late rate, zero-order share, seasonality
ratio, staleness) is MEASURED at the end and printed. None of them appears as a literal,
a clip, a target or a post-hoc adjustment.

V8 extends V7 (v8_generator_brief_and_validator.md). New mechanisms, all set-then-measure:
  §2.1 opening stock balance at a fixed pre-window date, rolled forward by the EXISTING
       inventory-transaction mechanism; every reported balance is an outcome of that roll.
  §2.2 forward requirement plan: a rolling 12-26 week horizon per published version, with
       the publish date carried, over the existing drift/revision mechanism.
  §2.5 the two missing feedback arcs -- shortage -> PO-line revision (a buyer acting on an
       expedite that was not raised or did not land) and sustained poor performance ->
       allocation share shifted away over months, which moves real ordered volume.
  §2.6 contract terms drawn from supplier_tier / supplier_type / business_class
       distributions. FEATURE-LEVEL ONLY -- see the banner written into
       allocation_disclaimer.txt; these terms are not quotable constraints.
  §2.7 NO cost or service parameter is synthesised. Carrying cost, ordering cost, shortage
       cost and freight structure are Rane business decisions and are absent by design.
"""
import numpy as np, pandas as pd, json, os, re, sys, argparse, hashlib, warnings
warnings.filterwarnings('ignore')
print = __import__('functools').partial(__builtins__.print if not isinstance(__builtins__, dict) else __builtins__['print'], flush=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=1001)
ap.add_argument('--out', default='db/gen_v8')
ap.add_argument('--spec', default='db/schema.sql')
ap.add_argument('--refdir', default='db')
ap.add_argument('--cap', type=float, default=8000.0)
# Order-policy cover. G1 asks for >= 6 PO lines per channel per year and the first v8 run
# measured 5.85-6.14 across five seeds -- four of them short. The lever is the order-up-to
# cover: order frequency goes as 1/cover, because the gap the policy refills is dhat*cover.
# 7.5 -> 6.5 is a MECHANISM change with the outcome re-measured, not a band moved.
ap.add_argument('--cover', type=float, default=6.5)
ap.add_argument('--seas', type=float, default=0.29)
ap.add_argument('--z', type=float, default=5.5)
# Scale overrides. The DEFAULTS are the readiness-gate G1 population (brief 2.4); the
# overrides exist so one code state can also emit a small smoke world. Any run below the
# G1 minimums is a smoke run and is labelled as such in manifest.json -- it is never a
# deliverable world.
ap.add_argument('--channels', type=int, default=16072)
ap.add_argument('--suppliers', type=int, default=420)
ap.add_argument('--parts', type=int, default=620)
ap.add_argument('--products', type=int, default=180)
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
    n_plants=7, n_suppliers=A.suppliers, n_parts=A.parts, n_products=A.products, n_customers=20,
    n_channels=A.channels, cold_start_channels=max(1, int(round(A.channels * 738 / 16072))),
    # capacity distribution per supplier tier (§13 left)
    cap_mu_log=np.log(A.cap), cap_sigma_log=0.40, cap_tier_bonus=0.35,
    # what a supplier DECLARES is not the latent capacity K. v7 published K itself as
    # declared_capacity_qty, which made capacity_strain (= ordered / K) computable in
    # closed form from two published columns -- a leak, not a task. v8 declares an
    # optimistic, annually-refreshed, noisy figure; K stays latent in _sim.npz only.
    cap_declared_bias=0.14, cap_declared_sigma=0.19, cap_declare_months=12,
    # order policy (§13 left) -- drives density AND idle share as OUTCOMES
    review_weeks=2, cover_weeks_mu=A.cover, cover_weeks_sd=A.cover*0.3, safety_cover_weeks=5.2,
    demand_rate_mu_log=np.log(11.0), demand_rate_sigma_log=1.05, demand_cv=0.30, service_z=A.z,
    # variance-preserving split of demand_rate_sigma_log:
    #   sigma_s^2 + sigma_i^2/(1-phi^2) = 0.85^2 + 0.1499^2/(1-0.97^2) = 0.7225 + 0.3800 = 1.1025 = 1.05^2
    rate_sigma_level=0.85, rate_phi=0.97, rate_sigma_innov=0.1499,
    forecast_alpha=0.10,         # MRP re-forecast: ~10-week EWMA of realised demand
    # lead-time process (§13 left)
    lead_base_frac=0.51, lead_sigma=0.34, congestion_beta=0.85, lead_disruption_mult=1.55,
    lead_short_beta=0.85,          # a part-covered line also ships later (§16 probe 3)
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
    lag_close_days=400.0,          # the accounting close: nothing is back-posted past it
    # §2.3 last bullet: a validity window is the one record that is legitimately FORWARD
    # dated. A capacity declaration and an allocation decision are registered before the
    # period they govern opens; the "lag" on those rows is negative by design, and the
    # validator checks that it is negative only there.
    announce_days=(5, 45),
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
    ls_plan_cut=0.18, ls_plan_weeks=6,   # §2.5 arc 5: a stop cuts the build that follows

    # ================= V8 §2.1 opening stock balance =================
    # A starting position per channel, sized off the safety-stock target and the demand
    # rate, with noise. It is posted to the ledger as an `adjustment` at a fixed
    # pre-window date; every later balance is the roll-forward of that posting plus the
    # receipts / issues / scrap / transfers the existing mechanism generates.
    opening_date='2015-12-28',
    open_cover_weeks_mu=6.2, open_cover_weeks_sd=2.4, open_cover_floor=0.5,
    open_zero_prob=0.035,          # part-plants that genuinely start the world empty

    # ================= V8 §2.2 forward requirement plan ==============
    plan_publish_weeks=4,          # a new plan version is cut every 4 weeks
    plan_horizon_min_w=12, plan_horizon_max_w=26,   # rolling horizon each version covers
    plan_bias_far=0.055,           # far-horizon optimism per 13 weeks of horizon
    plan_noise_near=0.07, plan_noise_far=0.26,      # revision noise, near vs far horizon
    plan_firm_weeks=6,             # inside this horizon the version is firm
    dem_publish_weeks=8,           # MRP netting cadence for part_demand_weekly

    # ================= V8 §2.5 shortage -> PO-line revision arc ======
    rev_pull_days=(4, 18),         # buyer pulls the promise date in by this many days
    rev_qty_step=0.35,             # or lifts the open quantity by this fraction
    rev_date_share=0.62,           # date revisions vs quantity revisions
    rev_max_lines_per_week=1200,
    rev_min_short_frac=0.05,       # an ack this far below the order is a recorded revision

    # ================= V8 §2.5 performance -> allocation arc =========
    alloc_review_months=3,         # buyers revisit the split quarterly
    alloc_window_w=26,             # trailing performance window they look at
    alloc_move_rate=0.22,          # how far the share moves toward the performance weights
    alloc_min_share=0.05,          # nobody is cut to zero inside a review cycle

    # ================= V8 §2.6 contract terms, FEATURES ONLY =========
    # Conditioned on supplier_tier / supplier_type / business_class, which already exist.
    # §2.6 and §2.7: these vary the features the arrival / fill / capacity heads read.
    # They are NOT inputs to a quotable allocation or delivery-schedule recommendation.
    ct_moq_base={'tier1': 48.0, 'tier2': 145.0},
    ct_moq_type={'manufacturer': 1.0, 'distributor': 0.42, 'specialty': 1.85},
    ct_moq_sigma=0.62,
    ct_lot_frac=(0.20, 0.55),      # lot size as a fraction of MOQ, uniform
    ct_cap_mult={'A': 3.4, 'B': 2.6, 'C': 2.0}, ct_cap_sigma=0.35,
    ct_commit_frac=(0.08, 0.40),   # min volume commitment as a fraction of the cap
    ct_penalty_log_mu=10.6, ct_penalty_log_sigma=0.95,
    ct_qual_p={'tier1': (0.86, 0.10, 0.04), 'tier2': (0.62, 0.24, 0.14)},

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
    """§2.3 recording lag: right-skewed, state-dependent, never negative.

    Two things bound the tail, and both are mechanisms rather than clips on the outcome:
      - the state response SATURATES. A disrupted plant records later, but a plant under
        five sigma of stress does not record five times later than one under one sigma;
        the backlog is worked off by people whose throughput has a ceiling.
      - a record that misses the ACCOUNTING CLOSE is posted at the close. The period is
        forced shut and the entry lands there. Without this, a lognormal sigma of 1.35 over
        a million draws put receipts on the books in 2029 for a world that ends in 2026.
    """
    med, sig, beta, _ = P['lag'][table]
    raw = rng.lognormal(np.log(med), sig, n) * (1 + beta * np.tanh(np.maximum(stress, 0)))
    return np.maximum(0.0, np.round(np.minimum(raw, P['lag_close_days']), 2))
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

_N_T2 = max(1, int(round(NS * 46 / 420)))
s_tier = np.where(np.arange(NS) < _N_T2, 'T2', 'T1')
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
# DECLARED capacity: refreshed every cap_declare_months, optimistic, noisy. This is the
# only capacity figure that reaches the CSVs.
DECL = np.zeros_like(K)
for m in range(NMONTH):
    if m % P['cap_declare_months'] == 0:
        _d = K[m] * np.exp(r.normal(P['cap_declared_bias'], P['cap_declared_sigma'], NS))
    DECL[m] = _d
DECL = np.maximum(np.round(DECL), 60.0)

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
# ---- §2.1 OPENING STOCK BALANCE -------------------------------------------------
# SET: the starting cover distribution (weeks of cover over the channel's demand rate,
# with noise, and a genuinely-empty slice). MEASURED: every balance from here on. The
# opening position is POSTED TO THE LEDGER as an `adjustment` at P['opening_date'], so
# the per-date balance is the roll-forward of that posting through the existing
# inventory-transaction mechanism and nothing else. v7 seeded on_hand directly and never
# posted it, which is why the ledger cumsum did not reproduce the simulator's balance.
_open_cover = np.maximum(P['open_cover_floor'],
                         r.normal(P['open_cover_weeks_mu'], P['open_cover_weeks_sd'], NCH))
_open_empty = r.random(NCH) < P['open_zero_prob']
on_hand = np.where(cold | _open_empty, 0, np.maximum(0, np.round(rate * _open_cover))).astype(np.int64)
OPEN_CH = on_hand.copy()
on_order = np.zeros(NCH, np.int64)
alt_boost = np.zeros(NCH)         # §10 alternate-sourcing volume shifted ONTO a channel

# ---- §2.5 arc 6: sustained poor performance -> allocation share -------------------
# The buyer's split of a part-plant across the channels that feed it. It starts even,
# is revisited every alloc_review_months against trailing delivered performance, and
# REWEIGHTS DEMAND between siblings at constant part-plant total -- so a supplier that
# keeps missing loses real volume in order history and the one that absorbs it sees its
# own load rise (arc 4, which runs through remK).
NOTCOLD = (~cold).astype(float)
_pp_live = np.maximum(1.0, np.bincount(PP, weights=NOTCOLD, minlength=NPP))
alloc_share = np.where(cold, 0.0, 1.0 / _pp_live[PP])
_alloc_decay = float(np.exp(-1.0 / P['alloc_window_w']))
perf_ord = np.zeros(NCH); perf_rec = np.zeros(NCH)
ALLOC = {k: [] for k in ('t', 'ch', 'share', 'reason')}
last_alloc_month = -99

# ---- §2.1 measured weekly position per part-plant (never set; checked against ledger)
PP_OH = np.zeros((T, NPP), np.int32); PP_OO = np.zeros((T, NPP), np.int32)
PP_ISS = np.zeros((T, NPP), np.int32); PP_TIN = np.zeros((T, NPP), np.int32)
PP_TOUT = np.zeros((T, NPP), np.int32); PP_DEM = np.zeros((T, NPP), np.int32)
PP_SS = np.zeros((T, NPP), np.int32)

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
po_rej = np.zeros(_CAP, np.int64)   # units rejected at receipt, reported by QC

# ---- §2.5 arc 2: shortage -> PO line revision -------------------------------------
REV = {k: [] for k in ('t', 'po', 'field', 'old', 'new', 'reason', 'by')}
promise_off = np.zeros(_CAP, np.int32)      # days the current promise has been pulled in
rev_mark = np.zeros(_CAP, bool)             # this line has already been revised once
rev_n = np.zeros(NCH, np.int32)             # revisions raised per channel, a real feature

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
util_obs_hist = np.zeros((NMONTH, NS))        # ordered / DECLARED -- the observable one
delivered_hist = np.zeros((NMONTH, NS)); month_delivered = np.zeros(NS)
sup_load_prev = np.zeros(NS)      # §10: previous month's utilisation degrades OTHER channels now

for t in range(T):
    perf_ord *= _alloc_decay; perf_rec *= _alloc_decay
    m = MONTHKEY[t]
    if m != cur_month:
        if cur_month >= 0:
            ordered_hist[cur_month] = month_ordered
            delivered_hist[cur_month] = month_delivered
            util_hist[cur_month] = month_ordered / K[cur_month]
            util_obs_hist[cur_month] = month_ordered / DECL[cur_month]
            sup_load_prev = util_hist[cur_month]
        # §10: last month's overload reduces this month's effective capacity
        carry = 1.0 - 0.18 * np.clip(sup_load_prev - 1.0, 0, 1.5)
        cur_month = m; remK = K[m].copy() * carry
        month_ordered = np.zeros(NS); month_delivered = np.zeros(NS)
        nwk = max(1, int((MONTHKEY == m).sum())); bank = remK / nwk
        # §2.5 arc 6: quarterly allocation review against trailing delivered performance
        if m - last_alloc_month >= P['alloc_review_months'] and t > 0:
            last_alloc_month = m
            seen = perf_ord > 1e-6
            tf = np.where(seen, perf_rec / np.maximum(perf_ord, 1e-9), np.nan)
            pp_mean = (np.bincount(PP, weights=np.where(seen, tf, 0.0), minlength=NPP)
                       / np.maximum(np.bincount(PP, weights=seen.astype(float), minlength=NPP), 1e-9))
            # weight = current share tilted by how far this channel sits from the
            # part-plant's own mean performance. Nobody is cut to zero in one review.
            tilt = np.where(seen, 1.0 + 2.5 * (tf - pp_mean[PP]), 1.0)
            tgt = np.maximum(alloc_share * tilt, 0.0)
            tgt_s = np.bincount(PP, weights=tgt, minlength=NPP)
            tgt = np.where(tgt_s[PP] > 1e-12, tgt / np.maximum(tgt_s[PP], 1e-12), alloc_share)
            new_share = (1 - P['alloc_move_rate']) * alloc_share + P['alloc_move_rate'] * tgt
            new_share = np.where(cold, 0.0, np.maximum(new_share, P['alloc_min_share'] / _pp_live[PP]))
            ns_s = np.bincount(PP, weights=new_share, minlength=NPP)
            new_share = np.where(ns_s[PP] > 1e-12, new_share / np.maximum(ns_s[PP], 1e-12), new_share)
            live = np.where(~cold)[0]
            ALLOC['t'].append(np.full(len(live), t)); ALLOC['ch'].append(live)
            ALLOC['share'].append(new_share[live] * 100.0)
            # 0 = unchanged, 1 = share cut (this supplier under-delivered), 2 = share lifted
            _d = new_share[live] - alloc_share[live]
            ALLOC['reason'].append(np.where(_d < -0.005, 1, np.where(_d > 0.005, 2, 0)).astype(np.int8))
            alloc_share = new_share
    # ---- demand and consumption
    _rate_eps = P['rate_phi'] * _rate_eps + r.normal(0, P['rate_sigma_innov'], NCH)
    dem_f = (rate * np.exp(_rate_eps) * SEAS[t] * TREND[t] * plant_f
             * (1 + dem_state[t][CL]) * r.lognormal(0, P['demand_cv'], NCH))
    dem_f = np.where(cold, 0, np.maximum(0, dem_f))
    # §2.5 arc 6: the allocation split moves volume BETWEEN the channels feeding a
    # part-plant, at constant part-plant total. It never creates or destroys requirement.
    _pp_tot = np.bincount(PP, weights=dem_f, minlength=NPP)
    _w = dem_f * alloc_share * _pp_live[PP]
    _pp_w = np.bincount(PP, weights=_w, minlength=NPP)
    dem_f = np.where(_pp_w[PP] > 1e-9, _w * _pp_tot[PP] / np.maximum(_pp_w[PP], 1e-9), dem_f)
    dem = dem_f.astype(np.int64)
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
        np.add.at(month_delivered, sup, deliv.astype(np.float64))
        bank = np.maximum(bank - used, 0) + remK / nwk      # unused capacity banks forward
        bank = np.minimum(bank, remK)
        # ---- §8 arrival: lane process, transit disruption, right-skewed
        base = contracted[idx] * P['lead_base_frac']
        lead = np.exp(r.normal(np.log(np.maximum(base, 3)), P['lead_sigma'], len(idx)))
        lead *= (1 + 0.55 * np.maximum(sup_state[t][sup], 0) + 0.9 * transit_state[t][CL[idx]]
                 + P['congestion_beta'] * np.clip(sup_load_prev[sup] - 0.70, 0, 2.0)
                 + (P['lead_disruption_mult'] - 1) * regime[t])
        # §16 probe 3: late and short are the SAME event seen twice. A supplier that can
        # only cover part of a line is a supplier whose queue is behind, and the covered
        # part moves at the back of that queue. v7 coupled the two only through the latent
        # stress state, which was far too weak to show up as co-occurrence on the line.
        _shortfall = 1.0 - np.divide(deliv.astype(float), np.maximum(q, 1.0))
        lead = lead * (1 + P['lead_short_beta'] * np.clip(_shortfall, 0, 1))
        lead = np.maximum(3, lead)
        arr_t = t + np.ceil(lead / 7).astype(int)      # may exceed T-1: those POs stay open
        nn = len(idx); sl = slice(npo, npo + nn)
        po_ch[sl] = idx; po_t[sl] = t; po_arr[sl] = arr_t; po_qty[sl] = qty[idx]
        po_deliv[sl] = deliv; po_lead[sl] = lead; po_pp[sl] = PP[idx]
        po_disp[sl] = t + np.maximum(1, np.ceil(lead * 0.35 / 7)).astype(int)
        po_open[sl] = True; po_ref[sl] = refuse
        # trailing performance the buyer can see at ack time: ordered vs committed
        perf_ord[idx] += qty[idx]; perf_rec[idx] += deliv
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
            # A rejection can never exceed what was delivered. Without the clamp, a line
            # that delivered nothing still drew a floor-of-one rejection, so `acc` went to
            # -1 and the channel's stock went negative -- and a negative `issue` on the next
            # week was then dropped by the `issue > 0` filter, silently putting one unit
            # into the simulator that never reached the ledger. One part-plant in 4,340,
            # found only by the exact §2.1 roll-forward check.
            rej = np.minimum(dq, np.where(r.random(len(dq)) < P['defect_line_prob'],
                             np.maximum(1, np.floor(dq * r.beta(P['reject_alpha'], P['reject_beta'], len(dq)))),
                             0)).astype(np.int64)
            acc = dq - rej
            np.add.at(on_hand, ci, acc); np.add.at(on_order, ci, -oq)
            po_open[bi] = False
            po_rej[bi] = rej            # the rejection quality_inspections will report
            # The receipt posts the FULL delivered quantity; the scrap posts the rejection
            # against it. v7 posted `acc` (already net of rejects) AND the scrap row, so
            # the ledger fell below the simulator's balance by the rejected units on every
            # part-plant that ever took a defect. v7's ledger check was
            # closing == opening + movements, which is true of any cumulative sum and
            # could not see it; §2.1's roll-forward check is what surfaced it.
            INV['t'].append(np.full(len(ci), t)); INV['ch'].append(ci)
            INV['typ'].append(np.full(len(ci), 0)); INV['qty'].append(dq)            # 0 = RECEIPT
            if rej.sum():
                s2 = rej > 0
                INV['t'].append(np.full(int(s2.sum()), t)); INV['ch'].append(ci[s2])
                INV['typ'].append(np.full(int(s2.sum()), 3)); INV['qty'].append(-rej[s2]) # 3 = SCRAP
    assert on_hand.min() >= 0, f'week {t}: stock went negative -- the ledger cannot represent it'
    if issue.any():
        s3 = issue != 0          # not `> 0`: a movement of any sign must reach the ledger
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

    # §2.1 MEASURED weekly position, recorded after every movement this week has landed.
    # The validator reconciles this against the independent cumulative sum of the emitted
    # inventory_transactions ledger; nothing here is written into the ledger.
    # np.rint, not a bare cast: bincount accumulates float64, so an exact integer total can
    # come back as ...999999998 and truncate one unit low. Three part-plants of 4,340 landed
    # exactly 1 short that way, and only the exact roll-forward check could see it.
    PP_OH[t] = np.rint(np.bincount(PP, weights=on_hand, minlength=NPP)).astype(np.int32)
    PP_OO[t] = np.rint(np.bincount(PP, weights=on_order, minlength=NPP)).astype(np.int32)
    PP_SS[t] = np.rint(PP_SAFETY).astype(np.int32)
    PP_ISS[t] = np.rint(np.bincount(PP, weights=issue, minlength=NPP)).astype(np.int32)
    PP_DEM[t] = np.rint(np.bincount(PP, weights=dem, minlength=NPP)).astype(np.int32)
    PP_TIN[t] = np.rint(np.bincount(PP, weights=give, minlength=NPP)).astype(np.int32)
    PP_TOUT[t] = np.rint(np.bincount(PP, weights=take, minlength=NPP)).astype(np.int32)

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

    # ---- §2.5 arc 2: SHORTAGE -> PO LINE REVISION.
    # The buyer whose expedite was never raised (no open line was a candidate) or did not
    # land still has to act: they revise the open line. A date revision pulls the promise
    # in -- a request, not a physical acceleration, so it moves `current_promise_date` and
    # nothing else, and the line is then measured late against the date the buyer asked
    # for. A quantity revision lifts the open ask. Both are recorded events with a
    # reason_code that traces back to the live episode.
    if len(shpp) and recent:
        cand2 = np.concatenate(recent)
        cand2 = cand2[po_open[cand2] & (po_arr[cand2] > t) & np.isin(po_pp[cand2], shpp)
                      & (~rev_mark[cand2])]
        if len(cand2):
            sev_c = sev_now[po_pp[cand2]]
            keep = r.random(len(cand2)) < np.clip(sev_c, 0, 1)
            cand2 = cand2[keep]
        if len(cand2) > P['rev_max_lines_per_week']:
            cand2 = r.choice(cand2, P['rev_max_lines_per_week'], replace=False)
        if len(cand2):
            is_date = r.random(len(cand2)) < P['rev_date_share']
            pull = r.integers(P['rev_pull_days'][0], P['rev_pull_days'][1] + 1, len(cand2))
            di = cand2[is_date]
            if len(di):
                promise_off[di] = pull[is_date]; rev_mark[di] = True
                REV['t'].append(np.full(len(di), t)); REV['po'].append(di)
                REV['field'].append(np.zeros(len(di), np.int8))          # 0 = promise_date
                REV['old'].append(np.zeros(len(di), np.int64))
                REV['new'].append(pull[is_date].astype(np.int64))
                REV['reason'].append(np.zeros(len(di), np.int8))         # 0 = shortage
                REV['by'].append(np.zeros(len(di), np.int8))             # 0 = buyer
                np.add.at(rev_n, po_ch[di], 1)
            qi = cand2[~is_date]
            if len(qi):
                add = np.maximum(1, np.floor(po_qty[qi] * P['rev_qty_step'])).astype(np.int64)
                REV['t'].append(np.full(len(qi), t)); REV['po'].append(qi)
                REV['field'].append(np.ones(len(qi), np.int8))            # 1 = qty
                REV['old'].append(po_qty[qi].copy()); REV['new'].append(po_qty[qi] + add)
                REV['reason'].append(np.zeros(len(qi), np.int8))
                REV['by'].append(np.zeros(len(qi), np.int8))
                rev_mark[qi] = True
                np.add.at(rev_n, po_ch[qi], 1)

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
util_obs_hist[cur_month] = month_ordered / DECL[cur_month]
delivered_hist[cur_month] = month_delivered
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
def _jsonable(v):
    if isinstance(v, tuple): return list(v)
    if isinstance(v, dict): return {k: (list(x) if isinstance(x, tuple) else x) for k, x in v.items()}
    return v
json.dump({k: _jsonable(v) for k, v in P.items()},
          open(OUT / 'mechanism_parameters.json', 'w'), indent=1, default=str)
np.savez_compressed(OUT / '_sim.npz', pt=pt, pch=pch, pq=pq, pd=pd_, pl=pl_, pa=pa, ppr=ppr, prf=prf,
                    CS=CS, CP=CP, CL=CL, cold=cold, util=util_hist, ordered=ordered_hist, K=K,
                    sup_state=sup_state, regime=regime, MONTHKEY=MONTHKEY, contracted=contracted,
                    DECL=DECL, util_obs=util_obs_hist,
                    it=cat(INV,'t').astype(int), ich=cat(INV,'ch').astype(int),
                    ityp=cat(INV,'typ').astype(int), iqty=cat(INV,'qty').astype(np.int64),
                    st=cat(SHORT,'t').astype(int), sch=cat(SHORT,'ch').astype(int), sq=cat(SHORT,'qty').astype(np.int64),
                    was_exp=was_exp, OPEN_CH=OPEN_CH, PP_OH=PP_OH, PP_OO=PP_OO, PP_ISS=PP_ISS,
                    PP_TIN=PP_TIN, PP_TOUT=PP_TOUT, PP_DEM=PP_DEM, delivered=delivered_hist,
                    alloc_t=cat(ALLOC,'t').astype(int), alloc_ch=cat(ALLOC,'ch').astype(int),
                    alloc_share=cat(ALLOC,'share'), alloc_reason=cat(ALLOC,'reason').astype(int),
                    rev_t=cat(REV,'t').astype(int), rev_po=cat(REV,'po').astype(int),
                    rev_field=cat(REV,'field').astype(int), rev_old=cat(REV,'old').astype(np.int64),
                    rev_new=cat(REV,'new').astype(np.int64), rev_n=rev_n,
                    promise_off=promise_off[:npo])
print('  simulation state saved ->', OUT / '_sim.npz')

# ==================================================================== STAGE 2: emit + derive
print('emitting transactional spine...')
WV = W.values
def wk_ts(t_arr, jitter_hi=6): return pd.to_datetime(WV[t_arr]) + pd.to_timedelta(r.integers(0, jitter_hi, len(t_arr)), unit='D')

ev_po = wk_ts(pt)                                     # PO creation event time
stress_po = sup_state[pt, CS[pch]]
lag_po_d = lag_days('po_lines', NPO, stress_po, r)
rec_po = ev_po + pd.to_timedelta(lag_po_d, unit='D')
nrec_po = never_mask('po_lines', NPO, r)
# §2.3: SOME RECORDS ARE NEVER ENTERED -- genuine right-censoring of the recording layer,
# not a modelling convenience. v7 drew this mask and then emitted every row anyway, so the
# mechanism was documented but absent. It is applied here to every table that nothing else
# references. The PO -> GRN -> inventory spine is left whole on purpose: those are an ERP's
# system of record, and dropping them would dissolve the conservation identity the rest of
# the world is checked against.
def entered(table, n):
    """True where the record was eventually keyed in at all."""
    return ~never_mask(table, n, r)


po_id = np.array([f'PO{i:09d}' for i in range(NPO)])
pol_id = np.array([f'POL{i:09d}' for i in range(NPO)])
promise = ev_po + pd.to_timedelta(contracted[pch], unit='D')
# §2.5 arc 2: a buyer's date revision is a REQUEST, not a physical acceleration. It moves
# the current promise and nothing else -- the line is then measured late against the date
# the buyer asked for, which is exactly what makes the revision a consequence with a cost.
poff = promise_off[:npo].astype(int)
cur_promise = promise - pd.to_timedelta(poff, unit='D')
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
    'current_promise_date': cur_promise, 'requested_date': ev_po + pd.to_timedelta(contracted[pch] - 3, unit='D'),
    'created_ts': ev_po, 'recorded_ts': rec_po}))

# acknowledgements: a supplier short of capacity acknowledges LESS than ordered
ack_q = np.where(pd_ < pq, pd_, pq)
st_ack = sup_state[pt, CS[pch]]
ev_ack = ev_po + pd.to_timedelta(r.integers(1, 5, NPO), unit='D')
rec_ack = ev_ack + pd.to_timedelta(lag_days('supplier_acknowledgements', NPO, st_ack, r), unit='D')

_ack_in = entered('supplier_acknowledgements', NPO)     # §2.3 never-entered acknowledgements
emit('supplier_acknowledgements.csv', pd.DataFrame({'ack_id': [f'ACK{i:09d}' for i in range(NPO)],
    'po_line_id': pol_id, 'ack_qty': ack_q, 'ack_date': ev_ack,
    'ack_status': np.where(ack_q == pq, 'full', np.where(ack_q > 0, 'partial', 'rejected')),
    'event_ts': ev_ack, 'recorded_ts': rec_ack})[_ack_in])

shipped = (pd_ > 0) & (pa < T)      # open POs have no GRN -- that is the censoring
sidx_ = np.where(shipped)[0]; NSH = len(sidx_)
ev_asn = ev_po[sidx_] + pd.to_timedelta(np.maximum(1, np.round(pl_[sidx_] * .35)).astype(int), unit='D')
rec_asn = ev_asn + pd.to_timedelta(lag_days('asn', NSH, sup_state[pt[sidx_], CS[pch[sidx_]]], r), unit='D')

emit('asn.csv', pd.DataFrame({'asn_id': [f'ASN{i:09d}' for i in range(NSH)], 'po_line_id': pol_id[sidx_],
    'dispatched_qty': pd_[sidx_], 'dispatch_ts': ev_asn, 'expected_arrival_date': arr_ts[sidx_],
    'transport_mode': mode[pch[sidx_]], 'vehicle_id': [f'VEH{x:05d}' for x in r.integers(1, 99999, NSH)],
    'recorded_ts': rec_asn})[entered('asn', NSH)])

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
# the rejection QC reports is the one the ledger actually scrapped, not a fresh draw --
# v7 drew it twice, so quality_inspections and inventory_transactions disagreed by
# construction and no accounting check could reconcile them.
rej = po_rej[:npo][sidx_].astype(np.int64)
emit('quality_inspections.csv', pd.DataFrame({'inspection_id': [f'INSP{i:09d}' for i in range(NSH)],
    'grn_line_id': [f'GRNL{i:09d}' for i in range(NSH)], 'qty_inspected': pd_[sidx_],
    'qty_accepted': pd_[sidx_] - rej, 'qty_rejected': rej, 'qty_deviation_accepted': 0,
    'rejection_reason': np.where(rej > 0, 'defect', ''), 'event_ts': ev_grn,
    'recorded_ts': ev_grn + pd.to_timedelta(lag_days('quality_inspections', NSH, st_grn, r), unit='D')
    })[entered('quality_inspections', NSH)])

# ---------------------------------------------------------------- §2.5 arc 2 revisions
# Two mechanism-driven sources, no random draw:
#   buyer   -- raised in the loop against an open line on a part-plant in a live shortage
#              episode whose expedite was never raised or did not land (REV registry)
#   supplier-- the acknowledgement came back materially short of what was ordered, which
#              is a recorded quantity revision on the line in any real ERP
print('PO line revisions (mechanism-driven, both directions)...')
_rv_t = cat(REV, 't').astype(int); _rv_po = cat(REV, 'po').astype(int)
_rv_f = cat(REV, 'field').astype(int); _rv_o = cat(REV, 'old').astype(np.int64)
_rv_n = cat(REV, 'new').astype(np.int64)
_sup_short = (pq - pd_) > np.maximum(1.0, P['rev_min_short_frac'] * pq)
_sp_po = np.where(_sup_short)[0]
REV_PO = np.concatenate([_rv_po, _sp_po])
REV_T = np.concatenate([_rv_t, pt[_sp_po]])
REV_FIELD = np.concatenate([_rv_f, np.full(len(_sp_po), 1)])          # 1 = qty
REV_OLD = np.concatenate([np.where(_rv_f == 0, ppr[_rv_po].astype(np.int64), _rv_o),
                          pq[_sp_po].astype(np.int64)])
REV_NEW = np.concatenate([np.where(_rv_f == 0, (ppr[_rv_po] - _rv_n).astype(np.int64), _rv_n),
                          pd_[_sp_po].astype(np.int64)])
REV_BY = np.concatenate([np.zeros(len(_rv_po), int), np.ones(len(_sp_po), int)])   # buyer / supplier
REV_REASON = np.concatenate([np.zeros(len(_rv_po), int), np.ones(len(_sp_po), int)])
_ro = np.lexsort((REV_T, REV_PO))
REV_PO, REV_T, REV_FIELD = REV_PO[_ro], REV_T[_ro], REV_FIELD[_ro]
REV_OLD, REV_NEW, REV_BY, REV_REASON = REV_OLD[_ro], REV_NEW[_ro], REV_BY[_ro], REV_REASON[_ro]
_first = np.r_[True, REV_PO[1:] != REV_PO[:-1]] if len(REV_PO) else np.zeros(0, bool)
_grp = np.cumsum(_first) - 1
REV_NUM = (np.arange(len(REV_PO)) - np.searchsorted(_grp, _grp, 'left') + 1) if len(REV_PO) else np.zeros(0, int)
NREV = len(REV_PO)
print(f'  revisions {NREV:,}  buyer {int((REV_BY==0).sum()):,}  supplier {int((REV_BY==1).sum()):,}')
REV_IN = entered('po_line_revisions', NREV)   # §2.3: some revisions are never keyed in
ev_rev = wk_ts(REV_T) if NREV else pd.DatetimeIndex([])
rec_rev = (ev_rev + pd.to_timedelta(lag_days('po_line_revisions', NREV, np.zeros(NREV), r), unit='D')
           if NREV else pd.DatetimeIndex([]))

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
# revisions per channel-week, bucketed on the VISIBLE week like every other store
if NREV:
    _rvw, _ = visible_week(ev_rev, rec_rev)
    # a revision nobody keyed in cannot be a feature -- REVW counts entered rows only
    _rin = (_rvw >= 0) & (_rvw < T) & REV_IN
    REVW = np.bincount(pch[REV_PO][_rin] * T + _rvw[_rin], minlength=NCH*T).reshape(NCH, T)
else:
    REVW = np.zeros((NCH, T), np.int64)
active = O > 0
awk = np.minimum(np.arange(T) + 1, 52)[None, :]
util_ch = util_obs_hist[MONTHKEY][:, CS].T   # observable load, not ordered/K (see §2.7 note)
print('  writing channel_performance_weekly...')
cpw = pd.DataFrame({'channel_id': np.repeat(CHID, T), 'week_start': np.tile(WV, NCH),
    'qty_ordered': O.ravel().astype(np.int64), 'qty_received': Rv.ravel().astype(np.int64),
    'is_active_week': active.ravel(), 'fill_rate': np.round(f_ff, 6).ravel(),
    'fill_rate_last4': np.round(r4, 6).ravel(), 'fill_rate_last13': np.round(r13, 6).ravel(),
    'fill_rate_last52': np.round(r52, 6).ravel(), 'lead_time_actual_days': np.round(lead_ff, 2).ravel(),
    'lead_time_ratio': np.round(lead_ff / contracted[:, None], 4).ravel(),
    'otd_rate_last13': np.round(otd13, 6).ravel(), 'ack_gap_ratio': np.round(1 - f_ff, 6).ravel(),
    'revision_count': REVW.ravel().astype(np.int64), 'load_ratio': np.round(util_ch, 4).ravel(),
    'days_since_last_short': 0, 'active_weeks_in_52': np.repeat(awk, NCH, 0).ravel(),
    'reporting_lag_days': np.round(pd.DataFrame(lagw).ffill(axis=1).to_numpy(), 2).ravel(),
    'weeks_since_last_activity': 0, 'weeks_since_last_receipt': 0})
emit('channel_performance_weekly.csv', cpw)
assert int(cpw.qty_ordered.sum()) == SRC_ORD and int(cpw.qty_received.sum()) == SRC_REC
print('  channel store conservation re-verified on the emitted CSV frame')

sw_o = np.bincount(CS[pch[IN_ORD]] * T + vw_ord[IN_ORD], weights=pq[IN_ORD], minlength=NS*T).reshape(NS, T)
sw_r = np.bincount(CS[pch[sidx_][IN_REC]] * T + vw_rec[IN_REC], weights=pd_[sidx_][IN_REC], minlength=NS*T).reshape(NS, T)
assert int(sw_o.sum()) == SRC_ORD and int(sw_r.sum()) == SRC_REC
_SREV = np.zeros((NS, T), np.int64); np.add.at(_SREV, CS, REVW)
# Cohort-consistent supplier fill: of the units ORDERED in this visible week, what share
# did the supplier commit? v7 divided receipts landing this week by orders placed this week
# -- two different cohorts -- which made a heavy ordering month look like a bad fill month
# and the month after look like a good one. That artefact, not the capacity arc, drove the
# sign of §16 probe 7. The qty_ordered / qty_received columns are untouched; they carry the
# conservation identity and are cohort-mixed by definition.
_snum = np.bincount(CS[pch[IN_ORD]] * T + vw_ord[IN_ORD],
                    weights=(line_fill[IN_ORD] * pq[IN_ORD]), minlength=NS*T)
_sden = np.bincount(CS[pch[IN_ORD]] * T + vw_ord[IN_ORD],
                    weights=pq[IN_ORD].astype(float), minlength=NS*T)
sf = np.where(_sden > 0, _snum / np.maximum(_sden, 1e-9), np.nan).reshape(NS, T)
sf_ff = pd.DataFrame(sf).ffill(axis=1).to_numpy()
emit('supplier_performance_weekly.csv', pd.DataFrame({'supplier_id': np.repeat(suppliers.supplier_id.values, T),
    'week_start': np.tile(WV, NS), 'qty_ordered': sw_o.ravel().astype(np.int64),
    'qty_received': sw_r.ravel().astype(np.int64),
    'revision_count': _SREV.ravel(),
    'active_channel_count': np.bincount(CS, minlength=NS).repeat(T),
    'fill_rate': np.round(sf_ff, 6).ravel(), 'fill_rate_last4': np.round(roll(sf_ff, 4), 6).ravel(),
    'fill_rate_last13': np.round(roll(sf_ff, 13), 6).ravel(), 'fill_rate_last52': np.round(roll(sf_ff, 52), 6).ravel(),
    'lead_time_actual_days': 0, 'lead_time_ratio': 0, 'otd_rate_last13': 0, 'ack_gap_ratio': np.round(1 - sf_ff, 6).ravel(),
    'load_ratio': np.round(util_obs_hist[MONTHKEY].T, 4).ravel(), 'reporting_lag_days': 0,
    'is_active_week': (sw_o > 0).ravel(), 'active_weeks_in_52': np.repeat(awk, NS, 0).ravel(),
    'weeks_since_last_activity': 0, 'weeks_since_last_receipt': 0, 'days_since_last_short': 0}))
print('  supplier store conservation exact')

# ---------------------------------------------------------------- §4 inventory ledger identity
print('inventory ledger (§2.1 opening balance posted, then rolled forward)...')
_mt, _mch, _mtyp, _mqty = (cat(INV,'t').astype(int), cat(INV,'ch').astype(int),
                           cat(INV,'typ').astype(int), cat(INV,'qty').astype(np.int64))
# §2.1: the opening position is a LEDGER POSTING at a fixed pre-window date, week index -1.
# Every balance after it is the roll-forward of this row through the same mechanism that
# produces receipts, issues, scrap and transfers -- it is never written as a level.
_op = np.where(OPEN_CH > 0)[0]
it = np.concatenate([np.full(len(_op), -1, int), _mt])
ich = np.concatenate([_op, _mch])
ityp = np.concatenate([np.full(len(_op), 6, int), _mtyp])           # 6 = adjustment
iqty = np.concatenate([OPEN_CH[_op].astype(np.int64), _mqty])
print(f'  opening postings {len(_op):,} channels, {int(OPEN_CH.sum()):,} units at {P["opening_date"]}')
TYPES = np.array(['receipt','issue_to_production','return','scrap','transfer_out','transfer_in','adjustment'])
_tsafe = np.maximum(it, 0)
ev_inv = pd.Series(wk_ts(_tsafe))
ev_inv[it < 0] = pd.Timestamp(P['opening_date'])
ev_inv = pd.DatetimeIndex(ev_inv)
lag_inv_d = lag_days('inventory_transactions', len(it), sup_state[_tsafe, CS[ich]], r)
rec_inv = ev_inv + pd.to_timedelta(lag_inv_d, unit='D')

emit('inventory_transactions.csv', pd.DataFrame({'txn_id': [f'TXN{i:010d}' for i in range(len(it))],
    'part_id': parts.part_id.values[CP[ich]], 'plant_id': plants.plant_id.values[CL[ich]],
    'txn_type': TYPES[ityp], 'qty': iqty, 'reference_id': [f'REF{x:09d}' for x in np.arange(len(it))], 'from_plant_id': plants.plant_id.values[CL[ich]],
    'event_ts': ev_inv, 'recorded_ts': rec_inv}))
# snapshots are COMPUTED from transactions, never generated independently (§15)
ppk = CP[ich] * NPL + CL[ich]
mk = np.where(it < 0, -1, MONTHKEY[_tsafe])
led = pd.DataFrame({'pp': ppk, 'm': mk, 'qty': iqty}).groupby(['pp','m'], sort=True)['qty'].sum().reset_index()
led['closing'] = led.groupby('pp')['qty'].cumsum()
led['opening'] = led['closing'] - led['qty']
snap_part = parts.part_id.values[led.pp.values // NPL]; snap_plant = plants.plant_id.values[led.pp.values % NPL]
# The snapshot carries the month's CLOSING balance, so it is dated at month END. v7 dated
# it at month start while holding the closing figure, which put the balance a month ahead
# of itself and made the snapshot irreconcilable with the ledger it was computed from.
_SNAPME = (pd.date_range('2015-12-01', periods=NMONTH + 1, freq='MS') + pd.offsets.MonthEnd(0))
snap_date = _SNAPME.values[np.clip(led.m.values + 1, 0, NMONTH)]
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

# ---- root cause, with CORROBORATION measured separately from assignment -------------
# The schema makes root_cause NOT NULL over a six-value enum, so every episode gets a
# label. What matters is whether the label is BACKED BY EVIDENCE already in the world.
# Each branch below is a distinct piece of evidence, re-derivable from the emitted CSVs:
#   quality_reject  a feeding receipt was rejected in the 26 weeks before it opened
#   supplier_delay  the worst feeding line arrived past its contracted lead time
#   supplier_short  the worst feeding line delivered less than was ordered
#   demand_spike    issued demand in the 4 weeks before ran above its own trailing mean
#   logistics       transit stress at that plant was elevated when it opened
#   plan_change     the allocation review cut a feeding supplier's share in the window
# Anything left over is labelled with the best available guess AND COUNTED AS
# UNCORROBORATED. CAUSE_SHARE is the corroborated share -- not the labelled share, which
# is 100% by construction and would be a gate that cannot fail.
ord_pp = CP[pch] * NPL + CL[pch]
# A line is evidence for a shortage only once its failure was VISIBLE. v7 searched on the
# week the line was PLACED, which credited a shortage in week w to a line placed in w-1
# that would not arrive late until w+8 -- something nobody could have known at w. The
# anchor is the week the failure surfaced: the acknowledgement for a short commitment, the
# promise date for a late one, the receipt for a rejection.
_late_line = pl_ > np.maximum(contracted[pch] - promise_off[:npo], 1)
_short_line = pd_ < pq
_rej_line = po_rej[:npo] > 0
_w_promise = pt + np.ceil(np.maximum(contracted[pch] - promise_off[:npo], 1) / 7.0).astype(int)
_w_vis = np.where(_short_line, pt, np.where(_rej_line, pa, _w_promise))
by = pd.DataFrame({'pp': ord_pp, 't': _w_vis, 'ch': pch,
                   'f': np.where(pq > 0, pd_ / pq, 1.0), 'late': _late_line,
                   'rej': po_rej[:npo]})
by = by[(by.f < 1.0) | by.late | (by.rej > 0)].sort_values('t')
grp = {k: v for k, v in by.groupby('pp')}
# largest feeder per part-plant, used only for the uncorroborated fallback
_vol = pd.DataFrame({'pp': ord_pp, 'ch': pch, 'q': pq}).groupby(['pp', 'ch']).q.sum()
_top_feeder = {int(k): int(v) for k, v in _vol.groupby('pp').idxmax().apply(lambda x: x[1]).items()}
# allocation cuts, by part-plant and week
_cut_pp = {}
if len(cat(ALLOC, 't')):
    _at = cat(ALLOC, 't').astype(int); _ach = cat(ALLOC, 'ch').astype(int); _ar = cat(ALLOC, 'reason').astype(int)
    _k = _ar == 1
    for _tt, _cc in zip(_at[_k], _ach[_k]):
        _cut_pp.setdefault(int(PP[_cc]), []).append((int(_tt), int(_cc)))
_DEMC = np.vstack([np.zeros((1, NPP)), np.cumsum(PP_DEM, 0)])     # prefix sums for the spike test

cause_sup, cause_ch, has_cause, _cause_kind = [], [], [], []
for ppv, tv in zip(pp_e, st_e):
    ppv = int(ppv); tv = int(tv)
    g = grp.get(ppv)
    if g is not None:
        w = g[(g.t <= tv) & (g.t >= tv - 26)]
        if len(w):
            i = w.f.idxmin()
            ch_ = int(w.at[i, 'ch'])
            kind = ('quality_reject' if int(w.rej.max()) > 0
                    else ('supplier_delay' if bool(w.at[i, 'late']) else 'supplier_short'))
            cause_ch.append(ch_); cause_sup.append(int(CS[ch_]))
            has_cause.append(True); _cause_kind.append(kind); continue
    a0, a1 = max(0, tv - 3), tv + 1
    b0, b1 = max(0, tv - 26), max(1, tv - 3)
    rec = (_DEMC[a1, ppv] - _DEMC[a0, ppv]) / max(a1 - a0, 1)
    base = (_DEMC[b1, ppv] - _DEMC[b0, ppv]) / max(b1 - b0, 1)
    if base > 0 and rec / base > 1.25:
        cause_ch.append(-1); cause_sup.append(-1); has_cause.append(True); _cause_kind.append('demand_spike')
    elif transit_state[min(tv, T - 1), ppv % NPL] > 0.5:
        cause_ch.append(-1); cause_sup.append(-1); has_cause.append(True); _cause_kind.append('logistics')
    else:
        cuts = [c for tt, c in _cut_pp.get(ppv, ()) if tv - 26 <= tt <= tv]
        # A requirement plan that moved is a cause in its own right. The safety-stock
        # target tracks the planner's running forecast, so a target that climbed over the
        # quarter puts the shelf below cover with no supplier having failed at all. This
        # branch is what the uncorroborated residual mostly was.
        _ss_now = PP_SS[min(tv, T - 1), ppv]
        _ss_then = PP_SS[max(0, tv - 13), ppv]
        if cuts:
            cause_ch.append(cuts[-1]); cause_sup.append(int(CS[cuts[-1]]))
            has_cause.append(True); _cause_kind.append('plan_change')
        elif _ss_then > 0 and _ss_now / _ss_then > 1.15:
            cause_ch.append(-1); cause_sup.append(-1)
            has_cause.append(True); _cause_kind.append('plan_change')
        else:
            tf = _top_feeder.get(ppv, -1)
            cause_ch.append(tf); cause_sup.append(int(CS[tf]) if tf >= 0 else -1)
            has_cause.append(False); _cause_kind.append('supplier_short')
cause_ch = np.array(cause_ch); cause_sup = np.array(cause_sup); has_cause = np.array(has_cause, bool)
CAUSE_KIND = np.array(_cause_kind)
CAUSE_SHARE = float(has_cause.mean()) if NE else 0.0
print('  root-cause mix: ' + ', '.join(f'{k}={v}' for k, v in
      pd.Series(CAUSE_KIND).value_counts().items()) if NE else '  no episodes')
print(f'  CORROBORATED share (the number that matters): {CAUSE_SHARE*100:.2f}%')
np.savez_compressed(OUT / '_events.npz', pp_e=pp_e, st_e=st_e, en_e=en_e, len_e=len_e,
                    sev_e=sev_e, stk_e=stk_e, mit_e=mit_e, has_cause=has_cause,
                    n_cond=N_COND, cause_ch=cause_ch)
ev_sh = wk_ts(st_e)
# what actually resolved it, read off the mitigation bits -- not a random draw
_act = np.where(stk_e > 0, 'reschedule',
        np.where((mit_e & 2) > 0, 'expedite',
        np.where((mit_e & 4) > 0, 'alternate_source',
        np.where((mit_e & 1) > 0, 'substitute', 'reschedule'))))   # inter-plant cover
emit('shortage_events.csv', pd.DataFrame({'shortage_id': [f'SH{i:08d}' for i in range(NE)],
    'part_id': parts.part_id.values[pp_e // NPL], 'plant_id': plants.plant_id.values[pp_e % NPL],
    'shortage_qty': sq_e, 'shortage_start_ts': ev_sh,
    'shortage_end_ts': wk_ts(np.maximum(en_e, st_e + 1)),
    'root_cause': CAUSE_KIND,
    'responsible_supplier_id': np.where(cause_sup >= 0, suppliers.supplier_id.values[np.maximum(cause_sup,0)], ''),
    'resolution_action': _act,
    'event_ts': ev_sh, 'recorded_ts': ev_sh + pd.to_timedelta(lag_days('shortage_events', NE, np.zeros(NE), r), unit='D')
    })[entered('shortage_events', NE)])

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
    'event_ts': ev_x, 'recorded_ts': ev_x + pd.to_timedelta(lag_days('expedite_events', NX, np.zeros(NX), r), unit='D')
    })[entered('expedite_events', NX)])

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
# §2.5 arc 2: revisions are the ones actually raised -- a buyer acting on a live shortage
# episode, or a supplier acknowledging short. There is no `r.random() < 0.34`.
emit('po_line_revisions.csv', pd.DataFrame({'revision_id': [f'REV{i:08d}' for i in range(NREV)],
    'po_line_id': pol_id[REV_PO], 'revision_number': REV_NUM,
    'field_changed': np.where(REV_FIELD == 0, 'promise_date', 'qty'),
    'old_value': REV_OLD.astype(str), 'new_value': REV_NEW.astype(str),
    'initiated_by': np.where(REV_BY == 0, 'buyer', 'supplier'),
    'reason_code': np.where(REV_REASON == 0, 'shortage', 'capacity_short'),
    'event_ts': ev_rev, 'recorded_ts': rec_rev})[REV_IN])

# ============================================================ §2.2 FORWARD REQUIREMENT PLAN
# A rolling 12-26 week horizon per PUBLISHED VERSION, carrying the date the version was
# published. The drift/revision mechanism underneath is unchanged: a version is optimistic
# and noisier the further out it looks, and successive versions converge on the actual.
# horizon_days is therefore an OUTCOME of the publish cadence and the drawn horizon --
# it is never a fixed 30.
print('§2.2 forward requirement plan (rolling multi-period, versioned)...')
prod = products.product_id.values; NPRD = len(prod)
prod_plant = r.integers(0, NPL, NPRD)                       # each product is built at one plant
_pbase = r.lognormal(np.log(220.0), 0.55, NPRD)             # weekly build rate per product
BUILD = np.maximum(0, np.round(_pbase[None, :] * SEAS[:, None] * TREND[:, None]
                               * r.lognormal(0, 0.18, (T, NPRD)))).astype(np.int64)
# §2.5 arc 5: a LINE STOP cuts the build that follows it, and the next published version
# of the plan sees the lower truth and revises down. The chain does not terminate at the
# stop: it turns back into the requirement. Only versions published AFTER the stop can
# know about it -- the ones published before are exactly the revisions that plan drift
# is supposed to be made of.
_ls_hit = np.zeros((T, NPRD))
if len(ls_t):
    for _t, _pp in zip(ls_t, ls_pp):
        _pl = int(_pp) % NPL
        _who = np.where(prod_plant == _pl)[0]
        _a, _b = int(_t), min(T, int(_t) + P['ls_plan_weeks'])
        _ls_hit[_a:_b, _who] += 1.0
BUILD = np.maximum(0, np.round(BUILD * np.maximum(
    1.0 - P['ls_plan_cut'] * np.minimum(_ls_hit, 3.0), 0.25))).astype(np.int64)
print(f'  arc 5: {int((_ls_hit > 0).sum()):,} product-weeks of build cut by a line stop')
T0 = int(np.searchsorted(W.values, np.datetime64(P['model_start'])))
T1 = int(np.searchsorted(W.values, np.datetime64(P['model_end'])))
HMIN, HMAX = P['plan_horizon_min_w'], P['plan_horizon_max_w']

def _ragged(reps):
    """offsets 1..k for each k in reps, concatenated"""
    reps = np.maximum(reps, 0)
    tot = int(reps.sum())
    if tot == 0: return np.zeros(0, int), np.zeros(0, int)
    idx = np.repeat(np.arange(len(reps)), reps)
    starts = np.r_[0, np.cumsum(reps)[:-1]]
    off = np.arange(tot) - np.repeat(starts, reps) + 1
    return idx, off

def _version_error(truth, h, sig_near, sig_far, bias_far):
    bias = 1.0 + bias_far * (h / 13.0)
    sig = sig_near + (sig_far - sig_near) * np.clip(h / HMAX, 0, 1)
    return (np.maximum(0, np.round(truth * bias * r.lognormal(0, sig))).astype(np.int64), sig)

_PJ, _TP, _TW, _PQ = [], [], [], []
for tp in np.arange(T0, T1, P['plan_publish_weeks']):
    H = np.minimum(r.integers(HMIN, HMAX + 1, NPRD), T - 1 - tp)
    j, off = _ragged(H)
    if not len(j): continue
    tw = tp + off
    q, _ = _version_error(BUILD[tw, j].astype(float), off.astype(float),
                          P['plan_noise_near'], P['plan_noise_far'], P['plan_bias_far'])
    _PJ.append(j); _TP.append(np.full(len(j), tp)); _TW.append(tw); _PQ.append(q)
PLAN = pd.DataFrame({'pi': np.concatenate(_PJ), 'tp': np.concatenate(_TP),
                     'tw': np.concatenate(_TW), 'q': np.concatenate(_PQ)})
PLAN = PLAN.sort_values(['pi', 'tw', 'tp'], kind='stable').reset_index(drop=True)
PLAN['plan_version'] = PLAN.groupby(['pi', 'tw']).cumcount() + 1
PLAN['h_w'] = PLAN.tw - PLAN.tp
_pl_date = pd.to_datetime(WV[PLAN.tp.values]); _pl_tgt = pd.to_datetime(WV[PLAN.tw.values])
emit('production_plan.csv', pd.DataFrame({
    'plan_id': [f'PLN{i:09d}' for i in range(len(PLAN))], 'plan_version': PLAN.plan_version.values,
    'plan_date': _pl_date, 'product_id': prod[PLAN.pi.values],
    'plant_id': plants.plant_id.values[prod_plant[PLAN.pi.values]],
    'target_period': _pl_tgt, 'period_grain': 'week', 'planned_qty': PLAN.q.values,
    'plan_type': np.where(PLAN.h_w.values <= P['plan_firm_weeks'], 'firm', 'rolling'),
    'is_firm': PLAN.h_w.values <= P['plan_firm_weeks'], 'event_ts': _pl_date,
    'recorded_ts': _pl_date + pd.to_timedelta(r.integers(0, 4, len(PLAN)), unit='D')}))
_vmax = int(PLAN.plan_version.max()); _hu = PLAN.h_w.nunique()
print(f'  plan versions per target period: max {_vmax}; distinct horizons {_hu}; '
      f'periods per published version (median) '
      f'{int(PLAN.groupby(["pi","tp"]).size().median())}')

# production_actual: what was actually built, weekly, over the model window
_ai = np.arange(T0, min(T, T1 + 1))
_apj = np.repeat(np.arange(NPRD), len(_ai)); _awk = np.tile(_ai, NPRD)
_act = np.maximum(0, np.round(BUILD[_awk, _apj] * r.lognormal(0, 0.09, len(_apj)))).astype(np.int64)
_ats = pd.to_datetime(WV[_awk])
emit('production_actual.csv', pd.DataFrame({'product_id': prod[_apj],
    'plant_id': plants.plant_id.values[prod_plant[_apj]], 'period': _ats, 'period_grain': 'week',
    'actual_qty': _act, 'scrapped_qty': (_act * 0.02).astype(np.int64),
    'rework_qty': (_act * 0.01).astype(np.int64), 'event_ts': _ats,
    'recorded_ts': _ats + pd.to_timedelta(r.integers(1, 9, len(_apj)), unit='D')}))

# plan_drift_features: V1 vs the latest version vs the actual, per target period
_ACT = np.zeros((T, NPRD), np.int64); _ACT[_awk, _apj] = _act
_g = PLAN.groupby(['pi', 'tw'])
_dr = pd.DataFrame({'orig': _g.q.first(), 'latest': _g.q.last(), 'n': _g.q.size(),
                    'sd': _g.q.std().fillna(0.0), 'mu': _g.q.mean(),
                    'h1': _g.h_w.first(), 'tp1': _g.tp.first()}).reset_index()
_dr['actual'] = _ACT[_dr.tw.values, _dr.pi.values]
emit('plan_drift_features.csv', pd.DataFrame({'product_id': prod[_dr.pi.values],
    'plant_id': plants.plant_id.values[prod_plant[_dr.pi.values]],
    'target_period': pd.to_datetime(WV[_dr.tw.values]), 'plan_date': pd.to_datetime(WV[_dr.tp1.values]),
    'horizon_days': (_dr.h1.values * 7).astype(np.int64), 'plan_version': _dr.n.values,
    'original_plan_qty': _dr.orig.values, 'latest_plan_qty': _dr.latest.values,
    'actual_qty': _dr.actual.values,
    'drift_ratio': np.round(_dr.actual.values / np.maximum(_dr.latest.values, 1), 4),
    'plan_revision_count': _dr.n.values - 1,
    'plan_volatility': np.round(_dr.sd.values / np.maximum(_dr.mu.values, 1e-9), 4),
    'is_firm': _dr.h1.values <= P['plan_firm_weeks'],
    'regime_flag': np.where(regime[_dr.tw.values] > 0.8, 'covid',
                     np.where(regime[_dr.tw.values] > 0, 'chip_shortage', 'normal')),
    'recorded_ts': pd.to_datetime(WV[_dr.tw.values]) + pd.to_timedelta(5, unit='D')}))

# part_demand_weekly: the same rolling-horizon discipline at part x plant, netted every
# dem_publish_weeks. The truth it forecasts is the demand the simulation actually issued.
print('§2.2 part_demand_weekly (rolling horizon per as-of date)...')
LIVE_PP = np.where(np.bincount(PP, weights=NOTCOLD, minlength=NPP) > 0)[0]
NLP = len(LIVE_PP)
_DJ, _DP, _DW, _DQ, _DS = [], [], [], [], []
for tp in np.arange(T0, T1, P['dem_publish_weeks']):
    H = np.minimum(r.integers(HMIN, HMAX + 1, NLP), T - 1 - tp)
    j, off = _ragged(H)
    if not len(j): continue
    tw = tp + off; ppv = LIVE_PP[j]
    truth = PP_DEM[tw, ppv].astype(float)
    q, sig = _version_error(truth, off.astype(float), P['plan_noise_near'],
                            P['plan_noise_far'], P['plan_bias_far'])
    _DJ.append(ppv); _DP.append(np.full(len(j), tp)); _DW.append(tw); _DQ.append(q); _DS.append(sig)
_dpp = np.concatenate(_DJ); _dtp = np.concatenate(_DP); _dtw = np.concatenate(_DW)
_dq = np.concatenate(_DQ); _dsig = np.concatenate(_DS)
_das = pd.to_datetime(WV[_dtp]); _dws = pd.to_datetime(WV[_dtw])
emit('part_demand_weekly.csv', pd.DataFrame({'part_id': parts.part_id.values[_dpp // NPL],
    'plant_id': plants.plant_id.values[_dpp % NPL], 'week_start': _dws, 'as_of_date': _das,
    'gross_requirement_p50': _dq,
    'gross_requirement_p90': np.round(_dq * np.exp(1.2816 * _dsig)).astype(np.int64),
    'horizon_days': ((_dtw - _dtp) * 7).astype(np.int64),
    'driving_products': prod[r.integers(0, NPRD, len(_dpp))],
    'recorded_ts': _das + pd.to_timedelta(r.integers(0, 3, len(_dpp)), unit='D')}))
print(f'  as-of dates {len(np.unique(_dtp))}; distinct horizon_days {len(np.unique(_dtw-_dtp))}')

# ============================================================ §2.1 MEASURED WEEKLY POSITION
# Every column here is read off the roll-forward of the opening posting. Nothing is set.
print('§2.1 inventory_position_weekly (measured roll-forward)...')
_c4 = pd.DataFrame(PP_ISS).rolling(4, min_periods=1).sum().to_numpy()
_c13 = pd.DataFrame(PP_ISS).rolling(13, min_periods=1).sum().to_numpy()
_i4 = pd.DataFrame(PP_TIN).rolling(4, min_periods=1).sum().to_numpy()
_o4 = pd.DataFrame(PP_TOUT).rolling(4, min_periods=1).sum().to_numpy()
_stale_num = np.bincount(ppk * T + np.clip(_wk_index(ev_inv), 0, T - 1), weights=lag_inv_d, minlength=NPP*T)
_stale_cnt = np.bincount(ppk * T + np.clip(_wk_index(ev_inv), 0, T - 1), minlength=NPP*T)
_stale = np.where(_stale_cnt > 0, _stale_num / np.maximum(_stale_cnt, 1), 0.0).reshape(NPP, T).T
_ipw_pp = np.tile(LIVE_PP, T); _ipw_t = np.repeat(np.arange(T), NLP)
_oh = PP_OH[_ipw_t, _ipw_pp].astype(np.int64)
_res = np.minimum(_oh, PP_DEM[_ipw_t, _ipw_pp].astype(np.int64))
_cons4 = _c4[_ipw_t, _ipw_pp].astype(np.int64)
_stl = np.round(_stale[_ipw_t, _ipw_pp]).astype(np.int64)
_wks = pd.to_datetime(WV[_ipw_t])
emit('inventory_position_weekly.csv', pd.DataFrame({'part_id': parts.part_id.values[_ipw_pp // NPL],
    'plant_id': plants.plant_id.values[_ipw_pp % NPL], 'week_start': _wks,
    'as_of_date': _wks + pd.to_timedelta(_stl, unit='D'), 'qty_on_hand': _oh, 'qty_blocked': 0,
    'qty_reserved': _res, 'qty_in_transit': PP_OO[_ipw_t, _ipw_pp].astype(np.int64),
    'qty_available': _oh - _res, 'open_po_qty': PP_OO[_ipw_t, _ipw_pp].astype(np.int64),
    'safety_stock_qty': PP_SS[_ipw_t, _ipw_pp].astype(np.int64),
    'days_of_supply': np.round(np.divide(_oh * 28.0, np.maximum(_cons4, 1)), 2),
    'consumption_4w': _cons4, 'consumption_13w': _c13[_ipw_t, _ipw_pp].astype(np.int64),
    'inter_plant_in_4w': _i4[_ipw_t, _ipw_pp].astype(np.int64),
    'inter_plant_out_4w': _o4[_ipw_t, _ipw_pp].astype(np.int64),
    'staleness_days': _stl, 'event_ts': _wks,
    'recorded_ts': _wks + pd.to_timedelta(np.maximum(_stl, 1), unit='D')}))


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
                  json.dumps({'channel': NCH}), 'v8.0', f'rane-v8-seed{SEED}', 'labels-v8',
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
                       int(HOR) if c else 0, 'forward simulated outcome', 'labels-v8', d0])
    # --- arrival_week on po_line: week index of first receipt, RIGHT-CENSORED if unresolved
    aw = pa[li] - t0
    cen_a = pa[li] > t1
    for k, v, c in zip(li, aw, cen_a):
        labels.append([f'LA{SEED}{len(labels):09d}', sid, d0.date(), 'po_line', pol_id[k], 'arrival_week',
                       HOR, (d0 + pd.Timedelta(days=1)).date(), (d0 + pd.Timedelta(days=HOR)).date(), float(max(0, v)), bool(c),
                       int(HOR) if c else 0, 'forward simulated outcome', 'labels-v8', d0])
    # --- capacity_strain on channel: supplier utilisation over the forward months
    _live_ch = np.where(~cold)[0]
    ch_s = r.choice(_live_ch, min(2500, len(_live_ch)), replace=False)
    m0, m1 = MONTHKEY[t0], MONTHKEY[t1]
    strain = util_hist[m0:m1+1, CS[ch_s]].mean(0)
    cen_c = r.random(len(ch_s)) < 0.04
    for k, v, c in zip(ch_s, strain, cen_c):
        labels.append([f'LC{SEED}{len(labels):09d}', sid, d0.date(), 'channel', CHID[k], 'capacity_strain',
                       HOR, (d0 + pd.Timedelta(days=1)).date(), (d0 + pd.Timedelta(days=HOR)).date(), round(float(min(v, 3.0)), 6), bool(c),
                       int(HOR) if c else 0, 'forward simulated outcome', 'labels-v8', d0])
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
                           int(HOR) if c else 0, 'forward simulated outcome', 'labels-v8', d0])
    # --- demand_drift on product_plant: the FORWARD ACTUAL against the plan that was
    # standing at t0. v7 computed a seasonal ratio times lognormal noise -- a number with
    # no relationship to production_plan, production_actual or plan_drift_features, which
    # is the brief's own definition of a plausible CSV rather than training data. Here it
    # is read off the same PLAN table the plan store is emitted from, and it is a
    # consequence of §2.5 arc 5 whenever a line stop cut the build in the window.
    # only inside the plan store's own window: outside it there is no standing plan to
    # drift against, and a label computed there would be an artefact of the boundary.
    _std = PLAN[(PLAN.tp <= t0) & (PLAN.tw > t0) & (PLAN.tw <= t1)] if (t0 >= T0 and t1 <= T1) else PLAN.iloc[:0]
    if len(_std):
        _latest = _std.sort_values('tp').groupby(['pi', 'tw']).q.last()
        _pl_sum = _latest.groupby(level=0).sum()
        _ac_sum = pd.Series({int(j): int(_ACT[t0 + 1:t1 + 1, int(j)].sum()) for j in _pl_sum.index})
        _dd = (_ac_sum / _pl_sum.replace(0, np.nan)).dropna()
        _dd = _dd[np.isfinite(_dd)]
        _sel = _dd.index.to_numpy()
        if len(_sel) > 900: _sel = r.choice(_sel, 900, replace=False)
        cen_d = r.random(len(_sel)) < 0.05
        for j, c in zip(_sel, cen_d):
            v = float(np.clip(_dd.loc[j], 0, 4))
            labels.append([f'LD{SEED}{len(labels):09d}', sid, d0.date(), 'product_plant',
                           f'{prod[int(j)]}|{plants.plant_id.values[prod_plant[int(j)]]}', 'demand_drift',
                           HOR, (d0 + pd.Timedelta(days=1)).date(), (d0 + pd.Timedelta(days=HOR)).date(),
                           round(v, 6), bool(c), int(HOR) if c else 0,
                           'forward actual / standing plan', 'labels-v8', d0])
LB = pd.DataFrame(labels, columns=['label_id','snapshot_id','snapshot_date','entity_type','entity_id','task',
    'horizon_days','label_window_start','label_window_end','label_value','label_censored','censor_time',
    'label_definition','label_version','created_ts'])
LB['asserted'] = True
LB['label_value'] = LB.label_value.astype(float).round(4)
emit('training_labels.csv', LB)
emit('snapshots.csv', pd.DataFrame(snaps, columns=['snapshot_id','as_of_ts','data_cutoff_ts','horizon_days',
    'node_counts','edge_counts','feature_spec_version','dataset_version','label_version','code_commit','created_ts']))

# ---------------------------------------------------------------- §2.5 arc 6: recorded allocation
# The share the buyer actually ran with, as it was revised quarterly against trailing
# delivered performance. Every row is a review outcome, not a 1/n placeholder, and the
# shares inside one (part, plant, effective_from) sum to 100 by construction.
_al_t = cat(ALLOC, 't').astype(int); _al_ch = cat(ALLOC, 'ch').astype(int)
_al_s = cat(ALLOC, 'share'); _al_r = cat(ALLOC, 'reason').astype(int)
_al_pp = PP[_al_ch]
_al_from = pd.to_datetime(WV[_al_t])
_al = pd.DataFrame({'part_id': parts.part_id.values[CP[_al_ch]],
                    'plant_id': plants.plant_id.values[CL[_al_ch]],
                    'supplier_id': suppliers.supplier_id.values[CS[_al_ch]],
                    'allocation_pct': np.round(_al_s, 2), 'effective_from': _al_from,
                    'pp': _al_pp, 'ch': _al_ch, 't': _al_t, 'reason': _al_r})
_al = _al.sort_values(['ch', 't'])
_al['effective_to'] = _al.groupby('ch')['effective_from'].shift(-1)
_al['change_reason'] = np.where(_al.reason.values == 1, 'performance_share_cut',
                        np.where(_al.reason.values == 2, 'performance_share_lift', 'review_no_change'))
_al['changed_by'] = [f'BUY{x:03d}' for x in (_al.ch.values % 40)]
# forward dated: the split is agreed and registered before the quarter it governs opens
_al['recorded_ts'] = _al.effective_from - pd.to_timedelta(
    r.integers(*P['announce_days'], len(_al)), unit='D')
emit('supplier_allocation.csv', _al)
print(f'  allocation reviews {len(_al):,}  share cut {int((_al.reason==1).sum()):,}  '
      f'lifted {int((_al.reason==2).sum()):,}')
# §2.6: qualification status conditioned on supplier tier. FEATURE-LEVEL VARIATION ONLY.
_qs_lab = np.array(['qualified', 'in_progress', 'potential'])
_qp = np.array([P['ct_qual_p'][t_] for t_ in np.where(s_tier[CS] == 'T2', 'tier2', 'tier1')])
_qs = _qs_lab[(r.random(NCH)[:, None] > np.cumsum(_qp, 1)).sum(1).clip(0, 2)]
alt = pd.DataFrame({'part_id': parts.part_id.values[CP], 'supplier_id': suppliers.supplier_id.values[CS],
    'plant_id': plants.plant_id.values[CL], 'qualification_status': _qs,
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

# ============================================================ §2.6 CONTRACT-TERM VARIATION
# FEATURE-LEVEL ONLY. Sampled from distributions conditioned on supplier_tier /
# supplier_type / business_class, which already exist on the supplier master. This
# un-degenerates the columns the arrival / fill / capacity heads read -- v7 shipped
# moq [10,10], lot_size [25,25], max_volume_cap [5000,5000], penalty [10000,10000].
#
# §2.6 / §2.7 HARD LIMIT: none of these terms may feed a QUOTED allocation or
# delivery-schedule recommendation. An invented spread validates the optimiser's
# sensitivity to the invention, not real constraint diversity. See
# allocation_disclaimer.txt, written beside this dataset.
print('§2.6 contract terms (tier/type/class-conditioned, FEATURES ONLY)...')
_cpair = pd.DataFrame({'s': CS, 'p': CP}).drop_duplicates().reset_index(drop=True)
_cs_i = _cpair.s.to_numpy(); _cp_i = _cpair.p.to_numpy(); NCT = len(_cpair)
_tier = np.where(s_tier[_cs_i] == 'T2', 'tier2', 'tier1')
_typ = suppliers.supplier_type.values[_cs_i]
_cls = suppliers.business_class.values[_cs_i]
_moq_mu = (np.array([P['ct_moq_base'][x] for x in _tier])
           * np.array([P['ct_moq_type'][x] for x in _typ]))
CT_MOQ = np.maximum(1, np.round(r.lognormal(np.log(_moq_mu), P['ct_moq_sigma']))).astype(np.int64)
CT_LOT = np.maximum(1, np.round(CT_MOQ * r.uniform(*P['ct_lot_frac'], NCT))).astype(np.int64)
_sup_nparts = np.maximum(1, np.bincount(_cs_i, minlength=NS))[_cs_i]
CT_CAP = np.maximum(CT_MOQ + 1, np.round(
    K.mean(0)[_cs_i] / _sup_nparts * np.array([P['ct_cap_mult'][x] for x in _cls])
    * r.lognormal(0, P['ct_cap_sigma'], NCT))).astype(np.int64)
CT_COMMIT = np.maximum(0, np.round(CT_CAP * r.uniform(*P['ct_commit_frac'], NCT))).astype(np.int64)
CT_BREAK = (CT_MOQ * r.integers(2, 9, NCT)).astype(np.int64)
CT_PEN = np.round(r.lognormal(P['ct_penalty_log_mu'], P['ct_penalty_log_sigma'], NCT), 2)
_cvf = pd.Timestamp('2017-01-01') + pd.to_timedelta(r.integers(0, 1200, NCT), unit='D')
CONTRACTS = pd.DataFrame({'contract_id': [f'CT{i:07d}' for i in range(NCT)],
    'supplier_id': suppliers.supplier_id.values[_cs_i], 'part_id': parts.part_id.values[_cp_i],
    'min_volume_commitment': CT_COMMIT, 'max_volume_cap': CT_CAP, 'moq': CT_MOQ,
    'lot_size': CT_LOT, 'price_break_qty': CT_BREAK, 'penalty_clause_inr': CT_PEN,
    'valid_from': _cvf.date, 'valid_to': (_cvf + pd.to_timedelta(r.integers(730, 2600, NCT), unit='D')).date})
print(f'  contracts {NCT:,}  moq [{CT_MOQ.min()},{CT_MOQ.max()}] median {int(np.median(CT_MOQ))}  '
      f'lot [{CT_LOT.min()},{CT_LOT.max()}]  cap [{CT_CAP.min()},{CT_CAP.max()}]')
# the part-plant planning record inherits the terms of the channels that feed it
_ct_key = _cs_i * NP + _cp_i
_ct_pos = {int(k): i for i, k in enumerate(_ct_key)}
_ch_ct = np.array([_ct_pos[int(k)] for k in (CS * NP + CP)])
PP_MOQ = np.round(np.bincount(PP, weights=CT_MOQ[_ch_ct].astype(float), minlength=NPP)
                  / np.maximum(1, np.bincount(PP, minlength=NPP))).astype(np.int64)
PP_LOT = np.round(np.bincount(PP, weights=CT_LOT[_ch_ct].astype(float), minlength=NPP)
                  / np.maximum(1, np.bincount(PP, minlength=NPP))).astype(np.int64)
PP_MOQ = np.maximum(PP_MOQ, 1); PP_LOT = np.maximum(PP_LOT, 1)

emit('supplier_capacity.csv', pd.DataFrame({'supplier_id': np.repeat(suppliers.supplier_id.values, NMONTH),
    'part_id': '', 'capacity_qty_per_month': DECL.T.ravel().astype(np.int64),
    'capacity_basis': 'declared', 'shift_basis': 2,
    'effective_from': np.tile(_MSTART.date_values, NS),
    'effective_to': np.tile(_MEND.date_values, NS),
    # forward dated: the declaration is on file before the month it governs
    'recorded_ts': np.tile(_MS.values, NS)
                   - pd.to_timedelta(r.integers(*P['announce_days'], NS*NMONTH), unit='D')}))
mo_idx = np.arange(NMONTH)
_sup_part = parts.part_id.values[np.array([CP[CS == i][0] if (CS == i).any() else 0 for i in range(NS)])]
# §3 worked example: observability is NEVER set. A supplier-month is constrained where
# what was ORDERED met or exceeded the latent ceiling, so delivered < ordered; the
# revealed estimate exists only there, and is the delivered volume that revealed it.
# Everywhere else revealed_capacity_est is genuinely null -- that is the mechanism, not
# a gap. declared_capacity_qty is the supplier's own optimistic declaration, not K.
_dmax12 = pd.DataFrame(delivered_hist).rolling(12, min_periods=1).max().to_numpy()
_dp95 = pd.DataFrame(delivered_hist).rolling(12, min_periods=1).quantile(0.95).to_numpy()
_constr = (ordered_hist >= K) & (delivered_hist < ordered_hist)
_evid = pd.DataFrame(_constr.astype(float)).rolling(12, min_periods=1).sum().to_numpy() / 12.0
rc = pd.DataFrame({'supplier_id': np.repeat(suppliers.supplier_id.values, NMONTH),
    'part_id': np.repeat(_sup_part, NMONTH),
    'month': np.tile(_MS.values, NS),
    'max_delivered_12m': _dmax12.T.ravel().astype(np.int64),
    'p95_delivered_12m': _dp95.T.ravel().astype(np.int64),
    'constrained_month_flag': _constr.T.ravel(),
    'declared_capacity_qty': DECL.T.ravel().astype(np.int64),
    'capacity_basis': np.where(_constr.T.ravel(), 'revealed', 'declared'),
    'revealed_capacity_est': np.where(_constr.T.ravel(), delivered_hist.T.ravel().round(), np.nan),
    'capacity_utilisation_observed': np.round(
        np.divide(ordered_hist, np.maximum(DECL, 1.0)).T.ravel(), 4),
    'evidence_strength': np.round(_evid.T.ravel(), 2)})
emit('revealed_capacity_monthly.csv', rc)
OBS_SHARE = float(_constr.mean())
print(f'  capacity observability (MEASURED, not set): {OBS_SHARE*100:.2f}% of supplier-months')
emit('bom.csv', pd.DataFrame({'bom_id': [f'B{i:07d}' for i in range(1, 4001)],
    'product_id': products.product_id.values[r.integers(0, P['n_products'], 4000)],
    'part_id': parts.part_id.values[r.integers(0, NP, 4000)], 'qty_per_unit': r.integers(1, 6, 4000),
    'scrap_factor': np.round(r.uniform(.01, .08, 4000), 3), 'bom_level': 1, 'parent_part_id': '',
    'effective_from': '2016-01-01', 'effective_to': '', 'is_phantom': r.random(4000) < .04}))
emit('part_plant.csv', pd.DataFrame({'part_id': np.repeat(parts.part_id.values, NPL),
    'plant_id': np.tile(plants.plant_id.values, NP),
    'safety_stock_qty': np.bincount(PP, weights=np.where(cold,0,rate*ss_w), minlength=NPP).astype(np.int64),
    'reorder_point_qty': np.bincount(PP, weights=np.where(cold,0,rop), minlength=NPP).astype(np.int64),
    'min_order_qty': PP_MOQ, 'lot_size': PP_LOT,
    'planning_lead_time_days': np.round(PP_COVER * 7).astype(np.int64).clip(7, 120),
    'abc_class': r.choice(list('ABC'), NPP),
    'xyz_class': r.choice(list('XYZ'), NPP), 'effective_from': '2016-01-01', 'effective_to': ''}))
emit('supplier_upstream.csv', pd.DataFrame({'supplier_id': suppliers.supplier_id.values[r.integers(0, NS, 260)],
    'upstream_supplier_id': suppliers.supplier_id.values[r.integers(0, _N_T2, 260)],
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

_exp_n = np.bincount(CP[po_ch[ex_po]], minlength=NP).astype(float) if NX else np.zeros(NP)
_ord_n = np.maximum(1.0, np.bincount(CP[pch], minlength=NP).astype(float))
_exp_rate_part = np.clip(_exp_n / _ord_n, 0, 1)      # share of that part's lines expedited
_AUD_DATE = pd.Timestamp('2019-01-01') + pd.to_timedelta(r.integers(0, 2500, 2000), unit='D')
_NPC = min(4000, NCH)
for nm, df in [('part_costs.csv', pd.DataFrame({'part_id': parts.part_id.values[CP[:_NPC]], 'supplier_id': suppliers.supplier_id.values[CS[:_NPC]],
        'unit_cost_inr': np.round(r.uniform(50, 2500, _NPC), 2),
        'freight_cost_inr': np.round(r.uniform(2, 250, _NPC)
                            * (1 + P['exp_freight_uplift'] * _exp_rate_part[CP[:_NPC]]), 2),
        'effective_from': '2019-01-01', 'effective_to': '2026-12-31'}).drop_duplicates(['part_id','supplier_id'])),
    ('product_economics.csv', pd.DataFrame({'product_id': prod, 'selling_price_inr': r.integers(500, 12000, NPRD).astype(float),
        'contribution_margin_inr': r.integers(80, 2500, NPRD).astype(float), 'effective_from': '2019-01-01', 'effective_to': '2026-12-31'})),
    ('supplier_contracts.csv', CONTRACTS),
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
GEN_VERSION = 'generator_v8'
# §5.6: every output carries seed, generator version, parameter version, code_commit,
# generation timestamp and dataset version. manifest.json holds them machine-readably;
# this is the copy that travels inside the data.
_TAGSTR = (f'{GEN_VERSION}|params-v8.0|rane-v8-seed{SEED}|labels-v8|commit={CODE_COMMIT}|seed={SEED}')
_YEARS = round((pd.Timestamp(P['world_end']) - pd.Timestamp(P['world_start'])).days / 365.25, 1)
cov = []
for f in sorted(OUT.glob('*.csv')):
    cov.append([f.name, P['world_start'], P['world_end'], _YEARS, 'complete', '', 'event',
                sum(1 for _ in open(f, encoding='utf8')) - 1, _TAGSTR, pd.Timestamp.now()])
emit('dataset_coverage.csv', pd.DataFrame(cov, columns=SCH['dataset_coverage.csv']))
# ============================================================ §3 / §5 PARAMETER FILE
# Mechanism parameters (SET) are written separately from measured outcomes (MEASURED).
# Nothing in the MEASURED block was chosen; each one is read off the emitted world. If a
# measured value misses its band, the fix is a SET parameter and a re-run -- never this file.
print('\nmeasuring outcomes (§3 right column -- none of these was set)...')
def _q(x, f):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.quantile(x, f)) if len(x) else float('nan')

_fl = line_fill[pq > 0]
_lead_obs = pl_[pa < T]
_lag_all = np.concatenate([lag_po_d, lag_inv_d])
_yrs = (W[-1] - W[0]).days / 365.25
_pp_bal = PP_OH[T - 1]
_led_fin = pd.DataFrame({'pp': ppk, 'q': iqty}).groupby('pp').q.sum()
_led_vec = np.zeros(NPP, np.int64); _led_vec[_led_fin.index.to_numpy()] = _led_fin.to_numpy()
ROLLFWD_EXACT = bool((_led_vec == PP_OH[T - 1].astype(np.int64)).all())
_rf_bad = int((_led_vec != PP_OH[T - 1]).sum())
print(f'  §2.1 ledger roll-forward == simulator balance for every part-plant: {ROLLFWD_EXACT} '
      f'({_rf_bad} mismatches of {NPP})')
_seas_pp = PP_DEM.sum(1)
_peak = float(np.max([_seas_pp[WMONTH == m].mean() for m in range(1, 13)]))
_trough = float(np.min([_seas_pp[WMONTH == m].mean() for m in range(1, 13)]))
_deg = np.bincount(PP[~cold], minlength=NPP)

MEASURED = {
 '__note__': 'Every value below is READ OFF the emitted world. None is set, clipped or targeted.',
 'entity_population': {
   'sourcing_channels': int(NCH), 'channels_ever_traded': int(len(np.unique(pch))),
   'cold_start_share_pct': round(float(cold.mean()) * 100, 3),
   'channel_week_rows': int(NCH * T),
   'channel_weeks_per_traded_channel': int(T),
   'po_lines_per_channel_per_year': round(float(NPO / max(1, len(np.unique(pch))) / _yrs), 3),
   'median_channel_degree_part_plant': float(np.median(_deg[_deg > 0]))},
 'fill_rate': {'mass_at_0_pct': round(float((_fl <= 1e-9).mean()) * 100, 3),
   'mass_at_1_pct': round(float((_fl >= 1 - 1e-9).mean()) * 100, 3),
   'mean': round(float(_fl.mean()), 5), 'p10': round(_q(_fl, .10), 5), 'p50': round(_q(_fl, .50), 5)},
 'lead_time_days': {'median': round(_q(_lead_obs, .50), 3), 'p90': round(_q(_lead_obs, .90), 3),
   'p99': round(_q(_lead_obs, .99), 3),
   'skewness': round(float(pd.Series(_lead_obs).skew()), 4),
   'late_vs_contract_pct': round(float((pl_[pa < T] > ppr[pa < T]).mean()) * 100, 3)},
 'recording_lag_days': {'p50': round(_q(_lag_all, .50), 3), 'p90': round(_q(_lag_all, .90), 3),
   'max': round(float(np.nanmax(_lag_all)), 3),
   'sd_over_mean': round(float(np.nanstd(_lag_all) / max(abs(np.nanmean(_lag_all)), 1e-9)), 4),
   'later_week_share_pct': round(float((_wk_index(rec_po) > _wk_index(ev_po)).mean()) * 100, 3),
   'never_recorded_share_pct': round(float(nrec_po.mean()) * 100, 3)},
 'capacity': {'observable_supplier_month_pct': round(OBS_SHARE * 100, 3),
   'revealed_est_non_null_pct': round(float(rc.revealed_capacity_est.notna().mean()) * 100, 3),
   'declared_over_latent_median': round(float(np.median(DECL / K)), 4)},
 'events_per_year': {'shortage_events': round(NE / _yrs, 2), 'line_stops': round(NL / _yrs, 2),
   'expedite_actions': round(NX / _yrs, 2), 'po_line_revisions': round(int(REV_IN.sum()) / _yrs, 2),
   'po_line_revisions_never_entered': int((~REV_IN).sum()),
   'shortage_condition_weeks': round(N_COND / _yrs, 2),
   'shortage_cause_corroborated_pct': round(CAUSE_SHARE * 100, 2),
   'shortage_root_cause_mix': {k: int(v) for k, v in pd.Series(CAUSE_KIND).value_counts().items()},
   'shortage_cause_labelled_pct': 100.0},
 'demand': {'peak_trough_ratio': round(_peak / max(_trough, 1e-9), 4)},
 'opening_stock_rollforward': {
   'opening_units_posted': int(OPEN_CH.sum()),
   'channels_with_opening_stock': int((OPEN_CH > 0).sum()),
   'part_plants_empty_at_open': int((np.bincount(PP, weights=OPEN_CH, minlength=NPP) == 0).sum()),
   'ledger_equals_simulator_balance': ROLLFWD_EXACT, 'mismatched_part_plants': _rf_bad,
   'balance_p50_at_cut': float(np.median(_pp_bal)), 'balance_max_at_cut': int(_pp_bal.max()),
   'weeks_with_any_negative_balance': int((PP_OH < 0).any(1).sum())},
 'forward_plan': {'versions_per_target_period_max': _vmax,
   'distinct_horizons_weeks': int(_hu),
   'median_periods_per_published_version': int(PLAN.groupby(['pi', 'tp']).size().median()),
   'min_periods_per_published_version': int(PLAN.groupby(['pi', 'tp']).size().min()),
   'part_demand_distinct_horizon_days': int(len(np.unique((_dtw - _dtp) * 7)))},
 'contract_terms_features_only': {
   'moq_distinct': int(len(np.unique(CT_MOQ))), 'moq_cv': round(float(CT_MOQ.std() / CT_MOQ.mean()), 4),
   'lot_size_distinct': int(len(np.unique(CT_LOT))),
   'max_volume_cap_distinct': int(len(np.unique(CT_CAP))),
   'penalty_distinct': int(len(np.unique(CT_PEN))),
   'qualification_status_shares': {k: round(float(v), 4) for k, v in
       pd.Series(_qs).value_counts(normalize=True).items()}},
 'feedback_arcs': {'allocation_reviews': int(len(_al)),
   'allocation_share_cuts': int((_al.reason == 1).sum()),
   'allocation_share_lifts': int((_al.reason == 2).sum()),
   'buyer_revisions': int((REV_BY == 0).sum()), 'supplier_revisions': int((REV_BY == 1).sum()),
   'expedited_po_lines': int(was_exp.sum()),
   'alt_sourcing_touched_channels': int((alt_boost > 0).sum())},
 'conservation': {'ordered_units': SRC_ORD, 'received_units': SRC_REC,
   'ledger_identity_exact': LEDGER_OK},
}
TAGS = {'seed': SEED, 'generator_version': GEN_VERSION, 'parameter_version': 'params-v8.0',
        'dataset_version': f'rane-v8-seed{SEED}', 'label_version': 'labels-v8',
        'code_commit': CODE_COMMIT, 'generated_at': pd.Timestamp.now().isoformat(),
        'is_smoke_run': bool(NCH < 15000),
        'schema': str(A.spec), 'rows_channel_week': int(NCH * T)}
json.dump({'tags': TAGS,
           'SET__mechanism_parameters': {k: _jsonable(v) for k, v in P.items()},
           'MEASURED__outcomes': MEASURED,
           'EXCLUDED__by_design': {
             '__why__': ('brief 2.7: carrying cost, ordering cost, freight structure and shortage '
                         'cost are Rane business decisions, not mechanisms. No distribution '
                         'produces a meaningful value; any number chosen determines the optimiser '
                         'answer directly, so testing against it only tests arithmetic.'),
             'carrying_cost_rate': None, 'ordering_cost_per_po': None,
             'shortage_cost_per_unit': None, 'freight_rate_structure': None,
             'truck_capacity': None, 'holding_cost_per_unit_week': None,
             'part_costs.unit_cost_inr': 'PLACEHOLDER - arbitrary draw, carried only because the '
                                         'schema requires the column. Not a Rane price.',
             'part_costs.freight_cost_inr': 'PLACEHOLDER - arbitrary draw. NOT a freight structure.',
             'expedite_events.premium_cost_inr': 'PLACEHOLDER - exp_premium_per_unit is a scale '
                                                 'constant, not a costed Rane rate.'}},
          open(OUT / 'parameters_v8.json', 'w'), indent=1, default=str)
json.dump(MEASURED, open(OUT / 'measured_outcomes.json', 'w'), indent=1, default=str)
json.dump(TAGS, open(OUT / 'manifest.json', 'w'), indent=1, default=str)
(OUT / 'allocation_disclaimer.txt').write_text(
 "ILLUSTRATIVE ONLY -- REAL CONSTRAINTS REQUIRED FOR A QUOTABLE RECOMMENDATION\n"
 "===========================================================================\n"
 "v8 generates contract terms (moq, lot_size, min_volume_commitment, max_volume_cap,\n"
 "penalty_clause_inr, qualification_status) from distributions conditioned on\n"
 "supplier_tier / supplier_type / business_class. Per the v8 brief 2.6 these are\n"
 "PREDICTION-TASK FEATURES ONLY.\n\n"
 "They are NOT valid inputs to a quoted allocation (UC 10.2) or delivery-schedule\n"
 "(UC 10.1) recommendation. An invented spread validates the optimiser's sensitivity\n"
 "to the invention, not real constraint diversity.\n\n"
 "No carrying cost, ordering cost, freight structure or shortage cost exists in this\n"
 "dataset (brief 2.7). UC 10.2 has no quotable output without a client-supplied\n"
 "shortage cost; that is a blocking item, not an assumption. Any optimiser run against\n"
 "this world must be labelled 'illustrative -- real constraints required'.\n")
print(f'  parameter file -> {OUT / "parameters_v8.json"}  (SET and MEASURED separated)')

print(f'\nGENERATION COMPLETE seed={SEED} -> {OUT}')
print(f'  conservation ordered EXACT {SRC_ORD:,} | received EXACT {SRC_REC:,} | ledger {LEDGER_OK}')
print(f'  opening-stock roll-forward reconciles to the ledger: {ROLLFWD_EXACT}')
print(f'  shortage events with identifiable upstream cause: {CAUSE_SHARE*100:.1f}%')

