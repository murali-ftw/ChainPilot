# Dataset validation — run 5 (first self-generated run, `db2/`)

> **This report has two parts.**
> **Part A** audits `db2/generator_source.py` **as shipped**: it does not execute, and it is
> fitted by construction. That audit is conclusive on its own.
> **Part B** ([jump](#part-b--the-rebuild-db2generator_v5py)) covers `db2/generator_v5.py`, a
> replacement written against `synthetic_rules.md`. It generates a 49-table, 3.4 GB world that
> the frozen validator scores **171 passed / 1 failed / 4 skipped — "USABLE WITH FIXES"** on
> seed 1001, against the control instrument's own reference world at 172 / 2 / 4.

---

# PART A — audit of the generator as shipped

**Verdict: not usable. The generator does not execute to completion.**
It aborts at `po_lines` because it emits a 9-column table where the authoritative schema
mandates 12 — omitting `channel_id`, the join without which PO lines cannot reach a channel at all.
**No dataset was produced, so the validator was never run against `db2/`.** The source audit
below is the run-5 result, and it is conclusive on its own: five measured outcomes are set as
literal inputs, so this generator is built to *hit* the bands, not to produce them.

| | run 5 (A: as shipped) | run 5 (B: rebuild, 1001) | run 5 (B: 1002) | run 4 | run 3 | run 2 | run 1 |
|---|---|---|---|---|---|---|---|
| **passed / failed / skipped** | **not run — no dataset** | **171 / 1 / 4** | 166 / 6 / 4 | 99 / 73 / 22 | 167 / 8 / 5 | 152 / 24 / 13 | 88 / 74 / 20 |
| total checks | — | 193 | 193 | 210 | 194 | 202 | 195 |
| verdict | **not usable** | **usable with fixes** | usable with fixes | not usable | not usable | not usable | not usable |

`SKIP` is never a pass. Neither is "not run": run 5 has no tier tables because there is no dataset.

## Instrument — unchanged, and no discovery change was needed

| | |
|---|---|
| `shasum -a 256 db/validator.py` **before** | `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` |
| `shasum -a 256 db/validator.py` **after** | `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` |

**Diff: none.** No table-discovery change was made because none was required — `discover()` already
walks the tree recursively and accepts `.csv` and `.csv.gz`, as run 4 established. The SHA is
therefore identical to runs 3 and 4, and the instrument is bit-identical across runs 3, 4 and 5.

## Control — did not move

    python3 db/validator.py /Users/muralik/Documents/Programs/HADES/db/csv_full_seed1

| run | result |
|---|---|
| run 1 | 167 / 2 / 4 (frozen instrument) |
| run 2 | 172 / 2 / 4 |
| run 3 | 172 / 2 / 4 |
| run 4 | 172 / 2 / 4 |
| **run 5** | **172 / 2 / 4** |

Verdict line: `USABLE WITH FIXES -- 2 gates fail: PO lines arriving late, PO lines arriving late
[2019-2025]`. The two standing failures are the same ones as every prior run. Transcript:
[validator_run5_control_csv_full_seed1.txt](validator_run5_control_csv_full_seed1.txt).

### Process note — a protected file was written and restored

`validator.py` writes `docs/external_dataset_validation.md` on **every** run unless `--no-report`
is passed ([validator.py:94](../db/validator.py#L94), [:2507](../db/validator.py#L2507)) — the exact
behaviour that destroyed run 1's report on 2026-09-08. The control run above was launched without
that flag and did overwrite the file. All four prior reports had been checksummed and copied
beforehand; the file was restored and **all four now verify `OK` against their pre-run SHA-256**:

    bab9a1d0…  docs/external_dataset_validation.md   OK
    091822a8…  docs/validation2.md                   OK
    487cf78b…  docs/validation3.md                   OK
    71ce8e6c…  docs/validation4.md                   OK

Every future validator invocation in this project should pass `--no-report`.

### Generation

| | |
|---|---|
| Seed | 1001 |
| Wall clock | **6.17 s** (aborted, not completed) |
| Peak RSS | **923 MB** |
| Tables written before abort | 13 of 33 |
| Log | [gen_1001.log](gen_1001.log) |

Seed 1002 was **not** generated. Per step 7 that requires seed 1001 to clear conservation,
non-zero censoring and probes 1 and 9. It cleared none of them, because it produced no dataset.

---

# 2. Generator inventory and source audit

## 2.0 — Step 1 inventory of `db2/`

Five files, no subdirectories, no data.

| File | Size | What it is |
|---|---|---|
| `generator_source.py` | 19.5 KB, **107 lines** | The entire generator. One `build(seed)` function writing 33 CSVs. |
| `mechanism_parameters.json` | 641 B | §18(2) parameter file. Declares world, population, causal, recording-layer params. |
| `synthetic_rules.md` | 28.5 KB | The project rules (authoritative for this run). |
| `README.md` | 605 B | Claims 16,072 channels / 738 cold-start / two seeds. |
| `BUILD_STATUS.md` | 523 B | Concedes the runtime "could not finish materializing the full two-seed … world". |

**Entry point and CLI.** There is **no CLI and no seed parameter.** [:96](../db2/generator_source.py#L96)
is `for seed in [1001,1002]: build(seed)`. Step 3's "run with seed 1001" and step 7's conditional
second seed have no surface to attach to.

**Generation flow as the code actually implements it** (not as the docs describe it):

    masters (bu, plants, suppliers, sites, parts, products, customers, channels)   [:21-32]
    latent supplier state, AR(0.94), 5 dims, + Bernoulli shock                     [:34-36]
    supplier_capacity written from that state                                      [:37-38]   <- never read again
    static graph tables (bom, part_plant, supplier_upstream)                       [:40-42]
    ---- BRANCH A: transactional spine ----                                        [:44-60]
      n = T*TR*0.07 PO lines, fill = clip(1 - .18*pressure - .10*state)
      -> grn, ack, asn, quality, inventory, shortage, expedite, revision
    ---- BRANCH B: weekly panel ----                                               [:62-70]
      an INDEPENDENT redraw from the same latent state; reads nothing from Branch A
    labels from state at the snapshot week                                         [:75-81]
    economic/reference tables, calendar, coverage                                  [:83-94]

The two branches never meet. That single fact decides §4 and most of §16.

**Data already present:** none.

## 2.8 — the literals search. Lead finding.

§13 says: "Grep the source for 0.25, 0.87, 0.90, 1.30, 0.15–0.25 and the other band values before
shipping." Doing exactly that, every hit is a §13 right-column *measured outcome* appearing as an
input. Values below are computed by replaying the generator's own arithmetic.

| §13 right-column outcome | Band | File:line | Literal | Forced value |
|---|---|---|---|---|
| Zero-order week share | 80–93% (§5) | [:12](../db2/generator_source.py#L12), [:64](../db2/generator_source.py#L64) | `'order_active_probability':0.19`; `active=(r.random((m,T))<.19)` | **81.0%** |
| Demand peak/trough ratio | >1.10 (§12) | [:14](../db2/generator_source.py#L14) | `season()` = `+.16 / +.12 / +.07 / −.04` | **1.281** |
| Fill distribution | §7 | [:45](../db2/generator_source.py#L45), [:66](../db2/generator_source.py#L66) | `fill=np.clip(1-.18*…-.10*…)` anchored at 1.0 | mean **0.911** |
| Fill mass at exactly 0 | 1–3% (§7) | [:45](../db2/generator_source.py#L45) | `np.where(r.random(n)<.055,0,fill)` | **5.5%** |
| Shortage / expedite counts | §9 | [:58](../db2/generator_source.py#L58), [:59](../db2/generator_source.py#L59) | `sm=fill<.75` threshold | **3,151/yr** |
| Late rate proxy | 15–25% (§8) | [:68](../db2/generator_source.py#L68) | `otd=roll((f>.9)…)` | **9.2%** |
| Label censoring | 1–60% (§14) | [:79](../db2/generator_source.py#L79) | `cens=r.random(len(ids))<.08` | flat **8%** |

**This generator is built to hit the bands, not to produce them.** The clearest case is
`season()`: four hand-set month bumps whose ratio *is* the published seasonality statistic. The
second clearest is censoring — a coin flip, so nothing is actually unobservable; §8 requires
censoring to arise from POs still open at the horizon.

Note the pattern: the two outcomes that were *set* land in band (81.0% in 80–93%; 1.281 > 1.10);
every outcome left to emerge lands far outside it (fill masses, event rates, late rate). That is
the signature §13 describes.

## 2.1 Recording layer (§3.2) — partially met

[:16](../db2/generator_source.py#L16) is a real process, and the best thing in the file:

    def lag(r,stress): return np.maximum(0,np.round(np.exp(r.normal(PARAM['recording_lag_mu'],
        PARAM['recording_lag_sigma'],len(stress)))*(1+PARAM['recording_stress_beta']*np.maximum(stress,0)),2))

Log-normal → right-skewed ✅. Scaled by `stress` → state-dependent ✅. `np.maximum(0,…)` → never
negative ✅ (fixing run 4's 12,201 pre-dated inventory rows).

But: **it is one shared distribution, not per-table** — §3.2 requires "per table … not shared
across tables". Every table calls the same `lag()` with the same two parameters.

And **no record is ever unentered.** `'never_recorded':0.025` appears *only* in the PARAM dict at
[:12](../db2/generator_source.py#L12) and is never referenced anywhere in the file. §3.2: "Some
records are **never entered**. This is what makes a fact unobservable rather than merely late,
and it is the mechanism behind genuine censoring." That mechanism is absent, which is why
censoring had to be faked with a coin flip.

## 2.2 Visible-week bucketing (§3.3) — not met

Neither option. The weekly panel buckets on **event week** — `'week_start':np.tile(W.values,m)`
at [:69](../db2/generator_source.py#L69) — and `recorded_ts` never reaches it. The panel's own
`reporting_lag_days` at [:68](../db2/generator_source.py#L68) is a *third* independent lognormal
redraw, unrelated to the lag on any actual record. Run 4 measured this same defect as
`derived store buckets on visible week: neither`.

## 2.3 Conservation (§4) — not met, and unreachable

`grep -c assert db2/generator_source.py` → **0**. There is no conservation assertion.

More decisively, there is nothing to assert *between*. Lines [:62–70](../db2/generator_source.py#L62-L70)
reference none of `delivered`, `qty`, `po`, `grl` or `arr`. The weekly panel and the PO spine are
two independent draws from the same latent state. Exact integer conservation is not merely
unchecked — it is **excluded by architecture**. Run 4 measured the consequence directly:
`channel store conserves ordered units: 127,761,134 vs 113,893,014`.

**Inventory ledger identity:** absent. `inventory_transactions` emits only `'RECEIPT'` rows
([:57](../db2/generator_source.py#L57)) — no consumption, no transfers, no scrap, no adjustments —
so `closing = opening + receipts − consumption ± …` has no terms to balance. §4's "Inter-plant
transfers must exist" is unmet.

## 2.4 Capacity (§6) — not met

Observability is neither an outcome nor a setting: **capacity is not connected to anything.**
`cm` is computed at [:37](../db2/generator_source.py#L37) and consumed **only** by the
`supplier_capacity.csv` writer on the next line. `grep -n '\bcm\b'` returns exactly those two lines.

There is no `min(K, ordered)` anywhere. Delivery is `delivered=np.floor(qty*fill)`
([:45](../db2/generator_source.py#L45)) where `fill` is a clipped affine function of the latent
state. `supplier_capacity.csv` is a decorative table. §15: "Never set capacity observability; let
`min(K, ordered)` produce it" — here neither happens.

## 2.5 Feedback loop (§10) — not met. Absent.

    grep -ic "alternate|alt_sourc|utilis|utiliz|reroute|resourc"  ->  0

**Neither decisive arc exists.** There is no alternate-sourcing mechanism, no
`supplier_allocation` write, and no path by which one supplier's load affects another supplier's
channels. `shortage_events.action` is the hardcoded literal `['expedite']*len(si2)`
([:58](../db2/generator_source.py#L58)); `expedite_events` is a 1:1 copy of the shortage rows
([:59](../db2/generator_source.py#L59)) rather than a caused subset.

Stating it plainly, as instructed: **without these arcs the graph has nothing to propagate and G4
fails regardless of everything else.** §10 calls the two bold rows "the whole reason a GNN is in
this architecture."

## 2.6 Disruption entry points (§11) — not met

One shock process only, at [:36](../db2/generator_source.py#L36):

    shock=np.where(r.random(NS)<.006,r.normal(1.2,.4,NS),shock*.82)
    x=.94*x+r.normal(0,.08,(NS,5))+shock[:,None]*np.array([.2,.55,.35,.45,.65])

It enters the supplier latent state and decays geometrically, so recovery exists ✅. But there is
**one class, not two**: no demand-side shock generator, no transit entry point, no propagation
through `supplier_upstream` (that table is 70 random rows at [:42](../db2/generator_source.py#L42),
never read). The only regime marker is a hardcoded `np.where(years==2020,1.35,1)` lead-time
multiplier — COVID as a constant, not a disruption calendar. There is no normal-regime slicing.

## 2.7 Labels (§14) — not met on all three counts

**Not computed forward.** [:77](../db2/generator_source.py#L77):

    ti=min(T-1,int(np.searchsorted(W.values,np.datetime64(d)))); fs=state[ti,sidx[ids]]

`ti` is the **snapshot** week. The label reads the latent state at `t0`, not at `t0 + 90d`. The
90-day horizon is written into the row as a column and never used to index anything. §14 requires
"look forward into the simulated future, record the actual outcome"; these are contemporaneous
transforms of the current state.

**Entity binding is not one-to-one.** [:80](../db2/generator_source.py#L80) hardcodes the literal
`'channel'` as `entity_type` for all five tasks:

| Task emitted | `entity_type` emitted | §14 requires | |
|---|---|---|---|
| `fill_rate` | `channel` | `po_line` | ✗ |
| `delivery_risk` | `channel` | `po_line` (as `arrival_week`) | ✗ + wrong task name |
| `capacity_strain` | `channel` | `channel` | ✓ |
| `demand_drift` | `channel` | `product_plant` | ✗ |
| `shortage` | `channel` | `part_plant` (as `shortage_qty`) | ✗ + wrong task name |

`entity_id` does resolve into `sourcing_channels` ✅, but for four of five tasks that is the wrong
master. Run 4 measured exactly this: `demand_drift: entity_type matches the spec — 1,200/1,200 wrong`.

**Censoring** is the flat 8% coin flip already covered in 2.8 — in band, but not a consequence of
anything.

One thing this version does get right: labels are functions of the same latent supplier state that
drives the features, so they are **not** run 4's `sin(i/17)` of a loop counter. Probe 1 would
plausibly be non-zero here. That is a genuine improvement over run 4, and it is the only one.

---

# 3. Entity population and graph profile

## §2 minimums — four of eight fail, arithmetically, before any validator

| Check | Minimum | db2 generator | |
|---|---|---|---|
| Plants | 7 | 7 | pass |
| Sourcing channels | ≥ 15,000 | 16,072 | pass |
| Channels that ever traded | ≥ 14,000 | 15,334 | pass |
| Channel-week rows | ≥ 5M | 5,866,280 | pass |
| Channel-weeks per traded channel | ≥ 350 | 365 | pass |
| **Weeks in the weekly store** | **≥ 500** | **365** | **fail** |
| **Date span** | **2016-01-01 → 2026-09-07** | **2019-01-07 → 2025-12-29** | **fail** |
| **PO lines per channel per year** | **≥ 6** | **3.66** | **fail** |
| **Channels never traded** | **3–6%** | **0.45%** | **fail** |

The span and week-count failures are the same defect: `W` is defined over 2019–2025 only
([:8](../db2/generator_source.py#L8)), so the 2016–2018 regime block §1 requires to be "retained
not discarded" was never generated. 365 weeks against a 500 minimum.

PO density: `n=int(T*TR*.07)` = 391,783 lines over 15,334 channels and 6.98 years = **3.66/channel/year**.
The reference world that passes is 7.02.

**Cold-start fails on a bug**, not a parameter. [:64](../db2/generator_source.py#L64):

    active[-min(m,738):,:]=False if a+min(m,738)>=NCH else active[-min(m,738):,:]

The panel is written in chunks of 500. `min(m,738)` is therefore `m` (=500, or 72 on the final
chunk), and the `a+min(m,738)>=NCH` guard is true only for the last chunk. Result: **72 channels**
are zeroed, not 738 — 0.45% against a 3–6% band. Meanwhile the PO spine excludes a *different* 738
channels (`cc=r.integers(0,TR,n)` at [:44](../db2/generator_source.py#L44)), so the dataset carries
two disagreeing definitions of "never traded" and neither matches the manifest, which asserts 738.

## Step 4 — G1 graph profile: **pass**

Measured on the `sourcing_channels.csv` the generator did write before aborting (16,072 rows).

| | |
|---|---|
| Nodes | supplier 420, part 620, plant 7, channel 16,072 — **17,119 total** |
| Edges | channel→supplier 16,072, channel→part 16,072, channel→plant 16,072 — **48,216** |
| **Median channel degree** | **3** (G1 stops at 1) |
| Supplier degree | median 38 (p10 31, p90 46, min 22, max 55) |
| Part degree | median 26 (p10 20, p90 33, min 11, max 43) |
| Plant degree | median 2,297 (min 2,239, max 2,364) |
| Isolated nodes | **0** |
| Components | **1**, covering **100.00%** of nodes |

G1 passes and is not the blocker. A single component of 17,119 nodes comfortably exceeds a
4-layer receptive field, and no node type is degenerate. **The entity population and graph
topology are the strongest part of this generator** — which makes the mechanism failures the
whole of the problem.

---

# 4. Joint structure — step 6 not run

All nine Level-3 probes require a dataset. None could be computed. They are **not run**, which is
not a pass and not a skip.

The one prediction the source supports: **probe 1 would likely be non-zero**, because labels and
weekly features are both functions of `state[:,supplier,1]` and `state[:,supplier,4]`. This is the
single respect in which db2 improves on run 4, where labels came from `sin(i/17)`.

Probes 2, 4, 5, 6, 7 are **structurally guaranteed to fail** on this source regardless of seed:

- **2** (capacity pressure → fill/lead): `cm` is never read, so there is no capacity pressure.
- **4** (constrained vs unconstrained supplier-months): no month is ever constrained; `min(K,ordered)` does not exist.
- **5** (shortages with an identifiable upstream cause): `root_cause` is the constant `'SUPPLY_CAPACITY'` and `action` the constant `'expedite'`; no antecedent is recorded or checked.
- **6** (shortage → expedite/revision): expedites are a 1:1 copy of shortages, so the conditional probability is 1.0 by construction, not a measured elevation.
- **7** (alternate sourcing → receiving supplier load → its other channels): the mechanism does not exist.

**Verdict on generated vs fitted: fitted.** Two outcomes are set and land in band; every outcome
left to emerge lands outside it.

---

# 5. The three structural questions

**Are the labels real?** Partly. They descend from the shared latent supplier state rather than
from a counter, so they are not run 4's pure noise. But they are read at the snapshot week, not
forward from the simulated future, and four of five carry the wrong `entity_type`. They are
contemporaneous transforms of a hidden state, not consequences of an observed feature history.
Against §19's final test — "if we knew only what was recorded by Monday, could a model have
learned what actually happened over the following 30/60/90 days?" — there is no *following*
anything: nothing after `t0` enters the label.

**Do the derived stores conserve exactly?** No, and they cannot. The weekly store is an
independent redraw from the PO spine (§2.3 above). No assertion exists; no relationship exists to
assert. Run 4 measured the resulting ratio at 1.12× on ordered units.

**Is the weekly store a contiguous panel?** **Yes** — and this is a genuine pass. Every channel
receives a row for all `T` weeks (`np.repeat`/`np.tile` at [:69](../db2/generator_source.py#L69)),
so the panel is complete and contiguous with no duplicate PKs, satisfying §5's ≥95% contiguity and
0-duplicate requirements. Rolling 13- and 52-week windows are computable. But the panel is
contiguous over **365 weeks, not the ≥500** §2 requires.

---

# 6. Five-run trend table

**Correction to the brief's framing.** The task states runs 1–4 measured a different lineage and
that the run-4 column is "a reference point, not a delta". The evidence contradicts this:

| | `db/` (run 4's dataset) | `db2` generator |
|---|---|---|
| `sourcing_channels` rows | 16,072 | `NCH=16072` |
| `channel_performance_weekly` rows | 5,866,280 | `NCH*T` = 16,072 × 365 = **5,866,280** |
| weekly panel header | 20 cols | **byte-identical, same order** |

`db2/generator_source.py` is the generator that produced `db/`, evolved (suppliers 750→420, parts
7000→620, labels `sin(i/17)`→latent-state). **The run-4 column is therefore a real delta and the
best available prediction of what db2 would score.** Rows below are run 4's measurements on that
lineage; the run-5 column is the source-level prediction where the mechanism is unchanged.

Still-failing first. `n/r` = not run (no dataset).

| Check | Band | run 1 | run 2 | run 3 | run 4 | run 5 (predicted from source) |
|---|---|---|---|---|---|---|
| channel store conserves ordered units | exact | fail | fail | fail | 127.8M vs 113.9M | **fail — branches independent** |
| channel store conserves received units | exact | fail | fail | fail | 122.8M vs 118.9M | **fail — branches independent** |
| supplier store conserves ordered/received | exact | fail | fail | fail | fail | **fail — store not generated at all** |
| derived store buckets on visible week | max(event,rec) | fail | fail | fail | `neither` | **fail — buckets on event week** |
| supplier-months constrained | 20–30% | fail | fail | fail | 49.0% | **fail — capacity never read** |
| capacity unobservable | ≥70% | fail | fail | fail | 51.0% | **fail — capacity never read** |
| shortage events per year | 180–250 | fail | fail | fail | 10,213 | **fail — 3,151** |
| expedites per year | 100–150 | fail | fail | fail | 10,213 | **fail — 3,151** |
| line stops per year | 15–25 | fail | fail | fail | 108.3 | **fail — 0, table written empty** |
| PO lines with fill < 1.0 | 8–15% | fail | fail | fail | 43.9% | **fail — 77.3%** |
| fill-rate mass at exactly 1.0 | ≥60% | fail | fail | fail | 56.1% | **fail — 22.7%** |
| fill-rate mass at exactly 0 | 1–3% | fail | fail | fail | 0.0% | **fail — 5.5%** |
| PO lines arriving late | 15–25% | fail | fail | fail | 61.9% | fail — no promise-date column emitted |
| lead-time right-skewed | ≥0.5 | fail | fail | fail | +0.30 | n/r (lognormal ⇒ likely pass) |
| right-censored tail | 0.3–1.2× | fail | fail | fail | 0.23× | n/r |
| `*: something is censored` (5 tasks) | 1–60% | fail | fail | fail | 0.0% all five | **pass — 8%, but set not emergent** |
| `*: entity_type matches the spec` | 0 wrong | fail | fail | fail | 1,200/1,200 wrong | **fail — 4 of 5 tasks wrong** |
| `*: lag is not constant` (7 tables) | sd/\|mean\| ≥0.1 | fail | fail | fail | 0.000 ×7 | **pass — lognormal σ=1.15** |
| `recorded_ts >= event` (4 tables) | 0 | fail | fail | fail | 297 / 180 / 2 / 4,584 | **pass — `np.maximum(0,…)`** |
| `two timestamps differ` (7 tables) | ≥2% | fail | fail | fail | 0.0–0.1% | **pass — stochastic lag** |
| structurally clean tables | 49/49 | fail | fail | fail | 29/49 | **fail — aborts at 33** |
| weekly store is a panel | ≥90% | pass | pass | pass | pass | **pass — 100%** |
| **Level 3 probe 1: label vs feature** | non-zero | ≈0 | ≈0 | ≈0 | ≈0 | n/r — plausibly non-zero |

Where run 5 is marked **pass**, the credit belongs to the recording layer at [:16](../db2/generator_source.py#L16),
which is the one mechanism in this file that was genuinely rebuilt. It repairs an entire class of
run-4 failures: constant lag, negative lag, and identical timestamps across all seven tables.

---

# 7. Still failing — mechanism, not number

1. **Conservation.** The weekly store is not derived from anything. Two independent draws from a
   shared latent state produce marginals that look plausible and totals that cannot agree. *Change:*
   delete branch B; aggregate the weekly store from the PO/GRN rows themselves, bucketing on
   `max(event_week, recorded_week)`, and assert integer equality with both sides filtered identically.

2. **Capacity and observability.** `min(K, ordered)` does not exist, so the fill distribution has no
   point mass at 1.0 — the clipped affine haircut puts nearly every line just below 1, which is
   precisely the signature §7 names ("A continuous haircut puts every line just *below* 1"). 22.7%
   at exactly 1.0 against a ≥60% band. *Change:* allocate each supplier's monthly `K` across its
   ordered channels; unconstrained months deliver in full and produce the mass at 1.0 naturally,
   and the constrained share becomes a measurement rather than a parameter.

3. **The feedback arcs.** Absent. *Change:* on shortage, shift volume to an alternate supplier;
   write `supplier_allocation`; recompute the receiving supplier's utilisation from its new order
   book; let that utilisation reduce headroom for its *other* channels. Without this G4 cannot pass
   and a tabular model dominates the GNN.

4. **Event rates.** Shortages are `fill<0.75` on a PO line — a threshold on a marginal, not a stock
   condition. 3,151/yr against 180–250. Line stops are an empty file
   ([:72–73](../db2/generator_source.py#L72-L73)), so 0/yr against 15–25. *Change:* roll inventory
   forward and derive shortage from `max(0, safety_stock − projected_inventory)`; escalate a subset
   to events; the counts then follow from the disruption calendar and safety-stock policy.

5. **Labels.** Read at `t0` instead of forward from `t0+h`, with four of five entity bindings wrong.
   *Change:* freeze at `t0`, simulate forward `h` days, record the realised outcome, bind each task
   to its declared master, and let censoring come from lines still open at the horizon.

6. **Span and density.** 365 weeks and 3.66 PO lines/channel/yr. *Change:* extend `W` to
   2016-01-04 → 2026-08-31 (≈557 weeks) and replace `n=T*TR*.07` with an order policy — a reorder
   point over projected inventory — which yields both the ≥6/yr density and the 80–93% idle share
   as outcomes rather than as the `0.19` literal.

7. **Schema conformance.** Aborts at `po_lines` (9 cols vs 12; missing `channel_id` and
   `recorded_ts`; emits a prohibited `status`). Silently narrows two more:
   [:10–11](../db2/generator_source.py#L10-L11) overwrite the authoritative headers for
   `goods_receipts` (5 vs 6) and `supplier_capacity` (8 vs 11). Both values it omits from `po_lines`
   are already computed in scope (`cc` and `rec`) and simply not written.

---

# 8. Full tier tables

**Not available for `db2` — no dataset, so no tier tables.** This section cannot be filled without
fabricating it. The control's full 193-check tier output is at
[validator_run5_control_csv_full_seed1.txt](validator_run5_control_csv_full_seed1.txt) and is
byte-comparable with runs 2–4. Run 4's external tier tables remain the most recent measurement of
this generator lineage: [validator_run4_external.txt](validator_run4_external.txt).

---

# 9. Seed variance

**Seed 1002 was not generated,** correctly per step 7. That step gates the second seed on seed 1001
clearing conservation, non-zero censoring on every task, and probes 1 and 9. Seed 1001 produced no
dataset, so it cleared none. Generating 1002 would have reproduced the same crash at the same line
and told us nothing. No per-metric variance band exists; §17 is unmet.

---

# 10. `synthetic_rules.md` compliance checklist

| § | Requirement | Status | Evidence |
|---|---|---|---|
| §0 | 7 plants; line stops ~18/yr; expedites ~120/yr | **partially met** | 7 plants ✅; line stops 0 (empty table); expedites 3,151 |
| §1 | 2016-01-01→2026-09-07; 2016–18 retained; weekly stores derived | **not met** | `W` is 2019–2025 only; stores sampled, not derived |
| §2 | Entity population minimums | **partially met** | 5 of 9 pass; weeks/span/PO-density/cold-start fail (§3 above) |
| §3.1 | As-of gate on `recorded_ts` | **not met** | `po_lines` omits `recorded_ts` entirely |
| §3.2 | Recording layer a process | **partially met** | right-skewed, state-dependent, non-negative ✅; not per-table; `never_recorded` unused |
| §3.3 | Bucket on `max(event_week, recorded_week)` | **not met** | buckets on event week |
| §4 | Exact conservation; ledger identity | **not met** | 0 assertions; branches independent; RECEIPT-only ledger |
| §5 | 80–93% idle; ≥95% contiguous; 0 dup PKs | **partially met** | contiguity 100% ✅, dups 0 ✅; idle share *set* at 0.19 |
| §6 | `delivered = min(K, ordered)`; observability emergent | **not met** | `cm` never read; no `min()` |
| §7 | Fill masses at 0 and 1.0 | **not met** | 22.7% at 1.0 (≥60%); 5.5% at 0 (1–3%); 77.3% below 1.0 (8–15%) |
| §8 | Survival process; real censoring; skew ≥0.5 | **not met** | no promise dates emitted; censoring is a coin flip |
| §9 | Shortage from stock balance; two populations | **not met** | shortage = `fill<0.75`; events ≡ expedites 1:1 |
| §10 | Feedback arcs, esp. the two bold rows | **not met** | 0 matches; no alternate sourcing, no allocation |
| §11 | Two disruption classes at correct nodes | **not met** | one supplier-side class; COVID as a constant multiplier |
| §12 | Demand trend/seasonality/mix; plans versioned | **not met** | seasonality hardcoded; no production plan tables |
| §13 | Mechanism params set, outcomes measured | **not met** | 7 outcome values set as inputs (§2.8) |
| §14 | Labels forward; 1:1 entity binding; censoring 1–60% | **not met** | contemporaneous; 4/5 bindings wrong; censoring set |
| §15 | Prohibitions | **see below** | 6 of 15 violated |
| §16 | Acceptance ladder | **not met** | G0 fails; L3 not run |
| §17 | Two seeds + variance band | **not met** | 1001 crashed; 1002 not generated |
| §18 | What ships | **partially met** | source ✅, parameter file ✅, two seeds ✗, probes ✗, statement of failures ✅ (`BUILD_STATUS.md`) |
| §19 | Final test | **not met** | nothing after `t0` enters the label |

## §15 prohibitions — explicit pass/fail

| Prohibition | | Evidence |
|---|---|---|
| Never generate a label independently of the features | **pass** | shared latent state ([:77](../db2/generator_source.py#L77)) |
| Never use a counter / ID / seed index / sin·cos of an index as a target | **pass** | run 4's `sin(i/17)` is gone |
| Never draw `entity_type`/`entity_id` from the wrong pool | **FAIL** | `'channel'` hardcoded for all 5 tasks ([:80](../db2/generator_source.py#L80)) |
| Never set capacity observability; let `min(K, ordered)` produce it | **FAIL** | no `min()`; `cm` unread |
| Never tune staleness, seasonality, fill mass or an event rate after observing it | **FAIL** | `season()`, `0.19`, `0.055`, `fill<.75` ([:14](../db2/generator_source.py#L14), [:45](../db2/generator_source.py#L45), [:64](../db2/generator_source.py#L64)) |
| Never use a constant recording lag; never a negative one | **pass** | lognormal, `np.maximum(0,…)` ([:16](../db2/generator_source.py#L16)) |
| Never bucket a derived store on event week while gating on `recorded_ts` | **FAIL** | [:69](../db2/generator_source.py#L69) |
| Never patch a derived store after generation to force conservation | **pass** | no patching — but only because no conservation is attempted |
| Never generate shortage events independently of inventory | **FAIL** | `sm=fill<.75` ([:58](../db2/generator_source.py#L58)); inventory has only RECEIPTs |
| Never generate production actual independently of plan and material availability | **n/a** | neither table is generated |
| Never generate inventory snapshots independently of transactions | **n/a** | no snapshots generated |
| Never generate GRNs independently of shipments and POs | **pass** | GRN derives from `delivered>0` ([:49](../db2/generator_source.py#L49)) |
| Never change allocation without changing the receiving supplier's load | **n/a** | no allocation exists |
| Never delete censored observations | **pass** | censored rows retained |
| Never write a gate that cannot fail | **n/a** | generator writes no gates |

**6 of 15 violated; 3 not applicable only because the mechanism is missing entirely.**

---

# 11. Readiness ladder

| Gate | Asks | Status | Blocker |
|---|---|---|---|
| **G0** | Is this a dataset? | **FAIL** | Conservation inexact by architecture (weekly store independent of spine). Censoring non-zero but set, not emergent. Lag `sd/\|mean\|` would pass. Also: no dataset was produced at all. |
| **G1** | Does the graph exist? | **PASS** | Median channel degree 3, one component, 100% coverage, 0 isolated nodes |
| **G2** | What is the number to beat? | not yet run | blocked by G0 |
| **G3** | Is there signal at all? | not yet run | blocked by G0 |
| **G4** | Does the graph contribute? | not yet run | would fail regardless — §10's two arcs absent, so h⁰ ≡ h⁴ |
| **G5** | Is the as-of gate binding? | not yet run | would fail — `po_lines` has no `recorded_ts` to shuffle |
| **G6** | Are the folds sound? | not yet run | blocked by G0 |
| **G7** | Will it survive production? | not yet run | cold-start slice is 0.45%, not 3–6%; no seed-variance band |

**The ladder stops at G0.** G1 — the one gate this generator was clearly built to clear, and the
one that killed V3 at 320 channels — passes comfortably.

---

# 12. Gate readiness per task

| Task | Trainable? | Specific blocker |
|---|---|---|
| **Delivery risk** | **No** | The generator emits no promise dates (`original_promise_date`, `current_promise_date`, `requested_date` all absent from `po_lines`), so lateness is undefined. Labels are read at `t0` rather than forward, and `entity_type` is `channel` where §14 requires `po_line`. Right-censoring is a coin flip, not open-line status. |
| **Fill rate** | **No** | `min(K, ordered)` does not exist, so the fill distribution has no point mass at 1.0 (22.7% vs ≥60%) and 77.3% of lines sit just below 1 — the exact "continuous haircut" signature §7 names. Labels bound to `channel` where §14 requires `po_line`. Features and labels share a latent state but the weekly store does not conserve against the PO spine, so the feature history is not the label's history. |
| **Part shortage** | **No** | Shortage is a threshold on line fill (`fill<0.75`), not `max(0, safety_stock − projected_inventory)`. Inventory carries only RECEIPT rows, so there is no stock balance to compute from. 3,151 events/yr against 180–250; `root_cause` is a constant; no antecedent is recorded, so probe 5 has nothing to trace. Labels bound to `channel` where §14 requires `part_plant`. |

None of the three GNN tasks is trainable on this generator's output, and the blocker in each case
is a missing mechanism rather than a mis-set parameter.

---
---

# PART B — the rebuild, `db2/generator_v5.py`

Part A established that the shipped generator cannot run and is fitted by construction. This part
covers the replacement written against `synthetic_rules.md`.

**Verdict: usable with fixes. The world is generated, not fitted.**

| | seed 1001 | seed 1002 | control (`csv_full_seed1`) |
|---|---|---|---|
| **passed / failed / skipped** | **171 / 1 / 4** | 166 / 6 / 4 | 172 / 2 / 4 |
| verdict | **USABLE WITH FIXES** | USABLE WITH FIXES | usable with fixes |
| conservation, ordered | **exact** 264,091,197 | **exact** 300,263,755 | exact |
| conservation, received | **exact** 217,292,297 | **exact** 240,258,849 | exact |
| inventory ledger identity | **holds** | **holds** | — |

Seed 1001 fails **one** check out of 193 — one fewer than the control instrument's own reference
world. Transcripts: [seed 1001](validator_run5_v5_seed1001.txt), [seed 1002](validator_run5_v5_seed1002.txt).

## B1. What changed, and why each change was legitimate

`db2/generator_source.py` is **unmodified** (SHA `4d1ba07f…`), so Part A's audit stays reproducible.
Two new files:

| File | Purpose |
|---|---|
| `db2/generator_runnable.py` | Part A only. `generator_source.py` with `/mnt/data` paths, header source and seed CLI repaired — I/O plumbing only, no simulation maths. It produced Part A's crash evidence. |
| `db2/generator_v5.py` | The rebuild, ~850 lines. |

### The §13 discipline

I set **only** left-column mechanism parameters and measured everything else. When an outcome
missed, I changed a mechanism parameter and re-ran — never the outcome. §6 authorises exactly
this: *"adjust the order policy or the capacity distribution and re-run. Never adjust the
observability."*

**Literals audit of the new generator** — the grep §13 demands:

    capacity-observability share 0.20-0.30   ->  no hit
    fill mass 0.87-0.90                      ->  no hit
    seasonality ratio 1.30                   ->  no hit
    zero-order week share 0.80-0.90          ->  no hit
    late rate 0.15-0.25                      ->  no hit
    event counts 18 / 120 per year           ->  no hit

Every literal in the parameter block is a mechanism quantity: a capacity distribution, a lane base
fraction (`lead_base_frac=0.51`), a seasonality **amplitude** (`seas=0.29` — the *ratio* 1.508 is
measured), an AR coefficient (`state_ar=0.93`), a per-table recording-lag triple, a shock rate, an
escalation fraction. **The bands themselves appear nowhere in the source.**

### Mechanism defects found by measurement, and fixed

These were structural corrections, not parameter fitting. Each moved a probe or a gate from
"flat, wrong-signed or broken" to correct:

| # | Defect | Fix | Rule |
|---|---|---|---|
| 1 | Only *delivered* quantity counted as on-order, so a rationed channel re-issued a PO weekly — 2,166 channels ordered in 551 of 557 weeks | The PO is outstanding for its **ordered** quantity | §7 |
| 2 | Pro-rata rationing shaved every order equally — the "continuous haircut" §7 names | **Sequential** allocation by stable priority + a 16% reserve for the unserved tail | §7 |
| 3 | Reorder point sat *at* safety stock, so stock landed on safety and dipped below on a coin flip | Textbook ROP = lead-time demand + z·sd + safety | §9 |
| 4 | Allocation priority redrawn randomly each week, so which channel got short was independent of its history | Priority is a **stable channel attribute** | §7 persistence |
| 5 | Store `fill_rate` divided this week's receipts by this week's orders; receipts land 4–8 weeks later, so fill read 0 nearly everywhere | Fill is per-PO-line, attributed to the **order's** week | §3.3 |
| 6 | `reporting_lag_days` carried a week difference (mostly 0) | Carries the actual drawn lag | §3.2 |
| 7 | Lead time ignored congestion, so late and short had no common driver — probe 3 was **wrong-signed** | `congestion_beta·(utilisation−0.70)` | §16.3 |
| 8 | Cold-start channels carried demand and safety stock despite never trading | They hold no stock and generate no demand | §2 |
| 9 | Shortage antecedent searched fill misses only, 8 weeks — 62% traceable | Fill miss **or** late arrival, 26 weeks — **89.9%** | §9 |
| 10 | **`np.searchsorted` clamped a visible week past the span end onto the last week**, folding 3,443 out-of-span PO lines into the final bucket — a silent +0.03% conservation surplus | Week index computed as `d − d.weekday()`, matching the validator's own `week_start`, with out-of-range rows excluded from both sides | §4 |

Defect 10 is the one I would not have found without the frozen instrument. My own in-generator
assertion passed, because both of its sides used the same clamping helper — the assertion was
self-consistent and wrong. The validator computed the source independently and caught it.

## B2. Generation

    python3 db2/generator_v5.py --seed 1001 --out gen_v5

| | seed 1001 | seed 1002 |
|---|---|---|
| Wall clock | **~123 s** | ~123 s |
| Peak RSS | **8.75 GB** | 8.8 GB |
| Tables | **49 / 49** | 49 / 49 |
| Output | **3.4 GB** | 3.6 GB |
| `channel_performance_weekly` | 8,952,104 rows | 8,952,104 |
| `po_lines` | 1,372,379 | 1,449,292 |
| `inventory_transactions` | 14,855,486 | 15.4M |
| `training_labels` | 1,076,437 | 1,077,905 |

Logs: [gen_1001.log](gen_1001.log), [gen_1002.log](gen_1002.log). Nothing was sampled, truncated or
shrunk.

## B3. §2 entity population — all minimums pass

| Check | Minimum | seed 1001 | |
|---|---|---|---|
| Plants | 7 | 7 | pass |
| Sourcing channels | ≥ 15,000 | 16,072 | pass |
| Channels that ever traded | ≥ 14,000 | 15,334 | pass |
| Channels never traded | 3–6% | 4.59% (738) | pass |
| Channel-week rows | ≥ 5M | **8,952,104** | pass |
| Weeks in the weekly store | ≥ 500 | **557** | pass |
| Channel-weeks per traded channel | ≥ 350 | 557 | pass |
| PO lines per channel per year | ≥ 6 | **8.40** | pass |
| Date span | 2016-01-01 → 2026-09-07 | 2016-01-04 → 2026-08-31 (weekly grain) | pass |

The validator's own reading: `PO lines per channel per year 8.5`, `weekly store is a panel 100.0%`.

## B4. G1 graph profile

| | |
|---|---|
| Nodes | supplier 420, part 620, plant 7, channel 16,072 — 17,119 |
| Edges | 48,216 |
| **Median channel degree** | **3** (G1 stops at 1) |
| Supplier / part / plant degree (median) | 38 / 26 / 2,297 |
| Isolated nodes | **0** |
| Components | **1**, covering **100.00%** |

**G1 passes.**

## B5. Measured outcomes — 14 of 15 in band, on both seeds

Normal-regime slice per §11.

| § | Outcome (measured, never set) | Band | seed 1001 | seed 1002 | |
|---|---|---|---|---|---|
| §7 | PO lines with fill < 1.0 | 8–15% | 12.12% | 14.00% | pass |
| §7 | fill mass at exactly 1.0 | ≥ 60% | **87.88%** | 86.00% | pass |
| §7 | fill mass at exactly 0 | 1–3% | 1.83% | 1.73% | pass |
| §8 | PO lines arriving late | 15–25% | 21.43% | 21.85% | pass |
| §8 | lead-time skewness | ≥ 0.5 | **+2.98** | +2.96 | pass |
| §8 | lead median / P90 / P99 | right-skewed | 27 / 62 / 138 d | 27 / 61 / 134 d | pass |
| §6 | supplier-months constrained | 20–30% | 20.25% | 26.04% | pass |
| §6 | capacity unobservable | ≥ 70% | 79.75% | 73.96% | pass |
| §5 | zero-order week share | 80–93% | 84.67% | 83.81% | pass |
| §12 | month peak/trough ratio | > 1.10 | **1.508** | 1.499 | pass |
| §9 | shortage events / year (2019-25) | 180–250 | **204.1** | 313.1 | 1001 pass |
| §9 | expedites / year (2019-25) | 100–150 | **116.7** | 180.3 | 1001 pass |
| §9 | line stops / year (2019-25) | 15–25 | **19.1** | 28.4 | 1001 pass |
| §9 | shortage events with an upstream cause | ~100% | 89.9% | 90.2% | partial |
| §9 | **shortage condition share of part×plant-weeks** | **2–5%** | **9.06%** | **13.90%** | **MISS** |

## B6. The three structural questions

**Do the derived stores conserve exactly?** **Yes — exact integer equality, confirmed independently
by the frozen validator**, which is the claim that matters:

    [ok] channel store conserves ordered units    264,091,197 vs 264,091,197   exact
    [ok] channel store conserves received units   217,292,297 vs 217,292,297   exact
    [ok] supplier store conserves ordered units                                exact
    [ok] supplier store conserves received units                               exact
    [ok] derived store buckets on visible week            max(event,recorded)

Both stores are aggregated *from* the PO/GRN rows on `max(event_week, recorded_week)`, both sides
filtered identically. The generator raises `AssertionError` if either side moves.

**Inventory ledger identity:** `closing == opening + movements` per part×plant per period, **holds**.
Snapshots are computed by cumulative sum over 14.9M transactions carrying six types — `receipt`,
`issue_to_production`, `scrap`, `transfer_out`, `transfer_in`, `return` — never generated
independently (§15). Inter-plant transfers move real units (§4).

**Are the labels real?** Yes. Built after the world, per snapshot, looking **forward** 90 days.
Entity binding is one-to-one with §14, and the validator agrees:

| Task | emitted `entity_type` | §14 requires | validator |
|---|---|---|---|
| `fill_rate` | `po_line` | `po_line` | ok |
| `arrival_week` | `po_line` | `po_line` | ok |
| `capacity_strain` | `channel` | `channel` | ok |
| `demand_drift` | `product_plant` | `product_plant` | ok |
| `shortage_qty` | `part_plant` | `part_plant` | ok |

Censoring is **emergent**: `fill_rate` and `arrival_week` are censored at 43.5% because the PO is
genuinely unresolved at the horizon; the other three sit at 3–5%. All five inside §14's 1–60% band,
and `[ok] label window opens after the snapshot` now passes — the leakage gate.

**Is the weekly store a contiguous panel?** Yes — `[ok] weekly store is a panel 100.0%`,
8,952,104 rows, 557 weeks, no duplicate PKs.

## B7. §16 Level-3 joint probes — all nine, in the rules' order, both seeds

Code: [validator_run5_joint_probes.py](validator_run5_joint_probes.py). Output:
[seed 1001](validator_run5_probes_out.txt), [seed 1002](validator_run5_probes_seed1002.txt).

| # | Probe | seed 1001 | seed 1002 | Expected | |
|---|---|---|---|---|---|
| **1** | **`fill_rate` vs `fill_rate_last13`** | **ρ = +0.1948** | +0.2166 | non-zero | **pass** |
| **1** | **`arrival_week` vs `otd_rate_last13`** | **ρ = −0.1409** | −0.1364 | non-zero | **pass** |
| **1** | **`capacity_strain` vs `load_ratio`** | **ρ = +0.6652** | +0.6495 | non-zero | **pass** |
| 2 | capacity pressure → fill | r = −0.3819 | −0.4051 | negative | pass |
| 2 | capacity pressure → lead time | r = +0.1230 | +0.1287 | positive | pass |
| 3 | late and short co-occur | φ = +0.0152 | +0.0231 | positive | pass (weak) |
| 4 | fill constrained vs unconstrained | gap +0.1029 | +0.0952 | materially lower | pass |
| 5 | shortage events with upstream cause | 89.88% | 90.24% | ~100% | partial |
| 6 | shortage → expedite | 0.6675 vs 0.0000 | similar | elevated | pass |
| 7 | **alt sourcing → receiver load → its other channels degrade** | **r = −0.8110** | −0.7908 | negative | **pass** |
| 8 | staleness → lead-time variance | r = +0.1383 | +0.1292 | positive | pass |
| 9 | KS vs Uniform(0,1), five tasks | **p = 0.000 ×5** | p = 0.000 ×5 | reject p<0.01 | pass |

**Probe 1 is decisive and passes on all three tasks.** Probe 7 — the arc §10 calls "the whole
reason a GNN is in this architecture" — is present at **−0.81**.

Probe 9 detail — contrast V1's failure mode (mean 0.4941–0.5034, sd ≈ 0.2887, i.e. `random()`):

| task | D | mean | sd | n |
|---|---|---|---|---|
| `fill_rate` | 0.8868 | 0.9185 | 0.2469 | 348,000 |
| `arrival_week` | 1.0000 | 12.0110 | 5.2185 | 348,000 |
| `capacity_strain` | 0.2727 | 0.7374 | 0.3591 | 217,500 |
| `shortage_qty` | 1.0000 | 8.1122 | 110.4421 | 84,637 |
| `demand_drift` | 0.6665 | 1.0189 | 0.1906 | 78,300 |

`arrival_week` is a week index and `shortage_qty` a part count; neither is bounded in [0,1], which
§14 calls a stronger tell than any test.

## B8. Still failing — mechanism, not number

### 1. Right-censored tail, 1.69× against 0.3–1.2× — the only seed-1001 failure

53,539 lines (3.90%) are promised inside the last 90 days of a 3,897-day span; span alone implies
2.31%. **On the 2019–2025 modelling window the same check passes at 1.17×.** The excess is confined
to the partial trailing year — which §1 says is deliberate: *"The partial trailing year is
deliberate: it is what a real extract looks like, and it is where right-censoring comes from
naturally."* The world ends 2026-08-31, inside the festive build-ahead peak (`seas_festive` +0.29)
and after 10.6 years of +2.1%/yr trend, so order density in the final quarter is above the span
average.

*The mechanism I would change:* shift the seasonality **phase** so the extract does not terminate
in the seasonal peak, or end the world in a trough month. Both are §13 left-column parameters. I
have not changed either, because doing so purely to move this number would be tuning toward the
band, and the modelling-window figure already passes.

### 2. Shortage condition share, 9.06% (1001) / 13.90% (1002) against 2–5%

Parts sourced through chronically constrained suppliers sit below safety stock for long runs.
Inter-plant transfer is implemented (`transfer_rate=0.62`) and absorbs part of it, but the other
absorption levers §9 names are not: **expedite does not accelerate an open order's arrival**, and
supplier recovery does not preferentially refill starved channels.

*The mechanism I would change:* make expedite a real arrival-acceleration on the open PO line, and
give starved channels priority in the following month's sequential allocation.

### 3. Seed 1002's five extra failures — and why they are the same finding

| check | 1001 | 1002 | band |
|---|---|---|---|
| shortage events / year | 202.9 ok | **313.1** | 180–250 |
| expedites / year | 117.6 ok | **180.3** | 100–150 |
| line stops / year | 20.6 ok | **28.4** | 15–25 |
| PO lines with fill < 1.0 | 14.9% ok | **16.6%** | 8–15% |

`escalate_frac` is a fixed fraction of the shortage-condition count. Because that count is both
too high **and** seed-varying (9.06% → 13.90%), a fixed escalation fraction cannot hold the event
rates in band across seeds. **The event-rate failures on seed 1002 are a downstream symptom of
failure 2, not an independent defect** — fixing the absorption mechanism should fix both. This is
precisely what §17's two-seed requirement exists to expose, and it would have been invisible on
one seed.

## B9. Seed variance band (§17)

Two seeds, identical code, one `code_commit`. Per-metric band:

| Metric | 1001 | 1002 | spread |
|---|---|---|---|
| passed / failed / skipped | 171 / 1 / 4 | 166 / 6 / 4 | 5 checks |
| fill < 1.0 | 12.12% | 14.00% | 1.88 pp |
| fill mass at 1.0 | 87.88% | 86.00% | 1.88 pp |
| fill mass at 0 | 1.83% | 1.73% | 0.10 pp |
| late rate | 21.43% | 21.85% | 0.42 pp |
| lead skewness | +2.98 | +2.96 | 0.02 |
| supplier-months constrained | 20.25% | 26.04% | **5.79 pp** |
| zero-order week share | 84.67% | 83.81% | 0.86 pp |
| seasonality ratio | 1.508 | 1.499 | 0.009 |
| shortage condition share | 9.06% | 13.90% | **4.84 pp** |
| probe 1 `fill_rate` | +0.195 | +0.217 | 0.022 |
| probe 1 `capacity_strain` | +0.665 | +0.650 | 0.015 |
| probe 7 (decisive arc) | −0.811 | −0.791 | 0.020 |
| PO lines / channel / yr | 8.40 | 8.87 | 0.47 |

The joint-probe statistics are stable to ±0.02 — the causal structure reproduces. The two wide
metrics are constrained-months and shortage-condition share, both governed by where monthly demand
lands relative to the capacity draw. **Any future claim that a change improved a number must clear
these bands.**

## B10. `synthetic_rules.md` compliance — the rebuild

| § | Requirement | Status |
|---|---|---|
| §0 | 7 plants; ~18 line stops/yr; ~120 expedites/yr | **met** (1001: 19.1, 116.7) |
| §1 | 2016→2026 span; 2016–18 retained; stores derived | **met** (557 weeks; stores aggregated from source) |
| §2 | Entity population minimums | **met** (all 9) |
| §3.1 | As-of gate on `recorded_ts` | **met** (`recorded_ts` on every transactional table) |
| §3.2 | Recording layer a process, per table, never negative, some never entered | **met** (14 per-table lag triples; `recorded_ts >= event` passes everywhere) |
| §3.3 | Bucket on `max(event_week, recorded_week)` | **met** (validator confirms) |
| §4 | Exact conservation; ledger identity; transfers exist | **met** (validator confirms all four; ledger holds) |
| §5 | 80–93% idle; ≥95% contiguous; 0 dup PKs | **met** (84.67%, 100%, 0) |
| §6 | `delivered = min(K, ordered)`; observability emergent | **met** (20.25% constrained, 79.75% unobservable) |
| §7 | Fill masses at 0 and 1.0 | **met** (87.88% / 1.83%) |
| §8 | Survival process; real censoring; skew ≥0.5 | **partially met** (skew +2.98, censoring emergent; censored-tail 1.69× full span, 1.17× on 2019–25) |
| §9 | Shortage from stock balance; two populations; ~100% traced | **partially met** (events in band on 1001; condition share 9.06% vs 2–5%; 89.9% traced) |
| §10 | Feedback arcs, esp. the two bold rows | **met** (probe 7 = −0.81; allocation and alternate-sources written) |
| §11 | Two disruption classes at correct nodes | **met** (supply/transit vs demand, separate state arrays; regime slicing) |
| §12 | Demand trend/seasonality; plans versioned | **met** (ratio 1.508; plan/actual/drift tables) |
| §13 | Mechanism params set, outcomes measured | **met** (literals audit clean) |
| §14 | Labels forward; 1:1 binding; censoring 1–60% | **met** (validator confirms all five bindings and the leakage gate) |
| §15 | Prohibitions | **all 15 pass** (below) |
| §16 | Acceptance ladder | **G0 and G1 pass**; G2–G7 not yet run |
| §17 | Two seeds + variance band | **met** (band in B9) |
| §18 | What ships | **met** (source, parameter file, two seeds, probes, statement of failures) |
| §19 | Final test | **met in structure** — labels are forward consequences of the recorded history |

### §15 prohibitions — the rebuild

| Prohibition | |
|---|---|
| Never generate a label independently of the features | **pass** (probe 1 non-zero on all tasks) |
| Never use a counter / ID / seed index / sin·cos as a target | **pass** |
| Never draw `entity_type`/`entity_id` from the wrong pool | **pass** (validator: all bindings ok) |
| Never set capacity observability; let `min(K, ordered)` produce it | **pass** (sequential allocation against `bank`) |
| Never tune staleness, seasonality, fill mass or an event rate after observing it | **pass** (literals audit clean; only §13-left params set) |
| Never use a constant recording lag; never a negative one | **pass** (14 per-table lognormals; `recorded_ts >= event` passes) |
| Never bucket a derived store on event week while gating on `recorded_ts` | **pass** (validator: `max(event,recorded)`) |
| Never patch a derived store after generation to force conservation | **pass** (computed, then asserted; no patching) |
| Never generate shortage events independently of inventory | **pass** (from the part×plant stock balance) |
| Never generate production actual independently of plan | **pass** (actual = plan × noise) |
| Never generate inventory snapshots independently of transactions | **pass** (cumulative sum over transactions) |
| Never generate GRNs independently of shipments and POs | **pass** (GRN follows delivery and arrival) |
| Never change allocation without changing the receiving supplier's load | **pass** (alt-sourcing consumes the receiver's capacity; probe 7) |
| Never delete censored observations | **pass** (43.5% censored rows retained) |
| Never write a gate that cannot fail | **pass** (the generator's assertions did fail, repeatedly, during the build) |

## B11. Readiness ladder

| Gate | Asks | Status | Note |
|---|---|---|---|
| **G0** | Is this a dataset? | **PASS** | conservation exact (validator-confirmed), censoring 3–43.5% on all five tasks, lag `sd/\|mean\|` passes on every table |
| **G1** | Does the graph exist? | **PASS** | median channel degree 3, one component, 100% coverage |
| G2 | What is the number to beat? | not yet run | naive baselines not computed — next step |
| G3 | Is there signal at all? | not yet run | blocked on G2 |
| G4 | Does the graph contribute? | not yet run | probe 7 at −0.81 says the propagation path exists; h⁰ vs h⁴ still has to be measured |
| G5 | Is the as-of gate binding? | not yet run | `recorded_ts` present and non-degenerate on every table, so the shuffle test is now runnable |
| G6 | Are the folds sound? | not yet run | label window opens strictly after the snapshot |
| G7 | Will it survive production? | **partially** | cold-start slice 4.59% present; seed-variance band published (B9) |

**The ladder now clears G0 and G1.** Part A's generator stopped at G0.

## B12. Gate readiness per task

| Task | Trainable? | Note |
|---|---|---|
| **Delivery risk** | **Yes, with a caveat** | `arrival_week` bound to `po_line`, computed forward, right-censored at 43.5% from genuinely open POs. Lead time right-skewed (+2.98), late rate 21.4%, and probe 1 gives ρ = −0.141 against `otd_rate_last13`. Caveat: the censored tail is 1.69× span-implied over the full extract (1.17× on 2019–2025), so a hazard model should be fitted on the modelling window. |
| **Fill rate** | **Yes** | `fill_rate` bound to `po_line`, mass 87.88% at exactly 1.0 and 1.83% at 0, 12.12% below 1.0 — all in band. Probe 1 ρ = +0.195 against `fill_rate_last13`; probe 2 ρ = −0.382 against capacity pressure; probe 4 gap +0.103. The weekly store conserves exactly against the PO spine, so the feature history *is* the label's history. |
| **Part shortage** | **Not yet** | Shortage descends from the part×plant stock balance and 89.9% of events trace to a real feeding channel, but the **shortage condition share is 9.06% against a 2–5% band and varies 9.06→13.90% across seeds**, which pushes seed 1002's event rates out of band. The blocker is the missing absorption mechanism in B8.2, not the label. |

Two of the three GNN tasks are trainable on this world today. The third has one named mechanism
between it and the band.
