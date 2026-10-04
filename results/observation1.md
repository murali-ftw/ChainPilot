# Observation 1 — What ChainPilot predicts, how it works, and where it stands

**Version 2.0** — rewritten after Phases 12–15. Version 1.0 is superseded in three places where its
conclusions have since been **disproved by measurement**, not merely refined: the capacity fix
(§4.5), the fill programme (§3), and the allocation blocker (§8). Each correction is flagged
in place.

**Purpose:** a single document covering every use case under implementation — the method behind it,
the mathematics of that method, why the method suits the real problem, how it has performed, and
what is planned where it has not.

**Scope of the evidence:** unless a world is named, every figure is measured on **v8, dataset seed
1001, fixed split** (train ≤ 2023, validation 2024, test 2025). Where v6/v7 figures appear they are
labelled as such. Bands are per metric, per configuration, per world and are **never borrowed**
between worlds.

**Companion reports:** `part2/phase-15.md`, `phase-14.md`, `phase-13.md`, `phase-12.md`,
`phase-11c.md`, `phase-0-1-v8.md`, `v8-clearance.md`; `part1/phase-8.md` and its predecessors for
the v6/v7 era.

**A note on reading this.** Every method carries a block marked **In plain terms** with an analogy.
The analogies illuminate the *mechanism* — including how it fails — not just the flavour. If a
section's mathematics is unfamiliar, read the analogy first and the equations second.

---

## 0. The use cases at a glance

| # | use case | what it answers | method | status |
|---|---|---|---|---|
| 1 | **Arrival timing** | when will material arrive, and will it run later than usual | Temporal-SHARE + discrete-time hazard | **ranking only** — real but small; no usable yes/no |
| 2 | **Fill rate** | what fraction of the order will show up | point-mass binned CDF, RPS loss | **ranking only** — recalibration does the work, not the architecture |
| 3 | **Supplier strain** | how close is this supplier to its demonstrated limit | Temporal-SHARE + quantile regression | **THE ONE SHIPPABLE ALERT** — 81% precision at 19% recall |
| 4 | **Shortage risk (ranking)** | which part-plants are most at risk | Temporal-SHARE + binary classifier | **diagnostic only** — base rate 8× reality |
| 5 | **Shortage quantity (simulation)** | how much short, and in which week | Monte Carlo stock roll-forward | **calibrated to the wrong world** — needs channel granularity |
| 6 | **Delivery schedule** | how much to order, in which week | mixed-integer linear program | **blocked on costs** |
| 7 | **Supplier allocation** | which supplier should get what share | candidate enumeration + constrained scoring | **worse than doing nothing** — and the blocker was misdiagnosed |
| 8 | **Transfer recommendation** | which plant should send stock, to whom | simulated donor safety check | **retire as an alert** |

Use cases 1–4 are learned models. 5 composes 1–3 into a simulation. 6–8 are decision layers that
consume 5.

---

## 0.5 Two ways to read every number in this document

**This section is new in v2.0 and it changes how the rest should be read.**

Until Phase 14, every result here was in **metric space** — CRPS, ROC-AUC, pinball loss. Those
answer *"is this model better than that one?"* Phase 14 re-scored every use case in **decision
space** — *"if a planner acts on this, how often are they right?"* — and most of the metric-space
wins did not survive the translation.

The reason is base rates. Three quarters of orders arrive in full. Nine weeks in ten are fine. When
one answer is right most of the time, a model has very little room to add value, and a good-looking
accuracy figure can be pure arithmetic.

**The rule adopted from Phase 14 onwards: no accuracy figure appears without the always-guess-the-
majority figure beside it, and the gap between them is the only part that is real.**

| use case | accuracy | guessing gets | real gain |
|---|---|---|---|
| Arrive within 4 weeks | 88.7% | 96.6% | **−7.9** |
| Below safety stock | 81.2% | 92.0% | **−10.8** |
| Part-plant at risk | 80.2% | 74.3% | **+5.9** |
| Arrives in full | 74.9% | 74.9% | **0.0** |
| Late vs contract | 74.2% | 73.0% | **+1.2** |
| **Demand exceeds capacity** | **67.8%** | **59.5%** | **+8.3** |
| Materially short (<0.95) | 55.5% | 75.5% | **−20.0** |
| Transfer donor | 51.4% | 50.4% | **+1.0** |

The highest accuracies in that table belong to the *worst* decisions. Only capacity and the
diagnostic shortage head beat guessing at all.

**Phase 15 then asked the follow-up question:** if a model only speaks when it is confident, how
often is it right? That converts every use case into a precision-at-coverage curve and produces the
operating points quoted throughout this document.

---

## 1. The shared foundation: Temporal-SHARE

Three of the four learned models sit on the same encoder. It has two stages, and the separation
matters: **time first, then structure.**

### 1.1 Stage one — the temporal encoder (TCN)

Each supplier-part-plant *channel* carries a weekly panel of 25 measured quantities (15 values plus
10 missingness indicators on v8). A dilated causal convolution compresses 52+ weeks of that panel
into one vector per channel.

For layer `l` with dilation `d_l` and kernel width 2:

```
h_t^(l+1) = ReLU( W_1^(l) · h_t^(l) + W_2^(l) · h_{t−d_l}^(l) ) + h_t^(l)
```

The residual term `+ h_t^(l)` is not cosmetic — an earlier specification omitted it and the stack
**could not learn at all** (C-index 0.5000 against a ridge probe's 0.6464).

Dilations double: 1, 2, 4, 8, 16, 32 across six layers. The receptive field is

```
RF = 1 + Σ_l (k − 1)·d_l = 1 + (1+2+4+8+16+32) = 64 weeks
```

which must exceed the 52-week window, or the model cannot see a full seasonal cycle. "Causal" means
each output at week *t* reads only weeks ≤ *t* — the architectural guarantee that a forecast never
consumes its own future.

**Why this is right for the problem.** Supplier behaviour is a trajectory, not a snapshot. A
supplier whose lead time has been stretching for six weeks is a different risk from one at the same
level but flat. A TCN sees both, at every timescale from one week to sixteen months, in one forward
pass — and unlike an RNN it does so in parallel, which is what makes 16,072 channels tractable.

> **In plain terms — reading a patient's chart.**
>
> A doctor reviewing a long illness doesn't read every entry with equal attention. They read the last
> week day by day, the last month week by week, the last year month by month — finer detail for
> recent history, broader strokes for old history. That is exactly what dilation does: the first
> layer looks one week back, the next two, the next four, doubling until the sixth layer reaches
> sixteen months. One pass, every timescale.
>
> **"Causal" means the later pages of the chart are physically covered.** The model cannot read an
> entry dated after the day it is making a judgement about. That is an architectural guarantee, not a
> discipline we have to remember.
>
> **And the residual connection is keeping the original chart open beside your summary.** Each layer
> writes a summary; the residual passes the raw entry forward alongside it. Remove it and by the
> sixth layer you are summarising a summary of a summary — which is precisely what happened: an
> earlier specification omitted the residuals and the model could not learn at all.

### 1.2 Stage two — the graph pass

The TCN output becomes the initial state `h⁰` of a graph with four node types (channel, supplier,
part, plant), 17,119 nodes, 48,216 edges and 6 directed relations. One or four rounds of message
passing follow. **This is not a spatio-temporal GNN** — time is fully consumed before the graph is
touched, which keeps the two concerns separable and testable.

**SHARE-lite** (used for arrival) — relation-specific transforms with relation-blind attention:

```
e_ij^r = LeakyReLU( aᵀ [ W_r h_i ‖ W_r h_j ] )
α_ij^r = softmax_j ( e_ij^r )
h_i'   = σ( W_0 h_i + Σ_r Σ_{j ∈ N_r(i)} α_ij^r W_r h_j )
```

**HeteroMP** (used for capacity and shortage) — learned per-relation means through a scalar gate:

```
m_r = mean_{j ∈ N_r(i)} h_j
h_i' = h_i + g ⊙ Σ_r W_r m_r ,   g learned
```

**Full SHARE** adds basis decomposition, `W_r = Σ_b a_rb V_b` with B = 10. It is **not used**, and
should not be: at hidden 128 the basis form costs 163,840 + 60 parameters per layer against 98,304
for free per-relation weights. Basis decomposition only pays above roughly R ≈ 20 relations; this
graph has R = 6. SHARE-lite is 468,608 encoder parameters against full SHARE's 730,992 with no
measured loss.

**Known waste (Phase 12, deviation 107).** The shipped SHARE-lite arrival encoder carries **49,152
structurally dead parameters**: last-layer messages into entity nodes never reach the channel
readout. They are computed and discarded every forward pass.

> **In plain terms — the morning huddle.**
>
> Picture every supplier-part-plant combination as one person who knows only their own situation.
> **h⁰ is everybody working in silence** — each acts on private information alone.
>
> **One round of message passing is one huddle.** Everyone reports to their supplier, their
> part-owner and their plant; those three aggregate what they heard and report back. Now this
> channel knows something it could not have known alone: that its supplier is simultaneously
> struggling on eleven other parts. **Four rounds (h⁴) means news travels four handshakes** —
> far enough that a plant's congestion reaches a channel that never touches that plant directly.
>
> **Attention (SHARE-lite) is listening harder to some voices than others.** In a real huddle you
> weight the report from the person whose situation most resembles your concern. The model learns
> those listening weights rather than being told them.
>
> **And here is why attention does less than you'd hope.** Half the room has exactly one supplier,
> one part and one plant to listen to. Listening harder to your only informant changes nothing —
> median channel degree is 3, so the softmax is over a single element and does no work at the
> channel. Attention only earns its keep one hop out, at suppliers, who have around 38 channels each
> to weigh.
>
> **HeteroMP is the plain version of the same huddle:** just average what each group says, then
> decide once how much to trust the huddle overall against your own read. That single "how much do I
> trust the room" dial is the gate.

### 1.3 Does the graph earn its place? — tested, and yes, but narrower than v1.0 claimed

This is the project's central architectural question and it was settled by a control designed to
kill the answer. Each node's neighbourhood was randomly permuted **within relation type**, so every
entity kept exactly the degree it had while the identity of its neighbours was destroyed. Three
arms were then compared: real graph, shuffled graph, no graph.

| | capacity | arrival |
|---|---|---|
| real beats shuffled | **5 of 5 dataset seeds** | **5 of 5 dataset seeds** |
| share of the h⁴-over-h⁰ gap attributable to the **edges** | 39.6–83.1%, mean **56.2%** | 86.9–115.8%, mean **100.2%** |
| shuffled worse than *no graph at all* | 0 of 5 | **3 of 5** |

Ten world-task combinations, every paired difference strictly positive. Under a null of no effect
that is roughly a 1-in-1000 coincidence. The shuffle itself was falsified three ways per world
(identity shuffle, 90%-preserved, all-to-one — each demonstrated firing), and 95% of edges
demonstrably moved endpoint.

> **In plain terms — the sabotaged org chart.**
>
> The worry this test exists to answer is a real one: maybe the huddle isn't useful at all, and the
> apparent benefit is just that a bigger model with more rounds of thinking does better regardless of
> *who* it talks to.
>
> So we sabotaged the org chart. **Same shape, wrong names.** Every supplier still has exactly 38
> channels reporting to them — we only changed *which* 38. Everyone still attends a huddle of the same
> size; they just hear about the wrong parts.
>
> Two possible outcomes, and they mean opposite things. If forecasts stay just as good, the huddle was
> only ever an excuse to think longer. If they get worse, the specific relationships carry real
> information.
>
> **They got worse, on all ten world-task combinations.** And on arrival something stronger happened:
> hearing the *wrong* supplier's news was worse than hearing *no* news. Being actively misinformed is
> worse than being uninformed — which is only possible if the information channel matters.

**Narrowed by Phase 12 C3.** A relation ablation found the **part relation carries no measurable
signal**: share moves +1.4% / −1.0% on arrival and +1.5% / −5.2% on capacity, with no full-vs-ablated
comparison disjoint. The ablated arm nonetheless stays disjoint from both h⁰ and the shuffled graph.

**So the correct statement is: the graph's value lives in the supplier and plant relations.** The
part relation, which has the highest message volume after channels (median degree 26), contributes
nothing measurable. A leaner encoder — dropping the part relation and the 49,152 dead parameters —
is a legitimate simplification to test, not a guess.

**Caveat.** v8's own generator diagnostic (G4) reports the opposite. Both seed-axis explanations for
that disagreement are closed — model seeds and dataset seeds both agree with us — so what remains is
an instrument difference (a different model, MAE rather than pinball, a different split).
Unexplained, not refuted.

---

## 2. Use case 1 — Arrival timing

### 2.1 What it predicts

For a purchase order channel, the distribution over which future week goods will arrive, and from
that, whether the arrival will be later than that channel has recently been running.

### 2.2 The method — discrete-time hazard

Rather than regressing on a date, the model predicts, for each future week *k*, the conditional
probability of arriving in that week **given it has not arrived yet**:

```
λ_k = P(T = k | T ≥ k, x) = σ( f_k(x) )
S(k) = Π_{j ≤ k} (1 − λ_j)                       survival
P(T = k) = λ_k · Π_{j < k} (1 − λ_j)             density
```

The log-likelihood splits by censoring status:

```
uncensored at week k :  log λ_k + Σ_{j<k} log(1 − λ_j)
censored  at week c  :          Σ_{j≤c} log(1 − λ_j)
```

**Why this formulation matters enormously here.** 48.5% of arrival rows are censored — the goods
had not arrived when the 90-day observation window closed, so we know only "later than this." A
regression must either discard those rows or invent a value for them. The hazard likelihood uses
every row: a censored row contributes the honest statement *"it survived past week c"*. The earlier
regression head trained on 55% of the population; the hazard head uses all 176,000 rows, asserted
every epoch. Switching to it produced a larger gain than the entire graph ever has.

**Real-world relevance.** A planner does not want a date, they want a distribution: *"70% chance by
week 6, 95% by week 9."* Safety stock and expediting decisions are made against tails, not means.

> **In plain terms — the hospital discharge question.**
>
> Ask a doctor "when will this patient go home?" and a good one does not name a date. They say:
> *"given they're still here on day three, there's about a 15% chance they leave tomorrow."* Ask that
> question for every remaining day and chain the answers, and you have a full distribution.
>
> **The real advantage is what happens to patients still in the ward when the study ends.** A
> regression has to invent a discharge date for them or throw their records away. The hazard
> formulation needs neither: it records *"still here on day 30"*, which is completely true and
> genuinely informative — it rules out every earlier day.
>
> **That is 48.5% of our purchase orders.** The old regression head studied only the patients who
> already went home — which systematically ignores the slow cases, the ones a planner most needs
> warning about.

### 2.3 How the reference metric was fixed

Lateness was originally scored against the **promise date**. Two independent problems:

1. **The promise date is not available at forecast time.** The labelled purchase order line is
   raised a median of 47 days *after* the snapshot.
2. **The scorer substituted a constant** for the promise arm before computing lateness. Scored
   honestly, the promise date's lateness ROC-AUC is exactly **0.5000** — structurally inevitable,
   since the promise is `order_date + contracted lead` and lateness is `actual lead > contracted
   lead`, so a per-channel constant cannot distinguish the two.

The metric is now defined against the **as-of channel median observed lead**, computable at the
forecast instant, with an as-of assertion demonstrated firing on three failure modes plus an
empty-set guard.

> **In plain terms — grading a weather forecaster against next week's newspaper.**
>
> You compare Monday's forecast against an "official prediction" that was published the following
> Thursday. It is not a competitor; it is a partial answer key. Any forecaster who fails to beat it
> looks weak, and any who beats it looks miraculous. Neither reading means anything.
>
> **And a second problem sat underneath it.** The scoring code quietly replaced the promise date with
> a flat constant before comparing. For five phases the reported comparison was "our model versus a
> fixed guess" while the label said "versus the promise date". The numbers were real; the sentence
> was not.

### 2.4 Performance — strong as a ranking, absent as a decision

**As a ranking** (v8 test, five model seeds):

| arm | lateness ROC-AUC |
|---|---|
| **head h⁴ (Temporal-SHARE)** | **0.70905** [0.70645, 0.71298] |
| b5flat LightGBM, same features | 0.70535 [0.70476, 0.70573] |
| head h⁰ (no graph) | 0.69999 [0.69842, 0.70133] |
| naive constant | 0.68697 |

All three comparisons disjoint. Arrival-week C-index: h⁴ **0.67438**, h⁰ 0.66133, best learned
baseline 0.64891 — also disjoint.

**As a decision (Phase 14, new in v2.0):** *"will this line be late against its contracted lead?"*

Censoring must be **resolved**, not excluded or zeroed — a censored line has T ≥ 13, so when its
reference is under 13 weeks it is *known late*, true for 95% of censored rows. That treatment choice
moves the same model's F1 from **0.42 to 0.84** (deviation 124).

Under it, **73% of lines are late anyway**, and:

- No arm beats "assume every line is late" disjointly.
- **The channel-constant ranker — the contracted term alone, no model — ties or beats every model**
  (treatment A: F1 0.732 and MCC 0.405, against h⁴'s 0.674 and 0.283). Deviation 126.
- Phase 15 reaches 0.869 precision at 0.396 recall, which is a **lift of only 1.19×** over the base.

**The "arrives within 4 weeks" form is near-empty.** At H = 1 there are **zero** positives and at
H = 2 only 37 of 36,000, because the label's clock starts at the snapshot, before most lines are even
raised (≈ 4.6-week order-raise wait). Only H = 4 is scoreable, at 0.159 precision / 0.528 recall on a
3.4% base — a 10× lift, but a ceiling of 0.35 precision.

### 2.5 Verdict: ship as a ranked watchlist, not an alert

**Why the ranking works:** the hazard head uses the censored half of the data; the encoder reads
channel state (supplier load, transit, prior utilisation) rather than the contracted term, which is
constant; and the graph contributes essentially all of the depth-over-no-depth gain.

**Why there is no yes/no product.** Lateness against a contract turns out to be mostly a property of
*which contract*, and the contract is a lookup. **v1.0 presented this use case as "works".
Corrected: it works as an ordering, not as a decision.**

**Client-readable claim:** *ranking purchase orders by how much later they will arrive than their
channel's recent average, the model separates late from on-time arrivals with ROC-AUC 0.709, ahead
of a gradient-boosted model (0.705) and a no-graph variant (0.700), disjointly at five seeds. It is
not a yes/no alert: 73% of lines run late against contract, and the contract term alone predicts
that as well as the model does.*

### 2.6 Open

- Week-level calibration (ECE-week) is seed-unstable at depth 4 — 0.176 to 0.310 across seeds on v8
  and the same on v6/v7. **Not quotable as a point value on any world.** The served distribution is
  h⁰'s recalibrated output while h⁴ supplies the ranking.
- Phase 8's v6/v7 arrival row quotes the wrong comparison and needs rewording.

---

## 3. Use case 2 — Fill rate

> **⚠ v1.0 correction.** Version 1.0 framed fill's problem as distribution *shape* and proposed
> reparameterising the head. Phases 13 and 15 tested four reshapings and closed the question. **The
> problem is calibration, not shape, and a post-hoc recalibration does most of the work no
> architecture change could.** §3.4–3.6 are rewritten.

### 3.1 What it predicts

For a purchase order line, the full distribution over *what fraction of the ordered quantity
arrives* — a number in [0, 1].

### 3.2 The method — binned CDF with explicit point masses

The distribution is not smooth. On v8, **74.9% of test mass sits at exactly 1.0** (validation 77.6%,
train 78.6% — the rate differs by split, deviation 127) and about 2% at exactly 0.

The target is discretised into **22 cells**: an atom at exactly 0, an atom at exactly 1.0, and 20
interior bins. The head emits a softmax over those cells, trained with the **ranked probability
score**:

```
RPS = (1 / (M−1)) · Σ_{m=1}^{M−1} ( F_m − 1{ y ≤ b_m } )²
```

Evaluation uses the exact, point-mass-aware CRPS:

```
CRPS(F, y) = ∫ ( F(x) − 1{x ≥ y} )² dx ,  atoms handled explicitly
```

**This distinction cost the project three phases.** A legacy CRPS implementation stepped the CDF at
bin edges and could not distinguish a mass at exactly 1.0 from one at 0.96. The point masses are
worth 18% / 10% on the exact integral and **0% on the legacy one**.

> **In plain terms — predicting a batsman's score.**
>
> Most innings end in one of two very specific ways: out for nothing, or not out at the close. A
> smooth bell curve cannot put a spike on *exactly zero* — it will always smear that probability
> across "0 to 4 runs", which is a different statement about the game.
>
> So you hand out probability across labelled buckets and reserve two for the outcomes that are
> exact: **exactly nothing**, and **exactly complete**. Fill rate has the same shape: three quarters
> of orders arrive exactly complete, and "95% arrived" is a shortage while "100% arrived" is not.
>
> **Why RPS rather than ordinary classification — guessing someone's age band.** If they are 55 and
> you guess 50–55, you were nearly right; 20–25 is badly wrong. Plain classification scores both as
> "incorrect". RPS charges you by *distance* across ordered buckets.
>
> **And the measurement trap we fell into for three phases:** our scoring code compared cumulative
> probabilities only at bucket *edges*, so it literally could not tell a spike at exactly 100% from
> one at 96%. "Fill is fine" survived because the ruler could not see the thing being measured.

### 3.3 Performance

v8 test fold, 5 seeds. **Raw and recalibrated are reported separately and must never be mixed** —
comparing a raw arm against a recalibrated one manufactured a false verdict in Phase 12
(deviation 80) and recurred in prose in Phase 13.

| arm | interior Σ\|err\| | exact CRPS | ROC-AUC P(fill=1) |
|---|---|---|---|
| 22-cell head, **raw** | 0.0878 [0.0764, 0.1039] | 0.13876 | 0.6204 |
| 22-cell head, **recalibrated** | **0.0279** | **0.13825** | **0.6205** |
| boundary reweight w=3, raw | 0.0820 | 0.13882 | **0.6222** |
| Beta interior, raw | 0.0932 | 0.13877 | 0.6188 |
| b5flat22, raw / recalibrated | 0.0252 / **0.0175** | 0.13969 / 0.13943 | 0.6049 |
| rolling-52 histogram (no seeds) | 0.0100 test, 0.0372 val | 0.15825 | 0.6054 |

*(Deviation 86: v1.0's "0.0445 against 0.0049" were **v6** figures quoted as v8.)*

### 3.4 Verdict: the head has the best discrimination and the worst calibration

Read the raw rows and a histogram appears to beat the neural head by a wide margin. Read the
recalibrated row and the picture inverts.

**The head knows *which* lines will fall short better than anything else** (ROC-AUC 0.6204 vs 0.6054;
CRPS 0.1388 vs 0.1583, both disjoint). **It is worst at saying *how much*** — until recalibrated,
which takes interior error from 0.0878 to 0.0279 while keeping the discrimination.

> **In plain terms — the doctor whose ranking is right and whose numbers are off.**
>
> A doctor reliably picks which patients are sickest but consistently overstates how sick. You do
> not replace the doctor; you keep the ranking and adjust the numbers afterwards. That adjustment is
> recalibration, and it turns out to be doing more work than any of the four architectures we tried.

**The head-shape programme is closed.** Four reparameterisations were tested (smooth Beta interior,
boundary reweight, history ratio alone, history ratio as the interior centre). None passes:

- The **Beta interior** fixes the `[0.95,1)` over-prediction erratically (0.14× to 1.22× by seed) and
  pushes the error into the low-fill cells, which then carry 68–80% of it — while degrading ROC-AUC
  disjointly.
- The **boundary reweight** is the best head arm and holds the phase's only disjoint gain anywhere
  (+0.002 ROC-AUC). It is not a reason to change anything on its own.
- **History ratios** (per part-supplier, per supplier-plant, and a Beta-binomial shrunk hierarchy)
  add nothing measurable over the head without them, and fail the ECE guard.
- `mean_of_ratios` disjointly beat `ratio_of_sums` as a point summary — the opposite of what was
  pre-registered, and relevant to how any history ratio is quoted.

**The error now lives in the low-fill cells `[0, 0.25)`**, for every arm and every baseline, once the
near-1 boundary is handled.

### 3.5 As a decision

*"Will this line arrive in full?"* — base rate 0.749. **Every arm's F1-optimal threshold collapses to
"always full."** Accuracy 0.7492 against a majority baseline of 0.7492. There is no yes/no product
here; PR-AUC 0.813 against a 0.749 base is the honest summary — a real but modest ranking signal.

*"Will this line be materially short?"* — at fill < 0.95 (base 0.245): precision 0.306, recall 0.645.
It catches 65% of short lines and **flags three for every one that is genuinely short**. Phase 15's
ceiling at any coverage is **0.577**; 0.85 precision is unreachable.

**Coarsening does not rescue it.** Pooling to supplier-month appears to "reach" 0.85 — but on a
**74% base rate**, with lift *falling* from 2.18 to 1.24. At supplier-quarter the base is 0.911 and
the answer is "always yes". Aggregation raises the base rate, not the signal (deviation 135).

### 3.6 Open

- **`shipped.json` still names a LightGBM the serving path cannot load** (deviation 46, now three
  datasets old). Build the loader so configuration determines what runs — but **do not swap the
  model**: b5flat22 is disjointly worse on CRPS *and* ROC-AUC, and ranking is the use case that
  depends on discrimination. The honest comparison is recalibrated-vs-recalibrated, 0.0279 → 0.0175.
- Fill's recalibrated ECE varies 9× across backtest windows. No single figure without its window.
- If any further fill work happens, it targets the low-fill cells — and the prior question is whether
  they are *predictable* (driven by supplier state, hence a feature problem) or noise.

---

## 4. Use case 3 — Supplier strain

> **⚠ v1.0 correction, twice over.** Version 1.0 said "the median works, the intervals do not" and
> prescribed conformal widening. **That prescription was tested and failed** — it made the next
> window worse. And version 1.0 understated the use case: measured as a *decision*, **this is the
> strongest thing in the project.** §4.4–4.5 are rewritten.

### 4.1 What it predicts

Per supplier-month, three quantiles (P10, P50, P90) of `capacity_strain` — demand against
demonstrated throughput. Above 1.0 means being asked for more than the supplier has ever sustained.

**Why "strain" and not "capacity".** True capacity is never observed — you only see what a supplier
shipped, bounded by what you ordered. Strain is observable. This reframing unblocked the use case.

### 4.2 The method — quantile regression via pinball loss

```
L_τ(y, q) = max( τ·(y − q) , (τ − 1)·(y − q) )
```

Asymmetric by construction: under-predicting is penalised `τ` per unit and over-predicting `(1−τ)`.
Minimising it drives `q` to the τ-th quantile with no distributional assumption — which matters
because strain is right-skewed and clipped.

> **In plain terms — the tailor cutting cloth.**
>
> A tailor faces a lopsided risk. Cut too short and the garment is ruined; too long and you waste a
> little cloth. So a sensible tailor deliberately cuts long — not because they expect to need the
> extra, but because the two mistakes cost different amounts.
>
> **Pinball loss is that lopsidedness written down.** Set τ = 0.9 and falling short is nine times as
> expensive as overshooting, so the cost-minimising number lands automatically at the 90th
> percentile. No bell curve required.
>
> **And here is the correction to what this document said in v1.0.** We said the tailor's margins
> were learned on last year's cloth and the fix was to cut longer. **We tried that, and it made the
> next garment worse.** The real problem is not the margin — it is that the tailor is measuring
> *last year's customer*. Their centre line is stale: when the customer's size shifts, the tailor
> follows only half to three-quarters of the way. Adding seam allowance to a garment cut for the
> wrong body does not make it fit.

### 4.3 Performance as a forecast

| metric | head h⁴ | h⁰ | LightGBM quantile | naive global |
|---|---|---|---|---|
| **pinball** (lower better) | **0.07248** | 0.07671 | 0.07990 | — |
| **80% coverage** (nominal 0.80) | 0.78391 | 0.80234 | 0.80344 | 0.82849 |
| P90 exceedance (nominal 0.10) | 0.13473 | 0.12065 | — | — |

Quantile crossing: **0.00000** — a real correctness property, not guaranteed by independent heads.

**The interval diagnosis, settled (Phase 12 B1).** Across a 16-window backtest, coverage ran
0.72–0.81 against nominal 0.80, below nominal in 12 of 16 windows, **in every model class including
LightGBM**. Conformalised quantile regression was then tested:

- exact on its own window (48/48),
- **worse on the next window** — h⁴ went from 9 of 16 covered to 5,
- level correlation ρ unchanged (0.65 / 0.80 / 0.83).

**The intervals were never too narrow. They are in the wrong place.** Each model's P50 follows only
**51–74%** of a level shift, and that lag explains post-CQR exceedance at Spearman **0.96**.

### 4.4 Performance as a decision — the project's one shippable alert

*"Will demand exceed this supplier's capacity over the 90-day horizon?"* Base rate 0.405.

| arm | F1 | precision | recall | accuracy (majority 0.595) | MCC | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **mp h⁴** | **0.635** [0.623, 0.648] | 0.591 | 0.695 | **0.678** | 0.358 | 0.677 | 0.750 |
| h⁰ | 0.601 | 0.563 | 0.645 | 0.653 | 0.299 | 0.612 | 0.708 |
| LightGBM quantile | 0.602 | 0.478 | 0.814 | 0.564 | 0.220 | 0.561 | 0.664 |
| naive channel | 0.577 | 0.405 | 1.000 | 0.405 | 0.000 | 0.430 | 0.532 |

**h⁴ is disjointly best on F1, MCC and PR-AUC, and it is the only non-diagnostic decision in the
project that beats its majority baseline on accuracy.**

**The shippable operating point (Phase 15, chosen on validation, held on test):**

> **81% precision at 19% recall, flagging about 10% of supplier-horizons. Lift 2.0× over base.**

And in three-way form, which is what a planner should actually see:

> *"Of 22,500 supplier-horizons, the system alerts on 3,829 as over capacity — right 76% of the
> time — clears 85, and passes the remaining 18,586 back to you."*

The 82.6% "no opinion" is a feature. A system that admits what it does not know earns the right to
be believed when it speaks.

**The level lag still costs something, but not this.** The test median sits at 0.890 against a
realised 0.967; the model puts 30.0% of channels over capacity where 40.5% are. Phase 15 built the
correct upper bound for fixing it — correcting each period by its *own* level, using future
information — and at matched recall 0.25 it gives 0.788 against the uncorrected 0.787.

**The level lag costs calibration, not discrimination.** It changes where the threshold lands and
what probability a planner reads off. It does not change the precision achievable at a given recall.
**Do not fund a level-tracking retrain on the strength of the alert product.** (v1.0's §4.5 estimated
"~3 hours" for this fix and called its premise sound. Both were wrong.)

### 4.5 Why this use case works when the others don't

Phase 15 tested whether capacity's advantage comes from its coarser grain (90-day channel aggregate
rather than per-week per-line). **It does not:** pooling capacity from channels to suppliers leaves
its lift unchanged (2.02 vs 2.00) and its precision unchanged (0.80 vs 0.81).

**What makes capacity work is the task.** Its label is a smooth, persistent utilisation ratio. Fill
and shortage are lumpy one-off events. **You cannot fix lumpy event prediction by adding events up** —
that raises the base rate, not the signal. This is a structural constraint on what this system can
be good at, and it is the most useful thing Phase 15 established.

---

## 5. Use case 4 — Shortage risk (ranking)

### 5.1 Method and performance

A binary head over part-plant-snapshot, trained with cross-entropy:

```
L = −[ y·log p + (1−y)·log(1−p) ] ,  p = σ(f(x))
```

v8 test fold: ROC-AUC **0.81587**, PR-AUC **0.65875**, against the best baseline's 0.77257 / 0.58889 —
disjoint on both.

**As a decision (first ever reported, Phase 14):** base rate 0.257; F1 **0.585**, precision 0.633,
recall 0.543, accuracy **0.802 against a majority of 0.743**, MCC 0.458. At high confidence
(Phase 15) it reaches **0.910 precision at 0.142 recall**, on 4% coverage.

> **In plain terms — the examiner who grades on the wrong cohort.**
>
> An examiner trained entirely on a remedial class where 26 in 100 failed becomes genuinely excellent
> at ordering students from weakest to strongest. But ask *"what fraction of this new class will
> fail?"* and they will say "about a quarter", because that is the world they learned in. In a normal
> class where three in a hundred fail, that answer is badly wrong while the ranking stays good.
>
> **That is our shortage head exactly.** v8 contains shortages at 25.7% against a real-world rate near
> 3% — roughly eight times too many, by construction.
>
> Hence the rule: **rank, never quantify.** ROC-AUC and PR-AUC measure ordering only, so they survive
> the distorted base rate. A probability does not, and none is quoted anywhere.

### 5.2 Verdict: internal watchlist only, never a client alert

The 0.910-precision operating point looks shippable and is not. **Precision is the metric most
sensitive to base rate**, and this head is graded on a population with ~8× the real shortage rate. At
a realistic base rate it would fall sharply — by how much is unmeasured, which is itself the reason
not to quote it.

**Before any investment here, re-measure on a realistic shortage rate.** That is a cheap
re-weighting, and it determines whether there is a product underneath.

---

## 6. Use case 5 — Shortage quantity (Monte Carlo simulation)

### 6.1 What it does

Composes the heads into a forward simulation of stock. For each part-plant, N paths over a 13-week
horizon:

```
I_w = I_{w−1} + A_w − C_w
```

- `I_0` — the opening balance. **This blocked the project for eight phases.** On v6/v7 no valid
  opening level existed; on v8 the ledger roll-forward reproduces the stated balance on
  **100.000000% of 2,253,420 rows, max difference 0**.
- `A_w` — arrivals: week `T` drawn from the hazard head, fraction `f` from the fill head's 22-cell CDF.
- `C_w` — consumption, from the forward requirement plan.

All paths are retained so percentiles are taken at the end and never summed from below.

> **In plain terms — playing the board game a thousand times.**
>
> You cannot derive the odds of losing a game with interacting rules algebraically. You play it a
> thousand times and count. Each "game" is one possible future: this order lands in week 5 at 90%
> fill; that one in week 8 at 60%; demand comes in above plan. Walk the ledger forward and see
> whether you drop below safety stock.
>
> **Two rules of playing matter more than they sound.** *Play whole games and count at the end* —
> combining the "90th percentile arrival" with the "90th percentile fill" is not a real game, it is
> two bad days stapled together. And *play with the real rules*.
>
> **v1.0 said one player instruction was wrong. It was four, and they are now fixed** — the reorder
> trigger double-counted, the lead time came from a quantity that is not a lead time, jitter was
> applied where the real world has none, and the stock already ordered and in transit was omitted
> entirely. **The remaining problem is different in kind: the game is watching the wrong weeks.**

### 6.2 Performance — the arc, and where it stopped

| stage | ratio to observed |
|---|---|
| original (v1.0 of this document) | **5.29×** over |
| after the order-policy fix (Phase 12 B2) | **1.475×** [1.464, 1.487] |
| against the **pre-rescue** reference (Phase 13 S1) | **1.129×** [1.121, 1.139] |

Phase 12 B2 closed **77% of the log-scale gap**: replacement rate 82.5% → 100.3%, shortfall 564 → 298
against 91 observed. Attribution: 5.29 → 2.78 from the order rule, lead and jitter; 2.78 → 1.49 from
adding the open pipeline.

**The S1 result, and its label.** Planners move stock between plants to prevent shortages, so the
observed record is *post-rescue*. Rebuilding the reference as "what would have happened without
intervention" gives **1.129×** — and the reference builder was falsified three ways:

| falsification | result |
|---|---|
| identity (add back zero transfers) | reproduces Phase 12 B2 **exactly**, per seed |
| wrong sign (subtract instead of add) | **6.55×** — moves away, as required |
| shuffled (permute transfers, total preserved) | **1.38–1.40×, never in band** — so attribution to specific rescued weeks is real |

**This is the most thoroughly verified result in the project.** It is also a statement about *what the
simulation represents*, not a better forecast: the observed world is still 1.475× over, and the
added-back transfers are future information relative to every t0.

### 6.3 As a decision — and the diagnosis that matters

*"Will this part-plant-week fall below safety stock?"* Base rate 0.080.

| | against observed | against pre-rescue |
|---|---|---|
| precision | 0.205 | **0.238** |
| recall | 0.469 | 0.491 |
| F1 | 0.286 | **0.321** (disjoint) |

**The cells the planners' transfers moved.** Of **12,401** part-plant-weeks that planners rescued:

- **34%** the simulation *had* flagged — these flip from false alarms to hits once the rescue is
  credited, which is why precision rises;
- **66%** it had **not flagged at all** — which is why recall barely moves.

> **S1 said the simulation's magnitude is right for the pre-rescue world. This says its *week
> selection* is the bigger problem.** Two-thirds of the weeks planners felt compelled to act on, the
> simulation never saw.

**And there is no threshold that rescues it.** Phase 15 found precision peaks at about 0.28 at 2–4%
coverage and then **falls** to 0.25 at the tightest coverage. The simulation's most confident weeks
are not its most accurate. Ceiling **0.284** — no operating point exists at any usable bar.

Coarsening helps here more than anywhere else (part × plant × month lifts 3.23 → 4.31 at 1% coverage)
and still tops out at 0.54–0.63.

### 6.4 Planned fix — one item, and it is structural

**Restate the simulation at channel granularity.** Four independent findings point at the same place:

1. Phase 12 **B3** added inter-plant transfers at **part-plant** granularity and overshot badly
   (298 → 20 against 91 observed) — part-plant state provably cannot represent the dynamics.
2. Phase 13 **S1** localised the entire residual gap to planner transfers.
3. Phase 14 showed **66% of rescues are on weeks the simulation never flags** — a week-selection
   problem, not a magnitude one.
4. Phase 15 measured the precision ceiling at 0.28 and showed coarsening cannot lift it.

Stage 0 of Phase 13 also confirmed the key: **a channel is exactly (part, supplier, plant)**, 16,072
of them, verified 1:1. The granularity this needs is already a first-class object in the graph.

**This is the only remaining modelling item in the project with a case behind it.**

### 6.5 The correlated-failure extension, deliberately deferred

On v8, within-group correlation is +0.2101 against cross-group +0.2055 — a separation of **0.0046**,
an order of magnitude inside the acceptance gate's own ±0.05 tolerance. A per-group ρ and a single
global ρ are statistically indistinguishable here, so the copula's central claim cannot be validated
on this world. Building it would produce a component whose correctness cannot be demonstrated.

### 6.6 Reproducibility defect — open

**Phase 12 B3's block regrouping silently changed the default simulation's random stream**
(deviation 122). Phase 14 had to load `order_policy.py` from git at `9e2d59d` to reproduce Phase 12
B2's numbers; with today's HEAD they do not reproduce. **A published headline result is not
reproducible from main.** Paths were also never stored — only aggregates. Both must be fixed before
any further simulation work.

---

## 7. Use case 6 — Delivery schedule (MILP)

### 7.1 The method

```
minimise   Σ_w [ c_hold·I_w + c_order·y_w + c_freight·⌈x_w / truck_cap⌉ ]

subject to I_w = I_{w−1} + x_w − d_w        stock balance
           I_w ≥ SS                          safety-stock floor
           x_w ≥ MOQ · y_w                   minimum order quantity
           x_w ≤ M · y_w                     linking
           x_w ≡ 0  (mod L)                  lot-size multiple
           y_w ∈ {0,1}                       order placed indicator
```

The integrality is essential: `y_w` makes "place an order at all" discrete, which creates the
ordering-cost-versus-holding-cost trade-off. Without it the answer degenerates to just-in-time.

> **In plain terms — grocery shopping with a small fridge.**
>
> Every trip costs an hour of your Saturday (ordering cost). Stocking up means food going off
> (holding cost). Eggs come by the dozen (lot size). They won't deliver under ₹500 (minimum order).
> You can't let the fridge run empty (safety stock floor).
>
> **The "integer" part is the trip itself.** You either go or you don't; there is no two-thirds of a
> trip. That yes/no is the entire source of the trade-off.
>
> **Why it cannot advise you on timing.** Nobody has told us what the fridge costs. With holding cost
> at zero, food never goes off — so buying everything on the first trip and everything on the last
> cost *exactly the same*, and the solver picks arbitrarily. It can tell you *how much*. It cannot
> tell you *when*. That is a missing price, not a solver limitation.

### 7.2 Status — blocked, and not on the solver

Built and verified on v6/v7; solves in milliseconds. It lacks an objective:

| needed | present in any dataset? |
|---|---|
| inventory holding / carrying cost | **no** |
| ordering / setup cost per PO | **no** |
| truck capacity and freight tariff | **no** |

With `c_hold = 0` the objective is **indifferent to timing**. The guide's own verification gate was
therefore **vacuous** — its premise is true of every part. It was replaced with five gates, each
demonstrated failing on constructed input.

v8 fixes the single-period limitation: 12–26 forward periods per plan version, refreshed every 56
days, 100% forward. The multi-period schedule is buildable and has not been rebuilt on v8.

**Not scoreable in Phase 14 or 15.** No synthetic cost was constructed, and none should be.

---

## 8. Use case 7 — Supplier allocation

> **⚠ v1.0 correction.** Version 1.0 named the shortage cost as the blocker. **Phase 12 A4 proved on
> data that it is not: across ₹100–10,000, 0 of 183 part-plants change winner.** The real blockers
> are a qualification rule and the seed bands. §8.2 is rewritten.

### 8.1 The method

Enumerate five candidate splits per part-plant — incumbent, all-to-cheapest, equal split, shift 20pp
to the cheapest alternative, cap any supplier at 70% — then score each:

```
score(s) = requirement · Σ_v share_v · (1 − E[fill_v]) · strain_penalty_v · c_short
         + Σ_v share_v · requirement · price_v
```

subject to constraints deciding whether a move is possible at all: tooling transferability,
qualification lead time, ramp-rate limit, minimum-volume commitment.

**The constraints are the product, not the objective.**

> **In plain terms — splitting the wedding order between three tailors.**
>
> Six weeks to the wedding, forty outfits, three tailors. Score each plan by what it costs plus what
> it costs you if something isn't ready.
>
> **But some plans are impossible at any price, and that is the real work.** Tailor A holds the only
> pattern. Tailor B can take at most 20% more work. Tailor C needs a trial garment approved, which
> takes three weeks you don't have. **Those facts decide whether the recommendation is usable; the
> pricing only decides which usable one is best.**
>
> **v1.0 ended this analogy by saying the decisive missing number was how bad it is if an outfit isn't
> ready. We then measured it, and it isn't.** Across a hundredfold range of that cost, **the
> recommended split never changed for a single one of 183 part-plants.** The thing actually stopping
> us is duller and more fixable: for half the part-plants, our own rules say **no tailor is allowed
> to take the work — including the one already doing it.**

### 8.2 Status — the recommender is currently worse than doing nothing

**Feasibility first (deviation 91):** on v8, **57 of 120 part-plants (47.5%) have no feasible
candidate at all**, because the qualification check rejects share held by a supplier marked
unqualified — the incumbent included.

**Precision against what actually happened (Phase 14):**

| | n | precision@1 | precision@2 |
|---|---|---|---|
| recommendation, **conditional on feasible** | 63 | 0.667 | 0.873 |
| incumbent's top supplier, same 63 | 63 | **0.651** | 0.873 |
| recommendation, **unconditional** | 120 | **0.350** | 0.458 |
| incumbent's top supplier, all 120 | 120 | **0.692** | 0.925 |

Conditional on feasibility it is indistinguishable from the status quo — and **92% of its "winners"
are "keep the incumbent"** (42 of 63). Unconditionally, **doing nothing beats the recommender by
almost 2×.**

**The shortage-cost blocker is closed, and the unlock was small.** A4 proved invariance on data (the
check fires on 38 part-plants at weight 1, so it is not vacuous). Quotable moved 25% → 31.7% on v6/v7
and 4.8% → 9.5% on v8; **actionable moved 17.5% → 19.2% and 3.2% → 7.9%** — two part-plants in 120.
The binding limit was always the seed bands. *(Deviation 90: v1.0's "25%" counted duplicate candidate
splits as runners-up; every exact top-two tie was the winner under another name.)*

Phase 15 could not curve this use case at all: 63 feasible part-plants leaves every coverage cell
under 50 alerts.

### 8.3 Planned

- **Settle the qualification semantics.** Until an incumbent can hold share it already holds, no v8
  allocation output is quotable. This is an hour's conversation, not a modelling task.
- Then re-measure. There may be no product underneath.
- The commitment period and penalty basis remain open: a quantity with no window cannot be enforced,
  and a penalty with no basis cannot be priced.

---

## 9. Use case 8 — Transfer recommendation *(new in v2.0)*

### 9.1 What it does

For each projected-short part-plant-week, propose an inter-plant transfer from a same-part plant with
surplus, recommending only if the donor's simulated shortage chance stays under a fitted threshold.

### 9.2 Status — retire as an alert

**Donor choice** (base rate 0.504): policy **0.557** precision, naive nearest-surplus 0.528, random
0.520. The policy is disjointly best at every θ — but **"always yes" scores F1 0.670 against the
policy's 0.270**, because half of all candidate plants shipped out that week anyway. **This is a
precision@1 claim and nothing more** (deviation 130). Phase 15: **NOT TUNABLE** on validation — lift
0.03, the confidence score carries no information about correctness.

**Recipient choice**: no rule has skill. Naive and random recommend on essentially every short week,
so their precision *is* the base rate, and the policy's filter never raises precision above it.

**Ground truth is inferred** (deviation 119): `inventory_transactions.from_plant_id` records the
**receiving** plant on all 1,482,011 transfer-in rows. The donor is never stored. Every donor figure
is measured against a same-week transfer-out proxy and cannot be tightened without a generator fix.

**Also: the quantity rule lifts recipients only to the median**, so post-transfer shortage chance is
≈0.5 by construction. The layer names *who*, not *how much*.

**Recommendation:** retire as an alert; keep only as a tie-break suggestion, labelled *"freight cost
and transfer lead time not weighed; validated against planner action, not planner intent; inherits an
uncalibrated simulation."*

---

## 10. Consolidated status and plan

### 10.1 Where each use case stands

| use case | works | does not | blocked by |
|---|---|---|---|
| Arrival timing | ranking, disjoint at 5 seeds | any yes/no — the contract term matches the model | nothing; it is at its ceiling |
| Fill rate | ranking; recalibration closes most of the calibration gap | any yes/no; head-shape programme closed | serving-path loader (config misdescribes what runs) |
| **Supplier strain** | **alert at 81% precision / 19% recall** | absolute probabilities (level lag) | — **this one ships** |
| Shortage ranking | ordering | any probability | base rate 8× reality — re-measure first |
| Shortage quantity | mechanism, gate, ledger; **1.129× pre-rescue, falsified 3 ways** | week selection: 66% of rescues unflagged | **channel granularity — ours to fix** |
| Delivery schedule | solver, constraints | objective | 3 cost parameters — **client** |
| Allocation | enumeration, constraints | beats doing nothing | qualification rule — **one conversation** |
| Transfer recommendation | donor precision@1 | detection of any kind | retire |

### 10.2 The plan, in order

**Ship:**

1. **Capacity as an alert** at the validation-chosen point, labelled *"81% right, catches 19%, flags
   about 10% of supplier-horizons."* Everything else goes out as a ranked watchlist or not at all.

**Fix (ours):**

2. **Reproducibility (deviation 122).** Main cannot reproduce Phase 12 B2. Pin or restore the random
   stream and store simulation paths. Everything downstream rests on this.
3. **Channel-granularity transfer model.** The only remaining modelling item with a case behind it,
   aimed at *week selection* rather than magnitude.
4. **Fill's serving loader** — so configuration determines what runs. **Do not swap the model.**
5. **Settle the allocation qualification rule.** One conversation; unblocks 47.5% of part-plants.
6. **Re-measure the shortage head at a realistic base rate** before investing in it.

**Do NOT fund** — each was tested and closed, not merely deprioritised:

- ~~Capacity level-tracking retrain~~ — the level lag costs calibration, not discrimination. Even a
  perfect per-period tracker (using future information) leaves the frontier unchanged.
- ~~Conformal interval widening~~ — exact on its own window, worse on the next.
- ~~Further fill head reparameterisation~~ — four arms, one disjoint gain of +0.002.
- ~~Coarser-grain reframing~~ — raises base rates, not lift.
- ~~Model-family agreement filtering~~ — dominated by tightening the best model's own threshold.

**Rane's to supply** — open across three datasets, and no work on our side closes them:

7. Inventory carrying cost.
8. Ordering / setup cost per purchase order.
9. Freight structure — truck capacity and tariff shape.
10. Shortage cost. *(Still needed for the MILP objective. Note it is now proven **not** to change
    allocation winners — that use of it is closed.)*

**Data defects to report to the generator's authors:**

11. `from_plant_id` records the receiving plant on all 1,482,011 transfer rows — the donor is never
    stored.
12. `purchase_orders.status` holds a single value (`OPEN`) on all 1,063,256 rows — a field that
    silently invites everyone to build on it.

---

## 11. What the honest scope is now

Fifteen phases in, ChainPilot is:

- **one alert product** — supplier capacity strain, at a measured and defensible operating point;
- **a set of ranked watchlists** — arrival lateness, material shortfall, shortage risk — each real as
  an ordering and none of them a yes/no;
- **one open modelling question** — restating the simulation at channel granularity;
- **two things blocked on Rane** — the cost parameters, and the allocation qualification rule.

That is much smaller than where this document started. It is also the first version of the scope that
survives measurement.

**The two claims that can go in front of a client today**, each with its label attached:

> *"The simulation reproduces what would happen without your planners' intervention to within about
> 13%. The gap to what actually happened is the value your planners add."*

> *"For supplier capacity, we alert on about one horizon in ten and are right roughly 8 times in 10,
> against a background rate of 4 in 10."*

---

## 12. Standing caveat

Every figure in this document is measured **within a synthetic world**: one generator produced both
the training and the test data, separated only by time. These are an optimistic upper bound on a real
extract, not a forecast of it.

The findings that transfer are the **mechanisms**, not the numbers:

- censoring must be modelled, not dropped — **and how it is resolved moves the headline by 2×**;
- point masses matter for fill, and the ruler must be able to see them;
- quantile intervals do not survive a level shift, **and widening them does not help**;
- the graph's edges carry information, concentrated in the supplier and plant relations;
- **smooth persistent quantities are predictable; lumpy one-off events are much less so, and
  aggregating them does not change that**;
- a system measured in metric space and a system measured in decision space are not the same system.

The only evidence that would support a production figure is a rolling-origin backtest on Rane's own
data.
