# Observation 1 — What ChainPilot predicts, how it works, and where it stands

**Purpose:** a single document covering every use case under implementation — the method behind it,
the mathematics of that method, why the method suits the real problem, how it has performed, and
what is planned where it has not.

**Scope of the evidence:** unless a world is named, every figure is measured on **v8, dataset seed
1001, fixed split** (train ≤ 2023, validation 2024, test 2025). Where v6/v7 figures appear they are
labelled as such. Bands are per metric, per configuration, per world and are **never borrowed**
between worlds.

**Companion reports:** `part2/phase-11c.md`, `phase-11b.md`, `phase-11a.md`, `phase-0-1-v8.md`,
`v8-clearance.md`; `part1/phase-8.md` and its predecessors for the v6/v7 era.

**A note on reading this.** Every method carries a block marked **In plain terms** with an analogy.
The analogies are chosen to illuminate the *mechanism* — including how it fails — not just to give a
flavour. If a section's mathematics is unfamiliar, read the analogy first and the equations second.

---

## 0. The use cases at a glance

| # | use case | what it answers | method | status |
|---|---|---|---|---|
| 1 | **Arrival timing** | when will material arrive, and will it be later than this channel usually runs | Temporal-SHARE + discrete-time hazard | **works** — lateness claim established at 5 seeds |
| 2 | **Fill rate** | what fraction of the order will actually show up | point-mass binned CDF, RPS loss | **partial** — wins the proper score, loses calibration |
| 3 | **Supplier strain** | how close is this supplier to its demonstrated limit | Temporal-SHARE + quantile regression | **partial** — median works, intervals do not |
| 4 | **Shortage risk (ranking)** | which part-plants are most at risk | Temporal-SHARE + binary classifier | **diagnostic only** — ranks well, levels unusable |
| 5 | **Shortage quantity (simulation)** | how much short, and in which week | Monte Carlo stock roll-forward | **not calibrated** — over-projects 5.29× |
| 6 | **Delivery schedule** | how much to order, in which week | mixed-integer linear program | **blocked on costs** |
| 7 | **Supplier allocation** | which supplier should get what share | candidate enumeration + constrained scoring | **blocked on costs** |

Use cases 1–4 are learned models. 5 composes 1–3 into a simulation. 6–7 are optimisers that consume
5 and require cost parameters that do not exist in any dataset.

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

**Why a graph at all.** Suppliers share plants, parts share suppliers, and a plant's congestion
spills across every channel it touches. A per-channel model cannot see that a supplier is
simultaneously late on eleven other parts. The graph makes that visible in one hop.

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

### 1.3 Does the graph actually earn its place? — tested, and yes

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

**On arrival the edges explain essentially the entire advantage**, with nothing measurable left for
depth and parameters. And on three worlds a randomised neighbourhood scores *below* having no
neighbourhood — aggregating over misleading neighbours costs more than having none. That is not a
pattern depth alone would produce.

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
> only ever an excuse to think longer, and the relationships were decoration. If they get worse, the
> specific relationships carry real information.
>
> **They got worse, on all ten world-task combinations.** And on arrival something stronger happened:
> hearing the *wrong* supplier's news was worse than hearing *no* news. Being actively misinformed is
> worse than being uninformed — which is only possible if the information channel matters.
>
> That is the test that could have ended this architecture, and it is the reason we can say the graph
> earns its place rather than merely asserting it.

**Caveat.** v8's own generator diagnostic (G4) reports the opposite. Both seed-axis explanations for
that disagreement are now closed — model seeds and dataset seeds both agree with us — so what
remains is an instrument difference (a different model, MAE rather than pinball, a different
split). Unexplained, not refuted.

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
The hazard form gives that directly, and handles the very common case of an order that simply has
not arrived yet.

> **In plain terms — the hospital discharge question.**
>
> Ask a doctor "when will this patient go home?" and a good one does not name a date. They say:
> *"given they're still here on day three, there's about a 15% chance they leave tomorrow."* Ask that
> question for every remaining day and chain the answers, and you have a full distribution — the
> probability of leaving on each day, and the probability of still being here by any given day.
>
> **The real advantage is what happens to patients still in the ward when the study ends.** A
> regression has to invent a discharge date for them or throw their records away. The hazard
> formulation needs neither: it records *"still here on day 30"*, which is completely true and
> genuinely informative — it rules out every earlier day.
>
> **That is 48.5% of our purchase orders.** Nearly half had not arrived when the 90-day window closed.
> The old regression head trained on the 55% that had, effectively studying only the patients who
> already went home — which systematically ignores the slow cases, the ones a planner most needs
> warning about. Switching to hazard was worth more than the entire graph.

### 2.3 How the reference metric was fixed — and why this matters

Lateness was originally scored against the **promise date**. Two independent problems:

1. **The promise date is not available at forecast time.** On this dataset the labelled purchase
   order line is raised a median of 47 days *after* the snapshot. Every arm, including ours, was
   being ranked on information no planner holds.
2. **The scorer substituted a constant** for the promise arm before computing lateness. Scored
   honestly, the promise date's lateness ROC-AUC is exactly **0.5000** — it carries no lateness
   information at all. That is structurally inevitable: the promise is `order_date + contracted
   lead`, and lateness is `actual lead > contracted lead`, so a per-channel constant cannot
   distinguish a late order from an on-time one.

The metric is now defined against the **as-of channel median observed lead** — the median lead over
receipts *recorded on or before t₀*, computable at the forecast instant, with an as-of assertion
demonstrated firing on three failure modes plus an empty-set guard.

> **In plain terms — grading a weather forecaster against next week's newspaper.**
>
> Suppose you want to know whether a forecaster beats "the official prediction". You compare their
> Monday forecast against an official prediction — but that official prediction was published the
> following Thursday. It is not a competitor; it is a partial answer key. Any forecaster who fails to
> beat it looks weak, and any who beats it looks miraculous. Neither reading means anything.
>
> That is exactly what the promise date was. The purchase order it belongs to is raised a median of
> 47 days *after* the moment we forecast. It is not a benchmark a planner could have used.
>
> **And a second, funnier problem sat underneath it.** The scoring code quietly replaced the promise
> date with a flat constant before comparing. So for five phases the reported comparison was
> "our model versus a fixed guess" while the label said "versus the promise date". The numbers were
> real; the sentence was not.
>
> **The fix is to grade against something Monday's reader actually had:** how fast this channel has
> been running lately, computed only from deliveries already recorded. Against that, the model wins
> cleanly — and, unexpectedly, wins by *more*, because the answer key had been so informative that it
> left no room for anyone to look good.

### 2.4 Performance

Lateness ROC-AUC, v8 test fold, five model seeds:

| arm | ROC-AUC |
|---|---|
| **head h⁴ (Temporal-SHARE)** | **0.70905** [0.70645, 0.71298] |
| b5flat LightGBM, same features | 0.70535 [0.70476, 0.70573] |
| head h⁰ (no graph) | 0.69999 [0.69842, 0.70133] |
| naive constant | 0.68697 |
| reference alone | 0.50000 (health check) |

**All three comparisons disjoint.** Arrival-week C-index: h⁴ **0.67438**, h⁰ 0.66133, best learned
baseline 0.64891 — also disjoint.

### 2.5 Verdict: successful, and correctly labelled for the first time

**Why it works:** the hazard head uses the censored half of the data; the encoder reads the channel
state that actually drives lateness (supplier load, transit state, prior utilisation) rather than
the contracted term, which is constant; and the graph contributes essentially all of the
depth-over-no-depth gain.

**What it cannot claim:** that it out-ranks the promise date at ordering arrivals. That comparison
is **retired**, not lost — the promise date is privileged information here, and the two quantities
are not available at the same instant. No feature change fixes that.

**Client-readable claim:** *ranking purchase orders by how much later they will arrive than their
channel's recent average — a reference computable from what was recorded by the forecast date — the
model separates late from on-time arrivals with ROC-AUC 0.709, ahead of a gradient-boosted model on
the same features (0.705) and a no-graph variant (0.700), across five training seeds with no
overlap.*

### 2.6 Open

- Week-level calibration (ECE-week) is seed-unstable at depth 4 — 0.176 to 0.310 across seeds on
  v8, and the same instability on v6 and v7. **Not quotable as a point value on any world.** This
  is why the served distribution is h⁰'s recalibrated output while h⁴ supplies the ranking.
- Phase 8's v6/v7 arrival row quotes the wrong comparison and needs rewording.

---

## 3. Use case 2 — Fill rate

### 3.1 What it predicts

For a purchase order line, the full distribution over *what fraction of the ordered quantity
arrives* — a number in [0, 1].

### 3.2 The method — binned CDF with explicit point masses

The distribution is not smooth. In practice most orders arrive complete, some arrive not at all, and
the interior is sparse. On v8, **79.9% of mass sits at exactly 1.0** and 2.1% at exactly 0.

So the target is discretised into **22 cells**: an atom at exactly 0, an atom at exactly 1.0, and 20
interior bins. The head emits a softmax over those cells, trained with the **ranked probability
score**:

```
RPS = (1 / (M−1)) · Σ_{m=1}^{M−1} ( F_m − 1{ y ≤ b_m } )²
```

where `F_m` is the predicted cumulative probability at bin edge `b_m`. RPS is a proper scoring rule
and — unlike cross-entropy over bins — it penalises being wrong by *a lot* more than being wrong by
a little, which is what you want when the bins are ordered.

Evaluation uses the exact, point-mass-aware CRPS:

```
CRPS(F, y) = ∫ ( F(x) − 1{x ≥ y} )² dx ,  atoms handled explicitly
```

**This distinction cost the project three phases.** A legacy CRPS implementation stepped the CDF at
bin edges and could not distinguish a mass at exactly 1.0 from one at 0.96. The point masses are
worth 18% / 10% on the exact integral and **0% on the legacy one** — which is why "fill is fine"
went unchallenged for so long.

**Real-world relevance.** "We'll ship 800 of your 1,000" is the single most common supplier
failure, and it is qualitatively different from a delay. A planner needs *P(complete)* specifically,
not an expected fraction — 0.9 expected fill could mean "always 90%" or "complete nine times in ten
and nothing the tenth", and those demand opposite responses.

> **In plain terms — predicting a batsman's score.**
>
> Most innings end in one of two very specific ways: out for nothing, or not out at the close. A
> smooth bell curve cannot put a spike on *exactly zero* — it will always smear that probability
> across "0 to 4 runs", which is a different statement about the game.
>
> So instead of fitting a curve, you hand out probability across labelled buckets, and you reserve two
> buckets for the two outcomes that are exact: **exactly nothing**, and **exactly complete**. That is
> the 22 cells — two atoms and twenty interior bins.
>
> Fill rate has the same shape: **80% of orders arrive exactly complete.** A smooth model spreads that
> spike across "95–100%", and "95% arrived" is a shortage while "100% arrived" is not. The business
> distinction lives exactly where the smooth model blurs.
>
> **Why RPS rather than ordinary classification — guessing someone's age band.** If they are 55 and
> you guess 50–55, you were nearly right. If you guess 20–25, you were badly wrong. Plain
> classification scores both as simply "incorrect". RPS charges you by *distance* across the ordered
> buckets, so being nearly right is rewarded — which is what you want when the buckets have an order.
>
> **And the measurement trap we fell into for three phases:** our original scoring code compared
> cumulative probabilities only at bucket *edges*, so it literally could not tell a spike at exactly
> 100% from one at 96%. The two atoms were worth 18% of the score under honest measurement and 0%
> under that one. "Fill is fine" survived unchallenged because the ruler could not see the thing being
> measured.

### 3.3 Performance

v8 test fold, recalibrated, 5 seeds where banded:

| metric | head | best baseline | verdict |
|---|---|---|---|
| **exact CRPS** (lower better) | **0.13827** | 0.13942 (recalibrated LightGBM-22) | **head wins, disjoint** |
| **ECE-22** (calibration, lower better) | 0.05583 | **0.01651** (rolling-52 as-of histogram) | **head loses, 3.4×** |

The head also loses ECE-22 to both recalibrated LightGBM arms. On v6/v7 the same pattern held in 12
of 16 backtest windows.

### 3.4 Verdict: partially successful, and the trade is uncomfortable

**Why the proper score works:** the point masses let the head put probability exactly where the data
is, which a smooth model cannot.

**Why calibration fails:** the head cannot fit its own training marginal on the rare interior cells
— Σ|error| 0.0445 against LightGBM's 0.0049. Diagnosed properly: it is not drift and not the loss
(three losses ablated, every difference inside band). It is head capacity on a sparse target.

**The trade, stated honestly:** the head buys roughly **1–2% on a proper score** and gives up
**10% to 240% on the calibration figure a planner reads off a screen**. A recalibration step (one
temperature, 22 biases, fitted on validation) improves ECE by 2.2–2.5× but does not close the gap,
and its fitted temperature swings 0.961–1.274 across windows — it must be refitted every retraining
and occasionally corrects in the wrong direction.

### 3.5 Open

- **The shipped configuration names a LightGBM that the serving path cannot load.** `shipped.json`
  says `b5flat22`; `loop.predict` has no LightGBM loader and silently returns the superseded neural
  head. Either build the loader or revert the config — a configuration that misdescribes what runs
  is worse than either honest state.
- Fill's recalibrated ECE varies 9× across backtest windows. No single fill calibration figure may
  be quoted without naming its window.

---

## 4. Use case 3 — Supplier strain

### 4.1 What it predicts

Per supplier-month, three quantiles (P10, P50, P90) of `capacity_strain` — roughly demand against
demonstrated throughput. Values sit near 0.87 on v8; above 1.0 means being asked for more than the
supplier has ever sustained.

**Why "strain" and not "capacity".** True capacity is never observed — you only see what a supplier
shipped, which is bounded by what you ordered. Strain is observable. This reframing unblocked the
use case; the original formulation was censored in a way no amount of modelling fixes.

### 4.2 The method — quantile regression via pinball loss

For target quantile τ:

```
L_τ(y, q) = max( τ·(y − q) , (τ − 1)·(y − q) )
```

Asymmetric by construction: under-predicting is penalised `τ` per unit and over-predicting `(1−τ)`.
Minimising it drives `q` to the τ-th quantile of the conditional distribution — no distributional
assumption required. Three heads at τ ∈ {0.1, 0.5, 0.9} give a band without assuming normality,
which matters because strain is right-skewed and clipped.

**Real-world relevance.** "This supplier will be at 85% of demonstrated capacity, and there is a
one-in-ten chance of above 110%" is directly actionable — it tells a buyer when to start a second
source *before* the failure, not after.

> **In plain terms — the tailor cutting cloth.**
>
> A tailor deciding where to cut faces a lopsided risk. Cut too short and the garment is ruined. Cut
> too long and you waste a little cloth. So a sensible tailor deliberately cuts long — not because
> they expect to need the extra, but because the two mistakes cost different amounts.
>
> **Pinball loss is that lopsidedness written down.** Charge τ for every unit you fall short and
> (1−τ) for every unit you overshoot. Set τ = 0.9 and falling short is nine times as expensive as
> overshooting, so the number that minimises your total cost lands automatically at the 90th
> percentile. **You get the "bad case" line without assuming anything about the shape of the
> distribution** — no bell curve required, which matters here because strain is skewed and clipped.
>
> Three tailors at τ = 0.1, 0.5 and 0.9 give you a low case, a middle case and a bad case.
>
> **Why the band currently fails, in the same terms.** The tailor learned their margins on last
> year's cloth, which shrank a certain amount in the wash. This year's cloth shrinks more. Their
> centre line is still right — they know how long a sleeve should be — but the safety margin they
> learned is now too tight, and the garments come up short more often than one in ten. That is exactly
> what we measured: coverage of 0.72–0.81 against a promised 0.80, and worst precisely when the
> period being forecast sits at a higher utilisation level than the period the margins were learned
> on. **Every model class failed this identically, including the gradient-boosted one** — so it is a
> property of learning margins at one level and applying them at another, not a flaw in our
> architecture. Which is why the fix has to widen the margin *in response to how much the cloth has
> changed*, and why simply cutting everything longer by a fixed amount would not work.

### 4.3 Performance

v8 test fold:

| metric | head h⁴ | h⁰ | LightGBM quantile | naive global |
|---|---|---|---|---|
| **pinball** (lower better) | **0.07248** | 0.07671 | 0.07990 | — |
| **80% coverage** (nominal 0.80) | 0.78391 | 0.80234 | 0.80344 | **0.82849** |
| P90 exceedance (nominal 0.10) | 0.13473 | 0.12065 | — | — |

Quantile crossing: **0.00000** — the three quantiles never invert, which is a real correctness
property and not guaranteed by independent quantile heads.

### 4.4 Verdict: the middle is right, the band is not

**Why the median works:** the graph is genuinely doing work here — capacity's label is a
supplier-level aggregate, so the readout node has a real neighbourhood (supplier median degree 38)
and attention has something to choose between. Pinball beats every baseline, disjointly.

**Why the intervals fail:** across a 16-window backtest, coverage ran 0.72–0.81 against nominal 0.80
and was below nominal in 12 of 16 windows — **in every model class, LightGBM included**. The
mechanism was isolated: exceedance correlates at ρ ≈ 0.8 with how far the evaluation window's level
sits above the window the quantiles were fitted on. Quantiles fitted at one utilisation level do not
transfer to another.

**Consequence: do not quote a capacity interval.** The median is quotable; the range is not.

### 4.5 Planned fix

The fix must be **level-aware**, because a fixed widening factor provably will not work:

- conformal widening keyed to recent drift — compute a nonconformity score on a trailing window and
  inflate the interval by its empirical quantile; or
- refit the quantile heads on a trailing window rather than a fixed calibration year.

Estimated ~3 hours. This is the one deferred model change whose premise still holds.

---

## 5. Use case 4 — Shortage risk (ranking)

### 5.1 Method and performance

A binary head over part-plant-snapshot, trained with cross-entropy:

```
L = −[ y·log p + (1−y)·log(1−p) ] ,  p = σ(f(x))
```

v8 test fold: ROC-AUC **0.81587**, PR-AUC **0.65875**, against the best baseline's 0.77257 / 0.58889
— disjoint on both. The strongest headline number in the project.

> **In plain terms — the examiner who grades on the wrong cohort.**
>
> Imagine an examiner trained entirely on a remedial class where 26 students in 100 failed. They become
> genuinely excellent at ordering students from weakest to strongest — put any hundred in front of them
> and the ranking will be right. But ask *"what fraction of this new class will fail?"* and they will
> say "about a quarter", because that is the world they learned in. In a normal class where three in a
> hundred fail, that answer is badly wrong while the ranking remains perfectly good.
>
> **That is our shortage head exactly.** v8 contains shortages at 25.7% against a real-world rate near
> 3% — roughly eight times too many, by construction. So *which* part-plants are most at risk is
> trustworthy; *how* risky any one of them is, is not.
>
> Hence the rule: **rank, never quantify.** ROC-AUC and PR-AUC only measure ordering, so they survive
> the distorted base rate. A probability does not, and none is quoted anywhere.
>
> **And why this head is diagnostic rather than the product:** it has learned a correlation with "a
> shortage happened", not the arithmetic of one. A planner needs a quantity and a week — *how many
> units short, in which week* — and that requires actually rolling the stock forward, which is the
> simulation below.

### 5.2 Verdict: ranks well, and its probabilities are unusable

**Why the ranking works:** shortage pools many channels at part × plant, so like capacity it has a
real neighbourhood for the graph to aggregate over.

**Why no probability may be quoted:** v8's shortage positive rate is 25.7% against a real operating
base rate near 3%, and v8's event rates run 4–6× over the bands Rane themselves stated. Any
probability from this head is **mis-calibrated in level by construction**. ROC-AUC and PR-AUC are
ranking metrics and are unaffected — which is precisely why they are what gets reported.

**Why it is diagnostic only:** a classifier learns a correlational shortcut to "will there be a
shortage." The product answer is a *quantity* and a *week*, and that requires the simulation below.

---

## 6. Use case 5 — Shortage quantity (Monte Carlo simulation)

### 6.1 What it does

Composes the three heads into a forward simulation of stock. For each part-plant, 1,000 paths over a
13-week horizon:

```
I_w = I_{w−1} + A_w − C_w
```

- `I_0` — the opening balance from the inventory store. **This is the number that blocked the
  project for eight phases.** On v6/v7 no valid opening level existed anywhere; on v8 the ledger
  roll-forward reproduces the stated balance on **100.000000% of 2,253,420 rows, max difference 0**.
- `A_w` — arrivals. For each open order, week `T` is drawn from the hazard head's recalibrated
  distribution and fraction `f` from the fill head's 22-cell CDF; the order contributes `qty × f` in
  week `T`.
- `C_w` — consumption, from the forward requirement plan.

Shortfall is measured against the safety stock floor, not zero. All 1,000 paths are retained so
percentiles are taken at the end and never summed from below — summing quantiles is a classic and
silent error.

**Why simulation rather than a formula.** Shortage is a *joint* event: late arrival AND partial fill
AND high demand, interacting through a stock level that carries state week to week. There is no
closed form. Sampling handles the interaction correctly and produces exactly what a planner needs —
*"85% chance of covering week 7; the shortfall, if it happens, is around 400 units."*

> **In plain terms — playing the board game a thousand times.**
>
> Suppose you want the odds of losing a particular board game. You could try to derive them
> algebraically — and for anything with interacting rules, you will fail. Or you can play it a thousand
> times and count.
>
> **Each "game" here is one possible future.** Roll the dice: this order lands in week 5 at 90% fill;
> that one in week 8 at 60%; demand comes in slightly above plan. Walk the stock ledger forward week
> by week and see whether you drop below safety stock, and by how much. Play a thousand times. The
> fraction of games you dipped is your probability; the size of the dips is your magnitude.
>
> **Two rules of playing that matter more than they sound.**
>
> *Play whole games and count at the end.* It is tempting to take the "90th percentile arrival" and the
> "90th percentile fill" and combine them — but that is not a real game, it is two separate bad days
> stapled together, and it usually gives an answer no actual future produces. We keep all thousand
> paths intact and take percentiles only at the finish.
>
> *Play with the real rules.* **This is where the simulation currently fails.** Our player is
> restocking by a rough guess — "buy roughly what you'll need" — instead of following the actual rules
> written on the board: reorder at this level, buy in lots of this size, never less than this minimum.
> Under the guess, arrivals replace only 85% of what gets consumed, so the player's stock trends
> downward every single game. Play a thousand games that way and almost all of them end in trouble —
> which is precisely our result: shortage predicted five times more often than it actually happened.
>
> **The board, the dice and the scorekeeping are all correct.** One player instruction is wrong.

### 6.2 Performance — the machinery is sound and the output is not

**Sound:**

| | |
|---|---|
| full grid | 4,340 part-plants × 61 snapshots × 13 weeks × 1,000 paths = 3.44 billion cells |
| wall-clock | **5.16 minutes** |
| identity gate | passes on correct replay; **fires on all five constructed failures**, including one unit wrong in one row of two million |
| as-of discipline | the assertion correctly dropped the 2016 snapshots where no label snapshot exists at or before them |
| orphan part-plants | the 113 with no channel are **carried with zero arrivals, not dropped** |

**Not sound:**

| | simulated | observed 2025 | ratio |
|---|---|---|---|
| part-plant-weeks below safety stock | **44.36%** | 8.38% | **5.29× over** |
| mean shortfall when short | 564 units | 95 units | — |

### 6.3 Why it fails — diagnosed precisely

**The order quantity is a placeholder.** The simulation currently guesses "order roughly the
horizon's requirement" instead of using the reorder point, minimum order quantity and lot size that
`part_plant` already carries. Over 13 weeks, arrivals replace only **85.4%** of consumption against
an opening level of 1,238 units. **The horizon drains by construction** — every path trends down, so
by the later weeks almost everything is below safety stock.

**Arrivals beyond week 13 are dropped** rather than carried, which compounds the drain.

**Demand is treated as near-certain.** Consumption is the plan's point forecast with a
`uniform(0.85, 1.15)` jitter — standard deviation 0.0865 — against a measured actual/planned
dispersion of **0.1265**. The placeholder **understates real plan error by 1.46×**. And because no
demand head exists in the shipped set, every interval the simulation emits reflects **supply
uncertainty only**.

**Consequence: no quantity from this simulation is usable as a forecast.** The gate, the ledger and
the as-of discipline are all correct; the draws are not.

### 6.4 Planned fix

1. **Replace the order heuristic with the real policy.** Trigger on `reorder_point_qty`; size with
   `min_order_qty` and `lot_size`; respect `planning_lead_time_days` when placing the order.
2. **Carry arrivals beyond the horizon** instead of discarding them.
3. **Widen consumption dispersion** to the measured sd 0.1265 as an interim, and label it as still
   representing plan error rather than demand uncertainty.
4. **Re-run the held-out validation.** If 5.29× closes toward 1.0, this becomes the first real
   forecast in the project. If it does not, the residual gap identifies precisely what else is
   missing — a far more informative failure than the present one.
5. **Build a demand head** before any interval is presented as total uncertainty.

### 6.5 The correlated-failure extension, deliberately deferred

The design calls for a copula so that suppliers in a group fail together. **It is deferred, and the
reason is a measurement, not a schedule:** on v8, within-group correlation is +0.2101 against
cross-group +0.2055 — a separation of **0.0046**, an order of magnitude inside the acceptance gate's
own ±0.05 tolerance. A per-group ρ and a single global ρ are statistically indistinguishable here,
so the copula's central claim cannot be validated on this world.

Building it would produce a component whose correctness cannot be demonstrated. And correlated
sampling layered on marginals that over-project by 5.29× would make the output worse, not better.
**Fix the marginals first.**

---

## 7. Use case 6 — Delivery schedule (MILP)

### 7.1 The method

A mixed-integer linear program over a weekly horizon:

```
minimise   Σ_w [ c_hold·I_w + c_order·y_w + c_freight·⌈x_w / truck_cap⌉ ]

subject to I_w = I_{w−1} + x_w − d_w        stock balance
           I_w ≥ SS                          safety-stock floor
           x_w ≥ MOQ · y_w                   minimum order quantity
           x_w ≤ M · y_w                     linking (M a valid big-M)
           x_w ≡ 0  (mod L)                  lot-size multiple
           y_w ∈ {0,1}                       order placed indicator
```

The integrality is essential: `y_w` makes "place an order at all" a discrete decision, which is what
creates the ordering-cost-versus-holding-cost trade-off. Without it the problem is a linear program
and the answer degenerates to just-in-time.

**Real-world relevance.** This is the question a buyer actually faces every week — order now and
carry stock, or wait and risk the floor. It needs no learned model, only correct constraints.

> **In plain terms — grocery shopping with a small fridge.**
>
> Every trip to the shop costs you an hour of your Saturday — that is the ordering cost. Stocking up
> means food sitting in the fridge going off — the holding cost. The shop sells eggs only by the dozen
> — the lot size. And they won't deliver for under ₹500 — the minimum order quantity. You also can't
> let the fridge run empty, because someone has to eat on Wednesday — the safety stock floor.
>
> The MILP is the timetable that answers: **how many trips this month, and how much on each one.**
>
> **The "integer" part is the trip itself.** You either go to the shop or you don't — there is no
> two-thirds of a trip. That yes/no is the whole source of the trade-off: if trips were free and
> divisible you would simply buy each item the moment you needed it. Make going to the shop cost
> something discrete and suddenly consolidating becomes worth it. Remove the integrality and the answer
> collapses to just-in-time.
>
> **Why it currently cannot advise you on timing.** Nobody has told us what the fridge costs. With
> holding cost at zero, food never goes off — so buying everything on the first trip and buying
> everything on the last trip cost *exactly the same*, and the solver picks between them arbitrarily.
> It can still tell you *how much* to buy (it knows about dozens and minimum deliveries). It cannot tell
> you *when*. That is not a solver limitation; it is a missing price, and only Rane has it.

### 7.2 Status — blocked, and not on the solver

Built and verified on v6/v7. The MILP solves in milliseconds. What it lacks is an objective:

| needed | present in any dataset? |
|---|---|
| inventory holding / carrying cost | **no** |
| ordering / setup cost per PO | **no** |
| truck capacity and freight tariff | **no** |

With `c_hold = 0` the objective is **indifferent to timing** — ordering everything in the first week
and everything in the last cost exactly the same. The guide's own verification gate ("a part with no
capacity constraint and zero holding cost must order in the last feasible week") is therefore
**vacuous**: its premise is true of every part. It was replaced with five gates, each demonstrated
failing on constructed input.

On v6/v7 the horizon was also single-period, because the plan held one week 30 days out. **v8 fixes
that** — 12 to 26 forward periods per plan version, refreshed every 56 days, 100% forward. The
multi-period schedule is now buildable and has not yet been rebuilt on v8.

### 7.3 Planned

- Rebuild multi-period on v8 now that the forward horizon exists.
- The three cost parameters are **client asks**. No generator can supply them without inventing the
  answer.

---

## 8. Use case 7 — Supplier allocation

### 8.1 The method

Enumerate five candidate splits per part-plant — incumbent, all-to-cheapest, equal split, shift 20pp
to the cheapest alternative, cap any supplier at 70% — then score each:

```
score(s) = requirement · Σ_v share_v · (1 − E[fill_v]) · strain_penalty_v · c_short
         + Σ_v share_v · requirement · price_v
```

subject to constraints that decide whether a move is possible at all: tooling transferability,
qualification lead time, ramp-rate limit, minimum-volume commitment.

**The constraints are the product, not the objective.** A recommendation to move volume to a
supplier who does not hold the tooling and cannot be qualified for six months is worse than no
recommendation — it destroys trust in everything else the system says.

> **In plain terms — splitting the wedding order between three tailors.**
>
> Six weeks to the wedding, forty outfits, three tailors. You could give everything to the cheapest,
> split it evenly, or shift a portion to the most reliable one. Score each plan by what it costs plus
> what it costs you if something isn't ready on the day.
>
> **But some plans are impossible at any price, and that is the real work.** Tailor A holds the only
> pattern for the embroidery, and it cannot be copied — so that portion cannot move, full stop.
> Tailor B can take on at most 20% more work per month without dropping quality. Tailor C has never
> made this design and needs a trial garment approved first, which takes three weeks you do not have.
>
> **Those three facts decide whether the recommendation is usable. The pricing only decides which of
> the usable ones is best.** A system that recommends Tailor C because they are cheapest, without
> knowing about the trial garment, is worse than a system that recommends nothing — you will never
> trust its next suggestion.
>
> **And the one number that decides everything is the one nobody has given us.** How bad is it, really,
> if an outfit isn't ready? If the answer is "mildly embarrassing", go with cheapest. If it is "the
> wedding is ruined", pay for the reliable tailor. We swept that number across a hundredfold range and
> the recommended split *changed* for a fifth of cases — with the flips clustered right around the
> figure we had invented. **Choosing that number ourselves is choosing the answer**, which is why it is
> a client ask and not a parameter we should be tuning.

### 8.2 Status — blocked on one number

On v6/v7, only **25% of part-plants** produced a recommendation stable across a shortage-cost sweep
and distinguishable from the runner-up; nine of those thirty said "change nothing". About 18%
actionable.

**The shortage cost is the blocker and it is not a tuning constant.** It sets the exchange rate
between unmet demand and purchase price — the entire trade-off the allocator exists to make. Swept
across two orders of magnitude, the recommended split *changed* for a fifth of part-plants, and the
flips clustered around the assumed ₹1,000. **Choosing that number ourselves means choosing the
answer.**

v8 improved the inputs substantially — allocation coverage from 19.5% to **97.05%**, capacity
ceilings populated, contract terms with real variation, a genuine 16.7% of alternate sources
unqualified so the qualification constraint can finally bind. But `is_approved` is still constant,
the minimum-volume commitment still carries no period, and the penalty still has no basis.

### 8.3 Planned

- Re-run on v8 with the real constraint data.
- Escalate the shortage cost from "assumption" to **blocking client ask**.
- Obtain the commitment period and penalty basis — the numbers vary now, but a quantity with no
  window cannot be enforced and a penalty with no basis cannot be priced.

---

## 9. Consolidated status and plan

### 9.1 Where each use case stands

| use case | works | does not | blocked by |
|---|---|---|---|
| Arrival timing | lateness ranking, disjoint at 5 seeds | week-level calibration at depth 4 | — |
| Fill rate | exact CRPS | marginal calibration (3.4× worse than a naive histogram) | serving-path loader |
| Supplier strain | median, pinball | intervals (coverage 0.72–0.81) | needs level-aware recalibration |
| Shortage ranking | ROC-AUC 0.816 | any probability | synthetic base rate 8× reality |
| Shortage quantity | mechanism, gate, ledger | **calibration (5.29× over)** | order policy — **ours to fix** |
| Delivery schedule | solver, constraints | objective | 3 cost parameters — **client** |
| Allocation | enumeration, constraints | recommendation stability | shortage cost — **client** |

### 9.2 The plan, in order

**Ours to fix:**

1. **Order policy in the simulation** (reorder point, MOQ, lot size, carry beyond horizon, widen
   demand dispersion), then re-validate against held-out outcomes. *Everything downstream waits on
   this.*
2. **Fill's serving path** — build the LightGBM loader or revert the config, and add an assertion
   that refuses to serve when the loaded model does not match the configured one.
3. **Capacity intervals** — conformal widening keyed to recent drift, or trailing-window
   recalibration. ~3 h.
4. **A demand head** — until one exists, every simulation interval is supply-only.
5. **Reword Phase 8's arrival row** to the comparison actually measured.
6. **Re-band the remaining 16 tight comparisons** at five seeds. Nineteen of twenty held in the
   sample already run, so this is expected to confirm rather than overturn — but that is an
   expectation, not a measurement.

**Rane's to supply** — no amount of work on our side closes these, and they have been open across
three datasets:

7. Inventory carrying cost.
8. Ordering / setup cost per purchase order.
9. Freight structure — truck capacity and tariff shape.
10. **Cost of running short of a critical part.** The single most sensitive number in the system.
    A rough figure with a stated basis is entirely sufficient; what matters is that it is theirs.

---

## 10. Standing caveat

Every figure in this document is measured **within a synthetic world**: one generator produced both
the training and the test data, separated only by time. These are an optimistic upper bound on a
real extract, not a forecast of it.

The findings that transfer are the **mechanisms** — that censoring must be modelled rather than
dropped, that point masses matter for fill, that quantile intervals do not survive a level shift,
that the graph's edges carry information — not the numbers. The only evidence that would support a
production figure is a rolling-origin backtest on Rane's own data.
