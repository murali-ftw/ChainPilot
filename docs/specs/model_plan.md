# ChainPilot × Rane — Model Plan

**How each model works: theoretically, mathematically, and in real life.**

Companion to `dataset_structure.md` (what the data is) and `data_plan.md` (which columns feed
what). This document explains the machinery itself.

Ten models cover all nine use cases. Only three are neural networks, and those three share
one encoder.

| # | Model | Used by | Class |
|---|---|---|---|
| 1 | BOM explosion | UC1, UC6 | Sparse linear algebra |
| 2 | Quantile drift model | UC1 | Order statistics → gradient boosting |
| 3 | Revealed capacity envelope | UC2 | Censored order statistics |
| 4 | **Temporal-SHARE encoder** | UC2, UC3, UC4 | TCN + relational GNN |
| 5 | Quantile regression head | UC2 | Pinball loss |
| 6 | Discrete CDF head | UC3 | CRPS loss |
| 7 | Discrete-time hazard head | UC4 | Survival analysis |
| 8 | Monte Carlo simulation | UC5, UC6 | Stochastic simulation + copula |
| 9 | LP / MILP | UC7, UC8 | Constrained optimisation |
| 10 | LightGBM | UC9, all baselines | Gradient-boosted trees |

---

## 1. BOM explosion

### Theory
A bill of materials is a **directed acyclic graph**. Products point to sub-assemblies point
to parts. Demand flows down it; shortage flows back up it. There is nothing to learn — the
structure is given and the arithmetic is exact.

The reason to state it formally rather than write a loop is that the formal version makes
multi-level BOMs, phantom assemblies and the upward exposure rollup fall out of the same
operation.

### Mathematics
Let $\mathbf{B}$ be the sparse matrix with $B_{jp} = $ quantity of part $p$ per unit of
product $j$. Direct requirements for a plan vector $\mathbf{d}$:

$$\mathbf{r}^{(1)} = \mathbf{B}^\top \mathbf{d}$$

For an $n$-level BOM, requirements cascade through sub-assemblies:

$$
\mathbf{r} = \big(\mathbf{B} + \mathbf{B}^2 + \dots + \mathbf{B}^n\big)^\top \mathbf{d} = \big((\mathbf{I}-\mathbf{B})^{-1} - \mathbf{I}\big)^\top \mathbf{d}
$$

The matrix $(\mathbf{I}-\mathbf{B})^{-1}$ is called the **total-requirements** or *gozinto*
matrix in MRP literature. In practice we never invert it — we sort the DAG topologically by
`bom_level` and iterate level by level, which is $O(|E|)$ instead of $O(n^3)$ and handles
scrap factors cleanly.

Scrap enters as a division, applied at each level:

$$r_p = \sum_j d_j \cdot \frac{q_{jp}}{1 - \sigma_p}$$

**The upward direction** (UC6) is the same matrix transposed, with a min instead of a sum —
because a product is blocked by its *scarcest* component, not the total of its shortages:

$$\text{buildable}(j) = \min_{p \in \text{BOM}(j)} \left\lfloor \frac{I_p}{q_{jp}} \right\rfloor$$

### In real life
A planner does this in Excel today, one product at a time, and it takes hours. The two
places it goes wrong in practice:

**Stale BOMs.** Engineering changes a part in March; the spreadsheet still has the old one in
June. Every requirement downstream is quietly wrong. This is why `effective_from` /
`effective_to` are mandatory rather than nice-to-have.

**Phantom assemblies.** Sub-assemblies that are built and consumed in the same operation
without ever being stocked. If you treat them as real, you plan inventory for something that
never sits still.

> Rane's ERP almost certainly runs this already as part of MRP. Read their gross-requirements
> output and use ours to *validate* theirs. Where they disagree, one of the two has a stale
> BOM — and finding that is worth more in week one than the explosion itself.

---

## 2. Quantile drift model

### Theory
The production plan is known; how much it will be *missed by* is not. Rather than forecast
demand, forecast the **error of a forecast that already exists**. This is a much easier
problem and it's why "80% accurate next month, worse at three months" is fixable.

Critically, we want a **distribution** rather than a point estimate. The downstream
simulation needs a conservative case, and safety stock is meaningless without a percentile.

### Mathematics
The drift ratio at horizon $h$:

$$\rho(j,l,\tau \mid h) = \frac{\text{actual}(j,l,\tau)}{\text{planned}(j,l,\tau \mid \text{plan made at } \tau-h)}$$

**Baseline — empirical quantiles.** No model at all. Bucket historical $\rho$ by (horizon
band, plant, product family) and read percentiles off directly:

$$\hat\rho_q = \text{Quantile}_q\big(\{\rho_i\}_{i \in \text{bucket}}\big)$$

With 7 years there are ~84 observations per (product, plant) and thousands per bucket. That
is plenty for a stable quantile, and it is completely transparent.

**Escalation — LightGBM with pinball loss**, only if buckets prove thin:

$$
\mathcal{L}_q(\rho, \hat\rho) = \begin{cases}
q \cdot (\rho - \hat\rho) & \text{if } \rho \ge \hat\rho \\
(q-1)(\rho - \hat\rho) & \text{otherwise}
\end{cases}
$$

**Why this loss produces a quantile.** Setting the expected derivative to zero gives
$P(\rho \le \hat\rho) = q$ at the optimum. The asymmetry is the whole trick: at $q=0.9$,
under-predicting costs nine times as much as over-predicting, so the model is pushed until
90% of outcomes sit below its estimate.

### In real life
The horizon conditioning is the entire value here. A plan made one month out and one made
three months out have completely different error profiles, and **that decay curve is the
client's stated problem.** Bucketing by horizon and reporting per-horizon — never averaged —
is most of the win.

Real drift is not symmetric. Plants under-build far more often than they over-build, so the
distribution is left-skewed and a mean would systematically mislead. Quantiles handle this
without needing to model the shape.

The one thing to watch: **plan revisions.** If a plan is revised weekly, "the plan at
horizon $h$" is ambiguous. Anchor on `plan_version` and treat firm and tentative periods
(`is_firm`) as separate populations — they behave nothing alike.

---

## 3. Revealed capacity envelope

### Theory
Capacity is never directly observed. What a goods receipt records is

$$D = \min(K, O)$$

— delivered is the smaller of what they *could* make and what you *asked for*. This is
**censored observation**, the same statistical structure as measuring how fast a car can go
by only ever driving it at 60 km/h.

The consequence: in any month where you ordered less than their capacity, the data carries
**zero information about capacity**. Averaging all months together therefore estimates your
own ordering behaviour, not their ability.

### Mathematics
Flag the months that carry evidence — those where the supplier visibly struggled:

$$
\text{constrained}(s,p,m) = \mathbb{1}\Big[f(s,p,m) < 1 \ \lor\ \tfrac{\text{LT}_{\text{actual}}}{\text{LT}_{\text{contracted}}} > 1+\delta\Big], \quad \delta \approx 0.15
$$

Then the envelope combines a hard lower bound with a robust upper estimate from constrained
months only:

$$
\hat{K} = \max\Big( \underbrace{\max_{m \in [t_0-12,\,t_0]} D_m}_{\text{they demonstrably did this}},\ \ \underbrace{\text{P95}\{D_m : \text{constrained}\}}_{\text{robust ceiling}} \Big)
$$

**Evidence strength** — how much of the estimate rests on actual evidence:

$$\eta = \frac{|\{m : \text{constrained}\}|}{12}$$

$\eta$ must widen the output band. Low $\eta$ means we are extrapolating, and the interface
has to show that honestly.

### In real life
Most manufacturers hold a stated capacity number that is contractual, negotiated years ago,
and never revised. It reflects what was agreed, not what the supplier can currently do —
they may have added a shift since, or lost a machine, or quietly reallocated your line to
another customer.

The revealed envelope is usually *more accurate than the number in the system*, which is
both the selling point and the diplomatic problem. Present it as "demonstrated capacity"
alongside the contractual figure rather than as a correction, and let the gap between them
start the conversation.

> **The UI rule this implies.** If a supplier has never been loaded past 90% of their
> historical max, a projection of 108% is beyond all evidence. Shade that region and widen
> the band. Showing it as a confident number is how a model loses a procurement team
> permanently.

---

## 4. Temporal-SHARE encoder

The one substantial neural component. Everything else in this document is either arithmetic
or a well-understood classical method.

### Theory
Two facts about the data force this architecture.

**Fact one: some entities have history, others don't.** A sourcing channel — supplier ×
part × plant — has 7 years of behaviour. A PO line is created, lives three weeks, and is
gone. Running a sequence model over a PO line is meaningless; running one over a channel is
exactly right. So the model splits nodes into **persistent** (channels, suppliers, parts,
plants) and **instance** (PO lines) and treats them differently.

**Fact two: thin entities need to borrow from thick ones.** A channel with 6 historical
orders has no usable statistics of its own. But it shares a supplier with 400 other orders,
a component category with 5,000, a supplier group and possibly a tier-2 parent. A graph is
the natural structure for that borrowing, and it is the thing a per-supplier spreadsheet
structurally cannot do.

**Why not an STGNN?** The classic spatio-temporal GNNs (DCRNN, STGCN, Graph WaveNet)
interleave time-mixing and graph-mixing layer after layer, and assume a **fixed node set**
whose values change. Our graph churns — half the nodes are PO lines that didn't exist last
month. Temporal-SHARE does all the time work first, per persistent node, then a single graph
pass. It sidesteps the churn entirely and costs far less.

### Mathematics

**Stage 1 — temporal encoding (dilated TCN).**

For each persistent node $c$, take its weekly feature sequence
$X_c = [\mathbf{x}_c(t_0-51), \dots, \mathbf{x}_c(t_0)] \in \mathbb{R}^{52 \times d}$.

A dilated causal convolution at layer $l$ with dilation $d_l = 2^l$:

$$
\mathbf{z}^{(l+1)}_t = \sigma\left(\sum_{k=0}^{K-1} \mathbf{W}^{(l)}_k \, \mathbf{z}^{(l)}_{t - d_l \cdot k} + \mathbf{b}^{(l)}\right)
$$

With kernel size $K=2$ and dilations $1, 2, 4, 8, 16$, the receptive field is
$1 + \sum_l (K-1)d_l = 32$ weeks after 5 layers — most of a year, with only
$\mathcal{O}(K \cdot L)$ parameters.

> **SUPERSEDED — build 6 layers, not 5.** `docs/benchmark_specification.md` §3.4 is the
> authority and specifies dilations 1–32 over **6 layers**, giving a **64-week** receptive
> field. With 5 layers the field is 32 weeks and the earliest 20 weeks of the 52-week input
> window are unreachable by the top layer. This section also predates `d_in = 38`.

*"Causal"* means position $t$ only ever sees $t' \le t$. This is a structural leakage
guarantee at the architecture level, not just a data-pipeline convention.

The node's temporal state is the final position: $\mathbf{s}_c = \mathbf{z}^{(L)}_{t_0}$.

**Why TCN over GRU.** Parallel across timesteps (a GRU is inherently sequential), no
vanishing gradients, and the receptive field is an explicit design choice rather than
something you hope the gates learn. At the measured sequence lengths — median 458 weeks per
channel, max 523 — this matters for training time.

**Stage 2 — graph encoding (SHARE), unchanged from HADES v3.**

Relations get a shared basis rather than independent weights — this is what protects thin
relation types from overfitting:

$$\mathbf{W}_r^{(l)} = \sum_{b=1}^{B} a_{rb}^{(l)} \mathbf{V}_b^{(l)}, \qquad B = 10$$

A relation never gets its own free parameter set, only a different mixing of $B$ shared
bases.

On top sits a single **relation-blind** attention scorer — one vector $\mathbf{a}$ shared by
every relation type:

$$
e_{ij} = \text{LeakyReLU}\Big(\mathbf{a}^\top \big[\mathbf{W}_r \mathbf{h}_i \,\Vert\, \mathbf{W}_r \mathbf{h}_j\big]\Big), \qquad
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}
$$

The scorer never sees *which* relation connects two nodes — only the content of the
already-transformed endpoints. That is the "relation-blind" part, and it is what recovers
the ability to single out one standout neighbour without reintroducing per-relation
parameters.

$$
\mathbf{h}_i^{(l+1)} = \sigma\left(\mathbf{W}_0^{(l)}\mathbf{h}_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)} \alpha_{ij}\, \mathbf{W}_r^{(l)} \mathbf{h}_j^{(l)}\right)
$$

Persistent nodes enter with $\mathbf{h}^{(0)}_c = \mathbf{s}_c$ (their temporal state);
instance nodes enter with a linear projection of their current features.

**Why this middle ground.** Fully independent per-relation attention (HGT) overfits thin
relation types — a relation with a few hundred edges doesn't have enough examples to fit its
own parameter set. Uniform aggregation (plain RGCN) loses the ability to weight one
important neighbour. The seven-architecture bake-off in HADES v3 confirmed SHARE as
best-or-tied on every task, and its nearest rival (SHARK, full per-relation attention) showed
**~10× worse seed-to-seed instability** at the same accuracy.

**Stage 3 — staleness gating.**

Rane's ERP is manually updated, so some records at $t_0$ are fresh and some are weeks old.
Each time-varying feature carries a learned trust weight decaying with its age:

$$
\tilde{x}_i = g_i \cdot x_i^{\text{obs}} + (1-g_i)\cdot x_i^{\text{prior}}, \qquad
g_i = \exp\big(-\max(0,\ w_i \Delta t_i + b_i)\big)
$$

Three rules make this work: **per feature** (not per node — `country` never changes and must
not be penalised for being old), **continuous** (a hard 30-day threshold makes 29 and 31
days behave completely differently), and **$\Delta t$ also passed as a plain feature**
(staleness decides trust *and* may itself be predictive — a PO record untouched for three
weeks might be stuck in customs). Precedent: GRU-D.

### Readout depth

HADES v3 fixed the readout depth per task — delay→h¹, shortage→h³, impact→h⁴ — after eight
attempts to learn it all lost to simply choosing it. **That mapping must be re-derived for
Rane, not inherited.** v3's own Layer 3 work found h⁴ was the *worst* depth for both latent
states that passed their gate, with h¹ and h² winning. Depth is task-specific in a way that
is not predictable in advance.

Expected pattern here: fill rate and timing read shallow (1 hop — the channel and its
supplier), capacity reads deeper (2–3 hops, reaching supplier group and tier-2).

### In real life

**What the temporal part buys you** is regime awareness. *"This supplier has been degrading
for four months"* and *"this supplier is consistently mediocre"* produce an identical 90-day
average, and they demand completely different responses. A fixed rolling window flattens them
together; a learned sequence encoder can represent the difference.

**What the graph part buys you** is coverage. Roughly half of Rane's 9,000 channels will have
too few orders for their own statistics. Without pooling, half the network is unpredictable
by construction.

**The honest caveat.** HADES v3's h⁰ diagnostic measured the graph's contribution on the
delay task at about **10% of the above-chance signal** — raw features scored 0.7516 AUC, the
full four-layer encoder 0.7802. Input quality dominated architecture by a wide margin. Run
the same diagnostic here before assuming the graph earns its keep on any given task, and
spend the first effort on features rather than layers.

---

## 5. Quantile regression head (capacity)

### Theory
Utilisation is a continuous quantity with genuinely asymmetric consequences: predicting 85%
when the truth is 105% causes a line stop; the reverse causes some unnecessary expediting.
A point estimate cannot express that. Quantiles can, and they compose directly into the band
the deck's chart wants.

### Mathematics
Three quantiles at three horizons — nine outputs:

$$\hat{u}_q(s, t_0+h), \quad q \in \{0.1, 0.5, 0.9\},\ h \in \{30,60,90\}$$

Total loss:

$$\mathcal{L} = \sum_{h}\sum_{q} \mathcal{L}_{\text{pinball}}\big(u_{\text{true}}(h),\ \hat u_q(h)\big)$$

**Monotonicity.** Nothing in the loss prevents $\hat u_{0.1} > \hat u_{0.9}$ on some inputs
— a nonsensical crossing. Two fixes: predict $\hat u_{0.1}$ plus non-negative increments
(via softplus) to the higher quantiles, or sort the outputs post-hoc. Prefer the first; it's
structural rather than cosmetic.

**Days-to-exceed** is derived, not learned:

$$\text{DTE} = \min\{h : \hat u_{0.5}(h) \ge 1\}$$

interpolated linearly between the 30/60/90 grid. Learning it separately would duplicate
information already in the curve and add error.

### In real life
Nine numbers per supplier is the right output because the deck's own table asks for exactly
that shape: utilisation at three horizons plus a days-to-exceed figure. The band is what
keeps the model honest — for a supplier never loaded past 90% of their historical max, the
P10–P90 band at $h=90$ should be visibly wide, and that width *is* the message.

---

## 6. Discrete CDF head (fill rate)

### Theory
Fill rate has an unusual distribution: a large point mass at exactly 1.0 (most orders arrive
complete), a smaller one at exactly 0 (nothing came), and a continuous smear between. A
Gaussian regression fits one smooth curve through this and is confidently wrong at both ends
— which are precisely the two outcomes a planner cares about.

A binned distribution handles point masses naturally, because a single bin can hold all the
mass sitting at one exact value.

### Mathematics
Bins with dedicated endpoints:

$$b_0 = \{0\},\quad b_1 = (0, 0.05],\ \dots,\ b_{18} = (0.95, 1),\quad b_{19} = \{1\}$$

Model outputs $\boldsymbol{\pi} = \text{softmax}(\mathbf{o})$ over 20 bins. Loss is CRPS
over the cumulative distribution:

$$
\mathcal{L}_{\text{CRPS}} = \sum_{b=0}^{19}\Big(\hat F(b) - \mathbb{1}[f_{\text{true}} \le b]\Big)^2, \qquad \hat F(b) = \sum_{b' \le b}\pi_{b'}
$$

**Why CRPS rather than cross-entropy.** Cross-entropy treats bins as unordered categories —
predicting bin 2 when the truth is bin 19 costs exactly the same as predicting bin 18. CRPS
knows the bins are ordered and penalises by distance. It is also a *proper* scoring rule:
its expected value is minimised by the true distribution, so the model has no incentive to
hedge.

**Alternative if bins prove too coarse:** zero-one-inflated Beta —
$\pi_0$, $\pi_1$, and $(\alpha,\beta)$ for the interior. Better specified, more fragile to
train. Start with bins.

### In real life
The output reads as a sentence a buyer understands: *"60% chance it arrives complete, 5%
chance nothing comes, and if it's partial, most likely around 80%."*

What actually gets reported is a quantity, not a probability:

$$\text{expected shortfall} = \texttt{qty\_ordered} \times \big(1 - \mathbb{E}[f]\big)$$

Three reasons. A planner can act on units directly. Units add up correctly across four
levels of rollup where probabilities do not. And a mediocre units estimate is still useful
for ranking, whereas a mediocre alert at 25% precision is just noise — which is exactly what
HADES v3's operating-point analysis found.

---

## 7. Discrete-time hazard head (arrival timing)

### Theory
Predicting *when* something arrives has a problem ordinary regression can't handle: at any
$t_0$, many POs haven't arrived yet. Their outcome is unknown — statistically
**right-censored**. You know only that $T > w_{\text{current}}$.

Dropping them keeps only the orders that already landed, which biases the training set
toward fast suppliers and makes every forecast quietly optimistic. Survival analysis exists
for exactly this, and a hazard model uses the partial information rather than discarding it.

### Mathematics
Hazard = conditional probability of arriving in week $w$ given it hasn't yet:

$$\lambda_w = P(T = w \mid T \ge w)$$

The model emits $\hat\lambda_w$ for $w = 1\dots12$ through independent sigmoids. Survival and
arrival probability follow:

$$S_w = \prod_{k<w}(1-\hat\lambda_k), \qquad P(T=w) = \hat\lambda_w \, S_w$$

The likelihood handles both complete and censored observations in one expression:

$$
\mathcal{L} = -\sum_\ell \sum_{w=1}^{\min(T_\ell,\,C_\ell)} \Big( \mathbb{1}[T_\ell = w]\log\hat\lambda_w + \mathbb{1}[T_\ell > w]\log(1-\hat\lambda_w) \Big)
$$

An open PO contributes $\log(1-\hat\lambda_w)$ for every week it has already survived. That
is real information — *"this order has now been open five weeks without arriving"* — and a
naive setup throws it away.

### In real life
The hazard shape itself is interpretable and worth showing. A supplier with a sharp spike at
week 0 and a long thin tail is *usually punctual, occasionally disastrous*. One with a broad
hump at weeks 2–3 is *reliably two weeks late* — which is a scheduling problem, not a risk
problem, and the fix is to adjust the planning lead time rather than chase the supplier.

Feeds the simulation directly:

$$\text{arriving}(p,l,w) = \sum_{\ell} q_\ell \cdot \mathbb{E}[f_\ell] \cdot P(T_\ell = w)$$

---

## 8. Monte Carlo simulation

### Theory
Shortage is not predicted — it is **computed** from a stock balance driven by three uncertain
inputs. Rather than learn shortage directly, roll the balance forward and propagate the
uncertainty through it.

Two reasons this beats a learned shortage model as the primary output.

**Auditability.** A procurement manager can follow every line of a stock roll-forward and
challenge any number in it. In a first deployment, where the real barrier is getting anyone
to trust the output at all, that is worth more than a couple of points of accuracy.

**Composability.** Change the allocation and re-run — you get the what-if for free. A learned
shortage model would have to be retrained or, worse, extrapolated outside its training
distribution.

### Mathematics
Per pass $n$, for each (part, plant), week by week:

$$
\begin{aligned}
I_w^{(n)} &= I_{w-1}^{(n)} + A_w^{(n)} - C_w^{(n)} \\
A_w^{(n)} &= \sum_\ell q_\ell f_\ell^{(n)} \mathbb{1}\big[T_\ell^{(n)} = w\big] + \min\big(\text{planned}_w, \hat K_w^{(n)}\big) \\
\text{Short}_w^{(n)} &= \max\big(0,\ \text{safety\_stock} - I_w^{(n)}\big)
\end{aligned}
$$

Shortage is measured against **safety stock**, not zero. Hitting zero is already a line stop;
the warning has to arrive earlier than that.

**The draws:**

$$f_\ell^{(n)} \sim \boldsymbol{\pi}_\ell \ \ (\text{UC3}), \qquad T_\ell^{(n)} \sim P(T_\ell = \cdot)\ \ (\text{UC4}), \qquad C_w^{(n)} \sim \hat\rho \ \ (\text{UC1})$$

**Correlated failure — the most important modelling decision here.** Independent sampling
badly understates risk. Two channels sharing a supplier group, a tier-2 parent or a transport
checkpoint fail together. Model it with a Gaussian copula:

$$
z_g^{(n)} \sim \mathcal{N}(0,1) \text{ per group } g, \qquad
f_\ell^{(n)} = \Pi_\ell^{-1}\Big(\Phi\big(\rho\, z_{g(\ell)}^{(n)} + \sqrt{1-\rho^2}\,\varepsilon_\ell^{(n)}\big)\Big)
$$

where $\Pi_\ell^{-1}$ is the inverse CDF from the binned head and $\rho$ is estimated from
historical co-occurrence of shortfalls within each group.

**Outputs** from $N=1000$ passes:

$$
P(\text{short}) = \tfrac{1}{N}\sum_n \mathbb{1}\big[\max_w \text{Short}^{(n)}_w > 0\big], \qquad
\text{earliest date} = \text{Quantile}_{0.1}\big(\{\min\{w : \text{Short}^{(n)}_w > 0\}\}\big)
$$

**Keep all $N$ paths.** The rollup in UC6 needs them.

### In real life
$N=1000$ over ~400 critical parts × 7 plants × 13 weeks is a few seconds of vectorised NumPy.
Not a performance concern.

The band is the product. *"Shortage between 18-Jun and 04-Jul, most likely 3,500 units,
could be as high as 6,200"* is honest and actionable. A single date with no window invites
exactly one response the first time it's wrong — the planner stops looking at the tool.

The **copula correlation is the differentiator.** A per-supplier risk table adds risks as if
independent and understates true exposure. This is the one thing in the whole system a
spreadsheet structurally cannot represent, and it is worth demonstrating explicitly in the
POC.

> **Quantified — and it is small.** This section used to say "understates true exposure
> **badly**". Measured on `csv_full_seed1`, independence understates the **supply-driven**
> P99 exposure by **+1.4%** at the ρ a modelling team could estimate from the emitted history
> (0.190), and by **+4.0%** at the ρ the config targets (0.350). Against *total* exposure
> those are +0.26% and +0.72%, because 86.2% of exposure at a 6-week horizon is a
> deterministic plan-vs-supply gap no supply outcome can move. Real, tail-concentrated, and
> small. Quote the supply-driven denominator with its null band, never the total alone —
> `docs/benchmark_specification.md` §11.6.2 and §11.6.4.

---

## 9. LP / MILP (schedule and allocation)

### Theory
Both problems have a decision, an objective and hard constraints, with an answer that is
*provably* optimal. Nothing needs to be learned. A solver finds the optimum and — unlike any
neural recommendation — can state exactly which constraint bound the solution.

> **Not reinforcement learning.** Someone will suggest it. There is no simulator good enough
> to train in, no reward signal at scale, and the problem is exactly solvable. A solver is
> better *and* explainable.

### Mathematics — allocation MILP

Decision: $x_{spm}$ = quantity of part $p$ from supplier $s$ in month $m$.

$$
\min_x \ \underbrace{c_{\text{short}}\,\mathbb{E}[\text{Short}(x)]}_{\text{piecewise-linear}} + \sum_{spm}\kappa_{sp}x_{spm} + \gamma\sum_{spm}\big|x_{spm} - x_{sp,m-1}\big|
$$

| Constraint | Formal |
|---|---|
| Demand met | $\sum_s x_{spm} = R(p,m)$ |
| Capacity | $x_{spm} \le \hat K_{spm}$ |
| Qualification | $x_{spm} = 0$ if channel not approved |
| Ramp | $x_{spm} \le x_{sp,m-1}(1+r_s)$ |
| MOQ / lot | $x_{spm} \ge \text{MOQ}\cdot y_{spm}$, $\ x \equiv 0 \bmod L$, $\ y \in \{0,1\}$ |
| Contract | $\sum_m x_{spm} \ge V^{\min}_{sp}$ |
| Tooling | $x_{spm} = 0$ if tool held elsewhere and not transferable |

**Handling $\mathbb{E}[\text{Short}(x)]$.** You cannot embed a Monte Carlo inside a MILP. Fit
a piecewise-linear approximation from simulation runs at a grid of allocations, solve, then
**re-simulate the chosen solution properly** to verify. The approximation guides the search;
the simulation validates the answer.

**Why integer.** Minimum order quantities and lot sizes make the feasible set discrete. You
cannot order 0.4 of a lot. That single fact moves the problem from LP to MILP and roughly
doubles the solve time — still milliseconds at this scale.

**Phase 1 is simpler and ships first.** Enumerate 5 candidate splits, re-run the simulation
under each, rank by expected shortage cost plus purchase cost. It matches the deck's
comparison table exactly and every number is explainable line by line. Reach for the MILP
only when the scenario space outgrows enumeration.

### In real life
**The constraints are the product, not the objective function.** Anyone can write a solver
that says "move 40% to Supplier B." What makes it usable in a plant is encoding why you often
can't: PPAP qualification takes months, the dies physically sit in Supplier A's plant, there
are minimum-volume penalties, and B cannot ramp from 30% to 45% next week.

And its job is to **price the trade-off, not eliminate risk.** The deck's own example: the
recommended split costs ₹40 lakh more and protects 3,300 units. Whether that's worth it is a
business decision. The model's job is to state it clearly and let a human make it.

Solver: OR-Tools CP-SAT or HiGHS. Both free, both far faster than needed here.

---

## 10. LightGBM

### Theory
Gradient-boosted trees are the correct default for tabular data with mixed types, non-linear
interactions and moderate sample sizes. They handle missing values natively, need almost no
preprocessing, and train in seconds.

They appear twice in this project: as the **model** for quality risk, and — more importantly
— as the **baseline every neural model must beat**.

### Mathematics
Fit an additive ensemble $F_M(x) = \sum_{m=1}^{M}\nu\, f_m(x)$ where each tree fits the
gradient of the loss at the current prediction:

$$
f_m = \arg\min_f \sum_i \left[ g_i f(x_i) + \tfrac{1}{2}h_i f(x_i)^2 \right] + \Omega(f),
\qquad g_i = \frac{\partial \mathcal{L}}{\partial F}, \ \ h_i = \frac{\partial^2 \mathcal{L}}{\partial F^2}
$$

Leaf values have a closed form, $w_j^* = -\dfrac{\sum_{i \in j} g_i}{\sum_{i \in j} h_i + \lambda}$,
which is why it is fast.

It accepts the same pinball loss as the quantile heads, so the baseline is directly
comparable to the neural model rather than measuring a different thing.

### In real life
**Take the baseline seriously.** HADES v3's h⁰ diagnostic found the graph contributed only
~10% of the above-chance signal on delay. On a real dataset with better features, LightGBM
on flat channel statistics may well match or beat temporal-SHARE on some tasks.

If it does, ship it. Keep the graph for the two places it is structurally required —
multi-tier propagation and cold-start channels with too little history for their own
statistics. Losing that comparison is a cheap, fast, good outcome, not a failure.

**One caution on how the baseline is built.** Flattening the network into scalars —
`dependency_count`, `supplier_concentration`, a precomputed `network_risk_score` — *is* the
LightGBM baseline. That is a legitimate and useful comparison, but be clear about what it
measures: once network structure is collapsed into a handful of columns, the graph has
nothing left to contribute, and a tie proves the flattening was adequate rather than that the
graph is useless. Run both, and keep them labelled as what they are.

---

## How the models compose

```
production_plan ──┬─► [1] BOM explosion ──► gross requirements
                  └─► [2] drift model ─────► P50/P90 demand
                                                    │
PO/GRN history ───► [4] Temporal-SHARE ──┬─► [5] quantile head ──► capacity
                          (one encoder)   ├─► [6] binned CDF ────► fill rate
                                          └─► [7] hazard head ───► arrival week
                                                    │
                          all of the above ─────────┴─► [8] Monte Carlo simulation
                                                              │
                                            ┌─────────────────┼─────────────────┐
                                            ▼                 ▼                 ▼
                                    shortage per       [1ᵀ] exposure      [9] allocation
                                    part/plant/week        rollup            & schedule
```

Three learned models, one shared encoder, one simulation. Everything else is linear algebra
or a solver.

---

## Validation

Every instrument in HADES v3 assumed five synthetic dataset seeds. **Rane is one world** —
that axis does not shrink, it disappears. Replacements:

| Concern | Approach |
|---|---|
| Generalisation | **Rolling-origin backtesting.** 8–12 origins for demand, quarterly for capacity and fill |
| Is a result real? | Reproduction floors recomputed across model-init seeds × time folds |
| Reporting | **Per-fold spread every time, never just the mean.** v3's hardest lesson: one fixed threshold produced recall from 32 to 94 out of 100 depending only on which world it ran in — the mean of 64.6 described nothing |
| Calibration | Fit on earlier folds, evaluate only on later ones. Never a random split — it leaks the future |
| Capacity metric | **Not accuracy.** Event recall with lead time: of historically significant capacity events, how many were flagged ≥N days early. Define "significant" jointly with Rane *before* building |
| Fill rate metric | CRPS and calibration of the predicted distribution — does the P10–P90 band contain 80% of outcomes? |
| Timing metric | Concordance index, plus weekly calibration of $P(T=w)$ |
| Shortage metric | Precision/recall **on critical parts only**, plus mean absolute error in units and in days on the earliest-risk date |
| Regime breaks | Exclude `regime_flag = 'covid'` rows from the primary training window; report performance on them separately as a robustness check. **Never average a structural break into normal operation** |
| Reproducibility | Every run stamped with `dataset_version`, `feature_spec_version`, `label_version`, `model_version` and `code_commit`. This is what supports *"if this had been deployed in Jan-2023, here is what it would have predicted"* — a claim that is unverifiable without them |

### Backtesting comes free from the schema

`model_outputs.csv` and `training_labels.csv` share (`snapshot_id`, `entity_id`, `task`).
Joining them puts every historical prediction beside what actually happened, at every past
snapshot, with no extra machinery:

```sql
SELECT o.snapshot_date, o.entity_id, o.point_estimate, o.p10, o.p90,
       l.label_value, l.label_censored
FROM   model_outputs o
JOIN   training_labels l USING (snapshot_id, entity_id, task)
WHERE  o.model_version = ? AND l.label_censored = FALSE
```

Keeping predictions out of the feature store is what makes this join safe. If outputs sat in
Group I, this query would be scoring the model against a table it had trained on.

**One thing to be explicit about internally:** uncertainty on real data will be *wider* than
v3's numbers suggest, not narrower. In v3 world-to-world variance was measurable because we
could generate more worlds. Here the same variance still exists — it is regime risk across
years, product mixes and macro conditions — but it becomes unmeasurable. Anyone reading a v3
confidence interval as a forecast of real-data performance is reading it wrong.
