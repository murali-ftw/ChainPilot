# Phase 21 Stage 4 — at placement: is the true order month worth anything when a PO is raised?

**Audience:** whoever decides what the system says the moment a planner raises a PO.
**Measured on:** v8 seed 1001. A **different decision point** from Stages 3 and 5: its rows, labels, base rates and
BASE are its own, and its numbers are **not** comparable with the snapshot arms.
**Instruments:** `phase21_proxy.py --task place` (fits `6248fae`; `_nl` arms `cbb1167`); `phase21_score.py place`
(`676ae0c`) → `score_place_v8.json` (as pre-registered) and `score_place_nl_v8.json` (no-leak). k = 10 (validation).

## Rows, labels, departures (pre-registration D10, stated before scoring)

- **Unit:** a PO line, scored at τ = the Monday 00:00 on or before its creation. Rows are the 218,881 distinct lines of
  v8's arrival label set, split by creation date: train 155,593 / val 31,494 / **test 31,794** (2025). 96.9% of test
  lines are receipted in the world; 2.1–3.1% (by fold) are censored.
- **Label:** lead in weeks. Late = lead > contracted lead (= receipt after `original_promise_date`).
  - **Test late rate: 32.9%** of receipted lines.
  - UC1-P base rate: **0.349** (censored lines count as late once their observed time passes the contract).
- **Departure from the Phase 15 / lateness definitions:** the spec's reference is defined from t0 with a fitted offset,
  which has no meaning at placement. The reference here is the **contract itself** (expected − promise), as D10 states.
- **Blocks** for intervals: 53 creation weeks. The per-"snapshot" precision spread is read per creation **month** (12).

## The pre-registered BASE leaks (deviation 199): its numbers are void

As pre-registered, BASE = the flat as-of channel features at τ + log1p(qty). It scored lateness AUC **0.987** and A3
**0.27 days**, with 92% of gain on `lead_time_actual_days`. The generator writes each line's **eventual** lead into its
channel's weekly row for the week the line was **ordered** (`generator_v8.py`, weekly stores:
`leadw[pch, vw_ord] = pl_`, forward-filled). So the panel row at a line's creation week contains that line's own
future lead.

Every pre-registered at-placement arm therefore reads the answer. Their results (BASE / L4 / L5 / L5_perm all
0.986–0.987, ALERT YES) are kept in `score_place_v8.json` and **are not used**. Every result below removes the three
columns built from `leadw` (`lead_time_actual_days`, `lead_time_ratio`, `otd_rate_last13`). The arms are named `_nl`,
so their identity is distinct.

## Arms (no-leak; 5-seed bands)

| arm | lateness AUC (vs contract) | A3 (days) | vs BASE |
|---|---|---|---|
| **BASE** (flat at creation) | **0.6948** [0.6944, 0.6956] | **12.65** [12.61, 12.69] | — |
| BASE + L4 | 0.6940 [0.6926, 0.6954] | **11.83** [11.82, 11.87] | lateness undetermined; **A3 better (−0.82 d)** |
| BASE + L5 (true order month) | 0.6931 [0.6920, 0.6941] | **11.78** [11.76, 11.82] | lateness **worse** (disjoint, −0.0017); A3 better |
| BASE + L5, permuted month | 0.6926 [0.6904, 0.6940] | 11.80 [11.68, 11.95] | lateness undetermined; A3 better |
| **standalone group rule** (KM median lead, k = 10) | **0.532** | **10.02** | block: lateness −0.162 [−0.174, −0.150]; **A3 +2.65 d [2.18, 3.13] better** |

Seed ensembles (block bootstrap over 53 creation weeks): L4 vs BASE lateness −0.0001 [−0.0024, +0.0022], A3 +0.84 d
[0.74, 0.92] better; **L5 vs L4 lateness −0.0009 [−0.0016, −0.0003] (worse)**, A3 +0.04 d [−0.01, 0.08] (undetermined).

## Gate verdicts

| family | verdict | why |
|---|---|---|
| **L4** | **PASS** (no control was pre-registered at placement) | A3 disjointly better (−0.82 d), lateness undetermined |
| **L5** (true month) | **FAIL** | disjointly **worse** on lateness AUC; and its permuted-month control is as good on A3 |
| **P4 / control (c): L5 vs L4** | **undetermined** (bands); ensemble block: L5 **worse** on lateness | the true month adds nothing |

Month slices (test, lateness AUC, 5-seed means; BASE / L4 / L5) show no month where L5 beats L4 by more than 0.002:
- Jan 0.644 / 0.637 / 0.639
- Feb 0.615 / 0.620 / 0.621
- May 0.646 / 0.655 / 0.653
- Sep 0.735 / 0.730 / 0.728
- Nov 0.688 / 0.683 / 0.680

The strong seasonality of the late rate (Stage 1: 0.21 in Feb → 0.50 in Sep) is already visible to BASE through the
as-of channel state at τ. The per-channel month cell (median 4–5 receipts) adds noise, not signal.

In the L5 fit the group block carries 24% of gain; the acknowledgement-ratio panel column (`ack_gap_ratio`, a BASE
feature) 14%; L5 itself 1.8% and L3 1.4%. LightGBM barely uses the month cells.

## Decision level (UC1-P: late vs contract at placement, censoring resolved; base rate 0.349)

| arm | class | REACHABLE (bar, lift) | P @ 1% / 5% / 10% / 20% | per-month P @ 5% min / median / max | accuracy (majority 0.651) |
|---|---|---|---|---|---|
| BASE, per seed | **ALERT** | PARTIAL (0.80, 2.22) | 0.853 / 0.791 / 0.750 / 0.700 | 0.659 / 0.787 / 0.881 | 0.697 |
| BASE + L4, per seed | WATCHLIST | NO, CEILING (the reachable condition fails on at least one seed) | 0.843 / 0.784 / 0.738 / 0.698 | 0.646 / 0.773 / 0.898 | 0.704 |
| BASE + L5, per seed | ALERT | PARTIAL (0.80, 2.25) | 0.847 / 0.788 / 0.738 / 0.698 | 0.650 / 0.774 / 0.889 | 0.701 |
| BASE + L5, 5-seed ensemble | ALERT | PARTIAL (0.80, 2.25) | 0.855 / 0.792 / 0.740 / 0.698 | 0.670 / 0.783 / 0.898 | 0.706 |
| standalone group rule | WATCHLIST | NO, CEILING; Stage B WEAKLY TUNABLE | 0.381 / 0.413 / 0.399 / 0.390 | 0.246 / 0.425 / 0.579 | 0.353 |

## What this would mean operationally

"When a planner raises a PO, the system states the expected week and a late-risk flag" is **supportable from the flat
as-of features alone**. The late list is an ALERT at this decision point: precision 0.79 at 5% coverage, 2.2× the
0.35 base rate. But one month ran at 0.66, so the 0.80 bar holds only on average.

The group statistics do not improve the flag. They **do** improve the expected week: −0.8 days of median error from
L4. The plain KM median of the channel's (shrunk) lead is better still on median error (10.0 days vs 12.65). It is a
poor ranker of lateness (0.53), though, so the **date** and the **flag** should come from different predictors. The
true order month is **not** worth adding at placement.
