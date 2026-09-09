"""Acceptance checks. A dataset is not usable until these pass.

Implements BUILD_PROMPT.md s7 in five groups:

    referential   every FK resolves; no orphans
    temporal      recorded_ts >= event_ts everywhere; no overlapping validity
                  windows; receipts never exceed what was ordered
    leakage       no hidden state in any CSV; no label window opening on or
                  before its snapshot; reporting lag genuinely displaces
                  receipts, and the derived stores conserve every unit of it
    statistical   the s4.5 base rates and the s7 distribution shapes
    volumes       the s1 row counts over the window they were stated for

Run standalone against an emitted directory:

    python validate.py csv_small_seed1
    python validate.py csv_full_seed1

The scale preset is read off the directory name (csv_<preset>_seed<N>), not
passed in. It selects the volume targets, the base-rate bands and the window
every statistical check runs over, so validating with the wrong one does not
skip checks -- it grades one world against another world's expectations and
reports confident, wrong failures. A directory that does not parse is an error
asking for an explicit --preset; an explicit --preset that contradicts the
directory is also an error.

Ground-truth checks (the realised correlation matrix, true constrained months)
need the generator's latent state and are skipped unless a World is passed in,
which generate_dataset.py does. Everything else is computed from the CSVs
alone -- the same position a reviewer with only the extract is in.

Why the base rates are gated on normal-regime rows
--------------------------------------------------
A regime is a change of parameters, not extra noise. COVID legitimately
doubles the shortage rate and triples lateness, so a span containing it cannot
sit inside a band describing normal operation. data_plan.md excludes covid rows
from the primary training window for the same reason. Both figures are
reported; the gate is on the normal-regime one.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import gzip
import math
import os
import re
import statistics as st
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (BASE_RATES, HIDDEN_STATE_ALLOWLIST, HIDDEN_STATE_COLUMNS,
                    HIDDEN_STATE_PATTERNS, SHAPE_TARGETS,
                    SYSTEM_GENERATED_TABLES, Config)

DATE = _dt.date

# Tables whose "when it happened" column is not called event_ts. Without this
# map the temporal check silently skips them, which is worse than not having
# the check at all.
# Minimum supplier pairs before the within-group correlation is gated rather
# than merely reported. See _check_correlation.
MIN_CORRELATION_PAIRS = 100

EVENT_COLUMN = {
    "purchase_orders": "created_ts", "po_lines": "created_ts",
    "po_line_schedules": "released_ts", "asn": "dispatch_ts",
    "goods_receipts": "receipt_ts",
    "shortage_events": "event_ts", "line_stop_events": "event_ts",
}


@dataclass
class Check:
    group: str
    name: str
    passed: bool
    value: str
    target: str = ""
    note: str = ""


@dataclass
class Result:
    checks: list = field(default_factory=list)

    def add(self, group, name, passed, value, target="", note=""):
        self.checks.append(Check(group, name, bool(passed), str(value),
                                 str(target), note))
        return passed

    @property
    def failed(self):
        return [c for c in self.checks if not c.passed]


# ===========================================================================
# Reading
# ===========================================================================

def rows(d: str, table: str):
    path = os.path.join(d, f"{table}.csv.gz")
    if not os.path.exists(path):
        return
    with gzip.open(path, "rt", newline="") as fh:
        yield from csv.DictReader(fh)


def header(d: str, table: str) -> list[str] | None:
    path = os.path.join(d, f"{table}.csv.gz")
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt", newline="") as fh:
        return next(csv.reader(fh))


def dt_(v):
    return _dt.datetime.fromisoformat(v) if v else None


def dd(v):
    return _dt.date.fromisoformat(v) if v else None


def ii(v, default=0):
    return int(v) if v not in (None, "") else default


def present(dirs, table):
    for d in dirs:
        if d and os.path.exists(os.path.join(d, f"{table}.csv.gz")):
            return d
    return None


# ===========================================================================
# schema.sql is the source of truth for structure
# ===========================================================================

def parse_schema(path: str):
    """Column order and foreign keys, read from the DDL.

    Hand-listing the foreign keys in the validator means the list drifts from
    the schema the moment either changes. Reading them out of schema.sql means
    a new reference is checked the day it is added.
    """
    sql = open(path).read()
    cols, fks = {}, []
    for m in re.finditer(r"CREATE TABLE (\w+)\s*\((.*?)\n\);", sql, re.S):
        table, body = m.group(1), m.group(2)
        names, depth = [], 0
        for raw in body.split("\n"):
            line = raw.split("--")[0].strip()
            if not line:
                continue
            if depth == 0:
                cm = re.match(r'"?([a-z_][a-z_0-9]*)"?\s+[A-Z]', line)
                if cm and not line.upper().startswith(
                        ("CONSTRAINT", "PRIMARY", "FOREIGN", "UNIQUE", "CHECK")):
                    names.append(cm.group(1))
                    ref = re.search(r"REFERENCES (\w+)\((\w+)\)", line)
                    if ref:
                        fks.append((table, cm.group(1), ref.group(1), ref.group(2)))
            depth += line.count("(") - line.count(")")
        cols[table] = names
    return cols, fks


# ===========================================================================
# Referential integrity
# ===========================================================================

def check_referential(res, dirs, schema_path) -> None:
    cols, fks = parse_schema(schema_path)

    # Non-negotiable 7: schema.sql matches the CSVs exactly -- same names, same
    # order. Checked against the written files, not merely trusted at write.
    bad_headers = []
    for table, declared in cols.items():
        for d in dirs:
            h = header(d, table)
            if h is not None:
                if h != declared:
                    bad_headers.append(table)
                break
    res.add("referential", "CSV headers match schema.sql", not bad_headers,
            f"{len(bad_headers)} mismatched", "0", ", ".join(bad_headers[:4]))

    parents = {}
    orphan_total = 0
    detail = []
    for table, col, ptable, pcol in fks:
        cd = present(dirs, table)
        pd_ = present(dirs, ptable)
        if not cd or not pd_:
            continue
        key = (ptable, pcol)
        if key not in parents:
            parents[key] = {r[pcol] for r in rows(pd_, ptable) if r.get(pcol)}
        keys = parents[key]
        n = 0
        for r in rows(cd, table):
            v = r.get(col)
            if v and v not in keys:
                n += 1
        if n:
            orphan_total += n
            detail.append(f"{table}.{col}->{ptable} ({n:,})")
    res.add("referential", f"all {len(fks)} foreign keys resolve",
            orphan_total == 0, f"{orphan_total:,} orphans", "0",
            "; ".join(detail[:3]))

    # The two the spec calls out by name.
    d = present(dirs, "po_lines")
    if d:
        chans = {r["channel_id"] for r in rows(present(dirs, "sourcing_channels"),
                                               "sourcing_channels")}
        n = sum(1 for r in rows(d, "po_lines") if r["channel_id"] not in chans)
        res.add("referential", "every po_line.channel_id is a real channel",
                n == 0, f"{n:,}", "0")
        lines = {r["po_line_id"] for r in rows(d, "po_lines")}
        gd = present(dirs, "grn_lines")
        if gd:
            n = sum(1 for r in rows(gd, "grn_lines")
                    if r["po_line_id"] not in lines)
            res.add("referential", "every grn_line.po_line_id is a real line",
                    n == 0, f"{n:,}", "0")


# ===========================================================================
# Temporal integrity
# ===========================================================================

def check_temporal(res, dirs, schema_path, lag_enabled=True) -> None:
    cols, _ = parse_schema(schema_path)
    violations = 0
    equal = 0
    checked_tables = 0
    detail = []
    for table, declared in cols.items():
        if "recorded_ts" not in declared:
            continue
        ev = EVENT_COLUMN.get(table, "event_ts")
        if ev not in declared:
            continue
        d = present(dirs, table)
        if not d:
            continue
        checked_tables += 1
        n = eq = 0
        for r in rows(d, table):
            a, b = r.get(ev), r.get("recorded_ts")
            if not a or not b:
                continue
            if b < a:
                n += 1
            elif b == a and table not in SYSTEM_GENERATED_TABLES:
                eq += 1
        violations += n
        equal += eq
        if n:
            detail.append(f"{table} ({n:,})")
    res.add("temporal", f"recorded_ts >= event_ts on all {checked_tables} event tables",
            violations == 0, f"{violations:,}", "0", "; ".join(detail[:3]))
    # Non-negotiable 4. --disable reporting_lag collapses the two timestamps on
    # purpose, so the check only applies when the mechanism is on.
    if lag_enabled:
        res.add("temporal", "recorded_ts never equals event_ts",
                equal == 0, f"{equal:,}", "0",
                f"exempt: {', '.join(sorted(SYSTEM_GENERATED_TABLES))}")

    # effective_from < effective_to wherever both are present
    bad = 0
    for table, declared in cols.items():
        if "effective_from" not in declared or "effective_to" not in declared:
            continue
        d = present(dirs, table)
        if not d:
            continue
        for r in rows(d, table):
            a, b = r.get("effective_from"), r.get("effective_to")
            if a and b and a >= b:
                bad += 1
    res.add("temporal", "effective_from < effective_to", bad == 0, f"{bad:,}", "0")

    # BOM validity windows must not overlap for the same (product, part).
    d = present(dirs, "bom")
    if d:
        spans = collections.defaultdict(list)
        for r in rows(d, "bom"):
            spans[(r["product_id"], r["part_id"])].append(
                (dd(r["effective_from"]), dd(r["effective_to"]) or DATE.max))
        overlaps = 0
        for v in spans.values():
            v.sort()
            for a, b in zip(v, v[1:]):
                if b[0] < a[1]:
                    overlaps += 1
        res.add("temporal", "no overlapping BOM windows per (product, part)",
                overlaps == 0, f"{overlaps:,}", "0")

    # Receipts must not exceed what was ordered, beyond a small over-delivery.
    d = present(dirs, "po_lines")
    if d:
        ordered = {r["po_line_id"]: ii(r["qty_ordered"]) for r in rows(d, "po_lines")}
        got = collections.Counter()
        gd = present(dirs, "grn_lines")
        if gd:
            for r in rows(gd, "grn_lines"):
                got[r["po_line_id"]] += ii(r["qty_received"])
        over = sum(1 for k, v in got.items() if v > ordered.get(k, 0) * 1.02)
        res.add("temporal", "sum(grn qty) <= qty_ordered x 1.02",
                over == 0, f"{over:,}", "0")


# ===========================================================================
# Leakage
# ===========================================================================

def check_leakage(res, dirs, schema_path, csv_dir, derived_dir) -> None:
    cols, _ = parse_schema(schema_path)
    hits = []
    scanned = 0
    for table in cols:
        for d in dirs:
            h = header(d, table)
            if h is None:
                continue
            scanned += 1
            for c in h:
                low = c.lower()
                if low in HIDDEN_STATE_ALLOWLIST:
                    continue
                if low in HIDDEN_STATE_COLUMNS or any(
                        p in low for p in HIDDEN_STATE_PATTERNS):
                    hits.append(f"{table}.{c}")
            break
    res.add("leakage", f"no hidden state in any of {scanned} CSVs",
            not hits, f"{len(hits)} columns", "0", ", ".join(hits[:4]))

    d = present(dirs, "training_labels")
    if d:
        bad = sum(1 for r in rows(d, "training_labels")
                  if r["label_window_start"] <= r["snapshot_date"])
        res.add("leakage", "every label window opens after its snapshot",
                bad == 0, f"{bad:,}", "0")
        # No binary -- or effectively constant -- labels anywhere
        # (dataset_structure.md s11).
        #
        # This used to test `len(distinct) <= 2`, which cannot fail on any
        # continuous target and would NOT have caught the defect it was meant
        # to: the original fill_rate label had THREE distinct values, all
        # within 0.0007 of 1.0. A model trained on it scores perfectly and has
        # learned nothing. Now uses the same relative-dispersion test as
        # check_labels, via a shared helper.
        vals = collections.defaultdict(list)
        cens = collections.Counter()
        tot = collections.Counter()
        for r in rows(d, "training_labels"):
            tot[r["task"]] += 1
            if r["label_censored"] == "true":
                cens[r["task"]] += 1
            elif r["label_value"]:
                vals[r["task"]].append(float(r["label_value"]))
        degenerate = []
        for t in sorted(tot):
            ok, stx = label_variation(vals.get(t, []))
            if not ok:
                degenerate.append(f"{t} (sd/|mean|={stx['spread']:.4f}, "
                                  f"{stx['ndistinct']} distinct)")
        res.add("leakage", "no binary or effectively-constant labels",
                not degenerate, f"{len(degenerate)} tasks", "0",
                "; ".join(degenerate) or
                f"all {len(tot)} tasks vary: sd/|mean| > {MIN_RELATIVE_SPREAD} "
                f"across >= {MIN_DISTINCT_VALUES} distinct values")
        censored_tasks = [t for t in tot if cens[t] > 0]
        res.add("leakage", "open outcomes are censored, not dropped",
                len(censored_tasks) >= 2,
                f"{len(censored_tasks)} of {len(tot)} tasks carry censored rows",
                ">= 2",
                ", ".join(f"{t} {cens[t]/tot[t]:.0%}" for t in sorted(censored_tasks)))

    d = present(dirs, "model_outputs")
    if d:
        n = sum(1 for _ in rows(d, "model_outputs"))
        res.add("leakage", "model_outputs is empty", n == 0, f"{n:,}", "0",
                "predictions never live in a feature table")

    # The as-of reconstruction must give a DIFFERENT answer from the naive one.
    # If it does not, either the reporting lag is absent or the derived layer is
    # quietly reading event_ts, and every leakage guarantee is decorative.
    if csv_dir:
        lines = {r["po_line_id"]: r["channel_id"] for r in rows(csv_dir, "po_lines")}
        rec_view = collections.Counter()
        ev_view = collections.Counter()
        for r in rows(csv_dir, "grn_lines"):
            ch = lines.get(r["po_line_id"])
            if not ch:
                continue
            q = ii(r["qty_received"])
            ev = dt_(r["event_ts"]).date()
            rc = dt_(r["recorded_ts"]).date()
            wk = ev - _dt.timedelta(days=ev.weekday())
            if rc <= wk + _dt.timedelta(days=6):
                rec_view[(ch, wk)] += q
            ev_view[(ch, wk)] += q
        keys = set(rec_view) | set(ev_view)
        differing = sum(1 for k in keys if rec_view[k] != ev_view[k])
        share = differing / len(keys) if keys else 0.0
        displaced = 1 - (sum(rec_view.values()) / max(1, sum(ev_view.values())))
        # DISPLACEMENT, not loss. This is the share of receipt quantity posted
        # after its event week closed, so it belongs to a later week than the
        # one it happened in.
        #
        # This used to be gated on a 2%-35% band under the name "as-of view
        # hides a material share of receipts", and that was the wrong shape of
        # test in a way that mattered. The number is computed from the raw
        # tables and says nothing about what the derived store did with those
        # units -- it reads identically whether they were deferred to a later
        # week or dropped on the floor. For five phases they were dropped, and
        # this gate passed comfortably throughout. Magnitude is not the
        # question; whether the units survived is, and that is `conservation`
        # below.
        #
        # So: report the magnitude, gate only that the mechanism is on. A
        # dataset built with `--disable reporting_lag` has zero displacement
        # and must still be visible here.
        res.add("leakage", "reporting lag displaces receipts",
                displacement_ok(displaced), f"{displaced:.1%} of receipt qty",
                "> 0",
                f"posted after their event week closed, so they belong to a "
                f"later week than they happened in; {share:.1%} of "
                f"channel-weeks are affected. MAGNITUDE IS REPORTED, NOT "
                f"GATED -- it is the same whether those units were deferred "
                f"or lost. See the conservation check for whether they "
                f"survived")

    # CONSERVATION. Every unit in the source appears in exactly one week of the
    # derived store. This is the check whose absence let the builder discard
    # 4.5% of active channel-weeks undetected, and it is deliberately a
    # reimplementation rather than a call into build_features: a checker that
    # shares its subject's arithmetic can only ever agree with it.
    if csv_dir and derived_dir and present([derived_dir], "channel_performance_weekly"):
        def vis_week(ev, rec):
            """Week in which a fact was both true and knowable."""
            a = ev - _dt.timedelta(days=ev.weekday())
            b = rec - _dt.timedelta(days=rec.weekday())
            return max(a, b)

        store_o = store_r = 0
        weeks = []
        for r in rows(derived_dir, "channel_performance_weekly"):
            store_o += ii(r["qty_ordered"])
            store_r += ii(r["qty_received"])
            weeks.append(r["week_start"])
        if weeks:
            lo_w = _dt.date.fromisoformat(min(weeks))
            hi_w = _dt.date.fromisoformat(max(weeks))
            known = {r["channel_id"] for r in rows(csv_dir, "sourcing_channels")}

            def counts(vw, ch):
                return ch in known and lo_w <= vw <= hi_w

            src_o = 0
            line_ch = {}
            for r in rows(csv_dir, "po_lines"):
                line_ch[r["po_line_id"]] = r["channel_id"]
                vw = vis_week(dt_(r["created_ts"]).date(),
                              dt_(r["recorded_ts"]).date())
                if counts(vw, r["channel_id"]):
                    src_o += ii(r["qty_ordered"])
            src_r = 0
            for r in rows(csv_dir, "grn_lines"):
                ch = line_ch.get(r["po_line_id"])
                if ch is None:
                    continue
                vw = vis_week(dt_(r["event_ts"]).date(),
                              dt_(r["recorded_ts"]).date())
                if counts(vw, ch):
                    src_r += ii(r["qty_received"])

            for label, src_q, got_q in (("ordered", src_o, store_o),
                                        ("received", src_r, store_r)):
                gap = (got_q - src_q) / src_q if src_q else 0.0
                res.add("leakage",
                        f"channel store conserves {label} units",
                        conserved_ok(src_q, got_q), f"{gap:+.4%}", "exactly 0",
                        f"channel_performance_weekly: source {src_q:,} vs "
                        f"store {got_q:,}. Every unit whose "
                        f"visible week falls in the store's span must appear in "
                        f"exactly one week of it. A shortfall means rows are "
                        f"being dropped -- which reads as an idle week, "
                        f"indistinguishable from a channel that genuinely "
                        f"ordered nothing")


# ===========================================================================
# Label sanity -- the standing gate
# ===========================================================================

# A task whose uncensored label_value has zero spread is a CONSTANT TARGET. It
# trains to near-zero loss on the first epoch and every metric says the model
# works. This is not hypothetical: training_labels.fill_rate shipped that way
# (docs/benchmark_specification.md s11.7) -- 837,318 uncensored rows all at
# exactly 1.0 -- because the censoring rule marked every settled short delivery
# as censored and left only the perfect ones behind.
# The threshold has to be RELATIVE. The defect this gate was written for had a
# standard deviation of about 1e-5 -- not zero, so any "sd > 0" test passes it
# while the target is constant for every practical purpose (three distinct
# values, all within 0.0007 of 1.0). A target is treated as effectively
# constant if its spread is under 1% of its own magnitude, or if it takes fewer
# than 5 distinct values. arrival_week is the reason the second threshold is 5
# and not higher: it is integer weeks 1-13 and legitimately has only 13.
MIN_RELATIVE_SPREAD = 0.01
MIN_DISTINCT_VALUES = 5


def label_variation(values):
    """(ok, stats) for a task's uncensored label values.

    ONE implementation, two callers -- check_leakage's no-degenerate-labels
    gate and check_labels' per-task report. Two gates computing the same thing
    from the same data is how the load-bearing one gets deleted as redundant.

    `ok` is False when the target is effectively constant. The threshold is
    RELATIVE because the defect this exists to catch had a standard deviation
    of ~1e-5 -- not zero, so `sd > 0` passed it -- across three distinct values
    all within 0.0007 of 1.0.
    """
    v = [x for x in values if x is not None]
    if not v:
        return False, dict(n=0, mean=0.0, sd=0.0, spread=0.0, ndistinct=0)
    n = len(v)
    mean = sum(v) / n
    sd = (sum((x - mean) ** 2 for x in v) / n) ** 0.5 if n > 1 else 0.0
    ndistinct = len({round(x, 6) for x in v})
    spread = sd / max(abs(mean), 1e-12)
    ok = spread > MIN_RELATIVE_SPREAD and ndistinct >= MIN_DISTINCT_VALUES
    return ok, dict(n=n, mean=mean, sd=sd, spread=spread, ndistinct=ndistinct)
# Censoring outside this band means the rule is broken rather than strict.
# Near 0: settled outcomes are being called observed when they are not, or the
# task is silently dropping its open rows. Near 1: almost nothing is observed
# and there is little left to learn from.
CENSORED_BAND = (0.01, 0.60)
# Right-censored tail size, as a multiple of what the span implies (90 days of
# a span-length window). Realised 0.575-0.582 across all three presets.
CENSORED_TAIL_RATIO = (0.30, 1.20)
# Displacement and conservation are separate questions and used to be one gate.
#
# `displaced` is the share of receipt quantity posted after its event week
# closed. It is a property of the reporting-lag mechanism, it is expected to be
# non-zero, and its MAGNITUDE says nothing about whether the derived store kept
# those units -- it reads the same whether they were deferred or discarded. So
# only its presence is asserted.
#
# `conserved` is the share of source quantity that reached the store. It is the
# one with a hard answer: exactly all of it.


def displacement_ok(share: float) -> bool:
    """Reporting lag is switched on and visibly moving receipts."""
    return share > 0


def conserved_ok(src_q: int, store_q: int) -> bool:
    """No unit was lost or duplicated between source and store."""
    return src_q == store_q


def check_labels(res, dirs) -> None:
    """Per task: does the target actually vary, and is censoring plausible?

    Applied to EVERY task, not just the one that broke. The failure this gate
    exists to catch is silent by construction -- a constant target looks like
    success -- so it has to run on tasks nobody currently suspects.
    """
    d = present(dirs, "training_labels")
    if not d:
        return
    n = collections.Counter()
    cens = collections.Counter()
    vals = collections.defaultdict(list)
    distinct = collections.defaultdict(set)
    for r in rows(d, "training_labels"):
        t = r["task"]
        n[t] += 1
        if r["label_censored"] == "true":
            cens[t] += 1
        elif r["label_value"] not in ("", None):
            v = float(r["label_value"])
            vals[t].append(v)
            distinct[t].add(round(v, 6))
    if not n:
        return
    for t in sorted(n):
        v = vals[t]
        varies, stx = label_variation(v)
        mean, sd, ndist, spread = (stx["mean"], stx["sd"], stx["ndistinct"],
                                   stx["spread"])
        frac = cens[t] / n[t]
        lo, hi = CENSORED_BAND
        stats = (f"n={n[t]:,} censored={frac:.1%} mean={mean:.4f} sd={sd:.4f} "
                 f"min={min(v):.4f} max={max(v):.4f} distinct={ndist:,} "
                 f"sd/|mean|={spread:.4f}"
                 if v else f"n={n[t]:,} censored={frac:.1%} -- NO uncensored rows")
        res.add("labels", f"{t}: uncensored target varies", varies,
                f"sd/|mean|={spread:.3f}", f"> {MIN_RELATIVE_SPREAD}",
                stats + ("" if varies else
                         ". EFFECTIVELY CONSTANT: this trains to near-zero loss "
                         "immediately and every metric reports success"))
        res.add("labels", f"{t}: censored fraction is plausible",
                lo < frac < hi, f"{frac:.1%}", f"{lo:.0%}-{hi:.0%}",
                "" if lo < frac < hi else
                ("censoring near zero -- settled and open outcomes are probably "
                 "not being distinguished" if frac <= lo else
                 "censoring this high leaves little observed signal to learn from"))


# ===========================================================================
# Statistical realism -- s4.5 base rates and s7 distribution shapes
# ===========================================================================

def _fill_population(csv_dir, cfg):
    """PO lines with an observable outcome, tagged by the regime they fell in.

    Lines promised inside the last 90 days of the span are right-censored --
    their delivery window runs past the end of the simulated world. They are
    the population the arrival-timing hazard consumes, and counting them as
    zero-fill would misstate every base rate.

    Buyer-cancelled lines are excluded for the same reason and it is the same
    mistake: the supplier was never given the chance to deliver, so scoring the
    line as a zero fill counts Rane changing its mind as a supplier failure.
    Left in, they inflate the fill-rate-at-zero spike and the share of
    supplier-months that look capacity-constrained. Identified by their
    short-close revision, which is how the emitted data records the
    cancellation (config.ProcurementParams.short_close_buyer_reason).
    """
    regime = {}
    for r in rows(csv_dir, "calendar"):
        regime.setdefault(dd(r["date"]), r["regime_flag"])
    horizon_end = cfg.preset.end - _dt.timedelta(days=90)

    lines = {}
    for r in rows(csv_dir, "po_lines"):
        lines[r["po_line_id"]] = {
            "q": ii(r["qty_ordered"]), "rec": 0, "acc": 0, "ch": r["channel_id"],
            "prom": dd(r["original_promise_date"]), "first": None,
            "cre": dt_(r["created_ts"]).date()}
    # Accepted quantity is resolved PER RECEIPT, with a fallback to the
    # received quantity where that receipt was never inspected. Summing
    # qty_accepted only over inspected receipts and comparing it against the
    # whole order makes every uninspected receipt look rejected -- and with
    # inspection coverage varying from 31% to 92% by plant, that turns a
    # coverage gap into a fake shortfall. data_plan.md UC3 specifies the
    # fallback for exactly this reason.
    accepted = {}
    for r in rows(csv_dir, "quality_inspections"):
        accepted[r["grn_line_id"]] = ii(r["qty_accepted"])
    for r in rows(csv_dir, "grn_lines"):
        L = lines.get(r["po_line_id"])
        if L:
            q = ii(r["qty_received"])
            L["rec"] += q
            L["acc"] += accepted.get(r["grn_line_id"], q)
            ev = dt_(r["event_ts"]).date()
            L["first"] = ev if L["first"] is None else min(L["first"], ev)
    cancelled = {r["po_line_id"] for r in rows(csv_dir, "po_line_revisions")
                 if r["reason_code"] == cfg.procurement.short_close_buyer_reason}
    observed, censored = {}, 0
    for k, L in lines.items():
        if k in cancelled:
            continue
        if L["prom"] and L["prom"] > horizon_end:
            censored += 1
        else:
            L["regime"] = regime.get(L["prom"], "normal")
            observed[k] = L
    return observed, censored, regime


def check_statistical(res, csv_dir, cfg, world=None) -> None:
    obs, censored, regime = _fill_population(csv_dir, cfg)
    if not obs:
        return
    chan = {}
    for r in rows(csv_dir, "sourcing_channels"):
        chan[r["channel_id"]] = (r["supplier_id"], r["part_id"],
                                 ii(r["contracted_lead_time_days"]))

    norm = {k: v for k, v in obs.items() if v["regime"] == "normal"} or obs
    fills = [min(1.0, v["acc"] / v["q"]) for v in norm.values() if v["q"] > 0]
    short = sum(1 for x in fills if x < 0.9999) / len(fills)
    zero = sum(1 for x in fills if x <= 1e-9) / len(fills)
    one = sum(1 for x in fills if x >= 0.9999) / len(fills)
    arrived = [v for v in norm.values() if v["first"]]
    late = sum(1 for v in arrived if v["first"] > v["prom"]) / max(1, len(arrived))

    def band(name, got, key):
        b = BASE_RATES[key]
        return res.add("statistical", name, b.lo <= got <= b.hi, f"{got:.1%}",
                       f"{b.lo:.0%}-{b.hi:.0%}", b.why)

    band("PO lines with fill < 1.0", short, "po_lines_fill_below_1")
    band("PO lines arriving late", late, "po_lines_late")
    res.add("statistical", "fill-rate spike at exactly 1.0",
            one >= SHAPE_TARGETS["fill_rate_spike_at_1_min"], f"{one:.1%}",
            f">= {SHAPE_TARGETS['fill_rate_spike_at_1_min']:.0%}",
            "a continuous haircut would put every line just below 1")
    lo, hi = SHAPE_TARGETS["fill_rate_spike_at_0"]
    res.add("statistical", "fill-rate spike at 0", lo <= zero <= hi,
            f"{zero:.1%}", f"{lo:.0%}-{hi:.0%}")
    # Right-censored share, as a FRACTION OF WHAT THE SPAN IMPLIES.
    #
    # This used to be `censored > 0`, which any non-degenerate dataset passes:
    # it tested existence, not adequacy. A flat rate band does not work either,
    # because the censored tail is a fixed 90-day window against a span that
    # varies by preset -- 4.7% at `small`, 1.4% at `full`, both correct.
    #
    # Normalising by 90/span makes it scale-invariant. The realised ratio is
    # 0.575 / 0.579 / 0.582 at small / mid / full -- stable to three decimals,
    # because ordering density is lower early in the span than late. The band
    # below is wide enough for that structure to shift and tight enough to
    # catch the tail collapsing or swallowing the dataset.
    span_days = max(1, (cfg.preset.end - cfg.preset.start).days)
    expected = 90.0 / span_days
    share = censored / max(1, censored + len(obs))
    ratio = share / expected if expected > 0 else 0.0
    lo_r, hi_r = CENSORED_TAIL_RATIO
    res.add("statistical", "right-censored tail is the right size",
            lo_r <= ratio <= hi_r, f"{ratio:.2f}x expected",
            f"{lo_r}-{hi_r}x",
            f"{censored:,} lines ({share:.2%}) promised inside the last 90 days "
            f"of a {span_days}-day span; outcome not yet observable. "
            f"Span alone implies {expected:.2%}")

    # --- supplier-months constrained, on months with activity ---------------
    # data_plan.md UC2: constrained = fill < 1 OR mean lead-time ratio > 1.15.
    mo = collections.defaultdict(lambda: [0, 0, []])
    for v in norm.values():
        meta = chan.get(v["ch"])
        if not meta or not v["prom"]:
            continue
        key = (meta[0], meta[1], v["prom"].replace(day=1))
        mo[key][0] += v["q"]
        mo[key][1] += v["rec"]
        if v["first"] and meta[2]:
            mo[key][2].append((v["first"] - v["cre"]).days / meta[2])
    con = sum(1 for o, g, lt in mo.values()
              if g < o or (lt and st.mean(lt) > 1.15))
    rate = con / len(mo) if mo else 0.0
    b = BASE_RATES["supplier_months_constrained"]
    res.add("statistical", "supplier-months constrained", b.lo <= rate <= b.hi,
            f"{rate:.1%}", f"{b.lo:.0%}-{b.hi:.0%}",
            "measured on months with activity; empty months are not evidence")
    res.add("statistical", "capacity unobservable",
            1 - rate >= SHAPE_TARGETS["supplier_months_unobservable_min"],
            f"{1 - rate:.1%}",
            f">= {SHAPE_TARGETS['supplier_months_unobservable_min']:.0%}",
            "delivered = min(K, ordered): unconstrained months carry no "
            "information about K at all")

    # --- lead time must be right-skewed, not Gaussian ----------------------
    lt = [(v["first"] - v["cre"]).days for v in arrived]
    if len(lt) > 30:
        m, sd = st.mean(lt), st.pstdev(lt)
        skew = (sum((x - m) ** 3 for x in lt) / len(lt) / sd ** 3) if sd > 0 else 0.0
        lt.sort()
        res.add("statistical", "lead-time distribution is right-skewed",
                skew >= SHAPE_TARGETS["lead_time_skewness_min"], f"{skew:+.2f}",
                f">= {SHAPE_TARGETS['lead_time_skewness_min']}",
                f"median {lt[len(lt)//2]}d, P90 {lt[int(.9*len(lt))]}d, "
                f"P99 {lt[int(.99*len(lt))]}d")

    # --- rare-event rates ---------------------------------------------------
    years = cfg.preset.years
    for table, key, label in (("shortage_events", "shortage_events_per_year",
                               "shortage events"),
                              ("line_stop_events", "line_stops_per_year",
                               "line stops"),
                              ("expedite_events", "expedites_per_year",
                               "expedites")):
        cnt = sum(1 for _ in rows(csv_dir, table))
        lo_, hi_ = cfg.rare_event_target(key)
        res.add("statistical", f"{label} per year", lo_ <= cnt / years <= hi_,
                f"{cnt / years:.1f}", f"{lo_:.0f}-{hi_:.0f}",
                BASE_RATES[key].why)

    # --- demand seasonality and the regime break ---------------------------
    by_month = collections.defaultdict(list)
    by_regime = collections.defaultdict(list)
    for r in rows(csv_dir, "production_actual"):
        p = dd(r["period"])
        by_month[p.month].append(ii(r["actual_qty"]))
        by_regime[regime.get(p, "normal")].append(ii(r["actual_qty"]))
    if len(by_month) >= 12:
        means = {m: st.mean(v) for m, v in by_month.items() if v}
        spread = max(means.values()) / max(1e-9, min(means.values()))
        res.add("statistical", "demand shows month-of-year seasonality",
                spread > 1.10, f"{spread:.2f}x peak/trough", "> 1.10",
                f"peak month {max(means, key=means.get)}, "
                f"trough {min(means, key=means.get)}")
    if by_regime.get("covid") and by_regime.get("normal"):
        ratio = st.mean(by_regime["covid"]) / st.mean(by_regime["normal"])
        res.add("statistical", "visible COVID break in demand", ratio < 0.95,
                f"{ratio:.2f}x normal", "< 0.95",
                "regimes shift the parameters, not just the noise")

    # --- correlated failure -------------------------------------------------
    if world is not None:
        _check_correlation(res, world)


def _check_correlation(res, world) -> None:
    """The project's key differentiator, measured on what the simulation
    produced rather than on what was requested."""
    try:
        import numpy as np
        import ground_truth
    except ImportError:
        return
    ground_truth.bind(world)
    labels, M = ground_truth.realised_correlation_matrix()
    if len(labels) < 5:
        return
    S = {s.supplier_id: s for s in world.suppliers}
    within, unrelated = [], []
    up = world.upstream_of
    # M is pairwise-complete and carries NaN where two suppliers never overlap
    # in enough bins to correlate. Those pairs are dropped, not counted as zero.
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            a, b = S[labels[i]], S[labels[j]]
            if a.supplier_group_id == b.supplier_group_id:
                within.append(v)
            elif not (set(up.get(a.supplier_id, ())) & set(up.get(b.supplier_id, ()))) \
                    and a.state != b.state and a.checkpoint != b.checkpoint:
                unrelated.append(v)
    lo, hi = SHAPE_TARGETS["within_group_shortfall_corr"]
    if within:
        v = float(np.mean(within))
        # Below this many pairs the mean is too noisy to gate on. At `small`
        # there are only ~40 supplier pairs sharing a group and the estimate
        # ranges 0.20-0.51 across seeds on identical parameters, so a pass or
        # fail there would report sampling noise, not a property of the data.
        # It is reported at every scale and gated where it can be measured.
        gate = len(within) >= MIN_CORRELATION_PAIRS
        res.add("statistical", "within-group shortfall correlation",
                (lo <= v <= hi) if gate else True, f"{v:+.3f}",
                f"{lo}-{hi}" if gate else f"{lo}-{hi} (not gated)",
                f"{len(within):,} supplier pairs sharing a group. A per-supplier "
                f"risk table adds these as if independent"
                + ("" if gate else
                   f". Fewer than {MIN_CORRELATION_PAIRS} pairs: reported, not gated"))
    if unrelated:
        v = float(np.mean(unrelated))
        cap = SHAPE_TARGETS["cross_group_shortfall_corr_max"]
        # DO NOT REMOVE THIS AS REDUNDANT. It is not a second opinion on the
        # within-group figure; it is the only thing standing between us and a
        # degenerate estimate.
        #
        # Correlations are measured on a sparse series after subtracting each
        # week's cross-sectional mean. When the event rate is low, almost every
        # entry becomes the negative of that mean, so EVERY pair correlates
        # toward 1 -- unrelated pairs included -- and the within-group figure
        # rises while the real structure disappears. Measured at `brutal`
        # difficulty: within-group 0.592 (higher than nominal's 0.434) with
        # unrelated at 0.560, on 0.9% non-zero observations.
        #
        # The within-group gate cannot see that. This one can, and does: it
        # fails at `hard` and `brutal` exactly as it should. The interpretable
        # quantity is the SEPARATION between the two, reported below.
        sep = float(np.mean(within)) - v if within else float("nan")
        res.add("statistical", "unrelated-supplier correlation", abs(v) < cap,
                f"{v:+.3f}", f"|x| < {cap}",
                f"{len(unrelated):,} pairs sharing no group, region, "
                f"tier-2 parent or checkpoint. Separation from within-group: "
                f"{sep:+.3f} -- this, not the within-group figure alone, is "
                f"what says the shared structure is real")


# ===========================================================================
# Volumes -- s1 row counts over the window they were stated for
# ===========================================================================

# (spec row count over 7 years at full scale, date column, what it scales with).
# Procurement volume is proportional to the channel count; planning volume is
# proportional to the product count. Scaling the planning tables by channels
# makes their target move with an unrelated quantity and fail on seeds where
# the channel draw happened to run high.
SPEC_VOLUMES = {
    "purchase_orders": (250_000, "created_ts", "channels"),
    "po_lines": (900_000, "created_ts", "channels"),
    "po_line_schedules": (1_800_000, "schedule_date", "channels"),
    "po_line_revisions": (400_000, "event_ts", "channels"),
    "supplier_acknowledgements": (700_000, "ack_date", "channels"),
    "grn_lines": (1_400_000, "event_ts", "channels"),
    "quality_inspections": (900_000, "event_ts", "channels"),
    "production_plan": (300_000, "target_period", "products"),
    "production_actual": (100_000, "period", "products"),
}


def check_volumes(res, csv_dir, cfg) -> None:
    """dataset_structure.md s1 states its counts for 7 years, and only at full
    scale. Below that the targets are scaled by the channel count, which is
    what the transaction volume is actually proportional to."""
    lo, hi = cfg.volume_window()
    # Scale on ACTIVE entities against s1's "active today" column, not on
    # distinct-over-history against its "distinct" column. The distinct counts
    # depend on how long the span is and on the churn draw, so a target built
    # from them moves with the seed; the active counts are what the preset
    # actually fixes.
    # Each basis is matched to the s1 column that actually predicts the volume.
    # Procurement volume tracks the DISTINCT channel count (14,000 over 7y):
    # every channel that ever existed placed orders while it did, and the
    # active count is not proportional across presets -- `small` carries 2.85
    # channels per part against `full`'s 1.81, because with three plants the
    # multi-plant BOM overlap is much higher. Planning volume tracks ACTIVE
    # products (1,200), which is what the preset fixes directly.
    n_channels = sum(1 for _ in rows(csv_dir, "sourcing_channels"))
    n_products = sum(1 for r in rows(csv_dir, "products")
                     if r.get("is_active") == "true")
    if not n_channels or not n_products:
        return
    scales = {"channels": n_channels / 14_000, "products": n_products / 1_200}
    years = (hi - lo).days / 365.25
    tol = cfg.volumes.tolerance
    for table, (spec, col, basis) in SPEC_VOLUMES.items():
        scale = scales[basis]
        d = present([csv_dir], table)
        if not d:
            continue
        n = 0
        for r in rows(d, table):
            v = r.get(col, "")
            if v and lo <= _dt.date.fromisoformat(v[:10]) <= hi:
                n += 1
        # At `full` the preset IS the configuration s1's counts describe, so
        # the target is the stated figure itself, pro-rated for the window
        # length. Applying a scale factor there would move the target with the
        # realised entity draw and quietly grade the generator against itself.
        if cfg.preset.name == "full":
            target = spec * (years / 7.0)
        else:
            target = spec * scale * (years / 7.0)
        if target < 50:
            continue
        err = n / target - 1
        # BUILD_PROMPT.md s1 asks for these volumes "at full". Below that the
        # target is an extrapolation on the channel count, and the channel
        # count is not proportional across presets -- `small` carries 2.85
        # channels per part against `full`'s 1.81 -- so a miss at `small` or
        # `mid` reports the extrapolation, not the data. Reported everywhere,
        # gated only where the figures were stated.
        gated = cfg.preset.name == "full"
        res.add("volumes", table, (abs(err) <= tol) if gated else True,
                f"{n:,}", f"{target:,.0f} +/-{tol:.0%}" if gated
                else f"~{target:,.0f} (not gated)", f"{err:+.0%}")


# ===========================================================================
# Reproducibility
# ===========================================================================

def digest(d: str) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for name in sorted(os.listdir(d)):
        if not name.endswith(".gz"):
            continue
        h.update(name.encode())
        with open(os.path.join(d, name), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def check_reproducibility(res, cfg, tmp_root) -> None:
    """Same seed gives byte-identical CSVs; a different seed gives a different
    world with the same statistical properties."""
    from generators import master, outcomes, planning, procurement
    from generators._common import WriterSet
    import ground_truth

    def emit(seed, out):
        c = Config.build(cfg.preset.name, seed=seed, disabled=cfg.disabled,
                         out_dir=out)
        w = master.build_world(c)
        ground_truth.bind(w)
        planning.build(w)
        master.finalise_demand(w)
        wr = WriterSet(out, c.gzip_level)
        master.emit(w, wr)
        planning.emit(w, wr)
        procurement.simulate(w, wr)
        outcomes.emit(w, wr)
        wr.close_all()
        return w

    a = os.path.join(tmp_root, "repro_a")
    b = os.path.join(tmp_root, "repro_b")
    c_ = os.path.join(tmp_root, "repro_c")
    emit(cfg.seed, a)
    emit(cfg.seed, b)
    w2 = emit(cfg.seed + 1, c_)
    same = digest(a) == digest(b)
    diff = digest(a) != digest(c_)
    res.add("reproducibility", "same seed gives byte-identical CSVs", same,
            "identical" if same else "DIFFERS", "identical")
    res.add("reproducibility", "a different seed gives a different world", diff,
            "differs" if diff else "IDENTICAL", "differs")
    # ... but the same statistical properties.
    other = Result()
    check_statistical(other, c_, Config.build(cfg.preset.name, seed=cfg.seed + 1),
                      world=w2)
    failed = [c.name for c in other.failed]
    res.add("reproducibility", "the other seed meets the same base rates",
            not failed, f"{len(other.checks) - len(failed)}/{len(other.checks)}",
            "all", ", ".join(failed[:3]))


# ===========================================================================
# Driver and report
# ===========================================================================

# ===========================================================================
# Gate self-test -- the standing rule
#
#   A check does not count as a gate until it has been demonstrated to FAIL on
#   a known-bad input.
#
# Three defects of the same shape have been found in this file by inspection
# rather than by the gates themselves:
#
#   `stddev > 0`            passed a target whose sd was 1e-5 across three
#                           values all within 0.0007 of 1.0
#   `censored_fraction`     three tasks reported exactly 0.0% because they
#                           were not testing observability at all
#   `len(distinct) <= 2`    could not fail on any continuous target, and would
#                           have missed the three-valued fill_rate defect
#
# Every one passed continuously while the thing it named was broken. A gate
# that has never been shown to fail is a comment with a green tick next to it.
#
# `python validate.py --self-test` runs each predicate against a fixture built
# to break it, and fails if the gate passes. New gates go in the table below
# with their fixture, or they do not count.
# ===========================================================================

def _gate_fixtures():
    """(name, predicate, bad_input, good_input) for every gate with a fixture.

    The predicate takes one argument and returns True when the data is
    acceptable, so a fixture passes the self-test when predicate(bad) is False
    and predicate(good) is True.
    """
    lo_c, hi_c = CENSORED_BAND
    lo_r, hi_r = CENSORED_TAIL_RATIO
    cap = SHAPE_TARGETS["cross_group_shortfall_corr_max"]
    corr_lo, corr_hi = SHAPE_TARGETS["within_group_shortfall_corr"]
    return [
        # The exact defect that shipped: three distinct values, sd ~1e-5, all
        # within 0.0007 of 1.0. `sd > 0` and `len(distinct) <= 2` both pass it.
        ("label varies (the shipped fill_rate defect)",
         lambda v: label_variation(v)[0],
         [1.0] * 900 + [0.9993] * 50 + [0.9999] * 50,
         [1.0] * 900 + [0.2, 0.35, 0.5, 0.66, 0.81] * 20),
        ("label varies (exactly binary)",
         lambda v: label_variation(v)[0], [0.0, 1.0] * 500,
         [i / 500 for i in range(1000)]),
        ("label varies (single constant)",
         lambda v: label_variation(v)[0], [1.0] * 1000,
         [i / 500 for i in range(1000)]),
        ("censored fraction in band (task silently drops open rows)",
         lambda f: lo_c < f < hi_c, 0.0, 0.15),
        ("censored fraction in band (nothing observed)",
         lambda f: lo_c < f < hi_c, 0.95, 0.15),
        ("right-censored tail size (tail collapsed)",
         lambda r: lo_r <= r <= hi_r, 0.0, 0.58),
        ("right-censored tail size (tail swallows the data)",
         lambda r: lo_r <= r <= hi_r, 4.0, 0.58),
        # Displacement: presence only. The old gate here banded the magnitude
        # (2%-35%) and passed for five phases while the store was discarding
        # exactly the units it was measuring.
        ("reporting lag displaces receipts (reporting_lag disabled)",
         displacement_ok, 0.0, 0.069),
        # Conservation: the check that would have caught that. Bad input is the
        # measured pre-fix shortfall at `full`; good input is equality.
        ("conservation (builder drops late-recorded rows)",
         lambda pair: conserved_ok(*pair),
         (5_697_236_276, 5_306_348_952), (5_697_236_276, 5_697_236_276)),
        ("conservation (builder double-counts a week)",
         lambda pair: conserved_ok(*pair),
         (6_211_613_371, 6_211_613_371 * 2), (6_211_613_371, 6_211_613_371)),
        ("unrelated correlation (degenerate estimator at low event rate)",
         lambda v: abs(v) < cap, 0.560, 0.013),
        ("within-group correlation (zero-filled estimator)",
         lambda v: corr_lo <= v <= corr_hi, 0.246, 0.403),
    ]


def self_test() -> bool:
    """Assert every gate with a fixture fails on its known-bad input."""
    width = 96
    print("=" * width)
    print("  GATE SELF-TEST -- does each gate actually fail on a bad input?")
    print("=" * width)
    bad = 0
    for name, pred, bad_in, good_in in _gate_fixtures():
        fails_on_bad = not pred(bad_in)
        passes_on_good = pred(good_in)
        ok = fails_on_bad and passes_on_good
        if not ok:
            bad += 1
        why = ("" if ok else
               ("  <- DID NOT FAIL on the bad input" if not fails_on_bad
                else "  <- rejected a GOOD input"))
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:<62}{why}")
    print("-" * width)
    if bad:
        print(f"  {len(_gate_fixtures()) - bad}/{len(_gate_fixtures())} gates "
              f"demonstrated. {bad} DO NOT DETECT THEIR OWN DEFECT.")
    else:
        print(f"  all {len(_gate_fixtures())} gates fail on a known-bad input "
              f"and pass on a good one")
    print("=" * width)
    return bad == 0


def validate(csv_dir, cfg, derived_dir=None, world=None, schema_path=None,
             reproducibility=False, tmp_root="/tmp") -> Result:
    schema_path = schema_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "schema.sql")
    dirs = [d for d in (csv_dir, derived_dir) if d]
    res = Result()
    check_referential(res, dirs, schema_path)
    check_temporal(res, dirs, schema_path, lag_enabled=cfg.on("reporting_lag"))
    check_leakage(res, dirs, schema_path, csv_dir, derived_dir)
    check_labels(res, dirs)
    check_statistical(res, csv_dir, cfg, world=world)
    check_volumes(res, csv_dir, cfg)
    if reproducibility:
        check_reproducibility(res, cfg, tmp_root)
    return res


def report(res: Result, cfg) -> bool:
    width = 96
    print()
    print("=" * width)
    print(f"  VALIDATION -- preset {cfg.preset.name}, seed {cfg.seed}, "
          f"{cfg.preset.start} to {cfg.preset.end}")
    if cfg.disabled:
        print(f"  disabled mechanisms: {', '.join(sorted(cfg.disabled))}")
    print("=" * width)
    group = None
    for c in res.checks:
        if c.group != group:
            group = c.group
            print(f"\n  {group.upper()}")
        mark = "ok " if c.passed else "FAIL"
        print(f"    [{mark}] {c.name:<46}{c.value:>16}   {c.target:<14}")
        if c.note:
            for line in _wrap(c.note, width - 14):
                print(f"           {line}")
    n = len(res.checks)
    bad = res.failed
    print("\n" + "-" * width)
    if bad:
        print(f"  {n - len(bad)}/{n} checks passed. FAILED: "
              f"{', '.join(c.name for c in bad)}")
    else:
        print(f"  all {n} checks passed")
    print("=" * width)
    return not bad


def _wrap(text, width):
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


# Directory naming is config.Config.build's out_dir:
#   csv_<preset>_seed<N>              (difficulty nominal)
#   csv_<preset>_<difficulty>_seed<N> (everything else)
DIR_NAME = re.compile(
    r"^csv_(?P<preset>small|mid|full)"
    r"(?:_(?P<difficulty>easy|nominal|hard|brutal))?"
    r"_seed(?P<seed>\d+)$")


def infer_from_dir(csv_dir: str) -> dict:
    """What preset/seed/difficulty a directory name declares itself to be.

    The preset decides the volume targets, the base-rate bands, the event-rate
    bands and the window every statistical check is evaluated over. Getting it
    wrong does not skip checks -- it grades one world against another world's
    expectations and reports confident, wrong failures. `--preset` used to
    default to `small` regardless of what was being validated, so the
    documented invocation against csv_mid_seed1 reported four failures that
    were entirely artefacts of the 3-year `small` window.

    So: read it off the directory, and refuse to guess.
    """
    name = os.path.basename(os.path.abspath(csv_dir.rstrip("/")))
    m = DIR_NAME.match(name)
    if not m:
        return {}
    return {"preset": m.group("preset"), "seed": int(m.group("seed")),
            "difficulty": m.group("difficulty") or "nominal"}


def resolve_preset(ap, csv_dir: str, preset: str | None, seed: int | None):
    """Reconcile the flags against the directory. Never silently default."""
    found = infer_from_dir(csv_dir)
    name = os.path.basename(os.path.abspath(csv_dir.rstrip("/")))
    if not found:
        if preset is None:
            ap.error(
                f"cannot infer the preset from directory name {name!r} "
                f"(expected csv_<small|mid|full>[_<difficulty>]_seed<N>). "
                f"Pass --preset explicitly. Validating against the wrong "
                f"preset grades the data against another world's bands.")
        return preset, (1 if seed is None else seed), None
    # An explicit flag that contradicts the data is a mistake, not an override.
    if preset is not None and preset != found["preset"]:
        ap.error(
            f"--preset {preset} contradicts directory {name!r}, which is a "
            f"{found['preset']} dataset. The preset is a property of the data, "
            f"not of the run; omit --preset.")
    if seed is not None and seed != found["seed"]:
        ap.error(
            f"--seed {seed} contradicts directory {name!r}, which is seed "
            f"{found['seed']}. Omit --seed.")
    return found["preset"], found["seed"], found["difficulty"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv_dir", nargs="?", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="assert every gate fails on a known-bad fixture; "
                         "needs no dataset")
    ap.add_argument("--derived", default=None,
                    help="directory holding the Group I feature stores")
    ap.add_argument("--preset", default=None,
                    help="scale preset; inferred from the directory name when "
                         "omitted. Passing one that contradicts the directory "
                         "is an error, not an override")
    ap.add_argument("--seed", type=int, default=None,
                    help="generator seed; inferred from the directory name "
                         "when omitted")
    ap.add_argument("--disable", default="", help="comma-separated mechanisms")
    ap.add_argument("--reproducibility", action="store_true",
                    help="re-run the generator twice to check byte-identity")
    ap.add_argument("--tmp", default="/tmp")
    a = ap.parse_args(argv)
    if a.self_test:
        return 0 if self_test() else 1
    if not a.csv_dir:
        ap.error("csv_dir is required unless --self-test is given")
    disabled = {m for m in a.disable.split(",") if m}
    preset, seed, difficulty = resolve_preset(ap, a.csv_dir, a.preset, a.seed)
    cfg = Config.build(preset, seed=seed, disabled=disabled,
                       **({"difficulty": difficulty} if difficulty else {}))
    if difficulty and difficulty != "nominal":
        print(f"  note: {difficulty} difficulty -- several base-rate gates are "
              f"definitions of `nominal` and are expected to fail. "
              f"Characterising, not certifying.\n", file=sys.stderr)
    derived = a.derived
    if derived is None:
        guess = os.path.join(a.csv_dir, "derived")
        derived = guess if os.path.isdir(guess) else None
    res = validate(a.csv_dir, cfg, derived_dir=derived,
                   reproducibility=a.reproducibility, tmp_root=a.tmp)
    return 0 if report(res, cfg) else 1


if __name__ == "__main__":
    raise SystemExit(main())
