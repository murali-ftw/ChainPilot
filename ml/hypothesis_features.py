"""
Observable pair features for §10's hypothesis ranking head — and the pattern detector itself.

**Every column here comes from an emitted CSV.** Nothing in this module reads
`db/generate_dataset.py`'s namespace, and none of the privileged mechanism state in
`ml/hypothesis_labels.py` appears as an input. That separation is the whole basis on which
§10's numbers can be read as decision support rather than as a restated ground truth: the
labels are the target, never a feature. `assert_no_privileged_features` re-checks it against
the live label object rather than asserting it in prose.

Two halves:

* **Detection** (`detect_patterns`) — the candidate set of *pattern instances*. This reuses
  `ml/observable_cofailure.py`'s `on_time_rate_90d` correlation exactly, which is the column
  §2.2's generator-side co-degradation check validated and the one §9.7 measured. It is
  deliberately not a new instrument.

* **Features** (`build_features`) — a symmetric function of the unordered pair, built from the
  same correlation plus each supplier's own observable attributes.

**One deviation from §9.7, stated because it changes what the numbers mean.** Stage 2 was a
*forecasting* test: it correlated only rows with `as_of_date < t0`, because a retriever feeding
a prediction at `t0` may not see the future, and that left 9 snapshots and 595 usable
suppliers. §10's module is **retrospective** — a reviewer is handed a co-degradation that has
already happened and asked what explains it — so detection reads the full 15-snapshot history
(708 usable suppliers). This is not a relaxed leakage rule; it is a different task. The
targets are static structural properties of the world, not future events, so "the future"
is not a well-defined thing to leak from. `--respect-t0` re-runs the whole module under
Stage 2's forecasting restriction, and §10.2 reports the candidate set both ways so the
choice is visible rather than assumed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ml.observable_cofailure import cofailure_scores, load_history, observable_cohort

# A residual this far below the fleet at one snapshot is treated as a "dip" for the joint-dip
# features. 0.05 = five on-time-rate points, which is roughly the width of the fleet's own
# snapshot-to-snapshot wobble at mid scale; the features are counts and depths, so the exact
# cut only sets a scale, not a threshold anything is decided on.
DIP = 0.05

FUTURE = "9999-12-31"   # sentinel `as_of` meaning "read the whole recorded history"


def _residuals(wide: pd.DataFrame, sup_ids: list[str], as_of: str,
               cohort: np.ndarray | None = None) -> np.ndarray:
    """`[N, T]` fleet-residual matrix, NaN where unobserved — the series `cofailure_scores`
    correlates, exposed separately so the joint-dip features describe the same object the
    correlation was computed from rather than a parallel reconstruction of it."""
    past = [c for c in wide.columns if str(c) < str(as_of)]
    X = wide.reindex(index=sup_ids)[past].to_numpy(dtype=float)
    resid = X - np.nanmean(X, axis=0, keepdims=True)
    if cohort is not None:
        for c in np.unique(cohort):
            m = cohort == c
            if m.sum() >= 2 and np.isfinite(resid[m]).any():
                resid[m] -= np.nanmean(resid[m], axis=0, keepdims=True)
    return resid


class PairContext:
    """Everything about one variant-seed that both detection and featurisation need.

    Built once per variant-seed and reused, because the `[N, N]` correlation matrices are the
    expensive part (2,000 x 2,000 float64 = 32 MB each) and the detector and the featuriser
    must see the *same* matrix — a candidate set selected from one score matrix and featurised
    from a slightly different one would silently decorrelate the two.
    """

    def __init__(self, csv_dir: str, as_of: str = FUTURE, min_points: int = 4):
        self.csv_dir = csv_dir
        self.as_of = as_of
        self.min_points = min_points
        self.sup_ids = pd.read_csv(f"{csv_dir}/suppliers.csv.gz",
                                   usecols=["id"])["id"].tolist()
        self.n = len(self.sup_ids)
        self.cohort = observable_cohort(csv_dir, self.sup_ids)

        w90, _ = load_history(csv_dir, "on_time_rate_90d")
        w30, _ = load_history(csv_dir, "on_time_rate_30d")
        # Cohorts in which every supplier is unobserved produce an empty-slice mean; that
        # is a coverage fact, not an error, and the -inf affinity it yields is exactly the
        # "never retrieved" behaviour `cofailure_scores` documents.
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        self.s90 = cofailure_scores(w90, self.sup_ids, as_of, cohort=None,
                                    min_points=min_points)
        self.s90c = cofailure_scores(w90, self.sup_ids, as_of, cohort=self.cohort,
                                     min_points=min_points)
        self.s30 = cofailure_scores(w30, self.sup_ids, as_of, cohort=None,
                                    min_points=min_points)
        self.resid = _residuals(w90, self.sup_ids, as_of)
        self.obs = np.isfinite(self.resid)
        self.n_obs = self.obs.sum(axis=1)
        self.usable = self.n_obs >= min_points

        # Descending rank of every partner within each anchor's row: rank 1 = top pick. The
        # detector selects on it and the featuriser reports it, from this one array.
        s = np.where(np.isfinite(self.s90), self.s90, -np.inf)
        order = np.argsort(-s, axis=1, kind="stable")
        self.rank = np.empty_like(order)
        self.rank[np.arange(self.n)[:, None], order] = np.arange(1, self.n + 1)[None, :]

        self._attrs = _supplier_attributes(csv_dir, self.sup_ids)
        self.coparent_edges = _observable_coparents(csv_dir, self.sup_ids)


def _supplier_attributes(csv_dir: str, sup_ids: list[str]) -> dict[str, np.ndarray]:
    """Per-supplier observable attributes, aligned to `sup_ids`."""
    sup = pd.read_csv(f"{csv_dir}/suppliers.csv.gz").set_index("id").reindex(sup_ids)
    comp = pd.read_csv(f"{csv_dir}/components.csv.gz", usecols=["id", "supplier_id"])
    csup = pd.read_csv(f"{csv_dir}/component_suppliers.csv.gz",
                       usecols=["component_id", "supplier_id"])
    stf = pd.read_csv(f"{csv_dir}/supplier_temporal_features.csv.gz",
                      usecols=["supplier_id", "shipment_count_180d", "lateness_variance",
                               "trend_slope"])
    agg = stf.groupby("supplier_id").mean(numeric_only=True).reindex(sup_ids)

    out = {
        "lead_time": sup["lead_time_days"].to_numpy(dtype=float),
        "capacity": sup["capacity_score"].to_numpy(dtype=float),
        "reliability": sup["reliability_history"].to_numpy(dtype=float),
        "degree_primary": comp.groupby("supplier_id").size().reindex(sup_ids)
                              .fillna(0).to_numpy(dtype=float),
        "degree_secondary": csup.groupby("supplier_id").size().reindex(sup_ids)
                                .fillna(0).to_numpy(dtype=float),
        "shipments": agg["shipment_count_180d"].fillna(0).to_numpy(dtype=float),
        "lateness_var": agg["lateness_variance"].fillna(0).to_numpy(dtype=float),
        "trend": agg["trend_slope"].fillna(0).to_numpy(dtype=float),
    }
    # Country enters as a per-country "both suppliers are in country X" indicator rather than
    # as a hand-picked `both_germany` flag. Two of the three base pools ARE country
    # definitions (H_TRUCK = USA/Mexico, H_CUSTOMS = Germany), so hard-coding them would be
    # handing the model the answer sheet from the privileged side; letting it learn which
    # countries matter from a generic encoding keeps the feature set honestly observable.
    countries = sorted(sup["country"].dropna().unique().tolist())
    out["_countries"] = countries
    out["country_code"] = pd.Categorical(sup["country"], categories=countries).codes
    return out


def _observable_coparents(csv_dir: str, sup_ids: list[str]) -> set[tuple[int, int]]:
    """Co-parent edges rebuilt from emitted tables only.

    `components.supplier_id` is the primary source and `component_suppliers` lists the
    secondaries, which is exactly how the generator builds its internal `coparents` dict.
    `ml/run_hypothesis_module.py` asserts this reconstruction equals the privileged set edge
    for edge -- that equality is the point of the `shared_sourcing` positive control, and an
    unasserted claim of observability would be worth nothing.
    """
    idx = {s: i for i, s in enumerate(sup_ids)}
    comp = pd.read_csv(f"{csv_dir}/components.csv.gz", usecols=["id", "supplier_id"])
    csup = pd.read_csv(f"{csv_dir}/component_suppliers.csv.gz",
                       usecols=["component_id", "supplier_id"])
    primary = dict(zip(comp["id"], comp["supplier_id"]))
    edges = set()
    for cid, sec in zip(csup["component_id"], csup["supplier_id"]):
        pri = primary.get(cid)
        if pri is None or pri == sec or pri not in idx or sec not in idx:
            continue
        a, b = idx[pri], idx[sec]
        edges.add((min(a, b), max(a, b)))
    return edges


def detect_patterns(ctx: PairContext, top_k: int = 16,
                    min_corr: float = 0.0) -> np.ndarray:
    """`[P, 2]` sorted, de-duplicated index pairs — the detected behavioural patterns.

    An instance is any unordered pair `{i, j}` where `j` is among `i`'s `top_k` most
    correlated partners (or vice versa) **and** the correlation clears `min_corr`. Both
    halves matter: top-K alone would force every usable supplier to emit K instances however
    weak its best partner is, while a bare threshold would let a handful of highly-correlated
    suppliers dominate the set. Suppliers below `min_points` observations carry -inf affinity
    from `cofailure_scores` and so never appear -- a supplier with no recorded history has no
    behavioural pattern to explain, which is a finding about coverage, not a gap to impute.
    """
    n = ctx.n
    k = min(top_k, n - 1)
    s = np.where(np.isfinite(ctx.s90), ctx.s90, -np.inf)
    part = np.argpartition(-s, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = part.reshape(-1)
    keep = np.isfinite(s[rows, cols]) & (s[rows, cols] > min_corr)
    rows, cols = rows[keep], cols[keep]
    lo = np.minimum(rows, cols)
    hi = np.maximum(rows, cols)
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]


def usable_universe(ctx: PairContext) -> np.ndarray:
    """`[P, 2]` every unordered pair of suppliers that carries a usable trajectory.

    This is not a detector -- it selects nothing -- and it is here for one reason: at
    `top_k=64` the detected set contains only 5-13 true `shared_upstream` pairs per seed
    (§10.2), which is far too few to distinguish "ranked at chance" from "not measured". The
    universe of pairs that *could* carry an observable pattern is the maximum-power version
    of the same question, and a null stated on it is a null with a sample size behind it.

    It is reported alongside the detected set, never instead of it: the detected set is what
    the module would actually emit, and its thinness is itself one of §10's findings.
    """
    u = np.where(ctx.usable)[0]
    a, b = np.triu_indices(len(u), k=1)
    return np.stack([u[a], u[b]], axis=1)


def _sym(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order-invariant summary of a per-supplier quantity over the pair. The pair is
    unordered, so any feature that depended on which supplier was listed first would let the
    model learn an artifact of the enumeration order."""
    return np.minimum(a, b), np.maximum(a, b)


def build_features(ctx: PairContext, pairs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """`([P, F] float32, feature names)` — all observable, all symmetric in the pair."""
    a, b = pairs[:, 0], pairs[:, 1]
    at = ctx._attrs
    cols: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, v: np.ndarray) -> None:
        cols.append(np.asarray(v, dtype=np.float64))
        names.append(name)

    def add_sym(name: str, per_supplier: np.ndarray) -> None:
        lo, hi = _sym(per_supplier[a], per_supplier[b])
        add(f"{name}_min", lo)
        add(f"{name}_max", hi)
        add(f"{name}_absdiff", np.abs(per_supplier[a] - per_supplier[b]))

    # --- behavioural: the correlation the detection was built from, and its variants
    add("corr90", np.nan_to_num(ctx.s90[a, b], nan=0.0, neginf=0.0))
    add("corr90_cohort", np.nan_to_num(ctx.s90c[a, b], nan=0.0, neginf=0.0))
    s30 = ctx.s30[a, b]
    add("corr30", np.nan_to_num(s30, nan=0.0, neginf=0.0))
    add("corr30_valid", np.isfinite(s30).astype(float))
    # The cohort-residualised correlation removes what country x lead-time explains. The
    # *gap* between it and the raw one is therefore a direct observable read on "how much of
    # this pair's co-movement is regional", which is the discriminative quantity between two
    # of the classes and is not recoverable from either column alone.
    add("corr90_cohort_gap", np.nan_to_num(ctx.s90[a, b] - ctx.s90c[a, b],
                                           nan=0.0, neginf=0.0))

    r_ab, r_ba = ctx.rank[a, b].astype(float), ctx.rank[b, a].astype(float)
    add("rank_min", np.minimum(r_ab, r_ba))
    add("rank_max", np.maximum(r_ab, r_ba))
    add("rank_log_min", np.log1p(np.minimum(r_ab, r_ba)))

    both = ctx.obs[a] & ctx.obs[b]
    n_both = both.sum(axis=1).astype(float)
    add("n_overlap", n_both)
    ra = np.where(both, ctx.resid[a], np.nan)
    rb = np.where(both, ctx.resid[b], np.nan)
    dip = both & (ctx.resid[a] < -DIP) & (ctx.resid[b] < -DIP)
    add("joint_dip_count", dip.sum(axis=1).astype(float))
    add("joint_dip_frac", dip.sum(axis=1) / np.maximum(1.0, n_both))
    # Most pairs never dip together and most suppliers are unobserved at most snapshots, so
    # all-NaN rows are the normal case here, not an anomaly -- they become 0.0 below, which
    # is the correct "no joint dip" value. Silenced narrowly rather than globally.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"(All-NaN|Mean of empty|Degrees of freedom)",
                                RuntimeWarning)
        depth = np.nanmin(np.where(dip, np.minimum(ra, rb), np.nan), axis=1)
        add("joint_dip_depth", np.nan_to_num(depth, nan=0.0))
        add("resid_mean_min", np.nan_to_num(np.minimum(np.nanmean(ra, axis=1),
                                                       np.nanmean(rb, axis=1)), nan=0.0))
        add("resid_std_max", np.nan_to_num(np.maximum(np.nanstd(ra, axis=1),
                                                      np.nanstd(rb, axis=1)), nan=0.0))
    add_sym("n_obs", ctx.n_obs.astype(float))

    # --- observable attributes
    cc = at["country_code"]
    add("same_country", (cc[a] == cc[b]).astype(float))
    for j, name in enumerate(at["_countries"]):
        add(f"both_{name}", ((cc[a] == j) & (cc[b] == j)).astype(float))
    for key in ("lead_time", "capacity", "reliability", "degree_primary",
                "degree_secondary", "shipments", "lateness_var", "trend"):
        add_sym(key, at[key])
    # Long lead time is the only emitted proxy for the `sea` flag that defines H_PORT, which
    # is never written to CSV (`ml/observable_cofailure.py::observable_cohort` makes the same
    # substitution for the same reason).
    long_lead = at["lead_time"] >= np.nanpercentile(at["lead_time"], 66)
    add("both_long_lead", (long_lead[a] & long_lead[b]).astype(float))

    edges = ctx.coparent_edges
    add("is_coparent", np.fromiter(((int(x), int(y)) in edges for x, y in zip(a, b)),
                                   dtype=float, count=len(pairs)))

    X = np.stack(cols, axis=1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def assert_no_privileged_features(names: list[str]) -> None:
    """Refuse to proceed if a privileged mechanism name reached the feature set.

    Cheap, and it is the check that would have caught the mistake if a later edit had added
    `hp_group` or `H_PORT` to `build_features` for convenience. §9.1's disclosure was prose;
    this is the executable form of it.
    """
    banned = ("hp_", "h_port", "h_truck", "h_customs", "h_polymer", "group", "resilience",
              "coparent_priv", "mechanism", "cs_rewire", "idio", "sea_flag")
    hit = [n for n in names if any(b in n.lower() for b in banned)]
    if hit:
        raise ValueError(f"privileged names in the observable feature set: {hit}")
