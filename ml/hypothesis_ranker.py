"""
The hypothesis ranking head, and the calibration machinery that is the actual deliverable.

`reports/layer3_testing.md` §10 does not claim to *discover* a hidden cause -- §9.8 closed that
question. It claims something weaker and checkable: given a detected co-degradation, produce a
**ranked, calibrated** distribution over candidate explanations, including an explicit
"unknown", and hand the decision to a human. The weaker claim is only worth anything if the
confidences are validated, so this module treats calibration as the primary measurement and
accuracy as secondary.

**Three decisions worth stating, because each one is a place a flattering number could hide.**

*Multi-label, not multi-class.* A pair can share a hidden parent *and* a port. Four independent
sigmoid heads let both be true; a softmax would force the model to trade one off against the
other and would make "distribution over hypotheses" mean something the world does not.

*Class weighting, and the damage it does.* Positives are rare enough (see §10.2's composition
table) that an unweighted fit predicts near-zero everywhere and scores well on every
threshold-free metric while being useless. `pos_weight` fixes the ranking and **wrecks the
calibration** -- deliberately so, and visibly: `evaluate` reports ECE both before and after
isotonic recalibration, and the gap between them is the size of the distortion. A model
reported only post-calibration would hide it.

*The split is by dataset seed, not by pair.* §9.8's finding is that co-membership is
distinguishable only by supplier *identity*. A random pair-level split leaves the same
suppliers on both sides, so a head with any capacity can memorise identities and post a
held-out number that means nothing -- exactly what Stage 1 did at §9.4. Folds are therefore
whole dataset seeds: the test seed's suppliers, groups and pools were generated independently
and the model has never seen any of them. Within each fold a further seed is held out as the
**calibration** split, so the isotonic fit never touches test data either.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ml.hypothesis_labels import CLASSES

N_BINS = 10


class HypothesisHead(nn.Module):
    """Small MLP -> one logit per hypothesis class.

    Deliberately small (two hidden layers of 64). The question is whether the observable
    signal *carries* the distinction, not whether a large model can memorise a few thousand
    pairs; §9.8 already established that capacity is not the binding constraint anywhere in
    this problem, so spending it here would only buy a less honest number.
    """

    def __init__(self, n_features: int, hidden: int = 64, n_classes: int = len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _standardise(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def train_head(Xtr: np.ndarray, Ytr: np.ndarray, Xva: np.ndarray, Yva: np.ndarray,
               epochs: int = 200, lr: float = 3e-3, weight_decay: float = 1e-4,
               seed: int = 0, device: str = "cpu",
               max_pos_weight: float = 50.0) -> dict:
    """Fit one head. Returns the model, the standardisation, and the epoch chosen.

    `pos_weight` is capped: at the rarest class's prevalence the uncapped ratio runs into the
    hundreds, and a weight that large makes the loss surface almost entirely about a handful
    of positives, which destabilises the fit far more than it helps the ranking. The cap is
    reported alongside the results so it is a stated choice rather than a silent one.
    """
    torch.manual_seed(seed)
    mu, sd = _standardise(Xtr)
    xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    ytr = torch.tensor(Ytr, dtype=torch.float32, device=device)
    xva = torch.tensor((Xva - mu) / sd, dtype=torch.float32, device=device)
    yva = torch.tensor(Yva, dtype=torch.float32, device=device)

    pos = Ytr.sum(axis=0).astype(float)
    neg = len(Ytr) - pos
    pw = np.clip(np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0), 1.0, max_pos_weight)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32,
                                                           device=device))

    model = HypothesisHead(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_epoch, best_state = float("inf"), 0, None
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xtr), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(xva), yva))
        if vl < best - 1e-6:
            best, best_epoch = vl, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "mu": mu, "sd": sd, "pos_weight": pw.tolist(),
            "best_epoch": best_epoch, "best_val_loss": best,
            "params": model.parameter_count()}


def predict(fit: dict, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """`[P, C]` sigmoid probabilities."""
    model = fit["model"]
    model.eval()
    x = torch.tensor((X - fit["mu"]) / fit["sd"], dtype=torch.float32, device=device)
    with torch.no_grad():
        return torch.sigmoid(model(x)).cpu().numpy()


# --------------------------------------------------------------------------- calibration


def isotonic_fit(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators isotonic regression, returned as a lookup `(x, y)` step.

    Written out rather than imported from sklearn so the mapping is inspectable and so the
    same code path runs when `IsotonicRegression`'s tie-handling changes between versions --
    this project has already been bitten once by a library-version-dependent result.
    """
    order = np.argsort(p, kind="stable")
    xs, ys = p[order].astype(float), y[order].astype(float)
    # PAVA: merge adjacent blocks until the fitted values are non-decreasing.
    vals = list(ys)
    wts = [1.0] * len(ys)
    i = 0
    while i < len(vals) - 1:
        if vals[i] <= vals[i + 1] + 1e-12:
            i += 1
            continue
        w = wts[i] + wts[i + 1]
        v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / w
        vals[i:i + 2] = [v]
        wts[i:i + 2] = [w]
        i = max(i - 1, 0)
    fitted, out = [], []
    for v, w in zip(vals, wts):
        fitted.extend([v] * int(round(w)))
    fitted = np.asarray(fitted[:len(xs)], dtype=float)
    return xs, fitted


def isotonic_apply(curve: tuple[np.ndarray, np.ndarray], p: np.ndarray) -> np.ndarray:
    xs, ys = curve
    if len(xs) == 0:
        return p
    return np.interp(p, xs, ys, left=ys[0], right=ys[-1])


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> tuple[float, list[dict]]:
    """Expected Calibration Error over **equal-count** bins, plus the reliability curve.

    Equal-count rather than equal-width, because these predictions pile up near zero: with
    equal-width bins nine of ten bins would hold a handful of points each and the ECE would
    be dominated by bins too small to estimate a rate in. Equal-count bins put the same
    number of instances behind every point of the reliability curve, which is the condition
    under which "41% means 41%" can be checked at all.
    """
    n = len(p)
    if n == 0:
        return float("nan"), []
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, min(n_bins, n))
    total, curve = 0.0, []
    for b in bins:
        if len(b) == 0:
            continue
        conf, acc = float(p[b].mean()), float(y[b].mean())
        total += len(b) / n * abs(acc - conf)
        curve.append({"n": int(len(b)), "confidence": conf, "empirical": acc,
                      "p_lo": float(p[b].min()), "p_hi": float(p[b].max())})
    return float(total), curve


def roc_auc(p: np.ndarray, y: np.ndarray) -> float | None:
    """Rank-based ROC AUC with midpoint tie handling; None when a class is degenerate."""
    y = y.astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # Average ranks within tied score groups, so a head that outputs a constant scores 0.5
    # instead of whatever the sort order happened to be.
    sp = p[order]
    start = 0
    for i in range(1, len(sp) + 1):
        if i == len(sp) or sp[i] != sp[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + i + 1) / 2.0
            start = i
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def average_precision(p: np.ndarray, y: np.ndarray) -> float | None:
    y = y.astype(bool)
    if y.sum() == 0:
        return None
    order = np.argsort(-p, kind="stable")
    hits = y[order]
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(prec[hits].sum() / hits.sum())


def evaluate(p_raw: np.ndarray, p_cal: np.ndarray, y: np.ndarray,
             n_bins: int = N_BINS) -> dict:
    """Per-class ranking and calibration quality. Never pooled into one headline number.

    §10's whole prediction is that the classes will behave *differently* -- `regional_logistics`
    close to observable and therefore workable, `shared_upstream` at chance per §9.8 -- so a
    macro average across them would average away the one result the section exists to state.
    """
    out = {}
    for i, cls in enumerate(CLASSES):
        yi = y[:, i]
        e_raw, _ = ece(p_raw[:, i], yi, n_bins)
        e_cal, curve = ece(p_cal[:, i], yi, n_bins)
        out[cls] = {
            "n": int(len(yi)), "positives": int(yi.sum()),
            "base_rate": float(yi.mean()),
            "auc": roc_auc(p_raw[:, i], yi),
            "ap": average_precision(p_raw[:, i], yi),
            "ece_raw": e_raw, "ece_calibrated": e_cal,
            "brier_raw": float(np.mean((p_raw[:, i] - yi) ** 2)),
            "brier_calibrated": float(np.mean((p_cal[:, i] - yi) ** 2)),
            "mean_pred_raw": float(p_raw[:, i].mean()),
            "mean_pred_calibrated": float(p_cal[:, i].mean()),
            "reliability": curve,
        }
    return out
