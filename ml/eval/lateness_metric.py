"""THE adopted lateness metric for arrival, and the guard that keeps the old one unquotable.

Phase 11C Stage C. Phase 11B §4.5 named reference (a) -- the channel's as-of median observed lead
-- but nothing used it. This module is what uses it.

THE DEFINITION (also recorded in docs/specs/lateness_metric.md so the next phase need not
re-derive it):

    R(channel, t0) = median{ lead_weeks : GRN lines of that channel RECORDED on or before t0 }
                     + c

    label  yl = arrival_week > R
    score  pl = prediction - R
    metric ROC-AUC(yl, pl)

`c` is a single global offset FITTED ON TRAINING ROWS ONLY AND NEVER REFITTED. It exists because
`arrival_week` is measured from t0 and contains two parts -- the wait until the order is raised,
and the lead once it is. Only the second varies by channel; the first is a property of the label's
sampling window and is common to every channel. So the channel-varying part of the reference is
the historical lead and the constant part is fitted once. `c` shifts every reference equally, so
it sets the label's balance and CANNOT affect ranking within a reference.

WHY THE PROMISE REFERENCE IS NOT THE METRIC. `promise_week` comes from a po_line raised ~7 weeks
AFTER the forecast instant (B9 = 0.00% on every v8 dataset seed). Worse, Phase 11C Stage A showed
that scoring any arm against it with a constant prediction ranks on -promise, which made Phase 8's
"beyond the promise date" a comparison against a naive constant. The promise reference is kept for
continuity and is carried under the `PRIVILEGED__` prefix, which `assert_no_privileged_headline`
makes LOAD-BEARING: a PRIVILEGED__ metric cannot be selected on or reported as a headline.

DEVIATION 65 STAYS NARROWED, NOT WITHDRAWN. Both sentences travel with any arrival lateness claim:
    * Against the promise date the claim is RETIRED -- that reference is not available at t0.
    * Against a t0-available reference the claim is ESTABLISHED at five model seeds.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "models"),
                os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data")]
import numpy as np

PRIVILEGED_PREFIX = "PRIVILEGED__"
ADOPTED = "asof_channel_lead"
DEVIATION_65 = (
    "Against the promise date the arrival head's lateness claim is RETIRED: that reference is "
    "not available at t0. Against a t0-available reference it is ESTABLISHED at five model seeds.")


def assert_no_privileged_headline(metrics, *, selected=None, headline=None):
    """The PRIVILEGED__ prefix is load-bearing, not decorative.

    Fires if a privileged metric is used to SELECT a configuration or is reported as a HEADLINE.
    Both are how a non-deployable number reaches a client. It can fail: pass any PRIVILEGED__ key
    as `selected` or `headline` and it raises.
    """
    priv = sorted(k for k in metrics if str(k).startswith(PRIVILEGED_PREFIX))
    if selected is not None and str(selected).startswith(PRIVILEGED_PREFIX):
        raise AssertionError(
            f"selection on a PRIVILEGED metric ({selected}). It reads information not available "
            f"at t0 and cannot choose a configuration.")
    if headline is not None:
        heads = [headline] if isinstance(headline, str) else list(headline)
        bad = [h for h in heads if str(h).startswith(PRIVILEGED_PREFIX)]
        if bad:
            raise AssertionError(
                f"PRIVILEGED metric(s) {bad} reported as a headline. They may appear only as "
                f"labelled context, never as the claim.")
    return dict(privileged_present=priv, n_privileged=len(priv))


def adopted_reference(world, lb, tr_mask, check_asof=True):
    """The adopted reference R, in weeks from t0, aligned to lb's rows."""
    from phase11b_lateness import build_reference
    return build_reference(ADOPTED, world, lb, tr_mask, check_asof=check_asof)


def lateness_scores(predictions: dict, Y, EV, R):
    """-> {arm: ROC-AUC} under the adopted metric. Every arm scores its OWN prediction.

    There is no constant-substitution path here. That is the defect Stage A closed
    (ml/tests/test_no_lateness_substitution.py), and reintroducing one would make two different
    baselines report the same number again.
    """
    from phase11b_lateness import lateness
    return {arm: lateness(P, Y, EV, R)[0] for arm, P in predictions.items()}
