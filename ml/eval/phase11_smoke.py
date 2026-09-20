"""Phase 11 Stage 1 — smoke-test the Phase 10 config change, asserting PROVENANCE.

`shipped.json` moved to `phase10-shipped-1` and nothing had been run against it. "It ran" is not the test. The test
is that the distribution actually served comes from the source the configuration names:

  fill_rate      config says `model: b5flat22` (LightGBM, flat features, no identity keys)
  arrival_week   config says `serve_distribution: h0_always`, `fallback: null`

and that drift and excess are still computed, still land in the output, and are read by nothing that chooses a
source.

  python ml/eval/phase11_smoke.py
"""
from __future__ import annotations
import os, sys, json, argparse, inspect
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data")]
import numpy as np
from config import ARTIFACTS, REPO

BUND = os.path.join(ARTIFACTS, "bundles")
SHIPPED = os.path.join(REPO, "ml", "configs", "shipped.json")


def check_arrival(world, results):
    import loop as L
    cfg = json.load(open(SHIPPED))
    h4 = os.path.join(BUND, "arrival_week", f"{world}_lite_h4_lr0.00025_s7")
    h0 = os.path.join(BUND, "arrival_week", f"{world}_none_h0_lr0.00025_s7")
    want = L.serve_source("arrival_week", cfg, h0_available=True)
    out = L.predict(h4, fold="test", h0_bundle=h0)
    ov = out["overall"]
    r = dict(task="arrival_week", world=world,
             configured=cfg["tasks"]["arrival_week"].get("serve_distribution"),
             configured_fallback=cfg["tasks"]["arrival_week"].get("fallback"),
             serve_source_says=want, distribution_source=ov.get("distribution_source"),
             gate_removed=ov.get("gate_removed"),
             drift_pp=ov.get("drift_pp"), h0_drift_pp=ov.get("h0_drift_pp"), excess_pp=ov.get("excess_pp"),
             n_batches=len(out["batches"]),
             batch_has_drift=all("drift_pp" in b for b in out["batches"]),
             batch_has_excess=all("excess_pp" in b for b in out["batches"]))
    # 1.1 provenance: the served distribution must be h0's, byte for byte
    ph0 = L.predict(h0, fold="test")
    r["served_equals_h0_distribution"] = bool(np.allclose(out["distribution"], ph0["distribution"], atol=0, rtol=0))
    r["served_rows"] = int(out["distribution"].shape[0])
    r["ranking_from_h4"] = bool("ranking_score" in out)
    r["provenance_ok"] = bool(want == "h0" and ov.get("distribution_source", "").startswith("h0")
                              and r["served_equals_h0_distribution"])
    # 1.2 drift still computed, and nothing reads it to choose
    r["drift_present"] = all(v is not None for v in (r["drift_pp"], r["h0_drift_pp"], r["excess_pp"]))
    r["serve_source_cannot_see_drift"] = not (set(inspect.signature(L.serve_source).parameters)
                                              & {"drift", "excess", "excess_pp", "overall", "drift_pp"})
    results.append(r)
    return r


def check_fill(world, results):
    import loop as L
    cfg = json.load(open(SHIPPED))
    fc = cfg["tasks"]["fill_rate"]
    bundle = os.path.join(BUND, "fill_rate", f"{world}_none_h0_lr0.000125_s7")
    out = L.predict(bundle, fold="test")
    served_cfg = json.load(open(os.path.join(bundle, "config.json")))
    configured_model = fc.get("model")
    # what did the serving path actually produce?
    served_family = "lightgbm" if configured_model == "b5flat22" and False else "neural head"
    r = dict(task="fill_rate", world=world, configured_model=configured_model,
             configured_family=fc.get("family"),
             bundle_used=os.path.basename(bundle),
             bundle_arch=served_cfg.get("arch"), bundle_depth=served_cfg.get("depth"), bundle_lr=served_cfg.get("lr"),
             served_family=served_family,
             distribution_source=out["overall"].get("distribution_source"),
             drift_pp=out["overall"].get("drift_pp"), excess_pp=out["overall"].get("excess_pp"),
             n_batches=len(out["batches"]), served_rows=int(out["distribution"].shape[0]))
    # PROVENANCE: loop.predict can only materialise a neural bundle. There is no b5flat22 bundle and no code path
    # in the serving loop that would load one.
    r["bundle_matches_configured_model"] = bool(configured_model in (None, "none") and served_cfg.get("arch") == "none")
    r["provenance_ok"] = bool(r["bundle_matches_configured_model"])
    r["b5flat22_bundle_exists"] = os.path.isdir(os.path.join(BUND, "fill_rate", f"{world}_b5flat22"))
    r["serving_path_can_load_lightgbm"] = bool([m for m in dir(L) if "lgbm" in m.lower() or "lightgbm" in m.lower()])
    results.append(r)
    return r


def main(out_path):
    results = []
    print("STAGE 1.1/1.2 — provenance of the served distribution\n")
    for w in ("v6", "v7"):
        r = check_arrival(w, results)
        print(f"arrival_week {w}: configured '{r['configured']}', fallback {r['configured_fallback']}, "
              f"serve_source -> '{r['serve_source_says']}'")
        print(f"   served distribution_source = {r['distribution_source']!r}; equals h0's distribution byte-for-byte "
              f"= {r['served_equals_h0_distribution']}; ranking still from h4 = {r['ranking_from_h4']}")
        print(f"   drift {r['drift_pp']:+.4f} pp | h0 drift {r['h0_drift_pp']:+.4f} pp | excess {r['excess_pp']:.4f} pp "
              f"| every batch carries drift = {r['batch_has_drift']} and excess = {r['batch_has_excess']}")
        print(f"   serve_source cannot see a drift value = {r['serve_source_cannot_see_drift']}")
        print(f"   PROVENANCE: {'OK' if r['provenance_ok'] else '**MISMATCH**'}\n")
    for w in ("v6", "v7"):
        r = check_fill(w, results)
        print(f"fill_rate {w}: configured model '{r['configured_model']}' (family {r['configured_family']})")
        print(f"   serving path loaded bundle {r['bundle_used']} -> arch={r['bundle_arch']}, depth={r['bundle_depth']}, "
              f"lr={r['bundle_lr']} — a {r['served_family']}")
        print(f"   a '{w}_b5flat22' bundle exists = {r['b5flat22_bundle_exists']}; "
              f"loop can materialise a LightGBM = {r['serving_path_can_load_lightgbm']}")
        print(f"   drift {r['drift_pp']:+.4f} pp | batches {r['n_batches']} | rows {r['served_rows']:,}")
        print(f"   PROVENANCE: {'OK' if r['provenance_ok'] else '**MISMATCH**'}\n")

    bad = [r for r in results if not r["provenance_ok"]]
    verdict = dict(checks=len(results), provenance_ok=len(results) - len(bad),
                   provenance_mismatch=[f"{r['task']}|{r['world']}" for r in bad])
    json.dump(dict(results=results, verdict=verdict), open(out_path, "w"), indent=1, default=float)
    print(f"VERDICT: {verdict['provenance_ok']} of {verdict['checks']} provenance checks pass; "
          f"mismatches: {verdict['provenance_mismatch'] or 'none'}")
    print(f"-> {out_path}")
    return len(bad)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ARTIFACTS, "phase11_smoke.json"))
    sys.exit(0 if main(ap.parse_args().out) == 0 else 1)
