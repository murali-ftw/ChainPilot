"""Phase 16 S4 -- the backward-compatibility gate on artifact identity. HARD: any mismatch is a STOP.

The Phase 16 axes (encoder_variant, traj_depths, traj_delta, pna_aggregators, low_degree_k) are omitted from every
name and identity when default. This proves it over EVERY stored artifact of Phases 0-15:

  (1) every bundle config.json (fixed-split and backtest): its directory name equals bundle_name(cfg) and its path
      ends in bundle_path_key(cfg), recomputed by the Phase 16 code;
  (2) for every such config, every naming function (config_name, bundle_name, bundle_path_key, pred_stem val/test,
      index_key, marker_name, identity_of) returns EXACTLY what the branch-point module (90a38ed) returns;
  (3) every queue completion marker (*.done carrying an identity): the stored identity equals identity_of recomputed,
      and the filename equals marker_name(identity);
  (4) every prediction-manifest entry: the stored identity equals identity_of(owner config) on the stored keys (the
      manifest predates the Phase 11-13 keys -- that schema difference is pre-existing, not Phase 16's), and the file
      is still the pred_stem of its owner;
  (5) every key of the backtest bundle index equals index_key of the bundle it points at;
  (6) score_name on every config-derived identity is unchanged against the branch point.
Then the gate is shown FIRING: one stored config is given encoder_variant='share_traj', every name must change; revert.
"""
from __future__ import annotations
import os, sys, json, glob, subprocess, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import phase12_common as C
import artifact_identity as AI

BRANCH_POINT = "90a38ed"
ART = C.ART


def old_module():
    src = subprocess.check_output(["git", "-C", C.REPO, "show", f"{BRANCH_POINT}:ml/artifact_identity.py"], text=True)
    m = types.ModuleType("artifact_identity_branch_point")
    exec(compile(src, f"{BRANCH_POINT}:ml/artifact_identity.py", "exec"), m.__dict__)
    return m


def names(M, c):
    return dict(config_name=M.config_name(c), bundle_name=M.bundle_name(c), bundle_path_key=M.bundle_path_key(c),
                pred_val=M.pred_stem(dict(c, origin=c.get("origin")), "val"),
                pred_test=M.pred_stem(dict(c, origin=c.get("origin")), "test"),
                index_key=M.index_key(dict(c, origin=c.get("origin"))), marker_name=M.marker_name(c),
                identity_of=M.identity_of(c))


def score_ident(c, split="test", calib="raw"):
    return {k: c.get(k) for k in AI.SCORE_AXES if k not in ("split", "calib", "arm")} | dict(split=split, calib=calib,
                                                                                              arm=c.get("arch"))


def run(OLD, fire=True):
    fails, n = [], dict(configs=0, name_fns=0, markers=0, manifest=0, index=0, score_names=0)
    cfgs = {}
    for p in sorted(glob.glob(os.path.join(ART, "**", "config.json"), recursive=True)):
        c = json.load(open(p)); d = os.path.dirname(p)
        if "task" not in c or "arch" not in c:
            continue
        cfgs[os.path.realpath(d)] = c
        n["configs"] += 1
        if os.path.basename(d) != AI.bundle_name(c):
            fails.append(("dirname", p, os.path.basename(d), AI.bundle_name(c)))
        if not d.endswith(AI.bundle_path_key(c)):
            fails.append(("path_key", p, AI.bundle_path_key(c)))
        a, b = names(OLD, c), names(AI, c)
        n["name_fns"] += len(a)
        for k in a:
            if a[k] != b[k]:
                fails.append(("name_fn", p, k, a[k], b[k]))
        for split in ("val", "test"):
            for calib in ("raw", "recal"):
                si = score_ident(c, split, calib); n["score_names"] += 1
                if OLD.score_name(si) != AI.score_name(si):
                    fails.append(("score_name", p, si))
    for p in sorted(glob.glob(os.path.join(ART, "**", "*.done"), recursive=True)):
        txt = open(p).read().strip()
        if not txt:
            continue                                     # Phase 12 C3 queue: empty sentinel, its own log-name scheme
        rec = json.loads(txt); ident = rec["identity"]; n["markers"] += 1
        if AI.identity_of(ident) != ident:
            fails.append(("marker_identity", p))
        if os.path.basename(p) != AI.marker_name(ident):
            fails.append(("marker_name", p, AI.marker_name(ident)))
    for mp in sorted(glob.glob(os.path.join(ART, "**", AI.MANIFEST), recursive=True)):
        man = json.load(open(mp))
        for f, e in man.items():
            n["manifest"] += 1
            c = cfgs.get(os.path.realpath(e["owner"]))
            if c is None:
                fails.append(("manifest_owner_missing", f, e["owner"])); continue
            new = AI.identity_of(c)
            if any(new.get(k) != v for k, v in e["identity"].items()) or set(new) - set(OLD.identity_of(c)):
                fails.append(("manifest_identity", f))
            stem = f.replace("RECAL_", "")
            if stem not in (AI.pred_stem(c, "val") + ".npz", AI.pred_stem(c, "test") + ".npz"):
                fails.append(("manifest_stem", f))
    for ip in sorted(glob.glob(os.path.join(ART, "**", "bundle_index.json"), recursive=True)):
        for k, e in json.load(open(ip)).items():
            n["index"] += 1
            c = cfgs.get(os.path.realpath(e["bundle"]))
            if c is None or AI.index_key(c) != k:
                fails.append(("index_key", k))
    out = dict(counts=n, n_fail=len(fails), fails=fails[:50])
    if fire:
        c = next(iter(cfgs.values()))
        before = names(AI, c)
        c2 = dict(c, encoder_variant="share_traj")
        after = names(AI, c2)
        changed = {k: before[k] != after[k] for k in before}
        default = names(AI, dict(c, encoder_variant=AI.INCUMBENT_ENCODER.get(c["arch"])))   # explicit default: no change
        reverted = names(AI, {k: v for k, v in c2.items() if k != "encoder_variant"})
        out["firing"] = dict(config=AI.bundle_path_key(c), nondefault_changes_every_name=all(changed.values()),
                             changed=changed, before=before["config_name"], after=after["config_name"],
                             explicit_default_unchanged=default == before, reverted_equal=reverted == before)
        assert out["firing"]["nondefault_changes_every_name"] and out["firing"]["explicit_default_unchanged"] \
            and out["firing"]["reverted_equal"], out["firing"]
    return out


def main():
    st = C.require_clean()
    OLD = old_module()
    out = run(OLD)
    out["stamp"] = st; out["branch_point"] = BRANCH_POINT
    print(json.dumps({k: out[k] for k in ("counts", "n_fail", "fails")}, indent=1, default=str)[:3000])
    print(json.dumps(out["firing"], indent=1)[:1500])
    print(C.dump(out, "phase16_identity_gate.json"))
    if out["n_fail"]:
        raise SystemExit("S4 GATE FAILED -- STOP")


if __name__ == "__main__":
    main()
