"""Phase 13 P1: the constant-substitution PATH (deviation 72) must be unreachable, statically.

test_no_lateness_substitution.py checks OUTCOMES on stored predictions. This checks the CODE: in
ml/eval/phase7_score.py no value derived from a constant prediction (`np.full(...)` passed to a scorer) may be
written to the key "roc_auc_late". The constant ranker may exist only under a key that says so.

Demonstrated capable of failing: `--falsify` re-inserts the pre-11C line into a copy and requires a violation.
"""
import ast, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "..", "eval", "phase7_score.py")


def violations(src):
    tree = ast.parse(src)
    const_vars = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            if any(isinstance(a, ast.Call) and getattr(a.func, "attr", "") == "full" for a in n.value.args):
                const_vars |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == "roc_auc_late":
                    names = {x.id for x in ast.walk(n.value) if isinstance(x, ast.Name)}
                    if names & const_vars:
                        bad.append(f"line {n.lineno}: roc_auc_late written from constant-fed {sorted(names & const_vars)}")
    return bad


def test_no_substitution_path():
    v = violations(open(TARGET).read())
    assert not v, "\n".join(v)


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        src = open(TARGET).read().replace(
            'out["roc_auc_late_CONSTANT_RANKER"] = aux_rank["roc_auc_late"]',
            'out["roc_auc_late"] = aux_rank["roc_auc_late"]')
        v = violations(src)
        print("FIRES" if v else "DID NOT FIRE", v); sys.exit(0 if v else 1)
    test_no_substitution_path(); print("PASS")
