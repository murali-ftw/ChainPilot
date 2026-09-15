"""Phase 8 2b — bands() must never silently return empty.

Phase 7 found `phase7_score.bands()` dropped every metric: the scorer returns tuples and the filter tested for lists.
These tests fail if that regresses, for scores as the scorer returns them (tuples) and as they come back from JSON (lists).

  venv/bin/python -m unittest ml/tests/test_phase7_bands.py
"""
import os, sys, json, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
import phase7_score as PS


def _scores(as_json):
    S = {}
    for s, (c, e) in zip((7, 17, 27), ((0.01, 0.10), (0.02, 0.12), (0.04, 0.11))):
        S[f"v6|fill_rate|lgbm22_id_s{s}"] = {"crps_exact": (c, c - 0.001, c + 0.001), "ece20": (e, e - 0.01, e + 0.01),
                                            "n_test": (36000,), "_file": "x.npz"}
    S["v6|fill_rate|naive_global_cdf"] = {"crps_exact": (0.5, 0.4, 0.6)}           # no seed suffix: not a band
    return json.loads(json.dumps(S)) if as_json else S


class BandsNeverEmpty(unittest.TestCase):
    def check(self, as_json):
        B = PS.bands(_scores(as_json))
        self.assertEqual(list(B), ["v6|fill_rate|lgbm22_id"], "bands() grouped the wrong entries")
        band = B["v6|fill_rate|lgbm22_id"]
        self.assertTrue(band, "bands() returned an empty metric dict -- the tuple/list bug is back")
        self.assertEqual(set(band), {"crps_exact", "ece20"})
        self.assertEqual(band["crps_exact"]["n"], 3)
        self.assertAlmostEqual(band["crps_exact"]["spread"], 0.03)
        self.assertAlmostEqual(band["ece20"]["mean"], 0.11)

    def test_tuples_from_the_scorer(self):
        self.check(as_json=False)

    def test_lists_from_json(self):
        self.check(as_json=True)


if __name__ == "__main__":
    unittest.main()
