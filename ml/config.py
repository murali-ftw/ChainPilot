"""Guide step 0.3 — which world each activity uses, and the frozen split."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORLDS = {
    "v6": os.path.join(REPO, "db", "gen_v6", "seed_1001"),
    "v7": os.path.join(REPO, "db", "gen_v7", "seed_1001"),
}
WORLDS_SEED2 = {
    "v6": os.path.join(REPO, "db", "gen_v6", "seed_1002"),
    "v7": os.path.join(REPO, "db", "gen_v7", "seed_1002"),
}
DEFAULT_WORLD = "v7"

# Frozen across runs 5-7 and every phase here. Do not change.
SPLIT = {"train_end": "2023-12-31", "val_end": "2024-12-31"}
# §16 fits on the modelling window, where the censored tail is 1.19x not 1.47x
FIT_WINDOW = ("2019-01-01", "2025-12-31")

ARTIFACTS = os.path.join(REPO, "ml", "artifacts")
CACHE = os.path.join(ARTIFACTS, "cache")

SPEC_TABLES = 49
EXPECT = {                      # measured, Stage A; identical masters across worlds
    "channels": 16072, "suppliers": 420, "parts": 620, "plants": 7,
    "supplier_groups": 84, "snapshots": 83, "weeks": 535,
}
