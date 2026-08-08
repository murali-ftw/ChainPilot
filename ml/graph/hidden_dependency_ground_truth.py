"""
Ground truth for the planted hidden-dependency ("H_POLYMER") validation scenario --
`db/README.md`'s "The hidden-dependency scenario (Transformer 2's target)" and
`db/generate_dataset.py`'s `H_POLYMER` set. Four suppliers, in four different
countries, supplying different (non-polymer-exclusive) component types, share an
UNMODELED upstream polymer plant -- no row, no edge, no column connects them in the
schema; only the correlated stress/degradation it causes is observable. This is the
one dependency-recovery scenario Transformer 2 (Step 7, Claim 3) exists to be tested
against (`docs/10_AI_ML_Documentation.md` §8.3, §9.4).

**Provenance -- how these 4 supplier IDs were obtained, not re-derived by re-running
the generator.** `db/generate_dataset.py`'s `H_POLYMER` selection is fully
deterministic given the `components` list's generation order (first-generated
supplier, by component insertion order, for each of 4 distinct NON-polymer
component_types, `db/generate_dataset.py` lines ~124-137) -- no random draw is used
for the selection itself. Re-running the generator was NOT an option here (it would
regenerate/reload the entire live dataset this whole project's history has run
against, a destructive operation far outside this round's scope). Instead, these 4
IDs were reconstructed by replicating the SAME deterministic algorithm directly
against `db/csv/components.csv` (whose row order is the generator's own literal
insertion order -- confirmed: row 2 is "Fastener 000", row 3 "Fastener 001", matching
the `ci`-indexed naming the generator writes), a read-only operation that touches
nothing in the live database. The result was cross-checked against the live
`suppliers`/`components` tables and matches `db/README.md`'s own published table
(4 distinct countries: USA, India, China, Vietnam; each member's own component types
collectively span all 5 categories; India's member specifically lacks `fastener`,
matching the README's table exactly) -- an exact match, not a guess.
"""

from __future__ import annotations

# The 4 planted H_POLYMER supplier IDs (v3, 800-supplier dataset). Verified live
# (2026-08-08) against `suppliers.country` and `components.component_type`:
#   ea344b5b-a6fa-5c52-a82d-6aca1e325dfd  USA-Poly Supply 10       (USA)
#   9a6dab5e-6ec0-5a4a-bf93-80f994e2582e  IND-Precision Supply 112 (India)
#   afa15ef0-34ab-58ff-ad47-8060502e5f84  CHI-Precision Supply 00  (China)
#   b3288dd6-e3cd-5ec3-a951-39986285505b  VIE-Alloy Supply 01      (Vietnam)
H_POLYMER_SUPPLIER_IDS: tuple[str, ...] = (
    "ea344b5b-a6fa-5c52-a82d-6aca1e325dfd",
    "9a6dab5e-6ec0-5a4a-bf93-80f994e2582e",
    "afa15ef0-34ab-58ff-ad47-8060502e5f84",
    "b3288dd6-e3cd-5ec3-a951-39986285505b",
)


def h_polymer_pairs() -> list[tuple[str, str]]:
    """All C(4,2)=6 unordered pairs among the 4 planted members, canonically
    ordered (a < b) to match `hidden_dependency_links.supplier_a_id <
    supplier_b_id`'s own ordering convention."""
    ids = sorted(H_POLYMER_SUPPLIER_IDS)
    return [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
