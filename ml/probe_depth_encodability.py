#!/usr/bin/env python3
"""
At which encoder depth does co-membership stop being present in SHARE's Supplier
representation?

The Stage 1 hard stop is phrased as a claim about the representation: *"that would mean
SHARE's Supplier representation lacks the structural information this task needs regardless
of how retrieval is trained."* That claim is testable directly, and it should be, because
"the representation lacks it" and "the representation has it at a depth retrieval does not
read" imply completely different next steps.

Transformer 2 retrieves over `z_impact` -- Variant A's bounded-residual blend, which under
the Markov prior for impact sits essentially at `h^4`. This script freezes a trained model
and fits a fresh probe head under the same InfoNCE objective on each of `h^0..h^4` and on
`z_impact`. A depth whose probe drops well below the chance loss `ln(N-1)` still CARRIES
co-membership; a depth whose probe stalls there has lost it.

**The permuted-label null is the point of the design.** A probe head with enough capacity
can memorise any labelling of 2,000 distinct inputs, so a low loss on its own proves
nothing. Every depth is therefore probed twice -- once on the real `HP_GROUPS` membership
and once on a seeded shuffle of the group assignment that preserves the group-size
distribution exactly -- and only the GAP between them is evidence. This is the same
permutation-null discipline `db/generate_dataset.py::_sep` applies to its own separability
checks, and for the same reason.

The probe is a diagnostic on the dataset and the trained encoder, in the tradition of
`ml/extract_hidden_state.py`: it reads privileged `HP_GROUPS` to score, and nothing it
produces is ever fed to a model that gets reported on.

    python3 ml/probe_depth_encodability.py --variants B,D --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data.loader import load_bundles, split_bundles  # noqa: E402
from ml.models.contrastive import GroupSupervision, build_supervision, infonce_loss  # noqa: E402
from ml.retrieval_metrics import merge, score_retrieval  # noqa: E402
from ml.train import train_model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO, "ml", ".cache")


def shuffled_supervision(sup: GroupSupervision, seed: int) -> GroupSupervision:
    """Reassign group ids across the SAME supplier rows, preserving every group's size.

    Shuffling membership rather than resampling it keeps the probe's task identical in
    difficulty -- same number of groups, same sizes, same anchor count -- so the only
    thing removed is the correspondence to the generator's real structure.
    """
    g = sup.group_id.clone()
    member_rows = torch.nonzero(g >= 0, as_tuple=True)[0]
    perm = member_rows[torch.randperm(len(member_rows),
                                      generator=torch.Generator().manual_seed(seed))]
    out = torch.full_like(g, -1)
    out[perm] = g[member_rows]
    anchor = torch.zeros_like(sup.anchor_mask)
    held = torch.zeros_like(sup.held_out_mask)
    anchor[perm] = sup.anchor_mask[member_rows]
    held[perm] = sup.held_out_mask[member_rows]
    return GroupSupervision(out, anchor, held, sup.n_supervised_groups, sup.n_held_out_groups)


def fit_probe(feats: torch.Tensor, sup: GroupSupervision, steps: int, device: str,
              seed: int = 0) -> tuple[float, torch.Tensor]:
    """Fit an MLP head on a FROZEN representation under InfoNCE; return the final loss and
    the resulting normalised embedding."""
    torch.manual_seed(seed)
    mlp = nn.Sequential(nn.Linear(feats.size(-1), 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    last = float("nan")
    for _ in range(steps):
        opt.zero_grad()
        loss = infonce_loss(mlp(feats), sup)
        loss.backward()
        nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        opt.step()
        last = float(loss.detach())
    with torch.no_grad():
        z = torch.nn.functional.normalize(mlp(feats), dim=-1)
    return last, z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default=os.path.join(REPO, "db", "csv_mid"))
    ap.add_argument("--hidden-dir", default=os.path.join(REPO, "out", "hidden_mid"))
    ap.add_argument("--variants", default="B,D")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--device", default="mps", choices=("cpu", "mps"))
    ap.add_argument("--out", default=os.path.join(REPO, "out", "depth_probe.json"))
    args = ap.parse_args()

    results = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            t_run = time.time()
            d = os.path.join(args.csv_dir, f"v{variant}_seed{seed}")
            bundles, _ = load_bundles(d, cache_dir=CACHE_DIR)
            tr, va, te = split_bundles(bundles)
            sup_ids = bundles[0].data["Supplier"].node_id
            groups = json.load(open(os.path.join(args.hidden_dir, f"{variant}_{seed}.json"))
                               )["hp_groups"]
            sup = build_supervision(groups, sup_ids, held_out_frac=0.3, seed=seed).to(args.device)
            sup_null = shuffled_supervision(sup, seed=1000 + seed).to(args.device)
            n = len(sup_ids)
            chance = float(np.log(n - 1))

            trd = [b.to(args.device) for b in tr]
            vad = [b.to(args.device) for b in va]
            ted = [b.to(args.device) for b in te]
            # The PLAIN arm -- the representation as it actually exists under the
            # finalized stack, with no contrastive pressure applied to it.
            r = train_model("rgcn_attn_variant_a_transformer2", trd, vad, epochs=args.epochs,
                            seed=0, device=args.device, hidden=128, num_bases=10)
            model = r["model"]
            model.eval()

            b = ted[0]
            with torch.no_grad():
                layers = model.encoder(b.data.x_dict, b.data.edge_index_dict)
                model(b.data.x_dict, b.data.edge_index_dict)
                z_imp = model._last_z_impact.detach()
            reps = {"raw_features": b.data["Supplier"].x.float().to(args.device)}
            for li, layer in enumerate(layers):
                reps[f"h^{li}"] = layer["Supplier"].detach()
            reps["z_impact"] = z_imp

            idx = {s: i for i, s in enumerate(sup_ids)}
            members = {gi: [idx[m] for m in g["members"] if m in idx]
                       for gi, g in enumerate(groups)}
            type_by_group = {gi: g["type"] for gi, g in enumerate(groups)}
            held = sup.held_out_mask.cpu().numpy()
            subset = {gi: ("held_out" if held[members[gi][0]] else "supervised")
                      for gi, g in enumerate(groups)
                      if g["type"] in ("A", "B") and members[gi]}

            rec = {"variant": variant, "seed": seed, "n_suppliers": n,
                   "chance_infonce": chance, "depths": {}}
            for name, feats in reps.items():
                obs, z = fit_probe(feats, sup, args.steps, args.device)
                null, _ = fit_probe(feats, sup_null, args.steps, args.device)
                s = (z @ z.T).float().cpu().numpy()
                np.fill_diagonal(s, -np.inf)
                pool = np.argsort(-s, axis=1)[:, :args.top_k]
                m = merge([score_retrieval(s, pool, members, type_by_group,
                                           top_k=args.top_k, subset_by_group=subset)])
                rec["depths"][name] = {
                    "infonce": obs, "infonce_permuted_null": null,
                    "gap_vs_null": null - obs,
                    "retrieval": {k: v for k, v in m.items()
                                  if k not in ("n_nodes", "top_k", "chance_discovery_rate")},
                }
                print(f"  v{variant} s{seed} {name:>13}: infonce {obs:.3f} "
                      f"(permuted null {null:.3f}, gap {null-obs:+.3f}, chance {chance:.3f})"
                      + (f"  held-out A R@k={m['A|held_out']['recall_at_k']:.4f}"
                         if "A|held_out" in m else ""), flush=True)
            results.append(rec)
            print(f"  v{variant} s{seed}: {time.time()-t_run:.0f}s", flush=True)

            for x in bundles:
                x.to("cpu")
            del bundles, trd, vad, ted, model, r
            if args.device == "mps":
                torch.mps.empty_cache()

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)

    print("\n## Co-membership encodability by encoder depth "
          "(InfoNCE after a frozen-representation probe; lower = more encodable)\n")
    print("| depth | InfoNCE | permuted-label null | gap | chance | held-out A/B R@K | chance |")
    print("|---|---|---|---|---|---|---|")
    names = list(results[0]["depths"]) if results else []
    for name in names:
        obs = np.mean([r["depths"][name]["infonce"] for r in results])
        null = np.mean([r["depths"][name]["infonce_permuted_null"] for r in results])
        ch = np.mean([r["chance_infonce"] for r in results])
        ho, hc = [], []
        for r in results:
            for key in ("A|held_out", "B|held_out"):
                v = r["depths"][name]["retrieval"].get(key)
                if v:
                    ho.append(v["recall_at_k"])
                    hc.append(v["chance"]["recall_at_k"])
        print(f"| {name} | {obs:.3f} | {null:.3f} | {null-obs:+.3f} | {ch:.3f} | "
              f"{np.mean(ho):.4f} | {np.mean(hc):.4f} |" if ho else
              f"| {name} | {obs:.3f} | {null:.3f} | {null-obs:+.3f} | {ch:.3f} | -- | -- |")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
