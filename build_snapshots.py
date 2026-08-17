"""
Precompute and cache all 15 graph snapshots as .pt files (HeteroData).
Run this once; train.py then just loads the cached files (~9s total to
build vs re-parsing CSVs every run).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
from data_loading import load_all
from graph_builder import IDMaps, build_snapshot, attach_labels, finalize

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "snapshots")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dfs = load_all()
    ids = IDMaps(dfs)
    snaps = dfs["graph_snapshots"].sort_values("t0").reset_index(drop=True)
    t_all = time.time()
    for i, row in snaps.iterrows():
        t0 = row["t0"]
        data, order_map, ship_map = build_snapshot(dfs, ids, t0)
        labels_this = dfs["training_labels"][dfs["training_labels"]["snapshot_id"] == row["id"]]
        data = attach_labels(data, ids, order_map, ship_map, labels_this, dfs["inventory"])
        data = finalize(data)
        path = os.path.join(OUT_DIR, f"snap_{i:02d}.pt")
        torch.save(data, path)
        counts = {nt: data[nt].x.shape[0] for nt in data.node_types}
        print(f"[{i:2d}] t0={t0.date()}  {counts}")
    print(f"built {len(snaps)} snapshots in {time.time() - t_all:.1f}s -> {OUT_DIR}")

if __name__ == "__main__":
    main()
