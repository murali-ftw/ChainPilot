import sys, glob, torch
sys.path.insert(0, "/home/claude/work/pipeline")
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from model import HeteroGAT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- load cached snapshots ----------------
paths = sorted(glob.glob("/home/claude/work/snapshots/snap_*.pt"))
snapshots = [torch.load(p, weights_only=False) for p in paths]
print(f"{len(snapshots)} snapshots loaded")

# Chronological split -- NEVER random across snapshots, since later months carry
# the extended 2025 disruption events and the hidden-dependency check is anchored
# at a specific date; a random split would let the model see the future.
train_snaps = snapshots[0:10]   # Jul 2024 - Apr 2025
val_snaps   = snapshots[10:12]  # May - Jun 2025
test_snaps  = snapshots[12:15]  # Jul - Sep 2025

metadata = snapshots[0].metadata()
in_channels = {nt: snapshots[0][nt].x.shape[1] for nt in snapshots[0].node_types}

model = HeteroGAT(metadata, in_channels, hidden_channels=64, num_layers=2, heads=4).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)

TASK_NODE = {"delay": "shipment", "shortage": "inventory", "impact": "supplier"}

def pos_weight_for(snaps, node_type):
    ys = torch.cat([s[node_type].y for s in snaps])
    labeled = ys[~torch.isnan(ys)]
    pos = (labeled == 1).sum().item()
    neg = (labeled == 0).sum().item()
    return torch.tensor(neg / max(pos, 1), dtype=torch.float)

pos_weights = {nt: pos_weight_for(train_snaps, nt) for nt in ["shipment", "inventory", "supplier"]}
print("pos_weights:", pos_weights)

def run_epoch(snaps, train: bool):
    model.train(train)
    total_loss = 0.0
    all_logits = {nt: [] for nt in ["shipment", "inventory", "supplier"]}
    all_targets = {nt: [] for nt in ["shipment", "inventory", "supplier"]}
    for data in snaps:
        data = data.to(DEVICE)
        if train:
            opt.zero_grad()
        logits, _ = model(data.x_dict, data.edge_index_dict)
        loss = 0.0
        for nt in ["shipment", "inventory", "supplier"]:
            y = data[nt].y
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                continue
            l = nn.functional.binary_cross_entropy_with_logits(
                logits[nt][mask], y[mask], pos_weight=pos_weights[nt].to(DEVICE))
            loss = loss + l
            all_logits[nt].append(logits[nt][mask].detach().cpu())
            all_targets[nt].append(y[mask].detach().cpu())
        if train:
            loss.backward()
            opt.step()
        total_loss += float(loss)

    metrics = {}
    for nt in ["shipment", "inventory", "supplier"]:
        if not all_logits[nt]:
            continue
        logit = torch.cat(all_logits[nt]).numpy()
        target = torch.cat(all_targets[nt]).numpy()
        if len(set(target.tolist())) < 2:
            continue
        metrics[nt] = {
            "auroc": roc_auc_score(target, logit),
            "auprc": average_precision_score(target, logit),
            "n": len(target), "pos": int(target.sum()),
        }
    return total_loss / len(snaps), metrics

NAME = {"shipment": "delay", "inventory": "shortage", "supplier": "impact"}

import os
N_EPOCHS = int(os.environ.get("N_EPOCHS", 18))
best_val = -1
for epoch in range(1, N_EPOCHS + 1):
    train_loss, train_m = run_epoch(train_snaps, train=True)
    with torch.no_grad():
        val_loss, val_m = run_epoch(val_snaps, train=False)
    val_score = sum(m["auroc"] for m in val_m.values()) / max(len(val_m), 1)
    tag = " *" if val_score > best_val else ""
    if val_score > best_val:
        best_val = val_score
        torch.save(model.state_dict(), "/home/claude/work/best_model.pt")
    vs = "  ".join(f"{NAME[nt]} AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} (n={m['n']},pos={m['pos']})"
                    for nt, m in val_m.items())
    print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  {vs}{tag}", flush=True)

print("\n=== Test (held-out, chronologically last 3 snapshots) ===")
model.load_state_dict(torch.load("/home/claude/work/best_model.pt"))
with torch.no_grad():
    test_loss, test_m = run_epoch(test_snaps, train=False)
for nt, m in test_m.items():
    print(f"{NAME[nt]:10s} AUROC={m['auroc']:.3f}  AUPRC={m['auprc']:.3f}  n={m['n']}  pos={m['pos']}")
