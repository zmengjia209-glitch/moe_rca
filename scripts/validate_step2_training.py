from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from moe_rca.data import CausalWindow, build_candidates, canonicalize_entity, discover_cases, load_case
from moe_rca.moe import HeterogeneousMoE, build_moe_batch
from moe_rca.training import LossConfig, moe_loss, target_indices, train_step

CASES = (
    "re1ob_adservice_cpu_1",
    "re1tt_ts-auth-service_cpu_1",
    "re2ob_checkoutservice_cpu_2",
    "re2ss_user_loss_1",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()

    refs = {r.case_id: r for r in discover_cases(args.raw_root)}
    cases, windows, entity_lists = [], [], []
    for case_id in CASES:
        ref = refs[case_id]
        case = load_case(ref, expert_projection=True)
        mt = pd.to_numeric(case.metrics["time"], errors="raise")
        window = CausalWindow(
            max(float(mt.min()), ref.inject_time_s - 300.0),
            min(float(mt.max()), ref.inject_time_s + 300.0),
        )
        # Candidate construction is complete before any GT file is opened.
        candidates = build_candidates(case, window)
        cases.append(case)
        windows.append(window)
        entity_lists.append(candidates.metric_schema_audited)

    batch = build_moe_batch(cases, windows, entity_lists)

    # First GT read: labels only, after Step-1 candidates and the MoE batch exist.
    gt = pd.read_parquet(args.raw_root / "cases.parquet", columns=["case", "root_cause_service"])
    gt = gt.set_index("case")
    roots = [canonicalize_entity(gt.loc[c, "root_cause_service"], refs[c].system) for c in CASES]
    targets = target_indices(batch.entity_names, roots)

    torch.manual_seed(7)
    model = HeterogeneousMoE(d_model=64, hidden_dim=96)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    cfg = LossConfig(expert_weight=0.5)

    model.eval()
    with torch.no_grad():
        initial_out = model(batch)
        initial = moe_loss(initial_out, batch, targets, cfg)
    history = []
    for step in range(args.steps):
        result = train_step(model, batch, targets, optimizer, config=cfg, grad_clip=5.0)
        if step in {0, args.steps - 1}:
            history.append({"step": step + 1, **result.__dict__})

    model.eval()
    with torch.no_grad():
        final_out = model(batch)
        final = moe_loss(final_out, batch, targets, cfg)
        pred = final_out.scores.argmax(dim=1)

    checks = {
        "finite_initial_loss": bool(torch.isfinite(initial.total)),
        "finite_final_loss": bool(torch.isfinite(final.total)),
        "loss_decreased": bool(final.total < initial.total),
        "expert_aux_active": bool(final.expert_terms > 0),
        "all_targets_in_step1_candidates": len(targets) == len(CASES),
        "all_four_fit_after_short_run": bool(torch.equal(pred.cpu(), targets.cpu())),
    }
    report = {
        "pass": all(checks.values()),
        "checks": checks,
        "cases": list(CASES),
        "roots": roots,
        "target_indices": targets.tolist(),
        "initial": {
            "total": float(initial.total), "fused": float(initial.fused),
            "expert": float(initial.expert), "expert_terms": initial.expert_terms,
        },
        "final": {
            "total": float(final.total), "fused": float(final.fused),
            "expert": float(final.expert), "expert_terms": final.expert_terms,
        },
        "history": history,
        "loss_config": {"expert_weight": cfg.expert_weight},
        "steps": args.steps,
        "device": "cpu",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
