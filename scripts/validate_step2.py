from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from moe_rca.data import CausalWindow, build_schema_candidates, discover_cases, load_case
from moe_rca.moe import HeterogeneousMoE, MODALITIES, build_moe_batch
from moe_rca.native_moe import NativeHeterogeneousMoE, build_native_moe_batch

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
    p.add_argument("--model-type", choices=["native", "statistical"], default="native")
    args = p.parse_args()

    refs = {x.case_id: x for x in discover_cases(args.raw_root)}
    missing = sorted(set(CASES) - set(refs))
    if missing:
        raise SystemExit(f"missing validation cases: {missing}")

    cases, windows, entity_lists = [], [], []
    per_case = []
    for case_id in CASES:
        ref = refs[case_id]
        window = CausalWindow(ref.inject_time_s - 300.0, ref.inject_time_s + 300.0)
        case = load_case(ref, expert_projection=True, window=window)
        _, audited = build_schema_candidates(case.metric_columns, ref.system)
        cases.append(case); windows.append(window); entity_lists.append(audited)
        per_case.append({
            "case": case_id, "system": ref.system, "entities": len(audited),
            "metric_channels": len(case.metric_columns),
            "has_logs": case.logs is not None, "has_traces": case.traces is not None,
            "window": [window.start_s, window.end_s],
        })

    torch.manual_seed(0)
    if args.model_type == "native":
        batch = build_native_moe_batch(cases, windows, entity_lists)
        model = NativeHeterogeneousMoE(d_model=128, hidden_dim=128).eval()
    else:
        batch = build_moe_batch(cases, windows, entity_lists)
        model = HeterogeneousMoE(d_model=64, hidden_dim=96).eval()
    with torch.no_grad():
        out = model(batch)

    candidate_scores = out.scores[batch.entity_mask]
    any_modality = out.modality_mask.any(-1)
    available_sums = out.gate_weights.sum(-1)[any_modality]
    unavailable_weights = out.gate_weights.masked_select(~out.modality_mask)
    checks = {
        "scores_finite": bool(torch.isfinite(candidate_scores).all()),
        "unavailable_gate_zero": bool(torch.equal(unavailable_weights, torch.zeros_like(unavailable_weights))),
        "available_gate_sums_one": bool(torch.allclose(available_sums, torch.ones_like(available_sums), atol=1e-6)),
        "padded_scores_neg_inf": bool(torch.isneginf(out.scores[~batch.entity_mask]).all()),
        "all_three_modalities_exercised": bool(out.modality_mask.any(dim=(0, 1)).all()),
    }
    report = {
        "pass": all(checks.values()), "checks": checks, "modalities": list(MODALITIES),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_type": args.model_type,
        "batch_shapes": {
            "entity_mask": list(batch.entity_mask.shape),
            "metric_values": list(batch.metric_values.shape),
            "metric_channel_features": list(batch.metric_channel_features.shape),
            "log": list((batch.log_token_ids if args.model_type == "native" else batch.log_features).shape),
            "trace": list((batch.trace_adjacency if args.model_type == "native" else batch.trace_features).shape),
            "gate_weights": list(out.gate_weights.shape),
            "scores": list(out.scores.shape),
        },
        "cases": per_case,
        "available_entity_modalities": {
            name: int(out.modality_mask[..., i].sum()) for i, name in enumerate(MODALITIES)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
