from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from moe_rca.data import build_schema_candidates, canonicalize_entity, discover_cases, load_case, window_around_injection
from moe_rca.moe import HeterogeneousMoE, build_moe_batch
from moe_rca.native_moe import NativeHeterogeneousMoE, build_native_moe_batch
from moe_rca.training import DistillationConfig, LossConfig, moe_loss, target_indices, train_step


def iter_batches(ids: list[str], batch_size: int):
    for i in range(0, len(ids), batch_size):
        yield ids[i:i + batch_size]


def main() -> None:
    p = argparse.ArgumentParser(description="Train heterogeneous RCA MoE")
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--model-type", choices=["native", "statistical"], default="native")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--before-s", type=float, default=300.0)
    p.add_argument("--after-s", type=float, default=300.0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--expert-weight", type=float, default=0.5)
    p.add_argument("--distill-teacher", type=Path, default=None)
    p.add_argument("--distill-weight", type=float, default=0.5)
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    manifest = json.loads(args.split_manifest.read_text())
    assignments = manifest["assignments"]
    refs = {r.case_id: r for r in discover_cases(args.raw_root) if r.case_id in assignments}
    if set(refs) != set(assignments):
        raise SystemExit("split manifest and discovered legal cases differ")

    # GT is experiment supervision only. Candidate constructors never receive this mapping.
    gt = pd.read_parquet(args.raw_root / "cases.parquet", columns=["case", "root_cause_service"])
    gt = gt.set_index("case")["root_cause_service"].to_dict()

    def make_model():
        if args.model_type == "native":
            return NativeHeterogeneousMoE(d_model=128, hidden_dim=128).to(device)
        return HeterogeneousMoE(d_model=64, hidden_dim=96).to(device)

    batch_builder = build_native_moe_batch if args.model_type == "native" else build_moe_batch
    model = make_model()
    teacher = None
    if args.distill_teacher is not None:
        teacher_state = torch.load(args.distill_teacher, map_location=device, weights_only=True)
        model.load_state_dict(teacher_state)
        teacher = make_model()
        teacher.load_state_dict(teacher_state)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_cfg = LossConfig(expert_weight=args.expert_weight)
    distill_cfg = DistillationConfig(weight=args.distill_weight, temperature=args.distill_temperature)

    def materialize(case_ids: list[str]):
        cases, windows, entity_lists = [], [], []
        for case_id in case_ids:
            ref = refs[case_id]
            window = window_around_injection(ref, args.before_s, args.after_s)
            case = load_case(ref, expert_projection=True, window=window)
            _, audited = build_schema_candidates(case.metric_columns, ref.system)
            cases.append(case); windows.append(window); entity_lists.append(audited)
        batch = batch_builder(cases, windows, entity_lists).to(device)
        roots = [canonicalize_entity(gt[c], refs[c].system) for c in case_ids]
        targets = target_indices(batch.entity_names, roots).to(device)
        return batch, targets

    def evaluate(case_ids: list[str], max_batches: int | None):
        model.eval(); totals = []; correct = total = 0
        with torch.no_grad():
            for bi, ids in enumerate(iter_batches(case_ids, args.batch_size)):
                if max_batches is not None and bi >= max_batches: break
                batch, targets = materialize(ids)
                out = model(batch); loss = moe_loss(out, batch, targets, loss_cfg)
                totals.append(float(loss.total)); correct += int((out.scores.argmax(1) == targets).sum()); total += len(ids)
                del batch, targets, out; gc.collect()
        return {"loss": sum(totals) / max(len(totals), 1), "top1": correct / max(total, 1), "cases": total}

    train_ids = sorted(c for c, s in assignments.items() if s == "train")
    val_ids = sorted(c for c, s in assignments.items() if s == "val")
    history = []
    best_val = float("inf"); best_state = None
    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch); rng.shuffle(train_ids)
        train_losses = []; distill_losses = []; distill_terms = 0
        for bi, ids in enumerate(iter_batches(train_ids, args.batch_size)):
            if args.max_train_batches is not None and bi >= args.max_train_batches: break
            batch, targets = materialize(ids)
            step = train_step(
                model, batch, targets, optimizer, config=loss_cfg,
                teacher_model=teacher, distill_config=distill_cfg, grad_clip=5.0,
            )
            train_losses.append(step.loss); distill_losses.append(step.distill_loss); distill_terms += step.distill_terms
            del batch, targets; gc.collect()
        val = evaluate(val_ids, args.max_val_batches)
        train_loss = sum(train_losses) / max(len(train_losses), 1)
        row = {
            "epoch": epoch, "train_loss": train_loss,
            "train_distill_loss": sum(distill_losses) / max(len(distill_losses), 1),
            "train_distill_terms": distill_terms,
            "val_loss": val["loss"], "val_top1": val["top1"], "val_cases": val["cases"],
        }
        history.append(row); print(json.dumps(row), flush=True)
        if val["loss"] < best_val:
            best_val = val["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, args.out_dir / "model.pt")
    report = {
        "device": str(device), "model_type": args.model_type,
        "training_stage": "distillation" if teacher is not None else "supervised",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "window": {"before_s": args.before_s, "after_s": args.after_s},
        "loss": {"expert_weight": args.expert_weight},
        "distillation": {
            "enabled": teacher is not None,
            "teacher_checkpoint": str(args.distill_teacher) if args.distill_teacher else None,
            "weight": args.distill_weight, "temperature": args.distill_temperature,
            "teacher_frozen": teacher is not None,
        },
        "history": history,
        "split_counts": {s: sum(v == s for v in assignments.values()) for s in ("train", "val", "test")},
        "candidate_protocol": "metric_schema_audited_canonical",
    }
    (args.out_dir / "training.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
