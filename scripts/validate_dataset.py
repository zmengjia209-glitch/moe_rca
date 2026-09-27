from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from moe_rca.data import (  # noqa: E402
    CausalWindow,
    build_candidates,
    canonicalize_entity,
    contract_violations,
    discover_cases,
    load_case,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RCAEval Step-1 full dataset validation")
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_checkpoint(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    rows, failures = {}, {}
    if not path.exists():
        return rows, failures
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["kind"] == "row":
            rows[rec["case"]] = rec["data"]
        else:
            failures[rec["case"]] = rec["data"]
    return rows, failures


def append_checkpoint(path: Path, kind: str, case: str, data: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, "case": case, "data": data}) + "\n")
        f.flush()


def pre_gt_pass(raw_root: Path, checkpoint: Path) -> tuple[pd.DataFrame, list[dict]]:
    refs = discover_cases(raw_root)
    row_map, failure_map = load_checkpoint(checkpoint)
    done = set(row_map) | set(failure_map)
    if done:
        print(f"RESUME completed={len(done)}/{len(refs)}", flush=True)

    for i, ref in enumerate(refs, 1):
        if ref.case_id in done:
            continue
        try:
            case = load_case(ref, candidate_projection=True)
            violations = contract_violations(case)
            metric_time = pd.to_numeric(case.metrics["time"], errors="coerce")
            metric_start, metric_end = float(metric_time.min()), float(metric_time.max())
            candidates = None
            if not violations:
                candidates = build_candidates(case, CausalWindow(metric_start, metric_end))
            row = {
                "case": ref.case_id,
                "suite": ref.suite,
                "system": ref.system,
                "inject_time": ref.inject_time_s,
                "metric_start": metric_start,
                "metric_end": metric_end,
                "legal": not violations,
                "violations": list(violations),
                "has_logs": case.logs is not None,
                "has_traces": case.traces is not None,
                "n_metric_channels": len(case.metric_columns),
                "metric_schema_raw": list(candidates.metric_schema_raw) if candidates else [],
                "metric_schema_audited": list(candidates.metric_schema_audited) if candidates else [],
                "telemetry_visible_validation": list(candidates.telemetry_visible) if candidates else [],
            }
            row_map[ref.case_id] = row
            append_checkpoint(checkpoint, "row", ref.case_id, row)
        except Exception as exc:
            failure = {"case": ref.case_id, "error": f"{type(exc).__name__}: {exc}"}
            failure_map[ref.case_id] = failure
            append_checkpoint(checkpoint, "failure", ref.case_id, failure)
        finally:
            if i % 25 == 0:
                gc.collect()
            completed = len(row_map) + len(failure_map)
            if completed % 50 == 0 or completed == len(refs):
                print(f"PRE_GT {completed}/{len(refs)} failures={len(failure_map)}", flush=True)
    order = [ref.case_id for ref in refs]
    rows = [row_map[c] for c in order if c in row_map]
    failures = [failure_map[c] for c in order if c in failure_map]
    return pd.DataFrame(rows), failures


def gt_join(pre: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    # First and only GT read. All pre-GT candidates are already checkpointed.
    gt = pd.read_parquet(
        raw_root / "cases.parquet",
        columns=["case", "root_cause_service", "fault"],
    )
    merged = pre.merge(gt, on="case", how="left", validate="one_to_one")
    merged["gt_canonical"] = merged.apply(
        lambda r: canonicalize_entity(r["root_cause_service"], r["system"]), axis=1
    )
    merged["gt_in_schema_raw"] = merged.apply(
        lambda r: str(r["root_cause_service"]).strip() in set(r["metric_schema_raw"]), axis=1
    )
    merged["gt_in_schema_audited"] = merged.apply(
        lambda r: r["gt_canonical"] in set(r["metric_schema_audited"]), axis=1
    )
    merged["gt_in_visible_validation"] = merged.apply(
        lambda r: r["gt_canonical"] in set(r["telemetry_visible_validation"]), axis=1
    )
    return merged


def summarize(joined: pd.DataFrame, failures: list[dict]) -> dict:
    legal = joined[joined["legal"]]
    excluded = joined[~joined["legal"]]
    return {
        "raw_cases": int(len(joined)),
        "legal_cases": int(len(legal)),
        "excluded_cases": int(len(excluded)),
        "pre_gt_failures": failures,
        "exclusions": excluded[["case", "violations"]].to_dict("records"),
        "legal_gt_in_schema_raw": int(legal["gt_in_schema_raw"].sum()),
        "legal_gt_in_schema_audited": int(legal["gt_in_schema_audited"].sum()),
        "legal_gt_in_visible_validation": int(legal["gt_in_visible_validation"].sum()),
        "legal_with_logs": int(legal["has_logs"].sum()),
        "legal_with_traces": int(legal["has_traces"].sum()),
    }


def assert_acceptance(report: dict) -> None:
    exclusions = report["exclusions"]
    reasons_ok = all(
        row["violations"] == ["inject_time_outside_metric_range"] for row in exclusions
    )
    checks = {
        "raw_cases_735": report["raw_cases"] == 735,
        "legal_cases_733": report["legal_cases"] == 733,
        "no_pre_gt_failures": not report["pre_gt_failures"],
        "two_rule_based_exclusions": report["excluded_cases"] == 2 and reasons_ok,
        "schema_raw_coverage": report["legal_gt_in_schema_raw"] == 733,
        "schema_audited_coverage": report["legal_gt_in_schema_audited"] == 733,
        "visible_validation_coverage": report["legal_gt_in_visible_validation"] == 733,
    }
    report["acceptance"] = checks
    report["pass"] = all(checks.values())
    if not report["pass"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"Step-1 acceptance failed: {failed}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / ".step1_pre_gt_checkpoint.jsonl"
    pre, failures = pre_gt_pass(args.raw_root, checkpoint)
    joined = gt_join(pre, args.raw_root)
    report = summarize(joined, failures)
    try:
        assert_acceptance(report)
    finally:
        out = args.out_dir / "step1_validation.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print(f"REPORT {out}", flush=True)
    checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
