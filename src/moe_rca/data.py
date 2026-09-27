from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

CASE_RE = re.compile(r"^re(?P<suite>[123])(?P<system>ob|ss|tt)_.+$")
SYSTEM_NAMES = {"ob": "OB", "ss": "SS", "tt": "TT"}


@dataclass(frozen=True)
class CaseRef:
    case_id: str
    path: Path
    suite: str
    system: str
    inject_time_s: int


@dataclass(frozen=True)
class CausalWindow:
    start_s: float
    end_s: float

    def contains(self, time_s: pd.Series) -> pd.Series:
        return time_s.ge(self.start_s) & time_s.le(self.end_s)


@dataclass
class CanonicalCase:
    ref: CaseRef
    metrics: pd.DataFrame
    logs: pd.DataFrame | None
    traces: pd.DataFrame | None

    @property
    def metric_columns(self) -> tuple[str, ...]:
        return tuple(c for c in self.metrics.columns if c != "time")


@dataclass(frozen=True)
class CandidateSets:
    metric_schema_raw: tuple[str, ...]
    metric_schema_audited: tuple[str, ...]
    telemetry_visible: tuple[str, ...]
    metric_visible: tuple[str, ...]
    log_visible: tuple[str, ...]
    trace_visible: tuple[str, ...]


def _read_inject_time(path: Path) -> int:
    text = path.read_text(encoding="utf-8").strip()
    if not text or not re.fullmatch(r"\d+", text):
        raise ValueError(f"invalid inject_time.txt: {path}")
    return int(text)


def discover_cases(raw_root: str | Path) -> list[CaseRef]:
    root = Path(raw_root)
    refs: list[CaseRef] = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        match = CASE_RE.match(path.name)
        if match is None:
            continue
        inject = path / "inject_time.txt"
        metrics = path / "metrics.parquet"
        if not inject.exists() or not metrics.exists():
            continue
        refs.append(
            CaseRef(
                case_id=path.name,
                path=path,
                suite=f"RE{match.group('suite')}",
                system=SYSTEM_NAMES[match.group("system")],
                inject_time_s=_read_inject_time(inject),
            )
        )
    return refs


def canonicalize_entity(entity: str, system: str) -> str:
    entity = str(entity).strip()
    if system == "OB" and entity == "frontendservice":
        return "frontend"
    return entity


def build_schema_candidates(
    metric_columns: Iterable[str], system: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw: set[str] = set()
    for column in metric_columns:
        if column == "time":
            continue
        if "_" not in column:
            raise ValueError(f"metric column has no entity/signal separator: {column!r}")
        entity, _signal = column.rsplit("_", 1)
        if not entity:
            raise ValueError(f"empty metric entity in column: {column!r}")
        raw.add(entity)
    audited = {canonicalize_entity(entity, system) for entity in raw}
    return tuple(sorted(raw)), tuple(sorted(audited))


def build_visible_candidates(
    metric_entities: Iterable[str],
    log_entities: Iterable[str],
    trace_entities: Iterable[str],
) -> tuple[str, ...]:
    return tuple(sorted(set(metric_entities) | set(log_entities) | set(trace_entities)))


def _read_optional_parquet(
    path: Path, columns: list[str] | None = None, filters=None
) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=columns, filters=filters)
    return None if frame.empty else frame


def load_case(
    ref: CaseRef, candidate_projection: bool = False, expert_projection: bool = False,
    window: CausalWindow | None = None,
) -> CanonicalCase:
    if candidate_projection and expert_projection:
        raise ValueError("candidate_projection and expert_projection are mutually exclusive")
    metric_filters = None if window is None else [("time", ">=", window.start_s), ("time", "<=", window.end_s)]
    metrics = pd.read_parquet(ref.path / "metrics.parquet", filters=metric_filters)
    if "time" not in metrics.columns:
        raise ValueError(f"{ref.case_id}: metrics missing time")
    if not metrics["time"].is_monotonic_increasing:
        metrics = metrics.sort_values("time", kind="stable").reset_index(drop=True)

    log_columns = ["timestamp", "container_name"] if candidate_projection else None
    log_filters = None if window is None else [("timestamp", ">=", window.start_s), ("timestamp", "<=", window.end_s)]
    logs = _read_optional_parquet(ref.path / "logs.parquet", columns=log_columns, filters=log_filters)
    if logs is not None:
        required = {"timestamp", "container_name"}
        if not candidate_projection:
            required.add("message")
        missing = required - set(logs.columns)
        if missing:
            raise ValueError(f"{ref.case_id}: logs missing {sorted(missing)}")
        logs = logs.copy()
        logs.insert(0, "time_s", pd.to_numeric(logs["timestamp"], errors="coerce"))
        logs.insert(1, "entity", logs["container_name"].map(lambda x: None if pd.isna(x) else canonicalize_entity(x, ref.system)))
        if not logs["time_s"].is_monotonic_increasing:
            logs = logs.sort_values("time_s", kind="stable").reset_index(drop=True)

    if candidate_projection:
        trace_columns = ["startTimeMillis", "serviceName"]
    elif expert_projection:
        trace_columns = [
            "startTimeMillis", "traceID", "spanID", "serviceName",
            "duration", "statusCode", "parentSpanID", "operationName",
        ]
    else:
        trace_columns = None
    trace_filters = None if window is None else [
        ("startTimeMillis", ">=", int(window.start_s * 1000)),
        ("startTimeMillis", "<=", int(window.end_s * 1000)),
    ]
    traces = _read_optional_parquet(ref.path / "traces.parquet", columns=trace_columns, filters=trace_filters)
    if traces is not None:
        required = {"serviceName", "startTimeMillis"}
        missing = required - set(traces.columns)
        if missing:
            raise ValueError(f"{ref.case_id}: traces missing {sorted(missing)}")
        traces = traces.copy()
        traces.insert(0, "time_s", pd.to_numeric(traces["startTimeMillis"], errors="coerce") / 1e3)
        traces.insert(1, "entity", traces["serviceName"].map(lambda x: None if pd.isna(x) else canonicalize_entity(x, ref.system)))
        if not traces["time_s"].is_monotonic_increasing:
            traces = traces.sort_values("time_s", kind="stable").reset_index(drop=True)

    return CanonicalCase(ref=ref, metrics=metrics, logs=logs, traces=traces)


def _finite_metric_entities(case: CanonicalCase, window: CausalWindow) -> set[str]:
    rows = case.metrics.loc[window.contains(pd.to_numeric(case.metrics["time"], errors="coerce"))]
    entities: set[str] = set()
    for column in case.metric_columns:
        values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        if np.isfinite(values).any():
            entity, _signal = column.rsplit("_", 1)
            entities.add(canonicalize_entity(entity, case.ref.system))
    return entities


def _event_entities(frame: pd.DataFrame | None, window: CausalWindow) -> set[str]:
    if frame is None:
        return set()
    rows = frame.loc[window.contains(pd.to_numeric(frame["time_s"], errors="coerce"))]
    out: set[str] = set()
    for value in rows["entity"].dropna():
        entity = str(value).strip()
        if entity:
            out.add(entity)
    return out


def build_candidates(case: CanonicalCase, window: CausalWindow) -> CandidateSets:
    if window.start_s > window.end_s:
        raise ValueError("causal window start must not exceed end")
    raw, audited = build_schema_candidates(case.metric_columns, case.ref.system)
    metric_visible = _finite_metric_entities(case, window)
    log_visible = _event_entities(case.logs, window)
    trace_visible = _event_entities(case.traces, window)
    visible = build_visible_candidates(metric_visible, log_visible, trace_visible)
    return CandidateSets(
        metric_schema_raw=raw,
        metric_schema_audited=audited,
        telemetry_visible=visible,
        metric_visible=tuple(sorted(metric_visible)),
        log_visible=tuple(sorted(log_visible)),
        trace_visible=tuple(sorted(trace_visible)),
    )


def window_around_injection(
    ref: CaseRef, before_s: float, after_s: float
) -> CausalWindow:
    if before_s < 0 or after_s < 0:
        raise ValueError("window extents must be non-negative")
    return CausalWindow(
        start_s=ref.inject_time_s - before_s,
        end_s=ref.inject_time_s + after_s,
    )


def contract_violations(case: CanonicalCase) -> tuple[str, ...]:
    errors: list[str] = []
    if case.metrics.columns.duplicated().any():
        errors.append("duplicate_metric_columns")
    metric_time = pd.to_numeric(case.metrics["time"], errors="coerce")
    if metric_time.isna().any() or metric_time.empty:
        errors.append("invalid_metric_time")
        return tuple(errors)
    start_s, end_s = float(metric_time.min()), float(metric_time.max())
    if not start_s <= case.ref.inject_time_s <= end_s:
        errors.append("inject_time_outside_metric_range")
    if not metric_time.is_monotonic_increasing:
        errors.append("metric_time_not_monotonic")
    if not case.metric_columns:
        errors.append("no_metric_channels")
    return tuple(errors)
