from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import re

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .data import CanonicalCase, CausalWindow, canonicalize_entity

MODALITIES = ("metric", "log", "trace")
METRIC_SIGNALS = (
    "cpu", "mem", "socket", "workload", "latency-50",
    "latency-90", "diskio", "error", "load", "latency",
)
LOG_FEATURE_DIM = 9
TRACE_FEATURE_DIM = 11


@dataclass(frozen=True)
class MoEBatch:
    entity_names: tuple[tuple[str, ...], ...]
    entity_mask: Tensor                 # [B, E]
    metric_values: Tensor               # [B, C, T]
    metric_value_mask: Tensor           # [B, C, T]
    metric_channel_mask: Tensor         # [B, C]
    metric_channel_features: Tensor     # [B, C, 10]
    metric_channel_entity: Tensor       # [B, C], -1 for padding
    log_features: Tensor                # [B, E, 9]
    trace_features: Tensor              # [B, E, 11]
    modality_mask: Tensor               # [B, E, 3]

    def to(self, device: torch.device | str) -> "MoEBatch":
        values = {}
        for name, value in self.__dict__.items():
            values[name] = value.to(device) if isinstance(value, Tensor) else value
        return MoEBatch(**values)


@dataclass(frozen=True)
class ExpertOutput:
    embedding: Tensor                   # [B, E, D]
    logit: Tensor                       # [B, E]
    confidence: Tensor                  # [B, E]
    mask: Tensor                        # [B, E]


@dataclass(frozen=True)
class MoEOutput:
    scores: Tensor                      # [B, E]
    gate_weights: Tensor                # [B, E, 3]
    fused_embedding: Tensor             # [B, E, D]
    expert_logits: Tensor               # [B, E, 3]
    expert_confidence: Tensor           # [B, E, 3]
    modality_mask: Tensor               # [B, E, 3]


def _signal_feature(signal: str) -> np.ndarray:
    try:
        idx = METRIC_SIGNALS.index(signal.lower())
    except ValueError as exc:
        raise ValueError(f"unknown metric signal suffix {signal!r}") from exc
    out = np.zeros(len(METRIC_SIGNALS), dtype=np.float32)
    out[idx] = 1.0
    return out


def _window_rows(frame: pd.DataFrame, time_col: str, window: CausalWindow) -> pd.DataFrame:
    time = pd.to_numeric(frame[time_col], errors="coerce")
    return frame.loc[window.contains(time)]


def _log_features(case: CanonicalCase, entities: Sequence[str], window: CausalWindow) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(entities), LOG_FEATURE_DIM), dtype=np.float32)
    present = np.zeros(len(entities), dtype=bool)
    if case.logs is None or case.logs.empty:
        return out, present
    rows = _window_rows(case.logs, "time_s", window)
    if rows.empty:
        return out, present
    duration = max(float(window.end_s - window.start_s), 1.0)
    entity_pos = {e: i for i, e in enumerate(entities)}
    for entity, group in rows.dropna(subset=["entity"]).groupby("entity", sort=False):
        pos = entity_pos.get(str(entity))
        if pos is None or group.empty:
            continue
        present[pos] = True
        n = len(group)
        messages = group["message"].fillna("").astype(str) if "message" in group else pd.Series([""] * n)
        lower = messages.str.lower()
        lengths = messages.str.len().to_numpy(dtype=np.float32)
        out[pos] = np.array([
            math.log1p(n), math.log1p(n / duration),
            float(lower.str.contains(r"error|exception", regex=True).mean()),
            float(lower.str.contains(r"fail|fatal", regex=True).mean()),
            float(lower.str.contains(r"warn", regex=True).mean()),
            float(lengths.mean() / 256.0), float(lengths.std() / 256.0) if n > 1 else 0.0,
            float(min(lengths.max() / 512.0, 4.0)), float(messages.nunique(dropna=False) / n),
        ], dtype=np.float32)
    return out, present


def _trace_features(case: CanonicalCase, entities: Sequence[str], window: CausalWindow) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(entities), TRACE_FEATURE_DIM), dtype=np.float32)
    present = np.zeros(len(entities), dtype=bool)
    if case.traces is None or case.traces.empty:
        return out, present
    rows = _window_rows(case.traces, "time_s", window)
    if rows.empty:
        return out, present
    duration_s = max(float(window.end_s - window.start_s), 1.0)
    entity_pos = {e: i for i, e in enumerate(entities)}
    for entity, group in rows.dropna(subset=["entity"]).groupby("entity", sort=False):
        pos = entity_pos.get(str(entity))
        if pos is None or group.empty:
            continue
        present[pos] = True
        n = len(group)
        dur = pd.to_numeric(group.get("duration", pd.Series(np.zeros(n))), errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
        status = pd.to_numeric(group.get("statusCode", pd.Series(np.nan, index=group.index)), errors="coerce")
        parent = group.get("parentSpanID", pd.Series([None] * n, index=group.index))
        operation = group.get("operationName", pd.Series([None] * n, index=group.index))
        out[pos] = np.array([
            math.log1p(n), math.log1p(n / duration_s),
            float((status.fillna(0) != 0).mean()), float(status.isna().mean()),
            math.log1p(float(np.mean(dur))), math.log1p(float(np.std(dur))),
            math.log1p(float(np.quantile(dur, .50))), math.log1p(float(np.quantile(dur, .90))),
            math.log1p(float(np.quantile(dur, .99))), float(parent.isna().mean()),
            float(operation.nunique(dropna=True) / n),
        ], dtype=np.float32)
    return out, present


def build_moe_batch(
    cases: Sequence[CanonicalCase],
    windows: Sequence[CausalWindow],
    entity_lists: Sequence[Sequence[str]],
) -> MoEBatch:
    """Build one heterogeneous batch. Candidate entity lists are supplied by Step 1."""
    if not (len(cases) == len(windows) == len(entity_lists)) or not cases:
        raise ValueError("cases, windows and entity_lists must have equal non-zero length")
    B = len(cases)
    E = max(len(x) for x in entity_lists)
    if E == 0:
        raise ValueError("each batch requires at least one candidate entity")

    metric_payloads = []
    max_c = max_t = 0
    log_payloads, trace_payloads = [], []
    for case, window, names in zip(cases, windows, entity_lists):
        entities = tuple(names)
        if len(set(entities)) != len(entities):
            raise ValueError(f"duplicate candidates in {case.ref.case_id}")
        entity_pos = {e: i for i, e in enumerate(entities)}
        mrows = _window_rows(case.metrics, "time", window)
        if mrows.empty:
            raise ValueError(f"{case.ref.case_id}: no metric samples inside requested window")
        channels = []
        for column in case.metric_columns:
            raw_entity, signal = column.rsplit("_", 1)
            entity = canonicalize_entity(raw_entity, case.ref.system)
            if entity not in entity_pos:
                continue
            values = pd.to_numeric(mrows[column], errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)
            channels.append((values, entity_pos[entity], _signal_feature(signal)))
        max_c = max(max_c, len(channels))
        max_t = max(max_t, len(mrows))
        metric_payloads.append(channels)
        log_payloads.append(_log_features(case, entities, window))
        trace_payloads.append(_trace_features(case, entities, window))

    metric_values = torch.zeros((B, max_c, max_t), dtype=torch.float32)
    metric_value_mask = torch.zeros((B, max_c, max_t), dtype=torch.bool)
    metric_channel_mask = torch.zeros((B, max_c), dtype=torch.bool)
    metric_channel_features = torch.zeros((B, max_c, len(METRIC_SIGNALS)), dtype=torch.float32)
    metric_channel_entity = torch.full((B, max_c), -1, dtype=torch.long)
    entity_mask = torch.zeros((B, E), dtype=torch.bool)
    log_features = torch.zeros((B, E, LOG_FEATURE_DIM), dtype=torch.float32)
    trace_features = torch.zeros((B, E, TRACE_FEATURE_DIM), dtype=torch.float32)
    modality_mask = torch.zeros((B, E, len(MODALITIES)), dtype=torch.bool)

    for b, (names, channels, log_pack, trace_pack) in enumerate(zip(entity_lists, metric_payloads, log_payloads, trace_payloads)):
        entity_mask[b, :len(names)] = True
        for c, (values, eidx, semantic) in enumerate(channels):
            finite = np.isfinite(values)
            if finite.any():
                metric_values[b, c, :len(values)] = torch.from_numpy(np.nan_to_num(values, nan=0.0))
                metric_value_mask[b, c, :len(values)] = torch.from_numpy(finite)
                metric_channel_mask[b, c] = True
                metric_channel_features[b, c] = torch.from_numpy(semantic)
                metric_channel_entity[b, c] = eidx
                modality_mask[b, eidx, 0] = True
        lf, lm = log_pack
        tf, tm = trace_pack
        log_features[b, :len(names)] = torch.from_numpy(lf)
        trace_features[b, :len(names)] = torch.from_numpy(tf)
        modality_mask[b, :len(names), 1] = torch.from_numpy(lm)
        modality_mask[b, :len(names), 2] = torch.from_numpy(tm)

    return MoEBatch(
        entity_names=tuple(tuple(x) for x in entity_lists), entity_mask=entity_mask,
        metric_values=metric_values, metric_value_mask=metric_value_mask,
        metric_channel_mask=metric_channel_mask, metric_channel_features=metric_channel_features,
        metric_channel_entity=metric_channel_entity, log_features=log_features,
        trace_features=trace_features, modality_mask=modality_mask,
    )


class DenseExpert(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, d_model: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, d_model), nn.GELU())
        self.logit_head = nn.Linear(d_model, 1)
        self.conf_head = nn.Linear(d_model, 1)

    def forward(self, x: Tensor, mask: Tensor) -> ExpertOutput:
        h = self.encoder(x)
        h = h * mask.unsqueeze(-1)
        logit = self.logit_head(h).squeeze(-1) * mask
        confidence = torch.sigmoid(self.conf_head(h).squeeze(-1)) * mask
        return ExpertOutput(h, logit, confidence, mask)


class MetricExpert(nn.Module):
    STAT_DIM = 8

    def __init__(self, hidden_dim: int, d_model: int):
        super().__init__()
        self.channel_encoder = nn.Sequential(
            nn.Linear(len(METRIC_SIGNALS) + self.STAT_DIM, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, d_model), nn.GELU(),
        )
        self.channel_weight = nn.Linear(d_model, 1)
        self.entity_refine = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.logit_head = nn.Linear(d_model, 1)
        self.conf_head = nn.Linear(d_model, 1)

    @staticmethod
    def _stats(x: Tensor, mask: Tensor) -> Tensor:
        count = mask.sum(-1).clamp_min(1)
        m = mask.to(x.dtype)
        mean = (x * m).sum(-1) / count
        centered = (x - mean.unsqueeze(-1)) * m
        var = centered.square().sum(-1) / count
        std = torch.sqrt(var + 1e-6)
        z = centered / std.unsqueeze(-1)
        T = x.shape[-1]
        idx = torch.arange(T, device=x.device).view(1, 1, T)
        first_idx = torch.where(mask, idx, T).amin(-1).clamp_max(T - 1)
        last_idx = torch.where(mask, idx, -1).amax(-1).clamp_min(0)
        first = z.gather(-1, first_idx.unsqueeze(-1)).squeeze(-1)
        last = z.gather(-1, last_idx.unsqueeze(-1)).squeeze(-1)
        min_z = z.masked_fill(~mask, float("inf")).amin(-1)
        max_z = z.masked_fill(~mask, float("-inf")).amax(-1)
        valid = mask.any(-1)
        min_z = torch.where(valid, min_z, torch.zeros_like(min_z))
        max_z = torch.where(valid, max_z, torch.zeros_like(max_z))
        t = idx.to(x.dtype) / max(T - 1, 1)
        tmean = (t * m).sum(-1) / count
        cov = ((t - tmean.unsqueeze(-1)) * z * m).sum(-1) / count
        tvar = (((t - tmean.unsqueeze(-1)) * m).square()).sum(-1) / count
        slope = cov / (tvar + 1e-6)
        finite_frac = count.to(x.dtype) / max(T, 1)
        scale = torch.log1p(std)
        return torch.stack((first, last, last - first, min_z, max_z, slope, finite_frac, scale), dim=-1)

    def forward(self, batch: MoEBatch) -> ExpertOutput:
        stats = self._stats(batch.metric_values, batch.metric_value_mask)
        channel_h = self.channel_encoder(torch.cat((stats, batch.metric_channel_features), dim=-1))
        channel_valid = batch.metric_channel_mask & (batch.metric_channel_entity >= 0)
        weight = torch.nn.functional.softplus(self.channel_weight(channel_h).squeeze(-1)) * channel_valid
        B, _, D = channel_h.shape
        E = batch.entity_mask.shape[1]
        idx = batch.metric_channel_entity.clamp_min(0)
        summed = torch.zeros((B, E, D), dtype=channel_h.dtype, device=channel_h.device)
        denom = torch.zeros((B, E), dtype=channel_h.dtype, device=channel_h.device)
        summed.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, D), channel_h * weight.unsqueeze(-1))
        denom.scatter_add_(1, idx, weight)
        mask = denom > 0
        h = self.entity_refine(summed / denom.clamp_min(1e-6).unsqueeze(-1)) * mask.unsqueeze(-1)
        logit = self.logit_head(h).squeeze(-1) * mask
        confidence = torch.sigmoid(self.conf_head(h).squeeze(-1)) * mask
        return ExpertOutput(h, logit, confidence, mask)


class HeterogeneousMoE(nn.Module):
    """Three modality experts with entity-wise masked dynamic routing."""

    def __init__(self, d_model: int = 64, hidden_dim: int = 96):
        super().__init__()
        self.metric_expert = MetricExpert(hidden_dim, d_model)
        self.log_expert = DenseExpert(LOG_FEATURE_DIM, hidden_dim, d_model)
        self.trace_expert = DenseExpert(TRACE_FEATURE_DIM, hidden_dim, d_model)
        self.modality_embedding = nn.Parameter(torch.randn(len(MODALITIES), d_model) * 0.02)
        self.router = nn.Sequential(nn.Linear(d_model + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.fusion_head = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, batch: MoEBatch) -> MoEOutput:
        metric = self.metric_expert(batch)
        log = self.log_expert(batch.log_features, batch.modality_mask[..., 1])
        trace = self.trace_expert(batch.trace_features, batch.modality_mask[..., 2])
        experts = (metric, log, trace)
        embedding = torch.stack([x.embedding for x in experts], dim=2)
        logits = torch.stack([x.logit for x in experts], dim=2)
        confidence = torch.stack([x.confidence for x in experts], dim=2)
        mask = torch.stack([x.mask for x in experts], dim=2) & batch.modality_mask
        router_h = embedding + self.modality_embedding.view(1, 1, len(MODALITIES), -1)
        gate_input = torch.cat((router_h, logits.unsqueeze(-1), confidence.unsqueeze(-1)), dim=-1)
        gate_logits = self.router(gate_input).squeeze(-1)
        any_modality = mask.any(dim=-1)
        safe_logits = gate_logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(safe_logits, dim=-1) * mask
        weights = torch.where(any_modality.unsqueeze(-1), weights / weights.sum(-1, keepdim=True).clamp_min(1e-9), torch.zeros_like(weights))
        fused = (embedding * weights.unsqueeze(-1)).sum(dim=2)
        scores = (logits * weights).sum(dim=-1) + self.fusion_head(fused).squeeze(-1)
        scores = scores.masked_fill(~batch.entity_mask, float("-inf"))
        return MoEOutput(scores, weights, fused, logits, confidence, mask)
