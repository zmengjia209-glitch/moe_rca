from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import hashlib
import math
import re

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .data import CanonicalCase, CausalWindow, canonicalize_entity
from .moe import ExpertOutput, METRIC_SIGNALS, MODALITIES, MoEOutput

LOG_VOCAB_SIZE = 8192
LOG_TOKEN_DIM = 64
TRACE_NODE_DIM = 8
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]*|[0-9A-Fa-f-]{8,}|\d+(?:\.\d+)?|[^\s]")


@dataclass(frozen=True)
class NativeMoEBatch:
    entity_names: tuple[tuple[str, ...], ...]
    entity_mask: Tensor                    # [B,E]
    metric_values: Tensor                  # [B,C,T]
    metric_value_mask: Tensor              # [B,C,T]
    metric_channel_mask: Tensor            # [B,C]
    metric_channel_features: Tensor        # [B,C,10]
    metric_channel_entity: Tensor          # [B,C]
    log_token_ids: Tensor                  # [B,E,L,K]
    log_token_mask: Tensor                 # [B,E,L,K]
    log_event_mask: Tensor                 # [B,E,L]
    log_event_time: Tensor                 # [B,E,L]
    trace_node_features: Tensor            # [B,E,8]
    trace_adjacency: Tensor                # [B,E,E], directed row-normalized calls
    modality_mask: Tensor                  # [B,E,3]

    def to(self, device: torch.device | str) -> "NativeMoEBatch":
        values = {}
        for name, value in self.__dict__.items():
            values[name] = value.to(device) if isinstance(value, Tensor) else value
        return NativeMoEBatch(**values)


def _signal_feature(signal: str) -> np.ndarray:
    idx = METRIC_SIGNALS.index(signal.lower())
    out = np.zeros(len(METRIC_SIGNALS), dtype=np.float32)
    out[idx] = 1.0
    return out


def _window_rows(frame: pd.DataFrame, time_col: str, window: CausalWindow) -> pd.DataFrame:
    time = pd.to_numeric(frame[time_col], errors="coerce")
    return frame.loc[window.contains(time)]


def _normalize_token(token: str) -> str:
    token = token.lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return "<num>"
    if len(token) >= 8 and re.fullmatch(r"[0-9a-f-]+", token):
        return "<id>"
    return token[:64]


def _token_id(token: str, vocab_size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return 1 + int.from_bytes(digest, "little") % (vocab_size - 1)


def _tokenize(message: str, max_tokens: int, vocab_size: int) -> list[int]:
    tokens = [_normalize_token(x) for x in _TOKEN_RE.findall(str(message))]
    if not tokens:
        tokens = ["<empty>"]
    return [_token_id(x, vocab_size) for x in tokens[:max_tokens]]


def _uniform_indices(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, cap).round().astype(np.int64))


def _log_payload(
    case: CanonicalCase,
    entities: Sequence[str],
    window: CausalWindow,
    *,
    max_events: int,
    max_tokens: int,
    vocab_size: int,
):
    E = len(entities)
    ids = np.zeros((E, max_events, max_tokens), dtype=np.int64)
    token_mask = np.zeros_like(ids, dtype=bool)
    event_mask = np.zeros((E, max_events), dtype=bool)
    event_time = np.zeros((E, max_events), dtype=np.float32)
    present = np.zeros(E, dtype=bool)
    if case.logs is None or case.logs.empty:
        return ids, token_mask, event_mask, event_time, present
    rows = _window_rows(case.logs, "time_s", window)
    entity_pos = {e: i for i, e in enumerate(entities)}
    duration = max(float(window.end_s - window.start_s), 1.0)
    for entity, group in rows.dropna(subset=["entity"]).groupby("entity", sort=False):
        pos = entity_pos.get(str(entity))
        if pos is None or group.empty:
            continue
        group = group.sort_values("time_s", kind="stable")
        take = _uniform_indices(len(group), max_events)
        group = group.iloc[take]
        present[pos] = True
        for j, row in enumerate(group.itertuples(index=False)):
            message = getattr(row, "message", "")
            toks = _tokenize(message, max_tokens, vocab_size)
            ids[pos, j, :len(toks)] = toks
            token_mask[pos, j, :len(toks)] = True
            event_mask[pos, j] = True
            t = float(getattr(row, "time_s"))
            event_time[pos, j] = float(np.clip((t - window.start_s) / duration, 0.0, 1.0))
    return ids, token_mask, event_mask, event_time, present


def _trace_payload(case: CanonicalCase, entities: Sequence[str], window: CausalWindow):
    E = len(entities)
    node = np.zeros((E, TRACE_NODE_DIM), dtype=np.float32)
    adj = np.zeros((E, E), dtype=np.float32)
    present = np.zeros(E, dtype=bool)
    if case.traces is None or case.traces.empty:
        return node, adj, present
    rows = _window_rows(case.traces, "time_s", window)
    if rows.empty:
        return node, adj, present
    entity_pos = {e: i for i, e in enumerate(entities)}
    for entity, group in rows.dropna(subset=["entity"]).groupby("entity", sort=False):
        pos = entity_pos.get(str(entity))
        if pos is None or group.empty:
            continue
        present[pos] = True
        n = len(group)
        dur = pd.to_numeric(group.get("duration", pd.Series(np.zeros(n), index=group.index)), errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
        status = pd.to_numeric(group.get("statusCode", pd.Series(np.nan, index=group.index)), errors="coerce")
        parent = group.get("parentSpanID", pd.Series([None] * n, index=group.index)).fillna("").astype(str)
        operation = group.get("operationName", pd.Series([None] * n, index=group.index))
        node[pos] = np.array([
            math.log1p(n), float((status.fillna(0) != 0).mean()),
            math.log1p(float(np.mean(dur))), math.log1p(float(np.quantile(dur, .50))),
            math.log1p(float(np.quantile(dur, .90))), math.log1p(float(np.quantile(dur, .99))),
            float(operation.nunique(dropna=True) / n), float((parent == "").mean()),
        ], dtype=np.float32)

    required = {"traceID", "spanID", "parentSpanID", "entity"}
    if required.issubset(rows.columns):
        span_map: dict[tuple[str, str], int] = {}
        trace_ids = rows["traceID"].astype(str).to_numpy()
        span_ids = rows["spanID"].astype(str).to_numpy()
        parent_ids = rows["parentSpanID"].fillna("").astype(str).to_numpy()
        child_entities = rows["entity"].fillna("").astype(str).to_numpy()
        for tr, sp, ent in zip(trace_ids, span_ids, child_entities):
            child = entity_pos.get(ent)
            if child is not None and sp:
                span_map[(tr, sp)] = child
        for tr, parent_sp, ent in zip(trace_ids, parent_ids, child_entities):
            child = entity_pos.get(ent)
            if child is None or not parent_sp:
                continue
            parent = span_map.get((tr, parent_sp))
            if parent is not None and parent != child:
                adj[parent, child] += 1.0
        adj = np.log1p(adj)
        row_sum = adj.sum(axis=1, keepdims=True)
        adj = np.divide(adj, row_sum, out=np.zeros_like(adj), where=row_sum > 0)
    return node, adj, present


def build_native_moe_batch(
    cases: Sequence[CanonicalCase],
    windows: Sequence[CausalWindow],
    entity_lists: Sequence[Sequence[str]],
    *,
    max_log_events: int = 64,
    max_log_tokens: int = 24,
    log_vocab_size: int = LOG_VOCAB_SIZE,
) -> NativeMoEBatch:
    if not (len(cases) == len(windows) == len(entity_lists)) or not cases:
        raise ValueError("cases, windows and entity_lists must have equal non-zero length")
    if max_log_events <= 0 or max_log_tokens <= 0 or log_vocab_size < 2:
        raise ValueError("invalid native batch limits")
    B = len(cases)
    E = max(len(x) for x in entity_lists)
    if E == 0:
        raise ValueError("each batch requires at least one candidate entity")

    metric_payloads, log_payloads, trace_payloads = [], [], []
    max_c = max_t = 0
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
        metric_payloads.append(channels)
        max_c = max(max_c, len(channels)); max_t = max(max_t, len(mrows))
        log_payloads.append(_log_payload(case, entities, window, max_events=max_log_events, max_tokens=max_log_tokens, vocab_size=log_vocab_size))
        trace_payloads.append(_trace_payload(case, entities, window))

    metric_values = torch.zeros((B, max_c, max_t), dtype=torch.float32)
    metric_value_mask = torch.zeros((B, max_c, max_t), dtype=torch.bool)
    metric_channel_mask = torch.zeros((B, max_c), dtype=torch.bool)
    metric_channel_features = torch.zeros((B, max_c, len(METRIC_SIGNALS)), dtype=torch.float32)
    metric_channel_entity = torch.full((B, max_c), -1, dtype=torch.long)
    entity_mask = torch.zeros((B, E), dtype=torch.bool)
    log_token_ids = torch.zeros((B, E, max_log_events, max_log_tokens), dtype=torch.long)
    log_token_mask = torch.zeros_like(log_token_ids, dtype=torch.bool)
    log_event_mask = torch.zeros((B, E, max_log_events), dtype=torch.bool)
    log_event_time = torch.zeros((B, E, max_log_events), dtype=torch.float32)
    trace_node_features = torch.zeros((B, E, TRACE_NODE_DIM), dtype=torch.float32)
    trace_adjacency = torch.zeros((B, E, E), dtype=torch.float32)
    modality_mask = torch.zeros((B, E, len(MODALITIES)), dtype=torch.bool)

    for b, (names, channels, lp, tp) in enumerate(zip(entity_lists, metric_payloads, log_payloads, trace_payloads)):
        n_ent = len(names); entity_mask[b, :n_ent] = True
        for c, (values, eidx, semantic) in enumerate(channels):
            finite = np.isfinite(values)
            if finite.any():
                metric_values[b, c, :len(values)] = torch.from_numpy(np.nan_to_num(values, nan=0.0))
                metric_value_mask[b, c, :len(values)] = torch.from_numpy(finite)
                metric_channel_mask[b, c] = True
                metric_channel_features[b, c] = torch.from_numpy(semantic)
                metric_channel_entity[b, c] = eidx
                modality_mask[b, eidx, 0] = True
        ids, tm, em, et, lpresent = lp
        log_token_ids[b, :n_ent] = torch.from_numpy(ids)
        log_token_mask[b, :n_ent] = torch.from_numpy(tm)
        log_event_mask[b, :n_ent] = torch.from_numpy(em)
        log_event_time[b, :n_ent] = torch.from_numpy(et)
        modality_mask[b, :n_ent, 1] = torch.from_numpy(lpresent)
        node, adj, tpresent = tp
        trace_node_features[b, :n_ent] = torch.from_numpy(node)
        trace_adjacency[b, :n_ent, :n_ent] = torch.from_numpy(adj)
        modality_mask[b, :n_ent, 2] = torch.from_numpy(tpresent)

    return NativeMoEBatch(
        entity_names=tuple(tuple(x) for x in entity_lists), entity_mask=entity_mask,
        metric_values=metric_values, metric_value_mask=metric_value_mask,
        metric_channel_mask=metric_channel_mask, metric_channel_features=metric_channel_features,
        metric_channel_entity=metric_channel_entity,
        log_token_ids=log_token_ids, log_token_mask=log_token_mask,
        log_event_mask=log_event_mask, log_event_time=log_event_time,
        trace_node_features=trace_node_features, trace_adjacency=trace_adjacency,
        modality_mask=modality_mask,
    )


class TemporalMetricExpert(nn.Module):
    def __init__(self, hidden_dim: int, d_model: int):
        super().__init__()
        temporal_dim = max(32, d_model // 4)
        self.temporal = nn.Sequential(
            nn.Conv1d(1, temporal_dim, 5, padding=2), nn.GELU(),
            nn.Conv1d(temporal_dim, temporal_dim, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(temporal_dim, temporal_dim, 3, padding=4, dilation=4), nn.GELU(),
        )
        self.time_score = nn.Conv1d(temporal_dim, 1, 1)
        self.channel_proj = nn.Sequential(
            nn.Linear(temporal_dim + len(METRIC_SIGNALS), d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.channel_weight = nn.Linear(d_model, 1)
        self.entity_refine = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model), nn.LayerNorm(d_model)
        )
        self.logit_head = nn.Linear(d_model, 1); self.conf_head = nn.Linear(d_model, 1)

    def forward(self, batch: NativeMoEBatch) -> ExpertOutput:
        x = batch.metric_values
        value_mask = batch.metric_value_mask & batch.metric_channel_mask.unsqueeze(-1)
        channel_valid = batch.metric_channel_mask & (batch.metric_channel_entity >= 0)
        B, C, T = x.shape
        flat_valid = channel_valid.reshape(-1)
        flat_x = x.reshape(B * C, T)[flat_valid]
        flat_mask = value_mask.reshape(B * C, T)[flat_valid]
        m = flat_mask.to(flat_x.dtype)
        count = m.sum(-1).clamp_min(1.0)
        mean = (flat_x * m).sum(-1) / count
        var = ((flat_x - mean.unsqueeze(-1)).square() * m).sum(-1) / count
        z = ((flat_x - mean.unsqueeze(-1)) / torch.sqrt(var + 1e-6).unsqueeze(-1)) * m
        feat = self.temporal(z.unsqueeze(1))
        time_logits = self.time_score(feat).squeeze(1).masked_fill(~flat_mask, -1e9)
        att = torch.softmax(time_logits, dim=-1) * m
        att = att / att.sum(-1, keepdim=True).clamp_min(1e-9)
        temporal = (feat * att.unsqueeze(1)).sum(-1)
        semantic = batch.metric_channel_features.reshape(B * C, -1)[flat_valid]
        valid_h = self.channel_proj(torch.cat((temporal, semantic), dim=-1))
        channel_h = x.new_zeros((B * C, valid_h.shape[-1]))
        channel_h[flat_valid] = valid_h
        channel_h = channel_h.reshape(B, C, -1)
        weight = torch.nn.functional.softplus(self.channel_weight(channel_h).squeeze(-1)) * channel_valid
        D = channel_h.shape[-1]; E = batch.entity_mask.shape[1]
        idx = batch.metric_channel_entity.clamp_min(0)
        summed = x.new_zeros((B, E, D)); denom = x.new_zeros((B, E))
        summed.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, D), channel_h * weight.unsqueeze(-1))
        denom.scatter_add_(1, idx, weight)
        emask = denom > 0
        h = self.entity_refine(summed / denom.clamp_min(1e-6).unsqueeze(-1)) * emask.unsqueeze(-1)
        return ExpertOutput(
            h,
            self.logit_head(h).squeeze(-1) * emask,
            torch.sigmoid(self.conf_head(h).squeeze(-1)) * emask,
            emask,
        )

class SemanticLogExpert(nn.Module):
    def __init__(self, d_model: int, vocab_size: int = LOG_VOCAB_SIZE, token_dim: int = LOG_TOKEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.event_proj = nn.Sequential(nn.Linear(token_dim, d_model), nn.GELU())
        self.time_proj = nn.Linear(1, d_model, bias=False)
        hidden = d_model // 2
        self.gru = nn.GRU(d_model, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(d_model, 1)
        self.refine = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.logit_head = nn.Linear(d_model, 1); self.conf_head = nn.Linear(d_model, 1)

    def forward(self, batch: NativeMoEBatch) -> ExpertOutput:
        ids = batch.log_token_ids; tmask = batch.log_token_mask
        tok = self.embedding(ids)
        denom = tmask.sum(-1, keepdim=True).clamp_min(1).to(tok.dtype)
        event = (tok * tmask.unsqueeze(-1)).sum(-2) / denom
        event = self.event_proj(event) + self.time_proj(batch.log_event_time.unsqueeze(-1))
        B, E, L, D = event.shape
        flat = event.reshape(B * E, L, D)
        lengths = batch.log_event_mask.sum(-1).reshape(-1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(flat, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=L)
        seq = seq.reshape(B, E, L, D)
        emask = batch.log_event_mask
        logits = self.attn(seq).squeeze(-1).masked_fill(~emask, -1e9)
        att = torch.softmax(logits, dim=-1) * emask
        att = att / att.sum(-1, keepdim=True).clamp_min(1e-9)
        mask = emask.any(-1)
        h = self.refine((seq * att.unsqueeze(-1)).sum(-2)) * mask.unsqueeze(-1)
        return ExpertOutput(h, self.logit_head(h).squeeze(-1) * mask, torch.sigmoid(self.conf_head(h).squeeze(-1)) * mask, mask)


class TraceGraphExpert(nn.Module):
    def __init__(self, hidden_dim: int, d_model: int, layers: int = 2):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(TRACE_NODE_DIM, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model), nn.LayerNorm(d_model))
        self.updates = nn.ModuleList([nn.Linear(d_model * 3, d_model) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.logit_head = nn.Linear(d_model, 1); self.conf_head = nn.Linear(d_model, 1)

    def forward(self, batch: NativeMoEBatch) -> ExpertOutput:
        mask = batch.modality_mask[..., 2]
        h = self.node_encoder(batch.trace_node_features) * mask.unsqueeze(-1)
        adj = batch.trace_adjacency
        for linear, norm in zip(self.updates, self.norms):
            outgoing = torch.bmm(adj, h)
            incoming = torch.bmm(adj.transpose(1, 2), h)
            update = torch.nn.functional.gelu(linear(torch.cat((h, incoming, outgoing), dim=-1)))
            h = norm(h + update) * mask.unsqueeze(-1)
        return ExpertOutput(h, self.logit_head(h).squeeze(-1) * mask, torch.sigmoid(self.conf_head(h).squeeze(-1)) * mask, mask)


class NativeHeterogeneousMoE(nn.Module):
    """Modality-native experts with the same entity-wise Router contract."""
    def __init__(self, d_model: int = 128, hidden_dim: int = 128, log_vocab_size: int = LOG_VOCAB_SIZE):
        super().__init__()
        self.metric_expert = TemporalMetricExpert(hidden_dim, d_model)
        self.log_expert = SemanticLogExpert(d_model, log_vocab_size)
        self.trace_expert = TraceGraphExpert(hidden_dim, d_model, layers=2)
        self.modality_embedding = nn.Parameter(torch.randn(len(MODALITIES), d_model) * 0.02)
        self.router = nn.Sequential(nn.Linear(d_model + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.fusion_head = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, batch: NativeMoEBatch) -> MoEOutput:
        experts = (self.metric_expert(batch), self.log_expert(batch), self.trace_expert(batch))
        embedding = torch.stack([x.embedding for x in experts], dim=2)
        logits = torch.stack([x.logit for x in experts], dim=2)
        confidence = torch.stack([x.confidence for x in experts], dim=2)
        mask = torch.stack([x.mask for x in experts], dim=2) & batch.modality_mask
        router_h = embedding + self.modality_embedding.view(1, 1, len(MODALITIES), -1)
        gate_input = torch.cat((router_h, logits.unsqueeze(-1), confidence.unsqueeze(-1)), dim=-1)
        gate_logits = self.router(gate_input).squeeze(-1)
        any_modality = mask.any(-1)
        weights = torch.softmax(gate_logits.masked_fill(~mask, -1e9), dim=-1) * mask
        weights = torch.where(any_modality.unsqueeze(-1), weights / weights.sum(-1, keepdim=True).clamp_min(1e-9), torch.zeros_like(weights))
        fused = (embedding * weights.unsqueeze(-1)).sum(dim=2)
        scores = (logits * weights).sum(-1) + self.fusion_head(fused).squeeze(-1)
        scores = scores.masked_fill(~batch.entity_mask, float("-inf"))
        return MoEOutput(scores, weights, fused, logits, confidence, mask)
