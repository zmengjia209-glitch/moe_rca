from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .moe import MODALITIES, MoEBatch, MoEOutput


@dataclass(frozen=True)
class LossConfig:
    expert_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.expert_weight < 0:
            raise ValueError("expert_weight must be non-negative")


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    fused: Tensor
    expert: Tensor
    expert_terms: int
    valid_cases: int


def target_indices(entity_names: Sequence[Sequence[str]], roots: Sequence[str]) -> Tensor:
    """Map GT names to Step-1 candidate positions. This function never builds candidates."""
    if len(entity_names) != len(roots):
        raise ValueError("entity_names and roots must have equal length")
    targets = []
    for names, root in zip(entity_names, roots):
        try:
            targets.append(tuple(names).index(root))
        except ValueError as exc:
            raise ValueError(f"GT entity {root!r} is absent from supplied Step-1 candidates") from exc
    return torch.tensor(targets, dtype=torch.long)


def _masked_entity_ce(logits: Tensor, available: Tensor, target: Tensor) -> Tensor:
    masked = logits.masked_fill(~available, float("-inf"))
    return F.cross_entropy(masked, target)


def moe_loss(
    output: MoEOutput,
    batch: MoEBatch,
    targets: Tensor,
    config: LossConfig = LossConfig(),
) -> LossOutput:
    if targets.ndim != 1 or targets.shape[0] != output.scores.shape[0]:
        raise ValueError("targets must have shape [B]")
    targets = targets.to(output.scores.device)
    if not torch.all(batch.entity_mask.gather(1, targets[:, None]).squeeze(1)):
        raise ValueError("every target must point to a real candidate entity")

    fused = F.cross_entropy(output.scores, targets)
    expert_losses: list[Tensor] = []
    for m in range(len(MODALITIES)):
        modality_available = output.modality_mask[..., m]
        target_visible = modality_available.gather(1, targets[:, None]).squeeze(1)
        informative = target_visible & modality_available.sum(dim=1).ge(2)
        for b in torch.nonzero(informative, as_tuple=False).flatten():
            expert_losses.append(
                _masked_entity_ce(
                    output.expert_logits[b : b + 1, :, m],
                    modality_available[b : b + 1],
                    targets[b : b + 1],
                )
            )
    expert = torch.stack(expert_losses).mean() if expert_losses else fused.new_zeros(())
    total = fused + config.expert_weight * expert
    return LossOutput(total, fused, expert, len(expert_losses), int(targets.numel()))


@dataclass(frozen=True)
class DistillationConfig:
    weight: float = 0.5
    temperature: float = 2.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("distillation weight must be non-negative")
        if self.temperature <= 0:
            raise ValueError("distillation temperature must be positive")


@dataclass(frozen=True)
class TrainStepOutput:
    loss: float
    fused_loss: float
    expert_loss: float
    expert_terms: int
    distill_loss: float = 0.0
    distill_terms: int = 0
    grad_norm: float = 0.0


def fused_to_expert_distillation(
    student: MoEOutput,
    teacher: MoEOutput,
    batch,
    targets: Tensor,
    config: DistillationConfig = DistillationConfig(),
) -> tuple[Tensor, int]:
    """Distill frozen fused-teacher rankings into each available modality expert."""
    temperature = config.temperature
    targets = targets.to(student.scores.device)
    terms: list[Tensor] = []
    for m in range(len(MODALITIES)):
        available = student.modality_mask[..., m] & teacher.modality_mask[..., m]
        target_visible = available.gather(1, targets[:, None]).squeeze(1)
        informative = target_visible & available.sum(dim=1).ge(2)
        for b in torch.nonzero(informative, as_tuple=False).flatten():
            support = available[b]
            teacher_prob = torch.softmax(teacher.scores[b, support].detach() / temperature, dim=0)
            student_log_prob = torch.log_softmax(student.expert_logits[b, support, m] / temperature, dim=0)
            terms.append(F.kl_div(student_log_prob, teacher_prob, reduction="sum") * (temperature ** 2))
    if not terms:
        return student.scores.new_zeros(()), 0
    return torch.stack(terms).mean(), len(terms)


def train_step(
    model: nn.Module,
    batch,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    config: LossConfig = LossConfig(),
    teacher_model: nn.Module | None = None,
    distill_config: DistillationConfig = DistillationConfig(),
    grad_clip: float | None = 5.0,
) -> TrainStepOutput:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    supervised = moe_loss(output, batch, targets, config)
    distill = output.scores.new_zeros(())
    distill_terms = 0
    if teacher_model is not None and distill_config.weight > 0:
        teacher_model.eval()
        with torch.no_grad():
            teacher_output = teacher_model(batch)
        distill, distill_terms = fused_to_expert_distillation(
            output, teacher_output, batch, targets, distill_config
        )
    total = supervised.total + distill_config.weight * distill
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite training loss")
    total.backward()
    if grad_clip is None:
        norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
        grad_norm = torch.stack(norms).norm(2) if norms else total.new_zeros(())
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return TrainStepOutput(
        loss=float(total.detach()),
        fused_loss=float(supervised.fused.detach()),
        expert_loss=float(supervised.expert.detach()),
        expert_terms=supervised.expert_terms,
        distill_loss=float(distill.detach()),
        distill_terms=distill_terms,
        grad_norm=float(grad_norm.detach()),
    )


def grouped_scenario_split(
    metadata,
    legal_cases: set[str],
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 17,
) -> dict[str, str]:
    """Split whole (dataset, root service, fault) groups; never split repetitions."""
    import random

    if not (0 < train_fraction < 1 and 0 <= val_fraction < 1):
        raise ValueError("invalid split fractions")
    test_fraction = 1.0 - train_fraction - val_fraction
    if test_fraction <= 0:
        raise ValueError("train_fraction + val_fraction must be < 1")
    required = {"case", "dataset", "root_cause_service", "fault"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"split metadata missing columns: {sorted(missing)}")

    frame = metadata[metadata["case"].isin(legal_cases)].copy()
    if set(frame["case"]) != set(legal_cases):
        absent = sorted(set(legal_cases) - set(frame["case"]))
        raise ValueError(f"legal cases missing from metadata: {absent[:5]}")
    frame["scenario_group"] = (
        frame["dataset"].astype(str) + "|" +
        frame["root_cause_service"].astype(str) + "|" + frame["fault"].astype(str)
    )

    case_to_split: dict[str, str] = {}
    rng = random.Random(seed)
    for dataset, part in frame.groupby("dataset", sort=True):
        groups = sorted(part["scenario_group"].unique().tolist())
        rng.shuffle(groups)
        n = len(groups)
        n_train = max(1, round(n * train_fraction))
        n_val = max(1, round(n * val_fraction)) if n >= 3 else 0
        if n_train + n_val >= n:
            n_train = max(1, n - n_val - 1)
        assignment = {}
        for i, group in enumerate(groups):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            assignment[group] = split
        for row in part[["case", "scenario_group"]].itertuples(index=False):
            case_to_split[str(row.case)] = assignment[str(row.scenario_group)]
    return case_to_split
