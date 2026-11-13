from __future__ import annotations
import torch
import torch.nn.functional as F
from .constants import SAFETY_COST


def ordinal_loss(logits: torch.Tensor, target: torch.Tensor):
    """Ordered classes 0<1<2; cumulative targets [y>0,y>1]."""
    thresholds = torch.arange(2, device=target.device)[None, :]
    labels = (target[:, None] > thresholds).float()
    return F.binary_cross_entropy_with_logits(logits, labels)


def expected_safety_cost(freshness_logits: torch.Tensor, target: torch.Tensor):
    matrix = freshness_logits.new_tensor(SAFETY_COST)
    probabilities = F.softmax(freshness_logits, dim=1)
    return (probabilities * matrix[target]).sum(dim=1).mean()


def diversity_penalty(expert_logits: torch.Tensor):
    """Average cosine similarity of per-expert batch-mean class distributions."""
    if expert_logits.shape[1] < 2:
        return expert_logits.new_zeros(())
    batch_mean = F.softmax(expert_logits, dim=-1).mean(dim=0)
    unit = F.normalize(batch_mean, dim=1)
    cosine = unit @ unit.T
    pairs = torch.triu_indices(cosine.shape[0], cosine.shape[1], offset=1, device=cosine.device)
    return cosine[pairs[0], pairs[1]].mean()


def balanced_weights(targets: list[int] | torch.Tensor, n_classes: int, device):
    target = torch.as_tensor(targets, dtype=torch.long)
    target = target[target >= 0]
    counts = torch.bincount(target, minlength=n_classes).float()
    inv = torch.zeros_like(counts)
    nonzero = counts > 0
    inv[nonzero] = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
    return inv.to(device)


def compute_loss(outputs, freshness, species, kind, fresh_weights, species_weights,
                 joint_weights, lambdas=(0.30, 0.30, 0.50, 0.05)):
    if kind == "joint24":
        joint_target = species * 3 + freshness
        return F.cross_entropy(outputs["joint_logits"], joint_target, weight=joint_weights)
    loss = F.cross_entropy(outputs["freshness_logits"], freshness, weight=fresh_weights)
    if kind == "multitask":
        return loss + lambdas[0] * F.cross_entropy(outputs["species_logits"], species, weight=species_weights)
    if kind.startswith("sgscfnet-"):
        species_term, ordinal_term, safety_term, diversity_term = lambdas
        loss = loss + species_term * F.cross_entropy(outputs["species_logits"], species, weight=species_weights)
        loss = loss + ordinal_term * ordinal_loss(outputs["ordinal_logits"], freshness)
        loss = loss + safety_term * expected_safety_cost(outputs["freshness_logits"], freshness)
        loss = loss + diversity_term * diversity_penalty(outputs["expert_logits"])
    return loss
