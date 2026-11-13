from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class TinyBackbone(nn.Module):
    """Compact backbone for CPU demonstrations and unit tests."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.num_features = 32

    def forward(self, images):
        return self.net(images)


def make_backbone(name: str, pretrained: bool):
    if name == "tiny_cnn":
        return TinyBackbone(), 32
    try:
        import timm
    except ImportError as exc:
        raise ImportError("Install project dependencies to use the pretrained image backbones") from exc
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    return model, int(model.num_features)


class FreshnessOnly(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True, **_):
        super().__init__()
        self.encoder, width = make_backbone(backbone, pretrained)
        self.freshness_head = nn.Linear(width, 3)

    def forward(self, x):
        return {"freshness_logits": self.freshness_head(self.encoder(x))}


class Multitask(nn.Module):
    def __init__(self, backbone: str, n_species: int, pretrained: bool = True, **_):
        super().__init__()
        self.encoder, width = make_backbone(backbone, pretrained)
        self.freshness_head = nn.Linear(width, 3)
        self.species_head = nn.Linear(width, n_species)

    def forward(self, x):
        z = self.encoder(x)
        return {"freshness_logits": self.freshness_head(z),
                "species_logits": self.species_head(z)}


class Joint24(nn.Module):
    def __init__(self, backbone: str, n_species: int, pretrained: bool = True, **_):
        super().__init__()
        self.encoder, width = make_backbone(backbone, pretrained)
        self.n_species = n_species
        self.joint_head = nn.Linear(width, 3 * n_species)

    def forward(self, x):
        joint_logits = self.joint_head(self.encoder(x))
        joint_prob = F.softmax(joint_logits, dim=1).reshape(-1, self.n_species, 3)
        fresh_prob = joint_prob.sum(dim=1).clamp_min(1e-12)
        return {"freshness_logits": fresh_prob.log(), "joint_logits": joint_logits}


class SGSCFNet(nn.Module):
    """Species-Gated Safety-Cost Fusion Network (hard or soft expert gating)."""
    def __init__(self, backbone: str, n_species: int, pretrained: bool = True,
                 gating: str = "hard", alpha: float = 0.4, **_):
        super().__init__()
        if gating not in {"hard", "soft"}:
            raise ValueError("gating must be 'hard' or 'soft'")
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be in [0,1]")
        self.encoder, width = make_backbone(backbone, pretrained)
        self.n_species = n_species
        self.gating = gating
        self.alpha = alpha
        self.species_head = nn.Linear(width, n_species)
        self.global_freshness_head = nn.Linear(width, 3)
        self.experts = nn.ModuleList([nn.Linear(width, 3) for _ in range(n_species)])
        self.fusion = nn.Linear(3, 3)
        nn.init.eye_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        self.ordinal_head = nn.Linear(width, 2)

    def forward(self, x):
        features = self.encoder(x)
        species_logits = self.species_head(features)
        routing = F.softmax(species_logits, dim=1)
        global_logits = self.global_freshness_head(features)
        expert_logits = torch.stack([head(features) for head in self.experts], dim=1)
        if self.gating == "soft":
            chosen = (routing.unsqueeze(-1) * expert_logits).sum(dim=1)
        else:
            idx = routing.argmax(dim=1)
            chosen = expert_logits[torch.arange(len(idx), device=idx.device), idx]
        mixed = self.alpha * global_logits + (1.0 - self.alpha) * chosen
        return {
            "freshness_logits": self.fusion(mixed),
            "species_logits": species_logits,
            "ordinal_logits": self.ordinal_head(features),
            "expert_logits": expert_logits,
            "routing": routing,
        }


def build_model(kind: str, backbone: str, n_species: int, pretrained=True, alpha=0.4):
    if kind == "freshness":
        return FreshnessOnly(backbone, pretrained=pretrained)
    if kind == "multitask":
        return Multitask(backbone, n_species=n_species, pretrained=pretrained)
    if kind == "joint24":
        return Joint24(backbone, n_species=n_species, pretrained=pretrained)
    if kind.startswith("sgscfnet-"):
        return SGSCFNet(backbone, n_species=n_species, pretrained=pretrained,
                        gating=kind.split("-", 1)[1], alpha=alpha)
    raise ValueError(f"Unknown model: {kind}")
