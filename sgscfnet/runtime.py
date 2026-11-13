from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import torch
from .models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(value, fp, indent=2, ensure_ascii=False)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["kind"], ckpt["backbone"], len(ckpt["species_map"]),
                        pretrained=False, alpha=ckpt.get("alpha", 0.4))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def transfer_compatible_weights(model, source_checkpoint):
    """Transfer all matching representations; initialize dataset-specific heads as needed."""
    own = model.state_dict()
    matched = {k: v for k, v in source_checkpoint["state_dict"].items()
               if k in own and own[k].shape == v.shape
               and not k.startswith(("species_head.", "experts."))}
    own.update(matched)
    model.load_state_dict(own)
    return sorted(matched)
