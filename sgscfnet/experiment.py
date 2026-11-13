from __future__ import annotations

import copy
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .constants import FRESHNESS_TO_ID
from .data import loader
from .losses import compute_loss, balanced_weights
from .metrics import scoring, choose_threshold
from .models import build_model
from .runtime import save_json, set_seed, transfer_compatible_weights


def _predict(model, dataloader, device):
    model.eval()
    probabilities, truth, species, paths, names = [], [], [], [], []
    with torch.inference_mode():
        for batch in dataloader:
            output = model(batch["image"].to(device, non_blocking=True))
            probabilities.append(torch.softmax(output["freshness_logits"], dim=1).cpu().numpy())
            truth.extend(batch["freshness"].numpy().tolist())
            species.extend(batch["species"].numpy().tolist())
            paths.extend(batch["path"])
            names.extend(batch["species_name"])
    return np.concatenate(probabilities), np.array(truth), np.array(species), paths, names


def predict_on_frame(model, frame, root, s_map, batch_size=64, workers=0, image_size=224,
                     device=None, threshold=0.0, prediction_csv=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dl = loader(frame, root, s_map, batch_size, workers, image_size=image_size)
    prob, truth, species, paths, names = _predict(model.to(device), dl, device)
    result = scoring(truth, prob, threshold=threshold, species=names)
    if prediction_csv is not None:
        pred = prob.argmax(axis=1)
        rows = pd.DataFrame({"path": paths, "species": names,
                             "true_freshness": truth, "pred_freshness": pred,
                             "p_highly_fresh": prob[:, 0], "p_fresh": prob[:, 1],
                             "p_not_fresh": prob[:, 2], "confidence": prob.max(axis=1),
                             "accepted": prob.max(axis=1) >= threshold})
        Path(prediction_csv).parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(prediction_csv, index=False)
    return result


def fit_model(train, validation, test, *, root, kind, backbone, species_map,
              output_dir, seed=1, epochs=25, patience=7, batch_size=64,
              workers=0, lr=2e-4, weight_decay=1e-4, image_size=224,
              alpha=0.4, lambdas=(0.30, 0.30, 0.50, 0.05),
              target_coverage=0.72, pretrained=True, transfer_checkpoint=None,
              linear_probe=False):
    """Train/validate/test once; checkpoint chosen only using validation macro-F1."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_species = len(species_map)
    if n_species < 1:
        raise ValueError("Training requires at least one species")
    model = build_model(kind, backbone, n_species, pretrained=pretrained, alpha=alpha)
    transferred = []
    if transfer_checkpoint is not None:
        transferred = transfer_compatible_weights(model, transfer_checkpoint)
    if linear_probe:
        model.encoder.requires_grad_(False)
    model.to(device)
    dl_train = loader(train, root, species_map, batch_size, workers, train=True, image_size=image_size)
    dl_val = loader(validation, root, species_map, batch_size, workers, image_size=image_size)
    dl_test = loader(test, root, species_map, batch_size, workers, image_size=image_size)
    y_train = [FRESHNESS_TO_ID[str(x).casefold()] for x in train["freshness"]]
    s_train = [species_map[str(x)] for x in train["species"]]
    w_f = balanced_weights(y_train, 3, device)
    w_s = balanced_weights(s_train, n_species, device)
    joint_y = [s * 3 + y for s, y in zip(s_train, y_train)]
    w_joint = balanced_weights(joint_y, 3 * n_species, device)
    optim = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr,
                  weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -1.0
    best_weights = None
    best_epoch = 0
    no_improvement = 0
    history = []
    start = time.perf_counter()

    for epoch in range(1, epochs+1):
        model.train()
        if linear_probe:
            model.encoder.eval()
        running, processed = 0.0, 0
        for batch in dl_train:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["freshness"].to(device, non_blocking=True)
            species = batch["species"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = compute_loss(outputs, targets, species, kind,
                                    w_f, w_s, w_joint, lambdas=lambdas)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += float(loss.detach()) * len(images)
            processed += len(images)
        scheduler.step()
        prob, y, _, _, _ = _predict(model, dl_val, device)
        val = scoring(y, prob)
        history.append({"epoch": epoch, "train_loss": running/max(processed,1),
                        "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"],
                        "val_safety_far": val["safety_far"]})
        if val["macro_f1"] > best_f1 + 1e-9:
            best_f1 = val["macro_f1"]
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break

    if best_weights is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_weights)
    vprob, vy, _, _, _ = _predict(model, dl_val, device)
    threshold = choose_threshold(vprob, target_coverage)
    val_metrics = scoring(vy, vprob, threshold=threshold)
    tprob, ty, ts, paths, names = _predict(model, dl_test, device)
    test_metrics = scoring(ty, tprob, threshold=threshold, species=names)
    test_metrics["elapsed_training_seconds"] = round(time.perf_counter() - start, 2)
    checkpoint = {
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "kind": kind, "backbone": backbone, "species_map": dict(species_map),
        "image_size": image_size, "alpha": alpha, "seed": seed,
        "lambdas": list(lambdas), "threshold": threshold,
        "target_coverage": target_coverage,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, output_dir / "best.pt")
    save_json(output_dir / "config.json", {k:v for k,v in checkpoint.items() if k != "state_dict"})
    save_json(output_dir / "validation.json", val_metrics)
    save_json(output_dir / "metrics.json", test_metrics)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame({"path": paths, "species": names,
                  "true_freshness": ty, "pred_freshness": tprob.argmax(axis=1),
                  "p_highly_fresh": tprob[:,0], "p_fresh": tprob[:,1],
                  "p_not_fresh": tprob[:,2], "confidence": tprob.max(axis=1),
                  "accepted": tprob.max(axis=1) >= threshold}).to_csv(
                      output_dir / "predictions.csv", index=False)
    return {"metrics": test_metrics, "validation": val_metrics,
            "best_epoch": best_epoch, "threshold": threshold,
            "transferred_parameters": transferred}
