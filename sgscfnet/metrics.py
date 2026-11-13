from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


def choose_threshold(probabilities, target_coverage=0.72):
    """Select confidence cutoff on validation predictions for target accepted coverage."""
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target coverage must be in (0,1]")
    confidences = np.max(np.asarray(probabilities), axis=1)
    count = max(1, int(round(len(confidences) * target_coverage)))
    return float(np.sort(confidences)[::-1][count - 1])


def scoring(truth, probabilities, threshold=0.0, species=None):
    truth = np.asarray(truth, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(truth) == 0:
        raise ValueError("Cannot evaluate an empty set")
    predictions = probabilities.argmax(axis=1)
    accepted = probabilities.max(axis=1) >= threshold
    unsafe = (truth == 2) & (predictions != 2)
    true_bad = truth == 2
    bad_accepted = true_bad & accepted
    result = {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "ordinal_mae": float(np.abs(truth - predictions).mean()),
        "safety_far": float(unsafe.sum() / max(1, true_bad.sum())),
        "safety_far_numerator": int(unsafe.sum()),
        "safety_far_denominator": int(true_bad.sum()),
        "threshold": float(threshold),
        "coverage": float(accepted.mean()),
        "accepted_n": int(accepted.sum()),
        "rejected_n": int((~accepted).sum()),
        "accepted_accuracy": float(accuracy_score(truth[accepted], predictions[accepted])) if accepted.any() else None,
        "accepted_safety_far": float((unsafe & accepted).sum() / bad_accepted.sum()) if bad_accepted.any() else None,
        "accepted_safety_far_numerator": int((unsafe & accepted).sum()),
        "accepted_safety_far_denominator": int(bad_accepted.sum()),
        "not_fresh_coverage": float(bad_accepted.sum() / max(1, true_bad.sum())),
        "unsafe_automatic_acceptance_fraction": float((unsafe & accepted).sum() / max(1, true_bad.sum())),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0,1,2]).tolist(),
    }
    if species is not None:
        s = np.asarray(species)
        result["per_species"] = {
            str(name): scoring(truth[s == name], probabilities[s == name], threshold=threshold)
            for name in sorted(set(s.tolist()))
        }
    return result
