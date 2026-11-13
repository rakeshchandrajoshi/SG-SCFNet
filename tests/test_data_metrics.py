from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sgscfnet.data import scan_ffe, stratified_split, FishDataset, grouped_day_split
from sgscfnet.metrics import scoring, choose_threshold


def test_ffe_scan_and_splits(tmp_path):
    for name in ["Salmon", "Tilapia"]:
        for label in ["Highly Fresh", "Fresh", "Not Fresh"]:
            folder = tmp_path / f"{name} - {label}"
            folder.mkdir()
            for i in range(6):
                Image.new("RGB", (24,24), color=(i*20,10,50)).save(folder / f"{i}.png")
    frame = scan_ffe(tmp_path)
    assert len(frame) == 36
    train, val, test = stratified_split(frame, 1)
    assert len(train) + len(val) + len(test) == 36
    dataset = FishDataset(train, tmp_path, {"Salmon": 0, "Tilapia": 1}, image_size=32)
    assert dataset[0]["image"].shape == (3,32,32)


def test_grouped_split(tmp_path):
    frame = pd.DataFrame({"path": [f"a{i}.png" for i in range(22)],
        "species": ["Tilapia"]*22, "freshness": ["Fresh"]*22,
        "day": [str(i//2) for i in range(22)]})
    train, val, test = grouped_day_split(frame, 1)
    assert not (set(train.day) & set(val.day))
    assert not (set(train.day) & set(test.day))
    assert not (set(val.day) & set(test.day))


def test_safety_metrics_and_threshold():
    y = [2,2,0,1]
    prob = np.array([[.8,.1,.1],[.1,.1,.8],[.8,.1,.1],[.1,.8,.1]])
    tau = choose_threshold(prob, 0.75)
    stats = scoring(y, prob, threshold=tau)
    assert stats["safety_far"] == 0.5
    assert stats["safety_far_numerator"] == 1
    assert stats["safety_far_denominator"] == 2
    assert 0 <= stats["coverage"] <= 1
