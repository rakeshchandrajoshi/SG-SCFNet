# SG-SCFNet

Python implementation for fish freshness classification using RGB images.

## Setup

Python 3.10+:

```bash
pip install -e .
```

## Data

Datasets are not included. FFE images should be arranged in folders named `<species> - <freshness>`, where freshness is `Highly Fresh`, `Fresh`, or `Not Fresh`.

For DaFiF, provide a CSV with columns `path,species,day,freshness` and image paths relative to the dataset root.

## Run

```bash
# Train FFE models
python -m scripts.train_ffe --images /path/to/FFE_Dataset --models sgscfnet-hard sgscfnet-soft --seeds 1 2 3

# Component ablation
python -m scripts.train_ablations --images /path/to/FFE_Dataset

# Leave-one-species-out evaluation
python -m scripts.train_loso --images /path/to/FFE_Dataset

# External evaluation
python -m scripts.external_dafif --mode zero-shot --images /path/to/DaFiF --manifest data/dafif_manifest.csv --ffe-checkpoint runs/ffe/sgscfnet-hard/seed_1/best.pt

# Single-image prediction
python -m scripts.infer --image /path/to/image.jpg --checkpoint runs/ffe/sgscfnet-hard/seed_1/best.pt
```

Additional scripts support dataset manifest preparation, fine-tuning, metric aggregation, and Grad-CAM.

## Tests

```bash
pip install -e ".[test]"
python -m pytest -q
```
