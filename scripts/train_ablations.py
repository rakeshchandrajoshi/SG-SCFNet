"""Run additive SG-SCFNet loss ablations and hard/soft routing comparisons."""
import argparse
from pathlib import Path
import pandas as pd
from sgscfnet.data import load_manifest, scan_ffe, species_map, stratified_split
from sgscfnet.experiment import fit_model

CONFIGURATIONS = {
    "base_global_experts": ("sgscfnet-hard", (0.30, 0.00, 0.00, 0.00)),
    "plus_ordinal": ("sgscfnet-hard", (0.30, 0.30, 0.00, 0.00)),
    "plus_safety": ("sgscfnet-hard", (0.30, 0.30, 0.50, 0.00)),
    "plus_diversity": ("sgscfnet-hard", (0.30, 0.30, 0.50, 0.05)),
    "full_hard": ("sgscfnet-hard", (0.30, 0.30, 0.50, 0.05)),
    "full_soft": ("sgscfnet-soft", (0.30, 0.30, 0.50, 0.05)),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--manifest")
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--output", default="runs/ablation")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()
    frame = load_manifest(args.manifest) if args.manifest else scan_ffe(args.images)
    mapping = species_map(frame)
    results = []
    for seed in args.seeds:
        train, val, test = stratified_split(frame, seed)
        for name, (kind, lambdas) in CONFIGURATIONS.items():
            path = Path(args.output) / name / f"seed_{seed}"
            run = fit_model(train, val, test, root=args.images, kind=kind,
                backbone=args.backbone, species_map=mapping, output_dir=path,
                seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                workers=args.workers, pretrained=not args.no_pretrained,
                lambdas=lambdas)
            results.append({"configuration": name, "seed": seed,
                "accuracy": run["metrics"]["accuracy"],
                "macro_f1": run["metrics"]["macro_f1"],
                "ordinal_mae": run["metrics"]["ordinal_mae"],
                "safety_far": run["metrics"]["safety_far"]})
            print(f"{name} seed={seed} macro-F1={results[-1]['macro_f1']:.4f}")
    pd.DataFrame(results).to_csv(Path(args.output) / "summary.csv", index=False)

if __name__ == "__main__":
    main()
