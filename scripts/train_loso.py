"""Leave-one-species-out FFE stress test."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sgscfnet.data import scan_ffe, load_manifest, species_map
from sgscfnet.experiment import fit_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--manifest")
    p.add_argument("--models", nargs="+", choices=["freshness","multitask","sgscfnet-hard"],
                   default=["freshness","multitask","sgscfnet-hard"])
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--seeds", nargs="+", type=int, default=[1,2,3])
    p.add_argument("--output", default="runs/loso")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()
    frame = load_manifest(args.manifest) if args.manifest else scan_ffe(args.images)
    rows = []
    for held in sorted(frame.species.unique()):
        pool = frame.loc[frame.species != held].reset_index(drop=True)
        test = frame.loc[frame.species == held].reset_index(drop=True)
        mapping = species_map(pool)
        strata = pool.species + "|" + pool.freshness
        for seed in args.seeds:
            train, val = train_test_split(pool, test_size=0.18,
                random_state=seed, stratify=strata)
            for kind in args.models:
                folder = Path(args.output) / kind / f"{held.replace(' ','_')}_seed_{seed}"
                result = fit_model(train.reset_index(drop=True), val.reset_index(drop=True),
                    test, root=args.images, kind=kind, backbone=args.backbone,
                    species_map=mapping, output_dir=folder, seed=seed,
                    epochs=args.epochs, batch_size=args.batch_size,
                    workers=args.workers, pretrained=not args.no_pretrained)
                rows.append({"held_out_species": held, "seed": seed, "model": kind,
                    "accuracy": result["metrics"]["accuracy"],
                    "macro_f1": result["metrics"]["macro_f1"],
                    "safety_far": result["metrics"]["safety_far"]})
                print(f"{held}: {kind} seed={seed} accuracy={rows[-1]['accuracy']:.4f}")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(args.output) / "summary.csv", index=False)

if __name__ == "__main__":
    main()
