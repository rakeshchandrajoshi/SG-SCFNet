"""Internal FFE experiments: freshness, multitask, joint24, SG-SCFNet hard/soft."""
import argparse
from pathlib import Path
import pandas as pd
from sgscfnet.data import scan_ffe, load_manifest, species_map, stratified_split
from sgscfnet.experiment import fit_model

MODELS = ("freshness", "multitask", "joint24", "sgscfnet-hard", "sgscfnet-soft")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--seeds", nargs="+", type=int, default=[1,2,3])
    p.add_argument("--output", default="runs/ffe")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--target-coverage", type=float, default=0.72)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--species-loss-weight", type=float, default=0.30)
    p.add_argument("--ordinal-loss-weight", type=float, default=0.30)
    p.add_argument("--safety-loss-weight", type=float, default=0.50)
    p.add_argument("--diversity-loss-weight", type=float, default=0.05)
    args = p.parse_args()
    frame = load_manifest(args.manifest) if args.manifest else scan_ffe(args.images)
    mapping = species_map(frame)
    rows = []
    for seed in args.seeds:
        train, val, test = stratified_split(frame, seed)
        for model in args.models:
            folder = Path(args.output) / model / f"seed_{seed}"
            result = fit_model(train, val, test, root=args.images,
                kind=model, backbone=args.backbone, species_map=mapping,
                output_dir=folder, seed=seed, epochs=args.epochs,
                patience=args.patience, batch_size=args.batch_size,
                workers=args.workers, lr=args.lr, weight_decay=args.weight_decay,
                alpha=args.alpha, target_coverage=args.target_coverage,
                lambdas=(args.species_loss_weight, args.ordinal_loss_weight,
                         args.safety_loss_weight, args.diversity_loss_weight),
                pretrained=not args.no_pretrained)
            rows.append({"model": model, "seed": seed,
                **{key: val for key,val in result["metrics"].items()
                   if isinstance(val, (float, int)) or val is None}})
            print(f"{model} seed={seed} accuracy={result['metrics']['accuracy']:.4f} "
                  f"macro_f1={result['metrics']['macro_f1']:.4f}")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(args.output) / "summary.csv", index=False)

if __name__ == "__main__":
    main()
