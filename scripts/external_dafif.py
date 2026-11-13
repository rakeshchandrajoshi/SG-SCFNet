"""DaFiF zero-shot evaluation and adaptation on species/day-labeled full-fish images."""
import argparse
from pathlib import Path
import pandas as pd
from sgscfnet.data import load_manifest, species_map, grouped_day_split
from sgscfnet.experiment import fit_model, predict_on_frame
from sgscfnet.runtime import load_checkpoint, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="Root of DaFiF full-fish images")
    p.add_argument("--manifest", required=True, help="Labeled DaFiF manifest with acquisition day")
    p.add_argument("--mode", choices=["zero-shot", "adapt"], required=True)
    p.add_argument("--ffe-checkpoint", help="SG-SCFNet checkpoint trained on FFE")
    p.add_argument("--initialization", choices=["ffe", "imagenet"], default="ffe")
    p.add_argument("--linear-probe", action="store_true")
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--seeds", nargs="+", type=int, default=[1,2,3])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--output", default="runs/dafif")
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()
    frame = load_manifest(args.manifest)
    if args.mode == "zero-shot":
        if not args.ffe_checkpoint:
            p.error("--ffe-checkpoint is required for zero-shot evaluation")
        model, ckpt = load_checkpoint(args.ffe_checkpoint)
        metrics = predict_on_frame(model, frame, args.images, ckpt["species_map"],
            batch_size=args.batch_size, workers=args.workers,
            image_size=ckpt["image_size"], threshold=ckpt.get("threshold", 0),
            prediction_csv=Path(args.output) / "zero_shot_predictions.csv")
        save_json(Path(args.output) / "zero_shot_metrics.json", metrics)
        print(f"DaFiF zero-shot accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}")
        return
    if args.initialization == "ffe" and not args.ffe_checkpoint:
        p.error("--ffe-checkpoint is required for FFE initialization")
    ckpt = load_checkpoint(args.ffe_checkpoint)[1] if args.initialization == "ffe" else None
    mapping = species_map(frame)
    rows = []
    for seed in args.seeds:
        train, val, test = grouped_day_split(frame, seed)
        out = Path(args.output) / args.initialization / ("linear_probe" if args.linear_probe else "full_finetune") / f"seed_{seed}"
        result = fit_model(train, val, test, root=args.images, kind="sgscfnet-hard",
            backbone=args.backbone, species_map=mapping, output_dir=out,
            seed=seed, epochs=args.epochs, batch_size=args.batch_size,
            workers=args.workers, pretrained=not args.no_pretrained and ckpt is None,
            transfer_checkpoint=ckpt, linear_probe=args.linear_probe)
        rows.append({"seed": seed, "accuracy": result["metrics"]["accuracy"],
                     "macro_f1": result["metrics"]["macro_f1"],
                     "safety_far": result["metrics"]["safety_far"]})
        print(f"DaFiF seed={seed}: accuracy={rows[-1]['accuracy']:.4f}")
    out = Path(args.output) / args.initialization / ("linear_probe" if args.linear_probe else "full_finetune")
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)

if __name__ == "__main__":
    main()
