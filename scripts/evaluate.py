"""Evaluate a saved checkpoint on any labeled CSV manifest."""
import argparse
from pathlib import Path
from sgscfnet.data import load_manifest
from sgscfnet.runtime import load_checkpoint, save_json
from sgscfnet.experiment import predict_on_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--output", default="runs/evaluation")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint)
    output = Path(args.output)
    threshold = checkpoint.get("threshold", 0.0) if args.threshold is None else args.threshold
    metrics = predict_on_frame(model, load_manifest(args.manifest), args.images,
        checkpoint["species_map"], batch_size=args.batch_size, workers=args.workers,
        image_size=checkpoint["image_size"], threshold=threshold,
        prediction_csv=output / "predictions.csv")
    save_json(output / "metrics.json", metrics)
    print(f"accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}, "
          f"Safety FAR={metrics['safety_far']:.4f}")

if __name__ == "__main__":
    main()
