"""Predict fish freshness from a single RGB image."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from sgscfnet.constants import FRESHNESS
from sgscfnet.data import image_transforms
from sgscfnet.runtime import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    transform = image_transforms(train=False, image_size=checkpoint["image_size"])
    with Image.open(args.image) as source:
        image = transform(source.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        result = model(image)
        probs = torch.softmax(result["freshness_logits"], dim=1)[0].cpu()
    conf = float(probs.max())
    threshold = checkpoint.get("threshold", 0.0) if args.threshold is None else args.threshold
    response = {"image": str(Path(args.image)),
                "prediction": FRESHNESS[int(probs.argmax())],
                "confidence": round(conf, 6),
                "probabilities": {k: round(float(v),6) for k,v in zip(FRESHNESS, probs)},
                "accepted": bool(conf >= threshold),
                "decision": "Automatic prediction" if conf >= threshold else "Manual inspection"}
    if "species_logits" in result:
        classes = {int(v): k for k, v in checkpoint["species_map"].items()}
        response["predicted_species"] = classes[int(result["species_logits"][0].argmax())]
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()
