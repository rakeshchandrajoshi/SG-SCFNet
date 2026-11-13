"""Create a Grad-CAM overlay for ConvNeXt-Tiny fish freshness inference."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sgscfnet.data import image_transforms
from sgscfnet.runtime import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--layer", default="stages.3.blocks.2")
    p.add_argument("--class-id", type=int, choices=[0,1,2])
    p.add_argument("--output", default="runs/gradcam.png")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    layer = model.encoder.get_submodule(args.layer)
    activation = {}

    def forward_hook(_layer, _inputs, output):
        activation["features"] = output
        output.register_hook(lambda grad: activation.__setitem__("gradient", grad))

    hook = layer.register_forward_hook(forward_hook)
    try:
        with Image.open(args.image) as opened:
            rgb = opened.convert("RGB")
        tensor = image_transforms(False, checkpoint["image_size"])(rgb).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)["freshness_logits"]
        target_class = int(logits.argmax(dim=1)) if args.class_id is None else args.class_id
        logits[0, target_class].backward()
        activ = activation["features"].detach()
        grad = activation["gradient"].detach()
        if activ.ndim != 4:
            raise ValueError("Selected layer must return spatial features with four dimensions")
        # ConvNeXt blocks may return NHWC or NCHW depending on backbone implementation.
        if activ.shape[1] < activ.shape[-1] and activ.shape[1] < 16:
            activ, grad = activ.permute(0,3,1,2), grad.permute(0,3,1,2)
        weights = grad.mean(dim=(2,3), keepdim=True)
        heatmap = torch.relu((weights * activ).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(heatmap, size=(checkpoint["image_size"], checkpoint["image_size"]),
                                mode="bilinear", align_corners=False)[0,0]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heat = (heatmap.cpu().numpy() * 255).astype(np.uint8)
        base = rgb.resize((checkpoint["image_size"], checkpoint["image_size"]))
        alpha = Image.fromarray((heat * 0.50).astype(np.uint8), "L")
        red = Image.new("RGB", base.size, (255, 35, 0))
        overlay = Image.composite(red, base, alpha)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        overlay.save(args.output)
        print(f"Saved class-{target_class} Grad-CAM: {args.output}")
    finally:
        hook.remove()

if __name__ == "__main__":
    main()
