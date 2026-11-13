"""Generate a CSV manifest from the FFE species × freshness directory tree."""
import argparse
from pathlib import Path
from sgscfnet.data import scan_ffe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="FFE dataset directory")
    p.add_argument("--output", default="data/ffe_manifest.csv")
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = scan_ffe(args.images)
    frame.to_csv(output, index=False)
    print(f"Indexed {len(frame)} FFE images from {frame.species.nunique()} species: {output}")

if __name__ == "__main__":
    main()
