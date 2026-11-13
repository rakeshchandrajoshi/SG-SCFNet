"""Create an external-domain manifest from labeled DaFiF image annotations."""
import argparse
from pathlib import Path
import pandas as pd
from sgscfnet.data import validate_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True,
                   help="CSV with columns path,species,day,freshness")
    p.add_argument("--images", required=True)
    p.add_argument("--output", default="data/dafif_manifest.csv")
    args = p.parse_args()
    data = pd.read_csv(args.annotations, dtype={"path": str, "species": str, "day": str, "freshness": str})
    if "day" not in data.columns:
        raise ValueError("DaFiF requires the acquisition-day column: day")
    validate_manifest(data)
    missing = [str(v) for v in data["path"] if not (Path(args.images) / str(v)).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image paths: {missing[:5]}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out, index=False)
    print(f"Wrote {len(data)} DaFiF records: {out}")

if __name__ == "__main__":
    main()
