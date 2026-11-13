from pathlib import Path
from PIL import Image
from sgscfnet.data import scan_ffe, species_map, stratified_split
from sgscfnet.experiment import fit_model
from sgscfnet.runtime import load_checkpoint


def test_training_checkpoint(tmp_path):
    for species in ["A", "B"]:
        for label in ["Highly Fresh", "Fresh", "Not Fresh"]:
            folder = tmp_path / f"{species} - {label}"
            folder.mkdir()
            for j in range(6):
                Image.new("RGB", (36,36), (j*12,40,75)).save(folder / f"sample_{j}.png")
    frame = scan_ffe(tmp_path)
    train,val,test = stratified_split(frame, 1)
    result = fit_model(train,val,test, root=tmp_path, kind="sgscfnet-hard",
        backbone="tiny_cnn", species_map=species_map(frame), output_dir=tmp_path / "run",
        pretrained=False, image_size=32, seed=1, epochs=1, batch_size=8, workers=0)
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert (tmp_path/"run"/"best.pt").is_file()
    assert (tmp_path/"run"/"predictions.csv").is_file()
    _,checkpoint = load_checkpoint(tmp_path/"run"/"best.pt")
    assert checkpoint["kind"] == "sgscfnet-hard"
