"""Regression checks for lossless disk round trips, including fresh processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

from image_protection import configure_cpu, load_key, protect, restore


def verify(key_path, report_path):
    configure_cpu()
    key_path = Path(key_path).resolve()
    script = Path(__file__).with_name("image_protection.py").resolve()
    rng = np.random.default_rng(918)
    results = {}
    # Non-square source, partial final blocks, single pixel and full masks.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = rng.integers(0, 256, size=(17, 23, 3), dtype=np.uint8)
        masks = {"single_pixel": np.zeros((17, 23), dtype=bool),
                 "irregular": rng.random((17, 23)) > 0.55,
                 "full": np.ones((17, 23), dtype=bool)}
        masks["single_pixel"][8, 11] = True
        for name, mask in masks.items():
            case = root / name
            case.mkdir()
            Image.fromarray(original).save(case / "source.png")
            Image.fromarray(mask.astype(np.uint8)*255).save(case / "mask.png")
            protect(case / "source.png", case / "mask.png", key_path, case / "protected.npz")
            isolated = case / "restore_only"
            isolated.mkdir()
            (isolated / "protected.npz").write_bytes((case / "protected.npz").read_bytes())
            (isolated / "key.npz").write_bytes(key_path.read_bytes())
            # Original and mask files are removed before starting a fresh interpreter.
            (case / "source.png").unlink()
            (case / "mask.png").unlink()
            process = subprocess.run(
                [sys.executable, "-s", str(script), "restore", "--protected", "protected.npz",
                 "--key", "key.npz", "--output", "restored.png"],
                cwd=isolated, env={**os.environ, "PYTHONNOUSERSITE": "1"},
                capture_output=True, text=True, check=True)
            restored = np.array(Image.open(isolated / "restored.png"))
            assert np.array_equal(original, restored), name
            with np.load(isolated / "protected.npz", allow_pickle=False) as data:
                outside = np.rint(data["image"][~mask]*255).astype(np.uint8)
                assert np.array_equal(outside, original[~mask]), name
            preview = np.array(Image.open(case / "protected.preview.png"))
            assert np.array_equal(preview[~mask], original[~mask]), name
            results[name] = {"exact_pixel_match": True, "background_unchanged": True,
                             "fresh_process_without_source": True,
                             "max_pixel_error": int(np.abs(original.astype(int)-restored).max()),
                             "restore": json.loads(process.stdout)}
        # A mismatched key must be rejected before attempting to decode.
        matrix, _ = load_key(key_path)
        from image_protection import key_identifier
        different = matrix[::-1].copy()
        wrong_key = root / "wrong.npz"
        np.savez_compressed(wrong_key, matrix=different, key_id=key_identifier(different))
        try:
            restore(case / "protected.npz", wrong_key, root / "wrong.png")
        except ValueError as error:
            assert "does not belong" in str(error)
            results["wrong_key_rejected"] = True
        else:
            raise AssertionError("Wrong key accepted")
        for name, mask in {"empty": np.zeros((17, 23), dtype=np.uint8),
                           "wrong_shape": np.ones((7, 8), dtype=np.uint8)*255}.items():
            Image.fromarray(original).save(root / "source.png")
            Image.fromarray(mask).save(root / "mask.png")
            try:
                protect(root / "source.png", root / "mask.png", key_path, root / "invalid.npz")
            except ValueError:
                results[name + "_mask_rejected"] = True
            else:
                raise AssertionError(name)
    Path(report_path).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    verify(sys.argv[1] if len(sys.argv)>1 else "outputs/key.npz",
           sys.argv[2] if len(sys.argv)>2 else "outputs/verification.json")
