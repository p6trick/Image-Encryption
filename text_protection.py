"""CPU text-guided image protection using CLIPSeg and a learned transform.

Examples:
    python text_protection.py run --image inputs/coffee.png --text cup --output-dir outputs/cup
    python text_protection.py examples
    python text_protection.py compare --image inputs/coffee.png --texts cup spoon
    python text_protection.py restore --protected outputs/cup/protected.npz --key outputs/cup/key.npz --output restored.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

from image_protection import configure_cpu, protect, restore, train_key

ROOT = Path(__file__).resolve().parent
MODEL_ID = "CIDAS/clipseg-rd64-refined"
MODEL_REVISION = "999e0328d9e10b484360c477313983f9afdd7050"
# The second value is passed to the model tokenizer, not just used as a label.
EXAMPLES = [("rocket", "rocket"), ("astronaut", "woman"), ("chelsea", "cat"),
            ("coffee", "cup"), ("camera", "camera"), ("eagle", "bird")]


class TextSegmenter:
    def __init__(self, model_id=MODEL_ID):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        started = time.perf_counter()
        cache = str(ROOT / ".cache" / "clipseg")
        local_model = ROOT / "models" / "clipseg"
        if (local_model / "model.safetensors").is_file():
            model_source = str(local_model)
            options = {"local_files_only": True}
        else:
            model_source = model_id
            options = {"revision": MODEL_REVISION, "cache_dir": cache}
        self.processor = CLIPSegProcessor.from_pretrained(
            model_source, use_fast=False, **options)
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            model_source, **options)
        self.model.to("cpu").eval().requires_grad_(False)
        self.model_id = model_id
        self.load_seconds = time.perf_counter()-started

    def segment(self, image_path, text, output_dir, threshold=0.5):
        if not text.strip():
            raise ValueError("Please provide a nonempty object description with --text")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1 (exclusive)")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        # Preserve the complete input, including its original dimensions.
        image.save(output_dir / "original.png")
        started = time.perf_counter()
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            resized = F.interpolate(logits.unsqueeze(1), size=(image.height, image.width),
                                    mode="bilinear", align_corners=False)[0, 0]
            probability = resized.sigmoid().cpu().numpy()
        seconds = time.perf_counter()-started
        # Absolute sigmoid score, never per-image min/max normalization.
        mask = probability > threshold
        np.save(output_dir / "probability.npy", probability)
        Image.fromarray(np.rint(probability*255).astype(np.uint8)).save(output_dir / "probability.png")
        Image.fromarray(mask.astype(np.uint8)*255).save(output_dir / "mask.png")
        overlay = np.array(image).copy()
        overlay[mask] = np.rint(0.55*overlay[mask] + 0.45*np.array([0, 255, 80])).astype(np.uint8)
        Image.fromarray(overlay).save(output_dir / "overlay.png")
        metrics = {"model": self.model_id, "model_revision": MODEL_REVISION,
                   "device": "cpu", "text_used": True, "text": text,
                   "threshold": threshold, "max_probability": float(probability.max()),
                   "selected_fraction": float(mask.mean()), "selected_pixels": int(mask.sum()),
                   "original_size": list(image.size), "segmentation_seconds": seconds}
        (output_dir / "segmentation.json").write_text(json.dumps(metrics, indent=2))
        return metrics


def draw_results(rows, destination, threshold):
    columns = [("original.png", "Original"), ("probability.png", "Text-conditioned score"),
               ("mask.png", f"Mask > {threshold:g}"),
               ("protected.preview.png", "Protected preview"), ("restored.png", "Restored")]
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(13, 2.75*len(rows)), squeeze=False)
    for index, (row, axis_row) in enumerate(zip(rows, axes)):
        for ax, (filename, title) in zip(axis_row, columns):
            directory = Path(row["directory"])
            if not directory.is_absolute():
                directory = ROOT / directory
            ax.imshow(Image.open(directory / filename), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if index == 0:
                ax.set_title(title, fontsize=10)
        axis_row[0].set_ylabel(f'Text: "{row["text"]}"', fontsize=10)
        axis_row[2].set_xlabel(f'Selected {row["selected_fraction"]:.1%}', fontsize=9)
        axis_row[-1].set_xlabel(f'Pixel error: {row["max_pixel_error"]}', fontsize=9)
    fig.suptitle("CLIPSeg | text + image | CPU", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(destination, dpi=150, facecolor="white")
    plt.close(fig)


def run_image(image_path, text, output_dir, *, segmenter, threshold=0.5, key_path=None):
    output_dir = Path(output_dir).resolve()
    metrics = segmenter.segment(image_path, text, output_dir, threshold)
    if metrics["selected_pixels"] == 0:
        # Do not leave an earlier successful run looking like this run's output.
        for filename in ["protected.npz", "protected.preview.png", "restored.png", "comparison.png", "metrics.json"]:
            (output_dir / filename).unlink(missing_ok=True)
        raise ValueError(f'No region selected for "{text}" (max score={metrics["max_probability"]:.3f}). '
                         'Inspect probability.png, choose another description or lower --threshold. '
                         'No protection has been applied.')
    local_key = output_dir / "key.npz"
    if key_path is not None:
        key_path = Path(key_path).resolve()
        if key_path != local_key:
            local_key.write_bytes(key_path.read_bytes())
    elif not local_key.exists():
        train_key(local_key)
    protection_metrics = protect(output_dir / "original.png", output_dir / "mask.png",
                                 local_key, output_dir / "protected.npz")
    # Fresh process, with no reference to the source, text or segmentation model.
    process = subprocess.run(
        [sys.executable, "-s", str(ROOT / "image_protection.py"), "restore",
         "--protected", str(output_dir / "protected.npz"), "--key", str(local_key),
         "--output", str(output_dir / "restored.png")],
        cwd=output_dir, env={**os.environ, "PYTHONNOUSERSITE": "1"},
        check=True, capture_output=True, text=True)
    original = np.array(Image.open(output_dir / "original.png"))
    restored = np.array(Image.open(output_dir / "restored.png"))
    mask = np.array(Image.open(output_dir / "mask.png")) > 127
    preview = np.array(Image.open(output_dir / "protected.preview.png"))
    error = np.abs(original.astype(int)-restored.astype(int))
    assert np.array_equal(original, restored), "Restored image differs from original"
    assert np.array_equal(preview[~mask], original[~mask]), "Background changed"
    try:
        saved_directory = output_dir.relative_to(ROOT).as_posix()
    except ValueError:
        saved_directory = str(output_dir)
    metrics.update({"directory": saved_directory, "exact_pixel_match": True,
                    "max_pixel_error": int(error.max()), "mse": float(np.mean(error**2)),
                    "background_unchanged": True,
                    "fresh_process_restoration": True,
                    "protection_seconds": protection_metrics["seconds"],
                    "restoration_seconds": json.loads(process.stdout)["seconds"]})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    draw_results([metrics], output_dir / "comparison.png", threshold)
    return metrics


def compare_prompts(image_path, texts, output_dir, *, segmenter, threshold=0.5):
    """Same image, same threshold, different text: demonstrate conditioning."""
    output_dir = Path(output_dir).resolve()
    if len(texts) < 2 or len(set(texts)) != len(texts):
        raise ValueError("Provide at least two distinct descriptions")
    rows = []
    for index, text in enumerate(texts):
        directory = output_dir / f"prompt_{index+1}"
        rows.append(segmenter.segment(image_path, text, directory, threshold))
    maps = [np.load(output_dir / f"prompt_{i+1}" / "probability.npy") for i in range(len(texts))]
    masks = [p > threshold for p in maps]
    pairs = []
    for index in range(1, len(texts)):
        union = np.logical_or(masks[0], masks[index]).sum()
        pairs.append({"texts": [texts[0], texts[index]],
                      "score_maps_differ": not np.array_equal(maps[0], maps[index]),
                      "changed_mask_pixels": int(np.count_nonzero(masks[0] != masks[index])),
                      "mask_iou": float(np.logical_and(masks[0], masks[index]).sum()/union) if union else 1.0})
    fig, axes = plt.subplots(len(texts), 3, figsize=(9, 3*len(texts)), squeeze=False)
    for index, (text, row, axis_row) in enumerate(zip(texts, rows, axes)):
        directory = output_dir / f"prompt_{index+1}"
        for ax, filename, title in zip(axis_row, ["original.png", "probability.png", "overlay.png"],
                                       [f'Text: "{text}"', "Text-conditioned score", "Selected region"]):
            ax.imshow(Image.open(directory / filename), cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        axis_row[-1].set_title(f'Selected {row["selected_fraction"]:.1%}', fontsize=10)
    fig.suptitle("Same image + different text", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "comparison.png", dpi=150, facecolor="white")
    plt.close(fig)
    report = {"image": str(image_path), "threshold": threshold, "prompts": rows, "comparisons": pairs}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def run_examples(segmenter, output_dir, threshold=0.5):
    from skimage import data
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_key = output_dir / "key.npz"
    if not shared_key.exists():
        train_key(shared_key)
    rows = []
    for index, (name, text) in enumerate(EXAMPLES, 1):
        image_path = ROOT / "inputs" / f"{name}.png"
        if not image_path.exists():
            image_path.parent.mkdir(exist_ok=True)
            Image.fromarray(getattr(data, name)()).convert("RGB").save(image_path)
        print(f'[{index}/{len(EXAMPLES)}] text="{text}"', flush=True)
        metrics = run_image(image_path, text, output_dir / name, segmenter=segmenter,
                            threshold=threshold, key_path=shared_key)
        rows.append(metrics)
        print(f'  selected={metrics["selected_fraction"]:.1%}, pixel error={metrics["max_pixel_error"]}', flush=True)
        print(f'  [Saved] {output_dir / name / "comparison.png"}', flush=True)
    draw_results(rows, output_dir / "gallery.png", threshold)
    for start in range(0, len(rows), 3):
        draw_results(rows[start:start+3], output_dir / f"gallery_{start//3+1}.png", threshold)
    comparison = compare_prompts(ROOT / "inputs" / "coffee.png", ["cup", "spoon"],
                                 output_dir / "text_comparison", segmenter=segmenter, threshold=threshold)
    if not comparison["comparisons"][0]["score_maps_differ"] or comparison["comparisons"][0]["changed_mask_pixels"] == 0:
        raise AssertionError("Different text did not change the predicted mask")
    report = {"model": segmenter.model_id, "device": "cpu", "text_used": True,
              "model_load_seconds": segmenter.load_seconds, "threshold": threshold,
              "example_count": len(rows), "examples": rows,
              "text_conditioning_check": comparison["comparisons"]}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=4)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Select an object using text, protect and verify restoration")
    run.add_argument("--image", type=Path, required=True)
    run.add_argument("--text", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--key", type=Path)
    run.add_argument("--threshold", type=float, default=0.5)
    examples = commands.add_parser("examples", help="Run six text-guided examples")
    examples.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "text_examples")
    examples.add_argument("--threshold", type=float, default=0.5)
    compare = commands.add_parser("compare", help="Change the text for the same image")
    compare.add_argument("--image", type=Path, required=True)
    compare.add_argument("--texts", nargs="+", required=True)
    compare.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "text_comparison")
    compare.add_argument("--threshold", type=float, default=0.5)
    dec = commands.add_parser("restore", help="Restore without loading a segmentation model")
    dec.add_argument("--protected", type=Path, required=True)
    dec.add_argument("--key", type=Path, required=True)
    dec.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    configure_cpu(args.threads)
    if args.command == "restore":
        result = restore(args.protected, args.key, args.output)
    else:
        if not 0 < args.threshold < 1:
            parser.error("threshold must be between 0 and 1 (exclusive)")
        if args.command == "run" and not args.text.strip():
            parser.error("--text must contain an object description")
        args.output_dir = args.output_dir.expanduser().resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Output directory] {args.output_dir}", flush=True)
        print("[Model] Loading CLIPSeg on CPU...", flush=True)
        segmenter = TextSegmenter()
        if args.command == "run":
            result = run_image(args.image, args.text, args.output_dir, segmenter=segmenter,
                               threshold=args.threshold, key_path=args.key)
        elif args.command == "examples":
            result = run_examples(segmenter, args.output_dir, args.threshold)
        else:
            result = compare_prompts(args.image, args.texts, args.output_dir,
                                     segmenter=segmenter, threshold=args.threshold)
    if args.command == "examples":
        print(f"[Saved gallery] {args.output_dir / 'gallery.png'}", flush=True)
        print(f"[Saved report] {args.output_dir / 'summary.json'}", flush=True)
        print(f"[Done] {result['example_count']} examples saved to {args.output_dir}", flush=True)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from None
