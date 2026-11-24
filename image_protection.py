"""Learned reversible region transforms for the CPU text-guided project.

The NPZ is the protected image; PNG is a preview. This educational linear
transform does not provide cryptographic security.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn


def configure_cpu(threads=4):
    torch.set_num_threads(threads)


class LinearTransform(nn.Module):
    def __init__(self, size=224, seed=42):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.lin = nn.Parameter(torch.randn(size, size, generator=generator,
                                            dtype=torch.float64) / np.sqrt(size))

    def forward(self):
        return self.lin.T @ self.lin


def key_identifier(matrix):
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype="<f8").tobytes()).hexdigest()


def train_key(path, size=224, steps=1000, lr=0.01, seed=42, tolerance=1e-10):
    """Train the transform only. CLIP and input images are not needed."""
    if size < 1 or steps < 1 or lr <= 0 or tolerance <= 0:
        raise ValueError("size, steps, lr and tolerance must be positive")
    started = time.perf_counter()
    en_model = LinearTransform(size, seed)
    before = en_model.lin.detach().clone()
    # FIX: the old notebook incorrectly used model.parameters() (CLIP).
    optimizer = torch.optim.Adam(en_model.parameters(), lr=lr)
    identity = torch.eye(size, dtype=torch.float64)
    history = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = (en_model() - identity).square().mean()
        value = loss.item()
        history.append({"step": step, "loss": value})
        if step % 100 == 0 or value <= tolerance or step == steps:
            print(f"step={step:4d} orthogonality_mse={value:.12g}", flush=True)
        if value <= tolerance or step == steps:
            break
        loss.backward()
        optimizer.step()
    matrix = en_model.lin.detach().numpy().copy()
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1e4:
        raise ValueError(f"Unstable transform (condition={condition}); retrain")
    changed = not torch.equal(before, en_model.lin.detach())
    if not changed or history[-1]["loss"] >= history[0]["loss"]:
        raise RuntimeError("Training did not update the matrix and reduce loss")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, matrix=matrix, key_id=key_identifier(matrix), version=1)
    report = {"device": "cpu", "matrix_size": size, "optimizer": "Adam(en_model.parameters())",
              "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"],
              "updates": history[-1]["step"], "parameters_changed": changed,
              "condition_number": condition, "seconds": time.perf_counter()-started,
              "history": history}
    path.with_suffix(".training.json").write_text(json.dumps(report, indent=2))
    return report


def load_key(path):
    with np.load(path, allow_pickle=False) as data:
        matrix = data["matrix"].astype(np.float64)
        identifier = str(data["key_id"].item())
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("Invalid key matrix")
    if key_identifier(matrix) != identifier:
        raise ValueError("Key identifier mismatch")
    return matrix, identifier


def protect(image_path, mask_path, key_path, output_path):
    """Pack selected RGB values into blocks; preserve every transformed value."""
    started = time.perf_counter()
    image = np.array(ImageOps.exif_transpose(Image.open(image_path)).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    if mask.shape != image.shape[:2] or not mask.any():
        raise ValueError("Mask must match image dimensions and contain selected pixels")
    matrix, identifier = load_key(key_path)
    normalized = image.astype(np.float64) / 255.0
    selected = normalized[mask].reshape(-1)
    size = len(matrix)
    padding = (-len(selected)) % size
    blocks = np.pad(selected, (0, padding)).reshape(-1, size)
    transformed = (blocks @ matrix.T).reshape(-1)
    protected = normalized.copy()
    protected[mask] = transformed[:len(selected)].reshape(-1, 3)
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("Protected image must use .npz to retain signed floating point values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, image=protected, mask=mask,
                        tail=transformed[len(selected):], key_id=identifier, version=1)
    # Display normalization is independent of the reversible data.
    preview = image.copy()
    display_values = protected[mask]
    lo, hi = float(display_values.min()), float(display_values.max())
    display_values = (display_values-lo) / (hi-lo) if hi > lo else np.zeros_like(display_values)
    preview[mask] = np.rint(display_values*255).astype(np.uint8)
    Image.fromarray(preview).save(output_path.with_suffix(".preview.png"))
    return {"seconds": time.perf_counter()-started, "selected_pixels": int(mask.sum()),
            "padding_values": padding, "protected_file": str(output_path)}


def restore(protected_path, key_path, output_path):
    """Restore using only on-disk protected data and key. No source or CLIP."""
    started = time.perf_counter()
    matrix, identifier = load_key(key_path)
    with np.load(protected_path, allow_pickle=False) as data:
        if str(data["key_id"].item()) != identifier:
            raise ValueError("This key does not belong to the protected image")
        image = data["image"].astype(np.float64)
        mask = data["mask"].astype(bool)
        tail = data["tail"].astype(np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("Invalid image or mask dimensions")
    selected = image[mask].reshape(-1)
    transformed = np.concatenate((selected, tail))
    if len(tail) != (-len(selected)) % len(matrix) or not np.isfinite(transformed).all():
        raise ValueError("Invalid protected payload")
    # y=Ux -> x=solve(U,y); training only approximates orthogonality.
    restored = np.linalg.solve(matrix, transformed.reshape(-1, len(matrix)).T).T.reshape(-1)
    image[mask] = restored[:len(selected)].reshape(-1, 3)
    result = np.rint(np.clip(image, 0, 1)*255).astype(np.uint8)
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError("Use .png for lossless restoration")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output_path)
    return {"seconds": time.perf_counter()-started, "restored_file": str(output_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--key", type=Path, required=True)
    train.add_argument("--size", type=int, default=224)
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--seed", type=int, default=42)
    enc = commands.add_parser("protect")
    enc.add_argument("--image", type=Path, required=True)
    enc.add_argument("--mask", type=Path, required=True)
    enc.add_argument("--key", type=Path, required=True)
    enc.add_argument("--output", type=Path, required=True)
    dec = commands.add_parser("restore")
    dec.add_argument("--protected", type=Path, required=True)
    dec.add_argument("--key", type=Path, required=True)
    dec.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    configure_cpu(args.threads)
    if args.command == "train":
        result = train_key(args.key, args.size, args.steps, seed=args.seed)
        result = {k: v for k, v in result.items() if k != "history"}
    elif args.command == "protect":
        result = protect(args.image, args.mask, args.key, args.output)
    else:
        result = restore(args.protected, args.key, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
