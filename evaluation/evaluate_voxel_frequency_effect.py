"""Evaluate the impact of voxel/Fourier round-trip on DGCNN accuracy."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from models.dgcnn import build_dgcnn_classifier
from utils.modelnet40 import ModelNet40
from utils.voxel import PointCloudToVoxel, VoxelToPointCloud
from utils.frequency import dft_3d, idft_3d
from .datasets import AdversarialExamplesDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing the ModelNet40 HDF5 files. Required unless "
            "--attacked-file is provided."
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the pretrained DGCNN weights (.t7 file).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1024,
        help="Number of points per shape (default: 1024).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=3,
        default=(64, 64, 64),
        help="Spatial resolution of the voxel grid (default: 64 64 64).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation (default: cuda if available).",
    )
    parser.add_argument(
        "--attacked-file",
        type=Path,
        default=None,
        help=(
            "Optional .pt file produced by generate_adversarial_pointclouds.py. "
            "When provided the evaluation runs on the saved adversarial samples "
            "instead of the original HDF5 split."
        ),
    )
    parser.add_argument(
        "--input-field",
        type=str,
        default="pointcloud",
        help=(
            "Dictionary key to read the input point cloud from each batch. "
            "Use 'attacked' to evaluate adversarial shapes."
        ),
    )
    parser.add_argument(
        "--compare-field",
        type=str,
        default=None,
        help=(
            "Optional second field to report baseline accuracy without the "
            "voxel/Fourier pipeline (e.g. compare clean vs adversarial inputs)."
        ),
    )
    return parser.parse_args()


def load_classifier(weights_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_dgcnn_classifier(num_classes=40)
    checkpoint = torch.load(weights_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        # Common checkpoints store the parameters under dedicated keys.
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError("Unsupported checkpoint format: expected dict")

    # Remove potential DataParallel prefixes.
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        new_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    return model


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    voxeliser: PointCloudToVoxel,
    devoxeliser: VoxelToPointCloud,
    device: torch.device,
    input_field: str,
    compare_field: Optional[str] = None,
) -> Tuple[float, float, Optional[float]]:
    baseline_correct = 0
    processed_correct = 0
    compare_correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if input_field not in batch:
                raise KeyError(
                    f"Batch is missing field '{input_field}'. Available keys: {list(batch.keys())}"
                )

            points = batch[input_field].to(device)
            labels = batch["cate"].view(points.size(0), -1)[:, 0].to(device)
            batch_size, num_points, _ = points.shape

            logits = model(points.transpose(1, 2))
            baseline_correct += (logits.argmax(dim=1) == labels).sum().item()

            voxels, transform = voxeliser(points)
            amplitude, phase = dft_3d(voxels)
            reconstructed_voxels = idft_3d(amplitude, phase)
            recovered_points = devoxeliser(
                reconstructed_voxels,
                transform,
                num_points=num_points,
                threshold=0.0,
            )

            logits_processed = model(recovered_points.transpose(1, 2))
            processed_correct += (
                logits_processed.argmax(dim=1) == labels
            ).sum().item()

            total += batch_size

            if compare_field is not None:
                if compare_field not in batch:
                    raise KeyError(
                        f"Batch is missing comparison field '{compare_field}'."
                    )
                compare_points = batch[compare_field].to(device)
                compare_logits = model(compare_points.transpose(1, 2))
                compare_correct += (
                    compare_logits.argmax(dim=1) == labels
                ).sum().item()

    baseline_acc = baseline_correct / total
    processed_acc = processed_correct / total
    compare_acc = None
    if compare_field is not None:
        compare_acc = compare_correct / total
    return baseline_acc, processed_acc, compare_acc


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.attacked_file is not None:
        dataset = AdversarialExamplesDataset(args.attacked_file)
    else:
        if args.dataset_root is None:
            raise ValueError(
                "--dataset-root must be provided when --attacked-file is not set."
            )
        dataset = ModelNet40(
            root=str(args.dataset_root),
            split=args.split,
            num_points=args.num_points,
            normalize=True,
            random_rotate=False,
        )

    if "cate" not in dataset[0]:
        raise RuntimeError("Dataset samples must provide a 'cate' label tensor.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = load_classifier(args.weights, device)

    grid_size = tuple(int(v) for v in args.grid_size)
    voxeliser = PointCloudToVoxel(grid_size=grid_size).to(device)
    devoxeliser = VoxelToPointCloud(grid_size=grid_size).to(device)

    baseline_acc, processed_acc, compare_acc = evaluate(
        model=model,
        dataloader=dataloader,
        voxeliser=voxeliser,
        devoxeliser=devoxeliser,
        device=device,
        input_field=args.input_field,
        compare_field=args.compare_field,
    )

    print(
        f"Baseline accuracy using '{args.input_field}': "
        f"{baseline_acc * 100:.2f}%"
    )
    print(
        "After voxel -> FFT -> iFFT -> devoxel pipeline accuracy: "
        f"{processed_acc * 100:.2f}%"
    )
    print(f"Accuracy drop: {(baseline_acc - processed_acc) * 100:.2f} percentage points")

    if compare_acc is not None:
        print(
            f"Reference baseline accuracy using '{args.compare_field}': "
            f"{compare_acc * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
