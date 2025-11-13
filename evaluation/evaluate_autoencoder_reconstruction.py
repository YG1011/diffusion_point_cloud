"""Evaluate classification accuracy before and after autoencoder reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from models.autoencoder import AutoEncoder
from models.dgcnn import build_dgcnn_classifier
from utils.modelnet40 import ModelNet40

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
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1024,
        help="Number of points per shape (default: 1024).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the pretrained DGCNN weights (.t7 file).",
    )
    parser.add_argument(
        "--ae-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint produced by train_ae.py containing the diffusion model.",
    )
    parser.add_argument(
        "--flexibility",
        type=float,
        default=0.0,
        help="Diffusion flexibility parameter used during decoding (default: 0.0).",
    )
    parser.add_argument(
        "--input-field",
        type=str,
        default="pointcloud",
        help=(
            "Dictionary key to read the input point cloud from each batch. "
            "Use 'attacked' for adversarial .pt files."
        ),
    )
    parser.add_argument(
        "--context-field",
        type=str,
        default=None,
        help=(
            "Field used to compute the autoencoder latent code. Defaults to the "
            "value passed to --input-field."
        ),
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable per-shape normalisation when loading ModelNet40.",
    )
    parser.add_argument(
        "--random-rotate",
        action="store_true",
        help="Randomly shuffle the order of points in ModelNet40 samples.",
    )
    parser.add_argument(
        "--save-reconstructions",
        type=Path,
        default=None,
        help="Optional path to store reconstructed point clouds as a .pt file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation (default: cuda if available).",
    )
    return parser.parse_args()


def load_classifier(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the strongest available classifier (DGCNN) for evaluation."""
    model = build_dgcnn_classifier(num_classes=40)
    checkpoint = torch.load(weights_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError("Unsupported checkpoint format: expected dict")

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[len("module.") :] if key.startswith("module.") else key
        cleaned[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading classifier checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in classifier checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    return model


def load_autoencoder(ckpt_path: Path, device: torch.device) -> AutoEncoder:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "args" not in checkpoint or "state_dict" not in checkpoint:
        raise RuntimeError(
            "Autoencoder checkpoint must contain 'args' and 'state_dict' entries."
        )

    args = checkpoint["args"]
    model = AutoEncoder(args)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def build_dataloader(args: argparse.Namespace) -> DataLoader:
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
            normalize=not args.no_normalize,
            random_rotate=args.random_rotate,
        )

    if "cate" not in dataset[0]:
        raise RuntimeError("Dataset samples must provide a 'cate' label tensor.")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(torch.device(args.device).type == "cuda"),
        drop_last=False,
    )


def evaluate(
    dataloader: DataLoader,
    classifier: torch.nn.Module,
    autoencoder: AutoEncoder,
    device: torch.device,
    input_field: str,
    context_field: str,
    flexibility: float,
    save_path: Optional[Path],
) -> Dict[str, float]:
    total = 0
    baseline_correct = 0
    recon_correct = 0
    saved_samples = []

    if flexibility < 0:
        raise ValueError("--flexibility must be non-negative.")

    with torch.no_grad():
        for batch in dataloader:
            if input_field not in batch:
                raise KeyError(
                    f"Batch is missing field '{input_field}'. Available keys: {list(batch.keys())}"
                )
            if context_field not in batch:
                raise KeyError(
                    f"Batch is missing context field '{context_field}'."
                )

            points = batch[input_field].to(device)
            context_points = batch[context_field].to(device)
            labels = batch["cate"].view(points.size(0), -1)[:, 0].to(device)

            logits_baseline = classifier(points.transpose(1, 2))
            baseline_correct += (logits_baseline.argmax(dim=1) == labels).sum().item()

            latents = autoencoder.encode(context_points)
            recon = autoencoder.decode(latents, num_points=points.size(1), flexibility=flexibility)
            logits_recon = classifier(recon.transpose(1, 2))
            recon_correct += (logits_recon.argmax(dim=1) == labels).sum().item()

            batch_size = points.size(0)
            total += batch_size

            if save_path is not None:
                recon_cpu = recon.detach().cpu()
                for idx in range(batch_size):
                    sample: Dict[str, torch.Tensor] = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            sample[key] = value[idx].detach().cpu()
                    sample["reconstructed"] = recon_cpu[idx]
                    saved_samples.append(sample)

    if total == 0:
        raise RuntimeError("Dataloader did not yield any samples.")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved_samples, save_path)

    baseline_acc = baseline_correct / total * 100.0
    recon_acc = recon_correct / total * 100.0
    return {
        "baseline_accuracy": baseline_acc,
        "reconstruction_accuracy": recon_acc,
        "accuracy_drop": baseline_acc - recon_acc,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataloader = build_dataloader(args)
    classifier = load_classifier(args.weights, device)
    autoencoder = load_autoencoder(args.ae_checkpoint, device)

    context_field = args.context_field or args.input_field

    results = evaluate(
        dataloader=dataloader,
        classifier=classifier,
        autoencoder=autoencoder,
        device=device,
        input_field=args.input_field,
        context_field=context_field,
        flexibility=args.flexibility,
        save_path=args.save_reconstructions,
    )

    print(
        "Baseline accuracy using '{field}': {acc:.2f}%".format(
            field=args.input_field, acc=results["baseline_accuracy"]
        )
    )
    print(
        "After autoencoder reconstruction accuracy: {acc:.2f}%".format(
            acc=results["reconstruction_accuracy"]
        )
    )
    print(
        "Accuracy drop: {drop:.2f} percentage points".format(
            drop=results["accuracy_drop"]
        )
    )


if __name__ == "__main__":
    main()
