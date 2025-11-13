"""Run diffusion-based purification on point clouds and evaluate with DGCNN."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from models.autoencoder import AutoEncoder
from models.dgcnn import build_dgcnn_classifier
from models.diffusion_sampler import DiffusionSampler
from utils.frequency import FrequencyGuidance, FrequencyMaskGenerator
from utils.modelnet40 import ModelNet40
from utils.voxel import PointCloudToVoxel, VoxelToPointCloud
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
        "--grid-size",
        type=int,
        nargs=3,
        default=(64, 64, 64),
        help="Spatial resolution of the voxel grid (default: 64 64 64).",
    )
    parser.add_argument(
        "--amplitude-radius",
        type=float,
        default=None,
        help="Radius (in voxels) of the low-pass mask applied to amplitudes.",
    )
    parser.add_argument(
        "--phase-radius",
        type=float,
        default=None,
        help="Radius (in voxels) of the low-pass mask applied to phases.",
    )
    parser.add_argument(
        "--phase-delta",
        type=float,
        default=None,
        help="Optional clipping range for phase correction (in radians).",
    )
    parser.add_argument(
        "--forward-noise-steps",
        type=int,
        default=None,
        help="Timestep used to predict a clean reference before sampling.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=None,
        help=(
            "Number of reverse diffusion steps. Defaults to the schedule length "
            "stored in the autoencoder checkpoint."
        ),
    )
    parser.add_argument(
        "--sampler-flexibility",
        type=float,
        default=0.0,
        help="Flexibility parameter controlling sigma interpolation (default: 0.0).",
    )
    parser.add_argument(
        "--input-field",
        type=str,
        default="attacked",
        help=(
            "Dictionary key to read the adversarial point cloud from each batch. "
            "Use 'pointcloud' when evaluating clean data."
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
        "--compare-field",
        type=str,
        default=None,
        help="Optional second field to report baseline accuracy for reference.",
    )
    parser.add_argument(
        "--save-purified",
        type=Path,
        default=None,
        help="Optional path to store purified point clouds as a .pt file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation (default: cuda if available).",
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
    return parser.parse_args()


def load_classifier(weights_path: Path, device: torch.device) -> torch.nn.Module:
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

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[len("module.") :] if key.startswith("module.") else key
        new_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")

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


def build_sampler(
    autoencoder: AutoEncoder,
    grid_size: Tuple[int, int, int],
    amplitude_radius: Optional[float],
    phase_radius: Optional[float],
    phase_delta: Optional[float],
    forward_noise_steps: Optional[int],
) -> Tuple[DiffusionSampler, PointCloudToVoxel, VoxelToPointCloud]:
    d, h, w = grid_size
    if amplitude_radius is None:
        amplitude_radius = min(d, h, w) / 4.0
    if phase_radius is None:
        phase_radius = amplitude_radius / 2.0

    mask_generator = FrequencyMaskGenerator(
        grid_size=grid_size,
        amplitude_radius=amplitude_radius,
        phase_radius=phase_radius,
    )
    frequency_guidance = FrequencyGuidance(
        mask_generator=mask_generator, phase_delta=phase_delta
    )

    voxeliser = PointCloudToVoxel(grid_size=grid_size)
    devoxeliser = VoxelToPointCloud(grid_size=grid_size)

    sampler = DiffusionSampler(
        model=autoencoder.diffusion.net,
        var_sched=autoencoder.diffusion.var_sched,
        pointcloud_to_voxel=voxeliser,
        voxel_to_pointcloud=devoxeliser,
        frequency_guidance=frequency_guidance,
        forward_noise_steps=forward_noise_steps,
    )
    autoencoder.diffusion.net.eval()
    return sampler, voxeliser, devoxeliser


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
    sampler: DiffusionSampler,
    device: torch.device,
    input_field: str,
    context_field: str,
    compare_field: Optional[str],
    forward_noise_steps: Optional[int],
    diffusion_steps: Optional[int],
    flexibility: float,
    save_path: Optional[Path],
) -> Dict[str, float]:
    total = 0
    baseline_correct = 0
    purified_correct = 0
    compare_correct = 0
    saved_samples: List[Dict[str, torch.Tensor]] = []

    if diffusion_steps is not None and diffusion_steps <= 0:
        raise ValueError("--diffusion-steps must be positive when provided.")

    max_steps = sampler.var_sched.num_steps
    if diffusion_steps is not None and diffusion_steps > max_steps:
        raise ValueError(
            f"Requested {diffusion_steps} diffusion steps but schedule only has {max_steps}."
        )

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
            purified = sampler.purify(
                x_adv=points,
                context=latents,
                start_step=forward_noise_steps,
                num_steps=diffusion_steps,
                flexibility=flexibility,
            )
            logits_purified = classifier(purified.transpose(1, 2))
            purified_correct += (logits_purified.argmax(dim=1) == labels).sum().item()

            if compare_field is not None:
                if compare_field not in batch:
                    raise KeyError(
                        f"Batch is missing comparison field '{compare_field}'."
                    )
                compare_points = batch[compare_field].to(device)
                compare_logits = classifier(compare_points.transpose(1, 2))
                compare_correct += (
                    compare_logits.argmax(dim=1) == labels
                ).sum().item()

            if save_path is not None:
                purified_cpu = purified.detach().cpu()
                for idx in range(points.size(0)):
                    sample: Dict[str, torch.Tensor] = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            sample[key] = value[idx].detach().cpu()
                    sample["purified"] = purified_cpu[idx]
                    saved_samples.append(sample)

            total += points.size(0)

    results = {
        "baseline_acc": baseline_correct / total,
        "purified_acc": purified_correct / total,
    }
    if compare_field is not None:
        results["compare_acc"] = compare_correct / total

    if save_path is not None:
        torch.save(saved_samples, save_path)

    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataloader = build_dataloader(args)
    classifier = load_classifier(args.weights, device)
    autoencoder = load_autoencoder(args.ae_checkpoint, device)

    grid_size = tuple(int(v) for v in args.grid_size)
    sampler, _, _ = build_sampler(
        autoencoder=autoencoder,
        grid_size=grid_size,
        amplitude_radius=args.amplitude_radius,
        phase_radius=args.phase_radius,
        phase_delta=args.phase_delta,
        forward_noise_steps=args.forward_noise_steps,
    )

    context_field = args.context_field if args.context_field is not None else args.input_field

    results = evaluate(
        dataloader=dataloader,
        classifier=classifier,
        autoencoder=autoencoder,
        sampler=sampler,
        device=device,
        input_field=args.input_field,
        context_field=context_field,
        compare_field=args.compare_field,
        forward_noise_steps=args.forward_noise_steps,
        diffusion_steps=args.diffusion_steps,
        flexibility=args.sampler_flexibility,
        save_path=args.save_purified,
    )

    print(
        f"Baseline accuracy using '{args.input_field}': "
        f"{results['baseline_acc'] * 100:.2f}%"
    )
    print(
        "After diffusion purification accuracy: "
        f"{results['purified_acc'] * 100:.2f}%"
    )
    print(
        f"Accuracy gain: {(results['purified_acc'] - results['baseline_acc']) * 100:.2f} percentage points"
    )

    if args.compare_field is not None and "compare_acc" in results:
        print(
            f"Reference accuracy using '{args.compare_field}': "
            f"{results['compare_acc'] * 100:.2f}%"
        )

    if args.save_purified is not None:
        print(f"Purified point clouds saved to {args.save_purified}")


if __name__ == "__main__":
    main()

