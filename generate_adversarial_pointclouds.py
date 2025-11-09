"""Command line utility for generating adversarial ModelNet40 point clouds."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from attacks import ATTACK_REGISTRY
from models.dgcnn import build_dgcnn_classifier
from utils.modelnet40 import ModelNet40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate adversarial point clouds for ModelNet40.')
    parser.add_argument('--dataset-root', type=str, required=True, help='Directory containing ModelNet40 HDF5 files.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Dataset split to attack.')
    parser.add_argument('--num-points', type=int, default=1024, help='Number of points per point cloud.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size used when iterating over the dataset.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--no-normalize', action='store_true', help='Disable per-shape normalisation.')
    parser.add_argument('--random-rotate', action='store_true', help='Randomly shuffle point order for each sample.')
    parser.add_argument('--seed', type=int, default=3, help='Random seed for attacks and sampling.')

    parser.add_argument('--attack', type=str, required=True, choices=sorted(ATTACK_REGISTRY.keys()),
                        help='Attack to apply to the dataset.')
    parser.add_argument('--steps', type=int, help='Number of optimisation steps used by iterative attacks.')
    parser.add_argument('--eps', type=float, help='Maximum perturbation radius for PGD-based attacks.')
    parser.add_argument('--alpha', type=float, help='Step size for PGD-based attacks.')
    parser.add_argument('--lr', type=float, help='Learning rate for optimisation-based attacks.')
    parser.add_argument('--num-attack-points', type=int,
                        help='Number of points to add/drop for PointAdd/PointDrop attacks.')
    parser.add_argument('--c', type=float, help='Regularisation constant for CW-like attacks.')
    parser.add_argument('--kappa', type=float, help='Confidence parameter used by CW/KNN attacks.')
    parser.add_argument('--no-random-start', action='store_true', help='Disable random start for PGD attacks.')

    parser.add_argument('--model-checkpoint', type=str, required=True,
                        help='Path to the pretrained DGCNN checkpoint (.pth).')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on.')
    parser.add_argument('--output-dir', type=str, default='data_attacked',
                        help='Base directory used to store adversarial examples.')
    return parser.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = build_dgcnn_classifier(num_classes=40)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ('model_state_dict', 'state_dict', 'net', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    else:
        raise ValueError(f'Unexpected checkpoint format at {ckpt_path!r}.')

    cleaned_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f'[Warning] Missing keys when loading checkpoint: {missing}')
    if unexpected:
        print(f'[Warning] Unexpected keys when loading checkpoint: {unexpected}')

    model.to(device)
    model.eval()
    return model


def build_attack(name: str, model: torch.nn.Module, device: torch.device, args: argparse.Namespace):
    name = name.lower()
    AttackCls = ATTACK_REGISTRY[name]
    kwargs: Dict[str, Any] = {'device': device, 'seed': args.seed}

    if AttackCls.__name__ == 'PGD_Linf':
        kwargs.update({
            'eps': args.eps if args.eps is not None else 0.05,
            'alpha': args.alpha if args.alpha is not None else 0.01,
            'steps': args.steps if args.steps is not None else 10,
            'random_start': not args.no_random_start,
        })
    elif AttackCls.__name__ == 'PGD_L2':
        kwargs.update({
            'eps': args.eps if args.eps is not None else 1.0,
            'alpha': args.alpha if args.alpha is not None else 0.02,
            'steps': args.steps if args.steps is not None else 10,
            'random_start': not args.no_random_start,
        })
    elif AttackCls.__name__ == 'PointDrop':
        kwargs.update({
            'num_points': args.num_attack_points if args.num_attack_points is not None else 200,
            'steps': args.steps if args.steps is not None else 10,
        })
    elif AttackCls.__name__ == 'PointAdd':
        kwargs.update({
            'num_points': args.num_attack_points if args.num_attack_points is not None else 200,
            'steps': args.steps if args.steps is not None else 7,
            'eps': args.eps if args.eps is not None else 0.05,
            'lr': args.lr if args.lr is not None else 0.01,
        })
    elif AttackCls.__name__ == 'CW':
        kwargs.update({
            'c': args.c if args.c is not None else 1.0,
            'kappa': args.kappa if args.kappa is not None else 0.0,
            'steps': args.steps if args.steps is not None else 50,
            'lr': args.lr if args.lr is not None else 0.01,
        })
    elif AttackCls.__name__ == 'KNN':
        kwargs.update({
            'kappa': args.kappa if args.kappa is not None else 0.0,
            'steps': args.steps if args.steps is not None else 200,
            'lr': args.lr if args.lr is not None else 0.01,
        })
    elif AttackCls.__name__ == 'VANILA':
        pass

    attack = AttackCls(model=model, **kwargs)
    serialisable: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            serialisable[key] = value
        else:
            serialisable[key] = str(value)
    return attack, serialisable


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset = ModelNet40(
        root=args.dataset_root,
        split=args.split,
        num_points=args.num_points,
        normalize=not args.no_normalize,
        random_rotate=args.random_rotate,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    model = load_model(args.model_checkpoint, device)
    attack, attack_params = build_attack(args.attack, model, device, args)

    dataset_name = getattr(dataset, 'dataset_name', dataset.__class__.__name__)
    attack_name = attack.name
    save_root = os.path.join(args.output_dir, f'{dataset_name}_{attack_name}')
    os.makedirs(save_root, exist_ok=True)

    metadata: Dict[str, Any] = {
        'dataset': dataset_name,
        'split': args.split,
        'num_points': args.num_points,
        'attack': attack_name,
        'attack_params': attack_params,
        'model_checkpoint': os.path.abspath(args.model_checkpoint),
        'device': str(device),
    }

    attacked_batches = attack.save(
        dataloader,
        root=save_root,
        args=metadata,
    )

    output_index = os.path.join(save_root, 'metadata.json')
    with open(output_index, 'w', encoding='utf-8') as fp:
        json.dump({
            'num_samples': len(attacked_batches),
            'metadata': metadata,
        }, fp, indent=4)
    print(f'Saved adversarial examples to {save_root}')


if __name__ == '__main__':
    main()
