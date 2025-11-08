"""Utilities to convert raw ShapeNetCore.v2 meshes into HDF5 point clouds."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import trimesh
from tqdm import tqdm

from utils.dataset import cate_to_synsetid, synsetid_to_cate


NORMALIZATION_CHOICES = {
    None,
    'none',
    'shape_unit',
    'shape_bbox',
    'shape_sphere',
}


class MeshLoadError(RuntimeError):
    """Raised when a mesh cannot be loaded."""


def _resolve_categories(categories: Sequence[str]) -> List[str]:
    if not categories:
        raise ValueError('`categories` must contain at least one category name or "all".')

    if len(categories) == 1 and categories[0].lower() == 'all':
        return sorted(list(cate_to_synsetid.keys()))

    resolved = []
    for cate in categories:
        cate = cate.strip()
        if not cate:
            continue
        if cate not in cate_to_synsetid:
            raise KeyError(f'Unknown category "{cate}". Available categories: {sorted(cate_to_synsetid)}')
        resolved.append(cate)
    if not resolved:
        raise ValueError('No valid categories were provided.')
    return sorted(resolved)


def _load_mesh(model_dir: Path) -> trimesh.Trimesh:
    """
    Try hard to find a usable mesh under `model_dir`.
    Priority: common normalized names; fallback: recursive search with many suffixes.
    """
    # 1) 常见相对路径（优先尝试）
    candidate_files = [
        'models/model_normalized.obj',
        'models/model_normalized.ply',
        'models/model.obj',
        'models/model.ply',
        'models/model.off',
        'models/model_normalized.glb',
        'models/model_normalized.gltf',
        'models/model.glb',
        'models/model.gltf',
        'model.obj',
        'model.ply',
        'model.off',
        'model.glb',
        'model.gltf',
        'models/model_normalized.obj.gz',
        'models/model.obj.gz',
        'model.obj.gz',
    ]
    for rel_path in candidate_files:
        p = model_dir / rel_path
        if p.exists():
            mesh = trimesh.load(p, force='mesh')
            if isinstance(mesh, trimesh.Scene):
                geoms = list(mesh.geometry.values())
                if not geoms:
                    raise MeshLoadError(f'No geometries found in scene: {p}')
                mesh = trimesh.util.concatenate(geoms)
            if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
                raise MeshLoadError(f'Unsupported or empty mesh: {p}')
            return mesh

    # 2) 递归兜底：搜一切可能的网格后缀
    exts = ('.obj', '.ply', '.off', '.stl', '.dae', '.glb', '.gltf', '.obj.gz')
    for p in model_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                mesh = trimesh.load(p, force='mesh')
                if isinstance(mesh, trimesh.Scene):
                    geoms = list(mesh.geometry.values())
                    if not geoms:
                        continue
                    mesh = trimesh.util.concatenate(geoms)
                if isinstance(mesh, trimesh.Trimesh) and mesh.vertices.size > 0:
                    return mesh
            except Exception:
                continue

    raise MeshLoadError(
        f'Unable to locate a usable mesh under {model_dir}. '
        f'Tried common names and recursive search for {exts}.'
    )



def _normalize_points(points: np.ndarray, mode: Optional[str]) -> Tuple[np.ndarray, np.ndarray, float]:
    if mode in (None, 'none'):
        return points, np.zeros(3, dtype=np.float32), 1.0

    if mode not in NORMALIZATION_CHOICES:
        raise ValueError(f'Unsupported normalization mode: {mode}')

    shift = np.zeros(3, dtype=np.float32)
    scale = 1.0

    if mode == 'shape_unit':
        shift = points.mean(axis=0)
        scale = float(points.std())
    elif mode == 'shape_bbox':
        max_xyz = points.max(axis=0)
        min_xyz = points.min(axis=0)
        shift = (max_xyz + min_xyz) / 2.0
        scale = float(np.max(max_xyz - min_xyz) / 2.0)
    elif mode == 'shape_sphere':
        shift = points.mean(axis=0)
        scale = float(np.max(np.linalg.norm(points - shift, axis=1)))

    if scale < 1e-8:
        scale = 1.0
    normalized = (points - shift) / scale
    return normalized.astype(np.float32), shift.astype(np.float32), float(scale)


def _sample_point_cloud(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    if points.shape[0] != num_points:
        # If the returned number of points is smaller, resample with replacement
        missing = num_points - points.shape[0]
        if missing > 0:
            repeat_idx = np.random.choice(points.shape[0], size=missing, replace=True)
            points = np.concatenate([points, points[repeat_idx]], axis=0)
    return points.astype(np.float32)


def _load_split_config(split_config_path: Path) -> Dict[str, Dict[str, List[str]]]:
    with split_config_path.open('r') as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError('Split configuration must be a JSON object mapping synset IDs to split dictionaries.')
    normalized: Dict[str, Dict[str, List[str]]] = {}
    for synset_id, splits in data.items():
        if not isinstance(synset_id, str):
            raise ValueError('Synset IDs in the split configuration must be strings.')
        if not isinstance(splits, dict):
            raise ValueError(f'Split definition for {synset_id} must be a dictionary.')
        normalized[synset_id] = {
            split: list(splits.get(split, [])) for split in ('train', 'val', 'test')
        }
    return normalized


def _generate_random_split(model_ids: Sequence[str], train_ratio: float, val_ratio: float) -> Dict[str, List[str]]:
    if train_ratio < 0 or val_ratio < 0 or (train_ratio + val_ratio) > 1:
        raise ValueError('Invalid split ratios. Ensure 0 <= train_ratio, val_ratio and train_ratio + val_ratio <= 1.')

    num_models = len(model_ids)
    indices = list(range(num_models))
    random.shuffle(indices)
    train_end = int(num_models * train_ratio)
    val_end = train_end + int(num_models * val_ratio)

    split = {
        'train': [model_ids[i] for i in indices[:train_end]],
        'val': [model_ids[i] for i in indices[train_end:val_end]],
        'test': [model_ids[i] for i in indices[val_end:]],
    }
    return split


def _collect_model_ids(category_dir: Path) -> List[str]:
    model_ids = []
    for item in category_dir.iterdir():
        if not item.is_dir():
            continue
        # Each model ID directory is expected to contain a mesh in a nested "models" folder
        mesh_dir = item / 'models'
        if mesh_dir.exists():
            model_ids.append(item.name)
        else:
            # Some datasets may place the mesh directly inside the model folder
            if any((item / candidate).exists() for candidate in ('model.obj', 'model.ply', 'model.off')):
                model_ids.append(item.name)
    model_ids.sort()
    return model_ids


def _process_category(
    synset_id: str,
    category_root: Path,
    model_ids_by_split: Dict[str, List[str]],
    num_points: int,
    normalization: Optional[str],
) -> Dict[str, np.ndarray]:
    category_results: Dict[str, List[np.ndarray]] = {split: [] for split in ('train', 'val', 'test')}
    for split, model_ids in model_ids_by_split.items():
        if not model_ids:
            continue
        for model_id in tqdm(model_ids, desc=f'{synsetid_to_cate[synset_id]} ({split})', leave=False):
            model_dir = category_root / model_id
            try:
                mesh = _load_mesh(model_dir)
                points = _sample_point_cloud(mesh, num_points)
                points, _, _ = _normalize_points(points, normalization)
                category_results[split].append(points)
            except MeshLoadError as exc:
                tqdm.write(f'[WARN] {exc}')
            except Exception as exc:  # pragma: no cover - unexpected issues are surfaced to the user
                tqdm.write(f'[WARN] Failed to process {model_dir}: {exc}')
        if category_results[split]:
            category_results[split] = np.stack(category_results[split], axis=0)
        else:
            category_results[split] = np.empty((0, num_points, 3), dtype=np.float32)
    return category_results


def _save_hdf5(
    output_path: Path,
    results: Dict[str, Dict[str, np.ndarray]],
    model_ids: Dict[str, Dict[str, List[str]]],
    compression: Optional[str] = 'gzip',
    compression_level: int = 4,
) -> None:
    """
    Save arrays to HDF5. Store model_ids as datasets (NOT attributes)
    to avoid HDF5 object header size limits.
    """
    # 变长 UTF-8 字符串 dtype
    str_dtype = h5py.string_dtype(encoding='utf-8', length=None)

    with h5py.File(output_path, 'w') as h5f:
        for synset_id, split_arrays in results.items():
            cate_group = h5f.create_group(synset_id)
            for split_name, array in split_arrays.items():
                # 点云数据
                cate_group.create_dataset(
                    split_name,
                    data=array,
                    compression=compression,
                    compression_opts=compression_level,
                    dtype=np.float32,
                )
                # 模型 ID 列表：单独的数据集，避免 attribute 过大
                ids = model_ids.get(synset_id, {}).get(split_name, [])
                ids_arr = np.array(ids, dtype=object)  # 先 object，再由 h5py 转成变长 utf-8
                cate_group.create_dataset(
                    f'{split_name}_model_ids',
                    data=ids_arr,
                    dtype=str_dtype
                )


def preprocess(
    dataset_root: Path,
    output_path: Path,
    categories: Sequence[str],
    num_points: int,
    normalization: Optional[str],
    split_config: Optional[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    compression: str,
    compression_level: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f'Output file {output_path} already exists. Use --overwrite to replace it.')

    resolved_categories = _resolve_categories(categories)
    random.seed(seed)
    np.random.seed(seed)

    if split_config is not None:
        split_data = _load_split_config(split_config)
    else:
        split_data = {}

    processed_results: Dict[str, Dict[str, np.ndarray]] = {}
    processed_model_ids: Dict[str, Dict[str, List[str]]] = {}

    for cate_name in resolved_categories:
        synset_id = cate_to_synsetid[cate_name]
        category_root = dataset_root / synset_id
        if not category_root.exists():
            raise FileNotFoundError(f'Category directory not found: {category_root}')

        all_model_ids = _collect_model_ids(category_root)
        if not all_model_ids:
            raise RuntimeError(f'No model directories found under {category_root}')

        if synset_id in split_data:
            provided_split = split_data[synset_id]
            split_ids = {}
            for split in ('train', 'val', 'test'):
                filtered = [mid for mid in provided_split.get(split, []) if (category_root / mid).is_dir()]
                split_ids[split] = filtered
            assigned = set().union(*split_ids.values())
            missing_models = [mid for mid in all_model_ids if mid not in assigned]
            if missing_models:
                tqdm.write(
                    f'[INFO] {len(missing_models)} models in {synset_id} were not present in the provided split. '
                    'They will be appended to the training split.'
                )
                split_ids['train'].extend(missing_models)
        else:
            split_ids = _generate_random_split(all_model_ids, train_ratio, val_ratio)

        processed_model_ids[synset_id] = split_ids
        processed_results[synset_id] = _process_category(
            synset_id=synset_id,
            category_root=category_root,
            model_ids_by_split=split_ids,
            num_points=num_points,
            normalization=normalization,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_hdf5(
        output_path=output_path,
        results=processed_results,
        model_ids=processed_model_ids,
        compression=compression,
        compression_level=compression_level,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Preprocess ShapeNetCore.v2 models into an HDF5 point cloud dataset.'
    )
    parser.add_argument('--dataset-root', type=Path, default=Path('/home/ubuntu/641/YYG/diffusion-point-cloud-main/data/ShapeNetCore.v2'), help='Path to the ShapeNetCore.v2 root directory.')
    parser.add_argument('--output', type=Path, default=Path('/home/ubuntu/641/YYG/diffusion-point-cloud-main/data/shapenet.hdf5'), help='Path to the output HDF5 file.')
    parser.add_argument(
        '--categories',
        type=str,
        default='all',
        help='Comma-separated category names (e.g., "airplane,chair"). Use "all" to include every category.',
    )
    parser.add_argument('--num-points', type=int, default=2048, help='Number of points to sample for each shape.')
    parser.add_argument(
        '--normalization',
        type=str,
        default='shape_bbox',
        choices=sorted([m for m in NORMALIZATION_CHOICES if m]),
        help='Normalization strategy applied to each sampled point cloud.',
    )
    parser.add_argument('--split-config', type=Path, help='Optional JSON file specifying train/val/test splits per synset ID.')
    parser.add_argument('--train-ratio', type=float, default=0.85, help='Train split ratio when no split config is provided.')
    parser.add_argument('--val-ratio', type=float, default=0.05, help='Validation split ratio when no split config is provided.')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed for shuffling and sampling.')
    parser.add_argument('--compression', type=str, default='gzip', help='Compression algorithm used by h5py.')
    parser.add_argument('--compression-level', type=int, default=4, help='Compression level for the HDF5 datasets.')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting the output file if it already exists.')
    args = parser.parse_args()

    categories = [cate.strip() for cate in args.categories.split(',') if cate.strip()]
    args.categories = categories
    if args.normalization == 'none':
        args.normalization = None
    return args


def main() -> None:
    args = parse_args()
    preprocess(
        dataset_root=args.dataset_root,
        output_path=args.output,
        categories=args.categories,
        num_points=args.num_points,
        normalization=args.normalization,
        split_config=args.split_config,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        compression=args.compression,
        compression_level=args.compression_level,
        overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main()
