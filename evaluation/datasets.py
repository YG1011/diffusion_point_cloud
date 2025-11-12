"""Reusable dataset helpers for evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class AdversarialExamplesDataset(Dataset):
    """Dataset wrapper around adversarial examples saved as ``.pt`` files."""

    def __init__(self, data_path: Path) -> None:
        super().__init__()
        raw = torch.load(data_path, map_location="cpu")
        if not isinstance(raw, list):
            raise RuntimeError(
                "Expected the adversarial file to contain a list of samples."
            )

        self.samples: List[Dict[str, torch.Tensor]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise RuntimeError(
                    "Unexpected sample format inside adversarial file."
                )

            sample: Dict[str, torch.Tensor] = {}
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    tensor = value.detach().clone()
                    if tensor.is_floating_point():
                        tensor = tensor.to(dtype=torch.float32)
                    sample[key] = tensor
                else:
                    raise RuntimeError(
                        f"Sample entry for key '{key}' is not a tensor: {type(value)}"
                    )
            self.samples.append(sample)

        if not self.samples:
            raise RuntimeError("No samples found inside the adversarial file.")

        self.dataset_name = "AdversarialExamples"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {key: value.clone() for key, value in sample.items()}

