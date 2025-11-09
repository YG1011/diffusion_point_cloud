"""Attack registry exposing ready-to-use adversarial attacks."""
from __future__ import annotations

from typing import Dict, Type

from .add import PointAdd
from .attack import Attack
from .cw import CW
from .drop import PointDrop
from .knn import KNN
from .pgd import PGD_Linf
from .pgdl2 import PGD_L2
from .vanila import VANILA

__all__ = [
    'Attack',
    'PointAdd',
    'PointDrop',
    'CW',
    'KNN',
    'PGD_Linf',
    'PGD_L2',
    'VANILA',
    'ATTACK_REGISTRY',
]

ATTACK_REGISTRY: Dict[str, Type[Attack]] = {
    'pointadd': PointAdd,
    'pointdrop': PointDrop,
    'cw': CW,
    'knn': KNN,
    'pgd_linf': PGD_Linf,
    'pgd': PGD_Linf,
    'pgd_l2': PGD_L2,
    'pgdl2': PGD_L2,
    'vanila': VANILA,
}
