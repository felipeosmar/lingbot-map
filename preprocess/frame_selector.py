"""Seletor de frames: portão de qualidade, seleção por movimento, orçamento, CLI."""
from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Iterable, Iterator

import cv2
import numpy as np
from tqdm import tqdm

from preprocess import frame_metrics as fm


@dataclass
class FrameStat:
    idx: int
    sharpness: float
    brightness: float
    clipped: float
    entropy: float
    hist: np.ndarray
    flow_prev: float  # fluxo do frame analisado anterior -> este (0.0 no primeiro)


def quality_gate(
    stats: list[FrameStat],
    min_entropy: float = 2.5,
    max_clipped: float = 0.6,
    min_brightness: float = 6.0,
    max_brightness: float = 250.0,
) -> list[int]:
    """Retorna as posições (em `stats`) que passam no portão de exposição/informação."""
    survivors = []
    for pos, s in enumerate(stats):
        if s.entropy < min_entropy:
            continue
        if s.clipped > max_clipped:
            continue
        if s.brightness < min_brightness or s.brightness > max_brightness:
            continue
        survivors.append(pos)
    return survivors
