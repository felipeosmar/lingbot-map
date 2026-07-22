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


def build_survivor_flow(stats: list[FrameStat], survivor_pos: list[int]) -> list[float]:
    """Fluxo do sobrevivente anterior a este, somando flow_prev dos frames pulados."""
    out: list[float] = []
    for j, pos in enumerate(survivor_pos):
        if j == 0:
            out.append(0.0)
            continue
        prev_pos = survivor_pos[j - 1]
        out.append(float(sum(stats[k].flow_prev for k in range(prev_pos + 1, pos + 1))))
    return out


def select_by_motion(
    survivors: list[FrameStat],
    survivor_flow: list[float],
    threshold: float,
    max_gap: int = 45,
    novelty_floor: float = 0.02,
) -> list[tuple[int, str]]:
    """Seleção sequencial: fecha segmento por movimento ou gap; escolhe o mais nítido."""
    if not survivors:
        return []
    selected: list[tuple[int, str]] = [(survivors[0].idx, "forced_first")]
    last = survivors[0]
    motion = 0.0
    segment: list[FrameStat] = []

    for j in range(1, len(survivors)):
        f = survivors[j]
        motion += survivor_flow[j]
        segment.append(f)
        gap = f.idx - last.idx
        hit_motion = motion >= threshold
        hit_gap = gap >= max_gap
        if not (hit_motion or hit_gap):
            continue
        cand = max(segment, key=lambda x: x.sharpness)
        novel = fm.histogram_distance(cand.hist, last.hist) >= novelty_floor
        if novel or hit_gap:
            reason = "max_gap" if hit_gap else "motion"
            selected.append((cand.idx, reason))
            last = cand
        motion = 0.0
        segment = []

    selected.sort(key=lambda t: t[0])
    return selected


def fit_budget(
    survivors: list[FrameStat],
    survivor_flow: list[float],
    budget: int,
    max_gap: int = 45,
    novelty_floor: float = 0.02,
    iters: int = 30,
) -> tuple[list[tuple[int, str]], float]:
    """Busca binária no threshold de movimento para o total cair em [0.85*budget, budget].

    threshold maior => menos frames. Se houver <= budget sobreviventes, mantém todos.
    """
    if len(survivors) <= budget:
        return [(s.idx, "forced_first" if i == 0 else "motion")
                for i, s in enumerate(survivors)], 0.0

    lo, hi = 0.0, max(sum(survivor_flow), 1e-6)
    best: tuple[list[tuple[int, str]], float] | None = None
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        sel = select_by_motion(survivors, survivor_flow, mid, max_gap, novelty_floor)
        n = len(sel)
        if n > budget:
            lo = mid                      # precisa filtrar mais
        else:
            best = (sel, mid)
            hi = mid                      # tenta aproximar do teto (mais frames)
        if best is not None and 0.85 * budget <= len(best[0]) <= budget:
            break
    if best is None:
        # piso forçado por max_gap excede o budget: devolve o menor conjunto possível
        best = (select_by_motion(survivors, survivor_flow, hi, max_gap, novelty_floor), hi)
    return best
