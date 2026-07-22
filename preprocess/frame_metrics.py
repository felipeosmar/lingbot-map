"""Funções puras de métrica de frame (grayscale uint8). Sem estado, sem I/O."""
from __future__ import annotations

import cv2
import numpy as np


def sharpness(gray: np.ndarray) -> float:
    """Variância do Laplaciano — proxy de foco/nitidez (maior = mais nítido)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure(gray: np.ndarray) -> tuple[float, float, float]:
    """Retorna (brilho médio 0-255, fração de pixels estourados/escuros, entropia em bits)."""
    brightness = float(gray.mean())
    clipped = float(np.mean((gray < 8) | (gray > 247)))
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    return brightness, clipped, entropy


def histogram(gray: np.ndarray, bins: int = 64) -> np.ndarray:
    """Histograma de intensidade normalizado (soma 1.0)."""
    h = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel().astype(np.float64)
    total = h.sum()
    return h / total if total > 0 else h


def histogram_distance(h_a: np.ndarray, h_b: np.ndarray) -> float:
    """Distância de variação total entre dois histogramas normalizados (0-1)."""
    return float(0.5 * np.abs(h_a - h_b).sum())


def flow_magnitude(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Magnitude média do fluxo óptico denso (Farneback) de a -> b."""
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(mag.mean())
