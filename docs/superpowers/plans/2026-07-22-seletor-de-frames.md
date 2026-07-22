# Seletor de Frames — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pré-processador standalone que seleciona frames de um vídeo por qualidade (anti-blur, exposição) e conteúdo (movimento, novidade), respeitando um orçamento, produzindo uma pasta de frames pronta para o `batch_demo --image_folder`.

**Architecture:** Dois módulos em `preprocess/`: `frame_metrics.py` (funções puras de métrica, sem estado) e `frame_selector.py` (portão de qualidade → seleção por movimento → ajuste de orçamento → escrita + CLI). Análise em grayscale reduzido; duas passagens na fonte (passe 1 = métricas/fluxo; passe 2 = escrita dos frames escolhidos na resolução original) para limitar memória.

**Tech Stack:** Python 3.10+, OpenCV (`opencv-python`), NumPy, tqdm, pytest. Sem GPU, sem PyTorch, sem o pacote do modelo.

## Global Constraints

- Sem GPU, sem PyTorch, sem importar `lingbot_map` — só `cv2`, `numpy`, `tqdm`.
- Todas as métricas operam em **grayscale reduzido** à largura `analysis_width` (default 320).
- Saída: frames renomeados `frame_%06d.png` em **ordem temporal**, resolução **original**.
- Nunca retornar vazio nem truncar silenciosamente — relaxar limiares e/ou registrar cortes no relatório.
- Orçamento default `900`; faixa de aterrissagem aceitável `[0.85·budget, budget]`.
- Fluxo entre frames = magnitude média do fluxo óptico Farneback.
- TDD: teste falha primeiro; commits frequentes.

---

### Task 1: Módulo de métricas puras (`frame_metrics.py`)

**Files:**
- Create: `preprocess/frame_metrics.py`
- Test: `tests/test_frame_metrics.py`

**Interfaces:**
- Consumes: nada (funções puras sobre `np.ndarray` grayscale `uint8`).
- Produces:
  - `sharpness(gray: np.ndarray) -> float`
  - `exposure(gray: np.ndarray) -> tuple[float, float, float]`  # (brightness 0-255, clipped_frac 0-1, entropy em bits 0-8)
  - `histogram(gray: np.ndarray, bins: int = 64) -> np.ndarray`  # normalizado, soma 1.0, shape (bins,)
  - `histogram_distance(h_a: np.ndarray, h_b: np.ndarray) -> float`  # total variation, 0-1
  - `flow_magnitude(gray_a: np.ndarray, gray_b: np.ndarray) -> float`  # >= 0

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_frame_metrics.py
import numpy as np
import cv2
import pytest
from preprocess import frame_metrics as fm


def _sharp_image():
    img = np.zeros((120, 160), np.uint8)
    img[:, ::8] = 255          # listras verticais nítidas
    return img


def test_sharpness_drops_with_blur():
    sharp = _sharp_image()
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=3)
    assert fm.sharpness(sharp) > fm.sharpness(blurred) * 2


def test_exposure_flags_dark_and_bright():
    dark = np.full((64, 64), 3, np.uint8)
    bright = np.full((64, 64), 252, np.uint8)
    b_dark, clip_dark, ent_dark = fm.exposure(dark)
    b_bright, clip_bright, _ = fm.exposure(bright)
    assert b_dark < 10 and clip_dark > 0.9 and ent_dark < 1.0
    assert b_bright > 245 and clip_bright > 0.9


def test_histogram_normalized_and_distance():
    a = np.random.RandomState(0).randint(0, 256, (64, 64), np.uint8)
    ha = fm.histogram(a)
    assert ha.shape == (64,)
    assert abs(ha.sum() - 1.0) < 1e-6
    assert fm.histogram_distance(ha, ha) < 1e-6
    b = np.full((64, 64), 0, np.uint8)
    hb = fm.histogram(b)
    assert fm.histogram_distance(ha, hb) > 0.5


def test_flow_magnitude_grows_with_shift():
    base = np.random.RandomState(1).randint(0, 256, (120, 160), np.uint8)
    shift1 = np.roll(base, 2, axis=1)
    shift5 = np.roll(base, 10, axis=1)
    m0 = fm.flow_magnitude(base, base)
    m1 = fm.flow_magnitude(base, shift1)
    m5 = fm.flow_magnitude(base, shift5)
    assert m0 < 0.5
    assert m5 > m1 > m0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_frame_metrics.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'preprocess.frame_metrics').

- [ ] **Step 3: Implement `preprocess/frame_metrics.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frame_metrics.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -f preprocess/frame_metrics.py tests/test_frame_metrics.py
git commit -m "feat(preprocess): funções puras de métrica de frame (nitidez, exposição, histograma, fluxo)"
```

---

### Task 2: Portão de qualidade (`quality_gate`)

**Files:**
- Create: `preprocess/frame_selector.py`
- Test: `tests/test_frame_selector.py`

**Interfaces:**
- Consumes: nada de tasks anteriores (usa só o dataclass local).
- Produces:
  - `@dataclass FrameStat` com campos: `idx:int, sharpness:float, brightness:float, clipped:float, entropy:float, hist:np.ndarray, flow_prev:float`
  - `quality_gate(stats: list[FrameStat], min_entropy: float = 2.5, max_clipped: float = 0.6, min_brightness: float = 6.0, max_brightness: float = 250.0) -> list[int]`  # retorna POSIÇÕES (índices em `stats`) que sobrevivem

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frame_selector.py
import numpy as np
import pytest
from preprocess import frame_selector as fs


def _stat(idx, sharp=100.0, bright=120.0, clip=0.0, ent=6.0, flow=1.0):
    return fs.FrameStat(idx=idx, sharpness=sharp, brightness=bright,
                        clipped=clip, entropy=ent, hist=np.ones(64) / 64, flow_prev=flow)


def test_quality_gate_drops_dark_bright_lowentropy():
    stats = [
        _stat(0),                             # ok
        _stat(1, bright=2.0, ent=0.5),        # escuro + baixa entropia -> fora
        _stat(2, bright=255.0, clip=0.95),    # estourado -> fora
        _stat(3, ent=1.0),                    # baixa entropia -> fora
        _stat(4),                             # ok
    ]
    survivors = fs.quality_gate(stats)
    assert survivors == [0, 4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frame_selector.py::test_quality_gate_drops_dark_bright_lowentropy -v`
Expected: FAIL (ModuleNotFoundError / AttributeError: no `FrameStat`).

- [ ] **Step 3: Implement dataclass + `quality_gate` in `preprocess/frame_selector.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_frame_selector.py::test_quality_gate_drops_dark_bright_lowentropy -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -f preprocess/frame_selector.py tests/test_frame_selector.py
git commit -m "feat(preprocess): FrameStat + portão de qualidade (exposição/entropia)"
```

---

### Task 3: Fluxo por sobrevivente + seleção por movimento

**Files:**
- Modify: `preprocess/frame_selector.py`
- Test: `tests/test_frame_selector.py`

**Interfaces:**
- Consumes: `FrameStat` (Task 2).
- Produces:
  - `build_survivor_flow(stats: list[FrameStat], survivor_pos: list[int]) -> list[float]`  # fluxo somado do sobrevivente anterior a este; [0]=0.0
  - `select_by_motion(survivors: list[FrameStat], survivor_flow: list[float], threshold: float, max_gap: int = 45, novelty_floor: float = 0.02) -> list[tuple[int, str]]`  # (FrameStat.idx, reason), reason ∈ {"forced_first","motion","max_gap"}, ordenado por idx

- [ ] **Step 1: Write the failing tests**

```python
def test_build_survivor_flow_sums_gaps():
    stats = [
        fs.FrameStat(0, 100, 120, 0, 6, np.ones(64)/64, 0.0),
        fs.FrameStat(1, 100, 120, 0, 6, np.ones(64)/64, 1.0),
        fs.FrameStat(2, 100, 120, 0, 6, np.ones(64)/64, 2.0),
        fs.FrameStat(3, 100, 120, 0, 6, np.ones(64)/64, 3.0),
    ]
    # sobreviventes pulam a posição 1: fluxo de 0->2 deve somar flow_prev[1]+flow_prev[2]
    flow = fs.build_survivor_flow(stats, [0, 2, 3])
    assert flow[0] == 0.0
    assert flow[1] == pytest.approx(3.0)   # 1.0 + 2.0
    assert flow[2] == pytest.approx(3.0)


def _mk(idx, sharp, flow, hist=None):
    h = hist if hist is not None else np.ones(64) / 64
    return fs.FrameStat(idx, sharp, 120, 0, 6, h, flow)


def test_select_picks_sharpest_in_segment_on_motion():
    # movimento acumula 1.0/frame; threshold 3 fecha segmento a cada ~3 frames
    survivors = [_mk(i, sharp=(10 if i != 2 else 999), flow=(0.0 if i == 0 else 1.0))
                 for i in range(5)]
    flow = [s.flow_prev for s in survivors]
    sel = fs.select_by_motion(survivors, flow, threshold=3.0, max_gap=100, novelty_floor=0.0)
    idxs = [i for i, _ in sel]
    assert idxs[0] == 0                     # primeiro sempre entra
    assert 2 in idxs                        # o mais nítido do 1º segmento (idx 2, sharp 999)
    assert sel[0][1] == "forced_first"


def test_select_max_gap_forces_keep_when_static():
    # sem movimento (flow=0) -> só o gap força seleções
    survivors = [_mk(i, sharp=10, flow=0.0) for i in range(10)]
    flow = [0.0] * 10
    sel = fs.select_by_motion(survivors, flow, threshold=5.0, max_gap=3, novelty_floor=0.0)
    idxs = [i for i, _ in sel]
    assert idxs[0] == 0
    assert any(r == "max_gap" for _, r in sel)
    assert max(np.diff([0] + idxs)) <= 3    # nunca deixa buraco > max_gap


def test_select_novelty_skips_near_duplicates():
    hA = np.zeros(64); hA[0] = 1.0
    hB = np.zeros(64); hB[63] = 1.0
    # 3 quase-idênticos (hA) depois 1 distinto (hB); threshold baixo dispararia em todos
    survivors = [_mk(0, 10, 0.0, hA), _mk(1, 10, 5.0, hA),
                 _mk(2, 10, 5.0, hA), _mk(3, 10, 5.0, hB)]
    flow = [s.flow_prev for s in survivors]
    sel = fs.select_by_motion(survivors, flow, threshold=1.0, max_gap=100, novelty_floor=0.3)
    idxs = [i for i, _ in sel]
    assert 0 in idxs and 3 in idxs          # primeiro e o distinto entram
    assert idxs.count(1) == 0               # duplicatas de hA são puladas por novidade
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_frame_selector.py -k "build_survivor_flow or select_" -v`
Expected: FAIL (AttributeError: no `build_survivor_flow` / `select_by_motion`).

- [ ] **Step 3: Implement in `preprocess/frame_selector.py`**

```python
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
            reason = "max_gap" if (hit_gap and not novel) else "motion"
            selected.append((cand.idx, reason))
            last = cand
        motion = 0.0
        segment = []

    selected.sort(key=lambda t: t[0])
    return selected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frame_selector.py -k "build_survivor_flow or select_" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -f preprocess/frame_selector.py tests/test_frame_selector.py
git commit -m "feat(preprocess): fluxo por sobrevivente + seleção por movimento/gap/novidade"
```

---

### Task 4: Ajuste ao orçamento (`fit_budget`)

**Files:**
- Modify: `preprocess/frame_selector.py`
- Test: `tests/test_frame_selector.py`

**Interfaces:**
- Consumes: `select_by_motion` (Task 3).
- Produces:
  - `fit_budget(survivors: list[FrameStat], survivor_flow: list[float], budget: int, max_gap: int = 45, novelty_floor: float = 0.02, iters: int = 30) -> tuple[list[tuple[int, str]], float]`  # (seleção, threshold_final)

- [ ] **Step 1: Write the failing tests**

```python
def test_fit_budget_keeps_all_when_under_budget():
    survivors = [_mk(i, 10, 1.0) for i in range(20)]
    flow = [0.0] + [1.0] * 19
    sel, thr = fs.fit_budget(survivors, flow, budget=100, max_gap=1000, novelty_floor=0.0)
    assert len(sel) == 20 and thr == 0.0


def test_fit_budget_lands_within_band():
    survivors = [_mk(i, 10, 1.0) for i in range(1000)]
    flow = [0.0] + [1.0] * 999
    budget = 100
    sel, thr = fs.fit_budget(survivors, flow, budget=budget, max_gap=10_000, novelty_floor=0.0)
    assert 0.85 * budget <= len(sel) <= budget
    assert thr > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_frame_selector.py -k fit_budget -v`
Expected: FAIL (AttributeError: no `fit_budget`).

- [ ] **Step 3: Implement `fit_budget` in `preprocess/frame_selector.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frame_selector.py -k fit_budget -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add -f preprocess/frame_selector.py tests/test_frame_selector.py
git commit -m "feat(preprocess): fit_budget (busca binária no threshold de movimento)"
```

---

### Task 5: Leitura da fonte + análise (`iter_*` + `analyze_source`)

**Files:**
- Modify: `preprocess/frame_selector.py`
- Test: `tests/test_frame_selector.py`

**Interfaces:**
- Consumes: `frame_metrics` (Task 1), `FrameStat` (Task 2).
- Produces:
  - `iter_video_frames(video_path: str, sample_step: int = 1) -> Iterator[tuple[int, np.ndarray]]`  # (idx original, BGR)
  - `iter_folder_frames(image_folder: str, exts: tuple[str, ...] = (".jpg", ".jpeg", ".png"), sample_step: int = 1) -> Iterator[tuple[int, np.ndarray]]`
  - `to_gray_small(bgr: np.ndarray, analysis_width: int) -> np.ndarray`
  - `analyze_source(frames: Iterable[tuple[int, np.ndarray]], analysis_width: int = 320) -> list[FrameStat]`  # preenche flow_prev sequencialmente

- [ ] **Step 1: Write the failing tests**

```python
import cv2
import os


def _write_video(path, frames):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def _color(v):
    return np.full((120, 160, 3), v, np.uint8)


def test_iter_video_frames_indices(tmp_path):
    frames = [_color(i * 20) for i in range(6)]
    p = str(tmp_path / "v.mp4")
    _write_video(p, frames)
    got = list(fs.iter_video_frames(p, sample_step=2))
    assert [i for i, _ in got] == [0, 2, 4]


def test_analyze_source_fills_stats_and_flow():
    base = np.random.RandomState(3).randint(0, 256, (120, 160, 3), np.uint8)
    frames = [(0, base), (1, base), (2, np.roll(base, 15, axis=1))]
    stats = fs.analyze_source(iter(frames), analysis_width=160)
    assert len(stats) == 3
    assert stats[0].flow_prev == 0.0
    assert stats[1].flow_prev < stats[2].flow_prev   # frame igual -> ~0; deslocado -> maior
    assert all(hasattr(s, "sharpness") for s in stats)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_frame_selector.py -k "iter_video or analyze_source" -v`
Expected: FAIL (AttributeError: no `iter_video_frames` / `analyze_source`).

- [ ] **Step 3: Implement in `preprocess/frame_selector.py`**

```python
def iter_video_frames(video_path: str, sample_step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
    step = max(1, int(sample_step))
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def iter_folder_frames(
    image_folder: str,
    exts: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    sample_step: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    paths = sorted(
        p for p in glob.glob(os.path.join(image_folder, "*"))
        if os.path.splitext(p)[1].lower() in exts
    )
    if not paths:
        raise ValueError(f"Nenhuma imagem em: {image_folder}")
    step = max(1, int(sample_step))
    for idx, p in enumerate(paths):
        if idx % step != 0:
            continue
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"Não foi possível ler a imagem: {p}")
        yield idx, img


def to_gray_small(bgr: np.ndarray, analysis_width: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w > analysis_width:
        new_h = max(1, round(h * analysis_width / w))
        bgr = cv2.resize(bgr, (analysis_width, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def analyze_source(
    frames: Iterable[tuple[int, np.ndarray]],
    analysis_width: int = 320,
) -> list[FrameStat]:
    """Passo 1: métricas por frame + fluxo do frame analisado anterior (mantém só o gray anterior)."""
    stats: list[FrameStat] = []
    prev_gray: np.ndarray | None = None
    for idx, bgr in tqdm(frames, desc="Analisando frames", unit="frame"):
        gray = to_gray_small(bgr, analysis_width)
        brightness, clipped, entropy = fm.exposure(gray)
        flow_prev = 0.0 if prev_gray is None else fm.flow_magnitude(prev_gray, gray)
        stats.append(FrameStat(
            idx=idx,
            sharpness=fm.sharpness(gray),
            brightness=brightness,
            clipped=clipped,
            entropy=entropy,
            hist=fm.histogram(gray),
            flow_prev=flow_prev,
        ))
        prev_gray = gray
    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frame_selector.py -k "iter_video or analyze_source" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add -f preprocess/frame_selector.py tests/test_frame_selector.py
git commit -m "feat(preprocess): leitura de vídeo/pasta + análise de métricas por frame"
```

---

### Task 6: Escrita das saídas (`write_outputs`)

**Files:**
- Modify: `preprocess/frame_selector.py`
- Test: `tests/test_frame_selector.py`

**Interfaces:**
- Consumes: `FrameStat` (Task 2), `iter_video_frames`/`iter_folder_frames` (Task 5).
- Produces:
  - `write_outputs(source_frames: Iterable[tuple[int, np.ndarray]], selected: list[tuple[int, str]], stats: list[FrameStat], out_dir: str, threshold: float, total_source: int, n_survivors: int, contact_sheet: bool = False) -> None`  # cria PNGs `frame_%06d.png`, `selection_manifest.csv`, `selection_report.txt` (+ `contact_sheet.png` se pedido)

- [ ] **Step 1: Write the failing test**

```python
def test_write_outputs_creates_frames_manifest_report(tmp_path):
    frames = [(i, _color(i * 10)) for i in range(6)]
    stats = fs.analyze_source(iter(frames), analysis_width=160)
    selected = [(0, "forced_first"), (4, "motion")]
    out = str(tmp_path / "sel")
    fs.write_outputs(iter(frames), selected, stats, out,
                     threshold=1.5, total_source=6, n_survivors=6)
    pngs = sorted(f for f in os.listdir(out) if f.endswith(".png"))
    assert pngs == ["frame_000000.png", "frame_000001.png"]  # renomeados sequencialmente
    assert os.path.exists(os.path.join(out, "selection_manifest.csv"))
    assert os.path.exists(os.path.join(out, "selection_report.txt"))
    with open(os.path.join(out, "selection_manifest.csv")) as fh:
        rows = list(__import__("csv").DictReader(fh))
    assert [r["src_frame_idx"] for r in rows] == ["0", "4"]
    assert rows[0]["reason"] == "forced_first"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frame_selector.py -k write_outputs -v`
Expected: FAIL (AttributeError: no `write_outputs`).

- [ ] **Step 3: Implement `write_outputs` in `preprocess/frame_selector.py`**

```python
def write_outputs(
    source_frames: Iterable[tuple[int, np.ndarray]],
    selected: list[tuple[int, str]],
    stats: list[FrameStat],
    out_dir: str,
    threshold: float,
    total_source: int,
    n_survivors: int,
    contact_sheet: bool = False,
) -> None:
    """Passo 2: re-lê a fonte e escreve os frames escolhidos + manifesto + relatório."""
    os.makedirs(out_dir, exist_ok=True)
    reason_by_idx = dict(selected)
    stat_by_idx = {s.idx: s for s in stats}
    ordered_idx = sorted(reason_by_idx)
    out_name = {src: f"frame_{n:06d}.png" for n, src in enumerate(ordered_idx)}

    written: list[np.ndarray] = []
    for idx, bgr in source_frames:
        if idx in out_name:
            cv2.imwrite(os.path.join(out_dir, out_name[idx]), bgr)
            if contact_sheet:
                written.append(cv2.resize(bgr, (160, 120), interpolation=cv2.INTER_AREA))

    fps_hint = 30.0
    with open(os.path.join(out_dir, "selection_manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["out_name", "src_frame_idx", "timestamp_s", "sharpness",
                    "flow_prev", "brightness", "entropy", "reason"])
        for idx in ordered_idx:
            s = stat_by_idx[idx]
            w.writerow([out_name[idx], idx, round(idx / fps_hint, 3),
                        round(s.sharpness, 2), round(s.flow_prev, 3),
                        round(s.brightness, 1), round(s.entropy, 3), reason_by_idx[idx]])

    with open(os.path.join(out_dir, "selection_report.txt"), "w") as fh:
        fh.write("Seletor de frames — relatório\n")
        fh.write(f"Frames na fonte (analisados): {total_source}\n")
        fh.write(f"Sobreviventes do portão de qualidade: {n_survivors}\n")
        fh.write(f"Selecionados: {len(ordered_idx)}\n")
        fh.write(f"motion_threshold final: {threshold:.4f}\n")
        if ordered_idx:
            fh.write(f"Índice de origem: {ordered_idx[0]}..{ordered_idx[-1]}\n")

    if contact_sheet and written:
        cols = 8
        rows = (len(written) + cols - 1) // cols
        grid = np.zeros((rows * 120, cols * 160, 3), np.uint8)
        for i, thumb in enumerate(written):
            r, c = divmod(i, cols)
            grid[r * 120:(r + 1) * 120, c * 160:(c + 1) * 160] = thumb
        cv2.imwrite(os.path.join(out_dir, "contact_sheet.png"), grid)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_frame_selector.py -k write_outputs -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -f preprocess/frame_selector.py tests/test_frame_selector.py
git commit -m "feat(preprocess): escrita de frames selecionados + manifesto CSV + relatório"
```

---

### Task 7: CLI `main()` + integração ponta-a-ponta + casos de borda

**Files:**
- Modify: `preprocess/frame_selector.py`
- Test: `tests/test_frame_selector.py`

**Interfaces:**
- Consumes: todas as funções anteriores.
- Produces:
  - `run(source_kind: str, source_path: str, out_dir: str, budget: int, sample_step: int, analysis_width: int, max_gap: int, novelty_floor: float, min_entropy: float, contact_sheet: bool) -> int`  # retorna nº de frames selecionados; relaxa portão se zerar
  - `main(argv: list[str] | None = None) -> None`  # argparse → run

- [ ] **Step 1: Write the failing tests**

```python
def test_run_end_to_end_respects_budget_and_drops_blur(tmp_path):
    rs = np.random.RandomState(7)
    frames = []
    for i in range(40):
        f = rs.randint(0, 256, (120, 160, 3), np.uint8)   # conteúdo variando (movimento)
        if i in (10, 11, 12):                              # trecho borrado
            f = cv2.GaussianBlur(f, (0, 0), sigmaX=6)
        frames.append(f)
    vpath = str(tmp_path / "v.mp4")
    _write_video(vpath, frames)
    out = str(tmp_path / "sel")
    n = fs.run(source_kind="video", source_path=vpath, out_dir=out, budget=12,
               sample_step=1, analysis_width=160, max_gap=1000,
               novelty_floor=0.0, min_entropy=0.0, contact_sheet=False)
    assert 0.85 * 12 <= n <= 12
    pngs = sorted(f for f in os.listdir(out) if f.endswith(".png"))
    assert len(pngs) == n
    # nomes sequenciais e ordenados
    assert pngs == [f"frame_{i:06d}.png" for i in range(n)]


def test_run_never_empty_when_gate_rejects_all(tmp_path):
    frames = [_color(2) for _ in range(8)]         # todos escuros -> portão reprovaria
    vpath = str(tmp_path / "v.mp4")
    _write_video(vpath, frames)
    out = str(tmp_path / "sel")
    n = fs.run(source_kind="video", source_path=vpath, out_dir=out, budget=100,
               sample_step=1, analysis_width=160, max_gap=1000,
               novelty_floor=0.0, min_entropy=2.5, contact_sheet=False)
    assert n >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_frame_selector.py -k "end_to_end or never_empty" -v`
Expected: FAIL (AttributeError: no `run`).

- [ ] **Step 3: Implement `run` + `main` in `preprocess/frame_selector.py`**

```python
def _source_iter(source_kind: str, source_path: str, sample_step: int):
    if source_kind == "video":
        return iter_video_frames(source_path, sample_step)
    return iter_folder_frames(source_path, sample_step=sample_step)


def run(
    source_kind: str,
    source_path: str,
    out_dir: str,
    budget: int = 900,
    sample_step: int = 1,
    analysis_width: int = 320,
    max_gap: int = 45,
    novelty_floor: float = 0.02,
    min_entropy: float = 2.5,
    contact_sheet: bool = False,
) -> int:
    stats = analyze_source(_source_iter(source_kind, source_path, sample_step), analysis_width)
    if not stats:
        raise ValueError("Fonte sem frames legíveis.")

    survivor_pos = quality_gate(stats, min_entropy=min_entropy)
    if not survivor_pos:                     # relaxa: nunca retorna vazio
        print("Aviso: portão de qualidade reprovou tudo — relaxando limiares.")
        survivor_pos = list(range(len(stats)))

    survivors = [stats[p] for p in survivor_pos]
    survivor_flow = build_survivor_flow(stats, survivor_pos)
    selected, threshold = fit_budget(survivors, survivor_flow, budget, max_gap, novelty_floor)

    write_outputs(
        _source_iter(source_kind, source_path, sample_step),
        selected, stats, out_dir, threshold,
        total_source=len(stats), n_survivors=len(survivors),
        contact_sheet=contact_sheet,
    )
    print(f"Selecionados {len(selected)} de {len(stats)} frames -> {out_dir}")
    return len(selected)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Seletor de frames (qualidade + conteúdo + orçamento)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_path", type=str)
    src.add_argument("--image_folder", type=str)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--budget", type=int, default=900)
    ap.add_argument("--sample_step", type=int, default=1)
    ap.add_argument("--analysis_width", type=int, default=320)
    ap.add_argument("--max_gap", type=int, default=45)
    ap.add_argument("--novelty_floor", type=float, default=0.02)
    ap.add_argument("--min_entropy", type=float, default=2.5)
    ap.add_argument("--contact_sheet", action="store_true")
    args = ap.parse_args(argv)

    kind = "video" if args.video_path else "folder"
    path = args.video_path or args.image_folder
    run(kind, path, args.output_dir, budget=args.budget, sample_step=args.sample_step,
        analysis_width=args.analysis_width, max_gap=args.max_gap,
        novelty_floor=args.novelty_floor, min_entropy=args.min_entropy,
        contact_sheet=args.contact_sheet)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full test suite**

Run: `python -m pytest tests/test_frame_metrics.py tests/test_frame_selector.py -v`
Expected: PASS (todos).

- [ ] **Step 5: Smoke test manual no vídeo real (opcional, se disponível localmente)**

Run:
```bash
python preprocess/frame_selector.py \
  --video_path VID_20260626_171646871.mp4 \
  --output_dir frames_sel/ --budget 900 --contact_sheet
cat frames_sel/selection_report.txt
```
Expected: relatório com ~≤900 selecionados; `frames_sel/` com PNGs sequenciais.

- [ ] **Step 6: Commit**

```bash
git add -f preprocess/frame_selector.py tests/test_frame_selector.py
git commit -m "feat(preprocess): CLI run/main + integração ponta-a-ponta e casos de borda"
```

---

## Notas de integração (pós-implementação)

Depois do seletor, a reconstrução na ctrob passa a consumir a pasta:

```bash
# gerar seleção (local, CPU):
python preprocess/frame_selector.py --video_path VID.mp4 --output_dir frames_sel/ --budget 900
# enviar a pasta e rodar streaming na ctrob:
rsync -a frames_sel/ ctrob@10.1.83.25:~/lingbot-inputs/frames_sel/
#   (na ctrob) batch_demo.py --image_folder ~/lingbot-inputs/frames_sel/ --mode streaming \
#              --no_render --save_predictions --keyframe_interval 1
```

Isso não altera o modelo; só melhora a entrada. A validação final é visual (viser),
comparando com as versões `streaming`/`windowed` já geradas.
