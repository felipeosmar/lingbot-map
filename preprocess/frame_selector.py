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

try:
    from preprocess import frame_metrics as fm
except ModuleNotFoundError:  # executado como script (python preprocess/frame_selector.py)
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def iter_video_frames(video_path: str, sample_step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Iterador de frames do vídeo com amostragem. Retorna (idx_original, BGR)."""
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
    """Iterador de frames da pasta de imagens com amostragem. Retorna (idx_original, BGR)."""
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
    """Converte BGR para grayscale e redimensiona se necessário."""
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


def write_outputs(
    source_frames: Iterable[tuple[int, np.ndarray]],
    selected: list[tuple[int, str]],
    stats: list[FrameStat],
    out_dir: str,
    threshold: float,
    total_source: int,
    n_survivors: int,
    contact_sheet: bool = False,
    output_ext: str = "png",
    jpeg_quality: int = 95,
    max_width: int = 0,
) -> None:
    """Passo 2: re-lê a fonte e escreve os frames escolhidos + manifesto + relatório.

    ``output_ext`` escolhe o formato ('png' ou 'jpg'); ``jpeg_quality`` (1-100) só
    vale para jpg; ``max_width`` > 0 redimensiona os frames de saída para essa largura
    (mantendo proporção) — juntos reduzem muito o tamanho para transferência.
    """
    os.makedirs(out_dir, exist_ok=True)
    ext = output_ext.lower().lstrip(".")
    imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)] if ext in ("jpg", "jpeg") else []
    reason_by_idx = dict(selected)
    stat_by_idx = {s.idx: s for s in stats}
    ordered_idx = sorted(reason_by_idx)
    out_name = {src: f"frame_{n:06d}.{ext}" for n, src in enumerate(ordered_idx)}

    def _maybe_resize(img: np.ndarray) -> np.ndarray:
        if max_width > 0 and img.shape[1] > max_width:
            new_h = max(1, round(img.shape[0] * max_width / img.shape[1]))
            return cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)
        return img

    written: list[np.ndarray] = []
    written_idx: set[int] = set()
    for idx, bgr in source_frames:
        if idx in out_name:
            cv2.imwrite(os.path.join(out_dir, out_name[idx]), _maybe_resize(bgr), imwrite_params)
            written_idx.add(idx)
            if contact_sheet:
                written.append(cv2.resize(bgr, (160, 120), interpolation=cv2.INTER_AREA))

    # Verificar que todos os frames selecionados foram efetivamente encontrados
    missing = [i for i in ordered_idx if i not in written_idx]
    if missing:
        raise ValueError(f"Frames selecionados ausentes na fonte ao reescrever: {missing[:10]}"
                         f"{'...' if len(missing) > 10 else ''} (total {len(missing)})")

    fps_hint = 30.0
    with open(os.path.join(out_dir, "selection_manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["out_name", "src_frame_idx", "timestamp_s_assume30fps", "sharpness",
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
        fh.write("Nota: timestamp_s_assume30fps assume 30 fps (o seletor não conhece o fps real da fonte).\n")
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
    output_ext: str = "png",
    jpeg_quality: int = 95,
    max_width: int = 0,
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
        output_ext=output_ext, jpeg_quality=jpeg_quality, max_width=max_width,
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
    ap.add_argument("--output_ext", type=str, default="png", choices=["png", "jpg", "jpeg"],
                    help="Formato dos frames de saída (jpg reduz muito o tamanho)")
    ap.add_argument("--jpeg_quality", type=int, default=95,
                    help="Qualidade JPEG 1-100 (só quando --output_ext jpg)")
    ap.add_argument("--max_width", type=int, default=0,
                    help="Redimensiona os frames de saída para esta largura (0 = resolução original)")
    args = ap.parse_args(argv)

    kind = "video" if args.video_path else "folder"
    path = args.video_path or args.image_folder
    run(kind, path, args.output_dir, budget=args.budget, sample_step=args.sample_step,
        analysis_width=args.analysis_width, max_gap=args.max_gap,
        novelty_floor=args.novelty_floor, min_entropy=args.min_entropy,
        contact_sheet=args.contact_sheet, output_ext=args.output_ext,
        jpeg_quality=args.jpeg_quality, max_width=args.max_width)


if __name__ == "__main__":
    main()
