import os

import cv2
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


def test_write_outputs_creates_frames_manifest_report(tmp_path):
    import os
    import csv
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
        rows = list(csv.DictReader(fh))
    assert [r["src_frame_idx"] for r in rows] == ["0", "4"]
    assert rows[0]["reason"] == "forced_first"
    assert "timestamp_s_assume30fps" in rows[0]  # verificar coluna renomeada
    with open(os.path.join(out, "selection_report.txt")) as fh:
        report = fh.read()
    assert "timestamp_s_assume30fps assume 30 fps" in report  # verificar nota na report


def test_write_outputs_raises_on_missing_selected_frames(tmp_path):
    """Verifica que write_outputs falha se frame selecionado não está presente na fonte."""
    import os
    frames = [(i, _color(i * 10)) for i in range(4)]  # frames 0, 1, 2, 3
    stats = fs.analyze_source(iter(frames), analysis_width=160)
    # Seleciona frames 0, 2, 99 (99 não existe)
    selected = [(0, "forced_first"), (2, "motion"), (99, "motion")]
    out = str(tmp_path / "sel")
    with pytest.raises(ValueError, match="Frames selecionados ausentes"):
        fs.write_outputs(iter(frames), selected, stats, out,
                         threshold=1.5, total_source=4, n_survivors=4)
    # Verificar que a exceção menciona o índice 99
    try:
        fs.write_outputs(iter(frames), selected, stats, out,
                         threshold=1.5, total_source=4, n_survivors=4)
    except ValueError as e:
        assert "99" in str(e)


def test_run_end_to_end_respects_budget_and_drops_blur(tmp_path):
    import os
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


def test_write_outputs_jpg_and_max_width(tmp_path):
    # frames 320x240; saída em jpg redimensionada para largura 80
    frames = [(i, np.full((240, 320, 3), i * 10, np.uint8)) for i in range(4)]
    stats = fs.analyze_source(iter(frames), analysis_width=160)
    selected = [(0, "forced_first"), (2, "motion")]
    out = str(tmp_path / "sel")
    fs.write_outputs(iter(frames), selected, stats, out,
                     threshold=1.0, total_source=4, n_survivors=4,
                     output_ext="jpg", jpeg_quality=90, max_width=80)
    files = sorted(os.listdir(out))
    jpgs = [f for f in files if f.endswith(".jpg")]
    assert jpgs == ["frame_000000.jpg", "frame_000001.jpg"]  # ext jpg + sequencial
    assert not any(f.endswith(".png") for f in files)         # nenhum png
    img = cv2.imread(os.path.join(out, "frame_000000.jpg"))
    assert img is not None and img.shape[1] == 80              # redimensionado p/ 80px de largura
    # manifesto referencia os nomes .jpg
    with open(os.path.join(out, "selection_manifest.csv")) as fh:
        rows = list(__import__("csv").DictReader(fh))
    assert rows[0]["out_name"] == "frame_000000.jpg"
