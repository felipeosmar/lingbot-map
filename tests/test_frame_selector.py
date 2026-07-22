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
