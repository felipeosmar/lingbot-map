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
