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
    # textura suave (rastreável pelo fluxo) num canvas maior; desloca por recorte (sem wraparound)
    rs = np.random.RandomState(1)
    canvas = cv2.GaussianBlur(rs.randint(0, 256, (160, 300), np.uint8), (0, 0), sigmaX=2)
    crop = lambda dx: canvas[:, dx:dx + 220]
    base, shift1, shift5 = crop(20), crop(22), crop(30)   # +0, +2 px, +10 px
    m0 = fm.flow_magnitude(base, base)
    m1 = fm.flow_magnitude(base, shift1)
    m5 = fm.flow_magnitude(base, shift5)
    assert m0 < 0.5
    assert m5 > m1 > m0
