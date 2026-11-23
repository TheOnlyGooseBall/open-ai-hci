import math
import random

import pytest
from conftest import H, W, make_ctx, make_hand

from open_ai_hci import EMAFilter, KalmanFilter1D, OneEuroFilter, SmoothingConfig, SmoothingProcessor
from open_ai_hci.processors.smoothing import IdentityFilter, make_filter_2d, map_to_screen


def test_ema_matches_the_prototype_formula():
    f = EMAFilter(0.20)
    assert f(100.0) == 100.0  # first sample passes through
    prev = 100.0
    for target in (200.0, 150.0, 400.0):
        expected = prev + (target - prev) * 0.20
        assert f(target) == pytest.approx(expected)
        prev = expected


def test_ema_validates_and_resets():
    with pytest.raises(ValueError):
        EMAFilter(0.0)
    with pytest.raises(ValueError):
        EMAFilter(1.5)
    f = EMAFilter(0.5)
    f(10)
    f.reset()
    assert f(99) == 99


def test_one_euro_passes_constants_and_reduces_jitter():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    assert all(f(5.0, t * 0.033) == pytest.approx(5.0) for t in range(30))

    rng = random.Random(7)
    noisy = [100 + rng.gauss(0, 3) for _ in range(300)]
    f.reset()
    smooth = [f(v, i * 0.033) for i, v in enumerate(noisy)]

    def spread(xs):
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    assert spread(smooth[50:]) < spread(noisy[50:]) * 0.5


def test_one_euro_beta_reduces_lag_on_fast_motion():
    def final(beta):
        f = OneEuroFilter(min_cutoff=1.0, beta=beta)
        out = 0.0
        for i in range(30):
            out = f(i * 40.0, i * 0.033)  # fast steady ramp
        return out

    target = 29 * 40.0
    assert target - final(beta=0.5) < target - final(beta=0.0)


def test_one_euro_survives_missing_or_repeated_timestamps():
    f = OneEuroFilter()
    f(1.0)  # no timestamp
    f(2.0, 1.0)
    assert math.isfinite(f(3.0, 1.0))  # same timestamp -> default dt
    assert math.isfinite(f(4.0, 0.5))  # going backwards -> default dt


def test_kalman_converges_and_tracks_constant_velocity():
    k = KalmanFilter1D(process_noise=2e5, measurement_noise=50.0)
    out = [k(100.0, i * 0.033) for i in range(60)]
    assert out[-1] == pytest.approx(100.0, abs=0.5)
    k.reset()
    for i in range(90):
        est = k(i * 10.0, i * 0.033)
    assert est == pytest.approx(89 * 10.0, abs=25)  # follows a ramp with small lag


def test_identity_and_factory():
    assert IdentityFilter()(3.3) == 3.3
    for kind in ("ema", "one_euro", "kalman", "none"):
        f = make_filter_2d(SmoothingConfig(kind=kind))
        x, y = f(10.0, 20.0, 0.0)
        assert (x, y) == (10.0, 20.0)  # all filters pass the first sample through
    with pytest.raises(ValueError):
        make_filter_2d(SmoothingConfig(kind="wat"))


def test_map_to_screen_without_margin_is_the_prototype_formula():
    assert map_to_screen(640, 360, 1280, 720, 1920, 1080) == (1920 / 1280 * 640, 1080 / 720 * 360)


def test_map_to_screen_with_margin_reaches_corners_and_clamps():
    x0, y0 = map_to_screen(128, 72, 1280, 720, 1920, 1080, margin=0.1)
    assert (x0, y0) == (0, 0)  # 10% inside the frame already maps to the screen corner
    x1, y1 = map_to_screen(1280, 720, 1280, 720, 1920, 1080, margin=0.1)
    assert (x1, y1) == (1919, 1079)  # clamped
    xm, ym = map_to_screen(640, 360, 1280, 720, 1920, 1080, margin=0.1)
    assert (xm, ym) == pytest.approx((959.5, 539.5))


def test_processor_smooths_and_holds_last_position_when_hand_lost():
    proc = SmoothingProcessor(SmoothingConfig(kind="ema", alpha=0.20), screen_size=(1920, 1080))
    ctx = make_ctx(hands=[make_hand((640, 360), (640, 460))])
    proc.process(ctx)
    c1 = ctx.cursor
    assert (c1.x, c1.y) == (pytest.approx(960.0), pytest.approx(540.0)) and c1.active
    ctx = make_ctx(hands=[make_hand((1280 - 1, 360), (1000, 460))])
    proc.process(ctx)
    c2 = ctx.cursor
    assert c2.x == pytest.approx(960.0 + (1919 / 1280 * 1279 - 960.0) * 0.2, rel=1e-3) or c2.x > 960
    assert c2.raw_x > c2.x  # smoothed value lags the raw target
    lost = make_ctx(hands=[])
    proc.process(lost)
    assert lost.cursor.active is False and (lost.cursor.x, lost.cursor.y) == (c2.x, c2.y)


def test_processor_without_any_history_has_no_cursor():
    proc = SmoothingProcessor(SmoothingConfig(), screen_size=(1920, 1080))
    ctx = make_ctx(hands=[])
    proc.process(ctx)
    assert ctx.cursor is None
