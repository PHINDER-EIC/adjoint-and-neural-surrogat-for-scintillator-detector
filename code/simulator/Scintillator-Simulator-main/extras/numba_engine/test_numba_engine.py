"""Tests for numba_engine.py. Run from the repo root:

    python -m pytest extras/numba_engine/test_numba_engine.py -q
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numba  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonBatch  # noqa: E402
from Photon_Simulator.core.photonsEngines.batchTransport import BatchTransport  # noqa: E402

import scenes as S  # noqa: E402
from numba_engine import NumbaEngine, NumbaTrackingConfig  # noqa: E402

RUNNING = (-1, -2)


def test_matches_engine_a_photon_by_photon_without_randomness():
    scene = S.deterministic_scene()
    batch = S.uniform_batch(500, 1)
    a = BatchTransport(scene, engine="approximate", seed=1, max_steps=200000).run(batch)
    d = NumbaEngine(scene).track_photon_batch(batch, seed=123)
    np.testing.assert_array_equal(a.batch.statuses, d.batch.statuses)
    np.testing.assert_array_equal(a.steps, d.steps)
    np.testing.assert_allclose(d.batch.interaction_points, a.batch.interaction_points, atol=1e-9)
    np.testing.assert_allclose(d.batch.times, a.batch.times, atol=1e-9)


def test_same_statistics_as_engine_a_reference_scene():
    scene = S.build_reference_scene()
    a = BatchTransport(scene, engine="approximate", seed=2).run(S.make_source_batch(2000, 2))
    d = NumbaEngine(scene).track_photon_batch(S.make_source_batch(200000, 3), seed=3)
    for codes in ([2], [4, 5, 6], [0, 7, 8]):
        pa = np.isin(a.batch.statuses, codes).mean()
        pd = np.isin(d.batch.statuses, codes).mean()
        se = np.sqrt(pa * (1 - pa) / 2000 + pd * (1 - pd) / 200000)
        assert abs(pd - pa) / se < 4, (codes, pa, pd)


def test_independent_of_thread_count():
    eng = NumbaEngine(S.v2_scene())
    batch = S.v2_photons(5000, 1)
    full = eng.track_photon_batch(batch, seed=5)
    numba.set_num_threads(1)
    try:
        one = eng.track_photon_batch(batch, seed=5)
    finally:
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    np.testing.assert_array_equal(full.batch.statuses, one.batch.statuses)
    np.testing.assert_array_equal(full.batch.times, one.batch.times)


def test_chunks_with_index_offset_reproduce_one_batch():
    eng = NumbaEngine(S.build_reference_scene())
    batch = S.make_source_batch(3000, 4)
    whole = eng.track_photon_batch(batch, seed=9)
    parts = []
    for start in range(0, 3000, 700):
        sl = slice(start, start + 700)
        part = PhotonBatch(batch.interaction_points[sl], batch.directions[sl], batch.times[sl], batch.statuses[sl])
        parts.append(eng.track_photon_batch(part, seed=9, index_offset=start).batch.times)
    np.testing.assert_array_equal(np.concatenate(parts), whole.batch.times)


def test_direction_length_does_not_matter():
    eng = NumbaEngine(S.build_reference_scene())
    b = S.make_source_batch(2000, 6)
    scaled = PhotonBatch(b.interaction_points, 3.7 * b.directions, b.times, b.statuses)
    r1 = eng.track_photon_batch(b, seed=1)
    r2 = eng.track_photon_batch(scaled, seed=1)
    np.testing.assert_array_equal(r1.batch.statuses, r2.batch.statuses)
    np.testing.assert_allclose(r1.batch.times, r2.batch.times, rtol=1e-12, atol=1e-12)


def test_inactive_photons_untouched():
    b = S.make_source_batch(1000, 7)
    st = b.statuses.copy(); st[::2] = 2
    b = PhotonBatch(b.interaction_points, b.directions, b.times, st)
    r = NumbaEngine(S.build_reference_scene()).track_photon_batch(b, seed=1)
    np.testing.assert_array_equal(r.batch.interaction_points[::2], b.interaction_points[::2])
    np.testing.assert_array_equal(r.steps[::2], 0)
    assert not np.isin(r.batch.statuses, RUNNING).any()


def test_truncation_flag():
    eng = NumbaEngine(S.build_reference_scene(), NumbaTrackingConfig(max_steps=5))
    r = eng.track_photon_batch(S.make_source_batch(2000, 8), seed=1)
    t = r.truncated
    assert t.any()
    assert np.isin(r.batch.statuses[t], RUNNING).all()
    assert (r.steps[t] == 5).all()
    assert not np.isin(r.batch.statuses[~t], RUNNING).any()


def test_air_gap_needs_air_volume():
    b = S.uniform_batch(20000, 9)
    without = NumbaEngine(S.air_gap_scene(False)).track_photon_batch(b, seed=1).batch.statuses
    with_air = NumbaEngine(S.air_gap_scene(True)).track_photon_batch(b, seed=1).batch.statuses
    assert (without == 2).sum() == 0
    assert 0.05 < (with_air == 2).mean() < 0.11


def test_lambertian_runs_and_differs_from_specular():
    b = S.make_source_batch(50000, 10)
    spec = NumbaEngine(S.lambertian_scene(1.0)).track_photon_batch(b, seed=1)
    diff = NumbaEngine(S.lambertian_scene(0.0)).track_photon_batch(b, seed=1)
    assert not np.isin(diff.batch.statuses, RUNNING).any()
    assert abs((spec.batch.statuses == 2).mean() - (diff.batch.statuses == 2).mean()) > 0.01


def test_photon_outside_everything_is_lost():
    b = PhotonBatch(np.array([[5.0, 50.0, 5.0]]), np.array([[1.0, 0, 0]]), np.zeros(1), np.array([-1], np.int32))
    r = NumbaEngine(S.build_reference_scene()).track_photon_batch(b)
    assert r.batch.statuses[0] == 1


def test_rejects_non_convex_volume():
    class Fake:
        name = "scintillator"
    scene = S.build_reference_scene()
    scene.volumes = list(scene.volumes) + [Fake()]
    with pytest.raises(TypeError):
        NumbaEngine(scene)
