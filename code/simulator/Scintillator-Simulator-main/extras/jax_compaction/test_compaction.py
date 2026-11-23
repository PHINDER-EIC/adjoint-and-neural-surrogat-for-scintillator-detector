"""Tests for compaction.py. Run from the repo root:

    python -m pytest extras/jax_compaction/test_compaction.py -q
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from Photon_Simulator.core.detectorSetup.referenceScene import build_reference_scene, make_source_batch  # noqa: E402
from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonBatch  # noqa: E402
from Photon_Simulator.core.photonsEngines.batchTransport import BatchTransport  # noqa: E402
from Photon_Simulator.core.photonsEngines.JaxEngine import JaxApproximateEngine, JaxTrackingConfig  # noqa: E402

from compaction import CompactedJaxEngine, enable_compaction, track_photon_batch_compacted  # noqa: E402

RUNNING = (-1, -2)


@pytest.fixture(scope="module")
def engine():
    return JaxApproximateEngine(build_reference_scene(),
                                JaxTrackingConfig(max_steps=20000, mixed_precision=True),
                                dtype=jnp.float64)


def _groups(statuses):
    s = np.asarray(statuses)
    return {"caught": np.mean(s == 2), "refl": np.mean(np.isin(s, [4, 5, 6])),
            "bulk": np.mean(np.isin(s, [0, 7, 8])), "lost": np.mean(np.isin(s, [1, 9, 10]))}


def test_same_statistics_as_lock_step(engine):
    n = 6000
    batch = make_source_batch(n, 11)
    base = _groups(engine.track_photon_batch(batch, key=3).statuses)
    comp = _groups(track_photon_batch_compacted(engine, batch, key=4).statuses)
    for k in base:
        sigma = np.sqrt((base[k] * (1 - base[k]) + comp[k] * (1 - comp[k])) / n) + 1e-12
        assert abs(comp[k] - base[k]) / sigma < 4, (k, base[k], comp[k])


def test_every_photon_finishes_and_has_steps(engine):
    out = track_photon_batch_compacted(engine, make_source_batch(1500, 2), key=0, segment_steps=4)
    s = np.asarray(out.statuses)
    assert not np.asarray(out.truncated).any()
    assert not np.isin(s, RUNNING).any()
    assert (np.asarray(out.steps) >= 1).all()


def test_reproducible_with_same_key(engine):
    batch = make_source_batch(800, 5)
    a = track_photon_batch_compacted(engine, batch, key=9)
    b = track_photon_batch_compacted(engine, batch, key=9)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_inactive_inputs_untouched_and_order_kept(engine):
    batch = make_source_batch(1000, 6)
    statuses = batch.statuses.copy()
    statuses[::2] = 2                                   # half the photons arrive already CAUGHT
    batch = PhotonBatch(batch.interaction_points, batch.directions, batch.times, statuses)
    out = track_photon_batch_compacted(engine, batch, key=1)
    np.testing.assert_array_equal(np.asarray(out.interaction_points)[::2], batch.interaction_points[::2])
    np.testing.assert_array_equal(np.asarray(out.times)[::2], batch.times[::2])
    np.testing.assert_array_equal(np.asarray(out.steps)[::2], 0)
    assert (np.asarray(out.steps)[1::2] >= 1).all()


def test_truncation_matches_step_budget(engine):
    out = track_photon_batch_compacted(engine, make_source_batch(1000, 7), key=2,
                                       segment_steps=4, max_steps=10)
    trunc = np.asarray(out.truncated)
    s = np.asarray(out.statuses)
    steps = np.asarray(out.steps)
    assert trunc.any()
    assert np.isin(s[trunc], RUNNING).all()
    assert (steps[trunc] == 10).all()
    assert (steps <= 10).all()
    assert not np.isin(s[~trunc], RUNNING).any()


def test_batchtransport_drop_in():
    scene = build_reference_scene()
    batch = make_source_batch(3000, 8)
    runner = enable_compaction(BatchTransport(scene, engine="jax", seed=42, chunk_size=1000),
                               segment_steps=8)
    assert isinstance(runner.engine, CompactedJaxEngine)
    assert runner.device_info["compaction"]["segment_steps"] == 8
    res = runner.run(batch)
    assert res.batch.size == 3000
    assert not res.truncated.any()
    assert not np.isin(res.batch.statuses, RUNNING).any()
    assert 0.5 < np.mean(res.batch.statuses == 2) < 0.7      # reference scene catches ~60%


def test_enable_compaction_rejects_cpu_runner():
    with pytest.raises(ValueError):
        enable_compaction(BatchTransport(build_reference_scene(), engine="approximate"))
