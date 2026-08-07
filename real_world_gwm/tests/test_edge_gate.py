"""Edge-alignment gate scoring (renderer/edge_gate.py)."""

import numpy as np

from real_world_gwm.renderer import edge_gate


def _scene(offset=0):
    """(rendered, observed): a white 'robot' rectangle; the observed frame
    contains the same rectangle (shifted by `offset`) on a textured floor."""
    rng = np.random.default_rng(0)
    observed = (rng.random((90, 120, 3)) * 40 + 60).astype(np.uint8)
    observed[30 + offset:60 + offset, 40 + offset:80 + offset] = 230
    rendered = np.zeros((90, 120, 3), dtype=np.uint8)
    rendered[30:60, 40:80] = 200
    return rendered, observed


def test_aligned_beats_shifted():
    rendered, observed = _scene(offset=0)
    aligned = edge_gate.silhouette_edge_score(rendered, observed)
    _, shifted_obs = _scene(offset=12)
    shifted = edge_gate.silhouette_edge_score(rendered, shifted_obs)
    assert aligned > 0.3
    assert shifted < aligned / 2


def test_empty_render_scores_zero():
    rendered = np.zeros((90, 120, 3), dtype=np.uint8)
    _, observed = _scene()
    assert edge_gate.silhouette_edge_score(rendered, observed) == 0.0


def test_perturbed_cam2world_keeps_position():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    p = edge_gate.perturbed_cam2world(c2w)
    assert np.allclose(p[:3, 3], [1.0, 2.0, 3.0])
    assert not np.allclose(p[:3, :3], np.eye(3))
    # pure rotation
    assert np.allclose(p[:3, :3] @ p[:3, :3].T, np.eye(3), atol=1e-12)


def test_probe_indices_spread_and_dedupe():
    assert edge_gate.probe_indices(range(100, 200)) == [119, 149, 179]
    assert edge_gate.probe_indices([5]) == [5]
    assert edge_gate.probe_indices([]) == []


def test_gate_scores_and_passes():
    rendered, observed = _scene(offset=0)
    misrendered = np.roll(rendered, 15, axis=1)

    def render_fn(cam2world):
        # identity camera -> aligned render; perturbed -> shifted render
        aligned = np.allclose(cam2world, np.eye(4))
        return (rendered if aligned else misrendered)[None]

    scores = edge_gate.gate_scores(render_fn, observed[None], np.eye(4))
    assert scores["score"] > scores["score_perturbed"]
    assert scores["margin"] == scores["score"] - scores["score_perturbed"]
    assert edge_gate.passes(scores)
    assert not edge_gate.passes(scores, min_score=scores["score"] + 0.1)
    assert not edge_gate.passes(scores, min_margin=scores["margin"] + 0.1)
