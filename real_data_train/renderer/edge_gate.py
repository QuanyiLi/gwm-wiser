"""Edge-alignment gate: is a joined DROID calibration actually right?

The KarlP join (scripts/prepare_droid_calibration.py) can produce wrong
candidates — stale calibrations, moved tripods, imperfect ChArUco fits
(droid-dataset#39). The gate renders the robot at a few probe frames with the
candidate calibration and checks that the rendered silhouette's edges land on
image edges *of matching orientation* in the observed frame, minus the hit
rate a randomly placed silhouette would get (the per-frame edge-density
baseline). This lift is contrast-invariant — verified on the local batch,
where raw edge-hit fractions scored aligned dark scenes below misaligned
bright ones. A second check compares the same probes under a deliberately
perturbed camera (PERTURB_DEG about the camera y axis): a correct calibration
beats its perturbation by a clear margin. Decision D-28: these two floors are
the admission criterion for every DROID stream — streams failing them never
enter the rendered tree.
"""

import numpy as np

PERTURB_DEG = 8.0        # ~18 px shift at DROID's fx~133 @320x180
DEFAULT_MIN_SCORE = 0.10   # mean oriented-edge lift floor (local calibration)
DEFAULT_MIN_MARGIN = 0.05  # true lift - perturbed lift floor
PROBE_FRACTIONS = (0.2, 0.5, 0.8)
EDGE_TOL_PX = 2.0          # contour-to-edge match distance
STRONG_EDGE_PCT = 88       # observed gradient-magnitude percentile


def silhouette_edge_score(rendered_rgb: np.ndarray,
                          observed_rgb: np.ndarray) -> float:
    """Above-chance oriented-edge alignment lift, roughly in [-0.2, 0.8].

    A rendered-silhouette contour pixel counts as a hit when a strong
    observed edge lies within EDGE_TOL_PX, weighted by |cos| of the angle
    between the contour normal and the observed gradient there. Chance level
    (edge density x the 2/pi random-orientation expectation) is subtracted,
    so 0 means "no better than dropping the silhouette anywhere" regardless
    of scene contrast or clutter. Returns 0.0 when the render is empty.

    rendered_rgb / observed_rgb: uint8 (H, W, 3); render is on black.
    """
    import cv2

    mask = (rendered_rgb.max(axis=2) > 8).astype(np.uint8)
    if mask.sum() < 50:
        return 0.0
    kernel = np.ones((3, 3), np.uint8)
    contour = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel) > 0
    if contour.sum() == 0:
        return 0.0

    gray = cv2.GaussianBlur(
        cv2.cvtColor(observed_rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0
    ).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    strong = mag >= np.percentile(mag, STRONG_EDGE_PCT)
    # Distance to the nearest strong edge, and that edge's orientation
    # propagated to every pixel via the distance-transform labels.
    dist, labels = cv2.distanceTransformWithLabels(
        (~strong).astype(np.uint8), cv2.DIST_L2, 3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    edge_idx = np.flatnonzero(strong.ravel())
    if edge_idx.size == 0:
        return 0.0
    lab_to_angle = np.zeros(labels.max() + 1, dtype=np.float32)
    lab_to_angle[labels.ravel()[edge_idx]] = np.arctan2(
        gy.ravel()[edge_idx], gx.ravel()[edge_idx])
    near = dist <= EDGE_TOL_PX
    obs_angle = lab_to_angle[labels]

    # Contour-normal direction from the smoothed mask gradient.
    mask_f = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    nx = cv2.Sobel(mask_f, cv2.CV_32F, 1, 0, ksize=3)
    ny = cv2.Sobel(mask_f, cv2.CV_32F, 0, 1, ksize=3)
    ren_angle = np.arctan2(ny, nx)

    agree = np.abs(np.cos(obs_angle - ren_angle))
    oriented_hit = float((near * agree)[contour].mean())
    chance = float(near.mean()) * (2.0 / np.pi)
    return oriented_hit - chance


def perturbed_cam2world(cam2world_cv: np.ndarray,
                        angle_deg: float = PERTURB_DEG) -> np.ndarray:
    """Rotate the camera about its own y (CV: down) axis — a pure image-space
    shift that keeps the robot mostly in frame but breaks alignment."""
    a = np.deg2rad(angle_deg)
    rot_y = np.array([
        [np.cos(a), 0.0, np.sin(a), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(a), 0.0, np.cos(a), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return np.asarray(cam2world_cv) @ rot_y


def probe_indices(candidates, n_probes: int = len(PROBE_FRACTIONS)) -> list:
    """Deterministic probe frames spread across the usable frame indices."""
    candidates = np.asarray(list(candidates))
    if candidates.size == 0:
        return []
    picks = [candidates[int(f * (candidates.size - 1))]
             for f in PROBE_FRACTIONS[:n_probes]]
    return sorted(set(int(p) for p in picks))


def gate_scores(render_fn, observed_frames: np.ndarray,
                cam2world_cv: np.ndarray) -> dict:
    """Score a candidate calibration on pre-selected probe frames.

    render_fn(cam2world_cv) -> uint8 (P, H, W, 3) robot-only renders of the
    probe frames under the given camera; called twice (true + perturbed).
    """
    true_frames = render_fn(cam2world_cv)
    pert_frames = render_fn(perturbed_cam2world(cam2world_cv))
    score = float(np.mean([
        silhouette_edge_score(r, o)
        for r, o in zip(true_frames, observed_frames)
    ]))
    perturbed = float(np.mean([
        silhouette_edge_score(r, o)
        for r, o in zip(pert_frames, observed_frames)
    ]))
    return {"score": score, "score_perturbed": perturbed,
            "margin": score - perturbed}


def passes(scores: dict, min_score: float = DEFAULT_MIN_SCORE,
           min_margin: float = DEFAULT_MIN_MARGIN) -> bool:
    return scores["score"] >= min_score and scores["margin"] >= min_margin
