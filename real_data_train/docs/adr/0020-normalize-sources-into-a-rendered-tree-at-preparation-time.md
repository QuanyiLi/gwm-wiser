---
status: accepted
---

# Normalize sources into a rendered tree at preparation time

*Amended 2026-08-07 (plan D-27): the robot-only container changed from PNG
per frame to one FFV1 lossless MKV per clip after cluster byte/inode quotas
made 60 M PNGs untenable at run-1 scale. Bit-exactness through the torchcodec
decode path is verified per clip at write time; schema v1 (PNG) trees remain
readable. The tree contract itself is unchanged.*

The training side consumes exactly one on-disk contract — the rendered tree
(`data/rendered/<source>/<clip_id>/{robot_only/*.png, meta.json}`) — instead
of per-source runtime adapters (the shape the plan originally sketched as
`adapters/molmoact2_droid/` + `adapters/molmobot/`).

Robot-only rendering is already an offline pass (ADR-0017 with D-15): the
render script must walk every source episode, camera stream, and frame, and
at that moment it holds every piece of alignment information training will
ever need — timestamps, the source RGB location, camera parameters, and
render provenance. Writing that into a normalized `meta.json` costs nothing;
after it, source formats (LeRobot v3.0 parquet/concatenated-AV1 on one side,
tar.zst scene packages with JSON-in-h5 states on the other) never reach the
training process. Full-scene RGB deliberately stays in the source videos —
extracting 17.8 M frames to images would multiply disk for no benefit — and
is decoded on the fly, which the per-frame `rgb_video`/`rgb_frame_start`
mapping makes uniform across both corpora.

Consequences: there is ONE window dataset, ONE audit, ONE held-out split
(deterministic hash of the camera-independent `episode_uid`), and ONE
visualization path; a new corpus is a new reader + render pass, zero training
changes. The cost is that `meta.json` is a versioned interface
(`schema_version`) that render and training sides must agree on, and any
change to rendering invalidates the tree (re-render, tracked by provenance
hashes). Runtime adapters would have avoided the persisted interface but
duplicated windowing, auditing, splitting, and visualization per source, and
would have put SAPIEN into dataloader workers — rejected.
