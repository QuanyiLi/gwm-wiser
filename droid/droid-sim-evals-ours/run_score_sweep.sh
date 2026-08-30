#!/usr/bin/env bash
# rat_scale regression sweep against a running gwm-server (:8901).
# Scores the scene1 proposals for both grasp instructions at the default
# (3.0), canonical (1.0), and full-trajectory uniform (none); rankings land in
# the proposals dir as scores_<tag>.json, RAT strips under
# runs/gwm_ab/rat_debug/<tag>/.
set -uo pipefail

PY=/root/code/gwm/gwm-wiser/droid/tiptop/.pixi/envs/default/bin/python
PROP=/root/code/gwm/gwm-wiser/droid/gwm_integrate_doc/proposals/scene1
H5=/root/code/gwm/gwm-wiser/droid/droid-sim-evals/tiptop_assets/external_scene1_0.h5
DBG=/root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours/runs/gwm_ab/rat_debug

for instr_pair in "cube:pick up the cube" "bowl:pick up the bowl"; do
  obj="${instr_pair%%:*}"; instr="${instr_pair#*:}"
  for scale in 3.0 1.0 none; do
    tag="${obj}_scale_${scale/./}"
    echo "=== $tag ==="
    $PY -m gwm_tiptop.score_client --proposals-dir "$PROP" --external-h5 "$H5" \
        --instruction "$instr" --tag "$tag" --rat-scale "$scale" --dump-dir "$DBG/$tag"
  done
done
