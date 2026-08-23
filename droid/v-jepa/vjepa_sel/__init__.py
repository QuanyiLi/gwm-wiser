"""V-JEPA 2-AC as an offline trajectory selector on the DROID-sim scene-6 tasks.

Package layout
  model.py       V-JEPA 2-AC wrapper: local checkpoint load, frame encoding,
                 autoregressive action-conditioned rollout, L1 energy.
  preprocess.py  the training-time image preprocessing (aspect clamp + 256x256).
  traj.py        candidate plan JSON -> EEF state / action sequence at the
                 model's 4 fps step via the shared Panda FK.
"""
