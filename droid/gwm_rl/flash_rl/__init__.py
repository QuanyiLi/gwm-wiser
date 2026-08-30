"""Vendored subset of FlashSAC (https://github.com/Holiday-Robot/FlashSAC, MIT, see LICENSE).

Only the pieces the trainer needs: the FlashSAC agent, its networks and update
rules, and the on-GPU uniform replay buffer. Package name kept as ``flash_rl``
under ``gwm_rl`` so the internal imports stay recognisable against upstream.
"""
