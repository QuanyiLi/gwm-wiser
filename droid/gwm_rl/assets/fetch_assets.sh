#!/usr/bin/env bash
# Fetch the YCB meshes scene8 streams from the Omniverse content bucket, plus
# the colour textures their OmniPBR materials reference relatively, so the
# scene has no network dependency at run time. Idempotent; ~15 MB.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/ycb"
BASE="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned"
mkdir -p "$HERE/ycb/Materials/Textures"
for f in 024_bowl.usd 011_banana.usd Materials/Textures/024_bowl_COLOR.png Materials/Textures/011_banana_COLOR.png; do
  if [ -s "$HERE/ycb/$f" ]; then echo "have $f"; continue; fi
  curl -sS -f -o "$HERE/ycb/$f" "$BASE/$f" && echo "fetched $f"
done
