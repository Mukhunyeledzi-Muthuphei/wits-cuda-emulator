#!/usr/bin/env bash
# Regenerates the live visualizations linked from the GitHub Pages site (docs/demo/).
# Run from anywhere: bash docs/build-demos.sh   (or: make site)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/docs/demo"
mkdir -p "$OUT"
make -C "$ROOT" >/dev/null

for name in 01_vector_add 02_missing_bounds_check 05_race_condition 06_grayscale_2d; do
    html="$OUT/$name.html"
    # A fixed seed keeps the pages stable between rebuilds. Buggy examples exit 1 by design.
    (cd "$ROOT" && WCU_SEED=1 WCU_TRACE_OUT="$html" bin/wcu run "examples/$name.cu" >/dev/null 2>&1) || true
    [ -s "$html" ] || { echo "no visualization for $name" >&2; exit 1; }
    # Strip this machine's paths, and keep the demos out of search results (the landing page is canonical).
    sed -i.bak \
        -e "s#$ROOT/##g" \
        -e "s#$ROOT#wits-cuda-emulator#g" \
        -e 's#<head>#<head><meta name="robots" content="noindex">#' \
        "$html"
    rm -f "$html.bak"
    echo "docs/demo/$name.html"
done
