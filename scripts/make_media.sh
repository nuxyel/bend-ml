#!/usr/bin/env bash
# Regenerates the README images (screenshots, banner, diagram, the deck frames) from the real command outputs and
# numbers.json; the launch video (v1.0) only with --video.
# Needs: node, ffmpeg, a Chromium-based browser (default /usr/bin/brave, or set $BRAVE).
# Usage: scripts/make_media.sh [--capture] [--video]
#   --capture   also re-run the commands and refresh docs/media/outputs/ (slow: runs GPT-2 and make check-full)
#   --video     also render the launch video, its poster and teaser
set -euo pipefail
cd "$(dirname "$0")/.."
capture=0; video=0
for a in "$@"; do case "$a" in --capture) capture=1;; --video) video=1;; *) echo "unknown option $a" >&2; exit 2;; esac; done
[ "$capture" = 1 ] && scripts/capture_outputs.sh
cd docs/media
[ -d node_modules ] || npm ci --silent
node shoot.mjs
node charts.mjs
node deck.mjs deck/readme.mjs deck/frames
if [ "$video" = 1 ]; then
  node render_video.mjs --poster
  node render_video.mjs               # writes bend-ml.mp4 (attach it to the release; it is not committed)
  ffmpeg -y -loglevel error -t 7 -i bend-ml.mp4 -vf "fps=15,scale=960:-1:flags=lanczos" -loop 0 -c:v libwebp -quality 72 -compression_level 6 teaser.webp
fi
echo "media ready in docs/media/"
