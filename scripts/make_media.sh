#!/usr/bin/env bash
# Regenerates the README images and the presentation video from the real command outputs.
# Needs: node, ffmpeg, a Chromium-based browser (default /usr/bin/brave, or set $BRAVE).
# Usage: scripts/make_media.sh [--capture] [--no-video]
#   --capture   also re-run the commands and refresh docs/media/outputs/ (slow: runs GPT-2 and make check-full)
#   --no-video  only the screenshots, charts and diagram
set -euo pipefail
cd "$(dirname "$0")/.."
capture=0; video=1
for a in "$@"; do case "$a" in --capture) capture=1;; --no-video) video=0;; *) echo "unknown option $a" >&2; exit 2;; esac; done
[ "$capture" = 1 ] && scripts/capture_outputs.sh
cd docs/media
[ -d node_modules ] || npm ci --silent
node shoot.mjs
node charts.mjs
if [ "$video" = 1 ]; then
  node render_video.mjs --poster
  node render_video.mjs               # writes bend-ml.mp4 (attach it to the release; it is not committed)
  ffmpeg -y -loglevel error -t 7 -i bend-ml.mp4 -vf "fps=15,scale=960:-1:flags=lanczos" -loop 0 -c:v libwebp -quality 72 -compression_level 6 teaser.webp
fi
echo "media ready in docs/media/"
