# Presentation video (60 s)

The video is **generated**, not recorded: `docs/media/video/scene.html` animates the real command outputs (`docs/media/outputs/`) and the measured numbers (`docs/media/numbers.json`), `docs/media/render_video.mjs` renders it frame by frame with a headless Chromium-based browser, and `ffmpeg` encodes it. It is 1920x1080, 30 fps, silent, with English captions.

The MP4 is not stored in git (it would stay in the history forever). It is attached to the GitHub release: `https://github.com/nuxyel/bend-ml/releases/download/v2.1.0/bend-ml.mp4`. The repository keeps `docs/media/poster.png` and the looping `docs/media/teaser.webp` used by the README.

## Regenerate

```bash
make media                                  # screenshots, charts, poster, teaser and the video
scripts/make_media.sh --capture             # also re-run the commands and refresh docs/media/outputs/ (slow)
scripts/make_media.sh --no-video            # only the README images
gh release upload v2.1.0 docs/media/bend-ml.mp4 --clobber
```

Needs `node`, `ffmpeg` and a Chromium-based browser (`/usr/bin/brave`, or set `$BRAVE`). To look at single frames: `node docs/media/render_video.mjs --stills 3,14,40`.

## Storyboard

| Time | Scene | Caption |
|---|---|---|
| 0–6 s | A `(2×3)·(4×5)` product: the grids collide, the inner dimensions 3 ≠ 4 turn red, and the real compiler error appears. | In bend-ml, this does not compile. |
| 6–10 s | The `bend-ml` wordmark and the tagline. | — |
| 10–19 s | `reshape` 2×6 → 5×3 is refused (`expected 12n / observed 15n`); 2×6 → 3×4 compiles because the proof `{==}` is just computing. | reshape needs a proof that the size is preserved. / 12 = 12: the proof is just computing. |
| 19–29 s | `bend bpe/main.bend --verdict` → `ALL PROOFS CHECK`, the `roundtrip` law, and the counters 24 laws · 5 packages · 0 @unsafe. | Proofs checked again by a Lean-proved kernel. / 24 laws. 5 packages. No @unsafe. |
| 29–43 s | GPT-2 small: loading (sped up, labelled), then the prompt and 8 tokens at their real pace (1.2 s), and the ids compared with PyTorch. | GPT-2 small, 124 M parameters, written in Bend. / Real time: the prompt and 8 tokens in 1.2 s. / Same tokens as PyTorch. |
| 43–53 s | Honest benchmark: MNIST epoch and GPT-2 time per token, v1 → v2 and PyTorch, with the note that PyTorch is still ahead. | Fast enough to be real. Honest about the gap. |
| 53–60 s | The 36 checks of `make check-full` roll by, then the repository card. | — |

## Posting

On X the video autoplays muted, so the captions carry the message; attach `bend-ml.mp4` to the first post of `x-thread.md`. A manual screen recording is still possible (`wf-recorder -f bend-ml.mp4`, then run the commands of the storyboard), but the generated video already shows the real outputs.
