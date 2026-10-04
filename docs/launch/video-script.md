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

## Look and motion

The identity is Bend's own: the colours of bend-lang.com (warm paper `#f2eee7`, ink `#4d4a44`, violet `#8b83b5`, green `#7e9a5e`, red `#c46a60`, orange `#c4845c`), one monospace family (a subset of iA Writer Mono S, renamed "Bend ML Mono" as its license requires), and Bend's blinking violet block cursor.

The film is one continuous paper plane with a camera that travels between the shots, instead of cuts. One object carries the story: the 26 matrix cells form the two matrices, jam, fold into the 3×3 logo glyph, reflow in the reshape, become the check boxes of the packages, the token chips of GPT-2 and the benchmark bars, and fly back into the glyph at the end. Cells move on a damped spring, the camera on exponential easing, and the frames are rendered at 60 fps and blended in pairs into 30 fps (motion blur).

## Storyboard

| Time | Shot | On screen |
|---|---|---|
| 0–7 s | close, slow push-in | `Mat<2, 3> · Mat<4, 5>`; the grids slide together, jam, and the mismatched column and row turn red; the real error; "this does not compile." |
| 7–12 s | to the title | the cells fold into the glyph above `bend-ml` and the violet cursor; "machine learning in Bend 2". |
| 12–21 s | pan right | `reshape`: 12 cells in 15 slots, the 3 empty slots turn red, "12 ≠ 15: rejected."; the slots become 3×4, the cells settle in violet, "12 = 12: it compiles." and the call with `{==}`. |
| 21–31 s | pan down | the five packages with their law counts, 24 laws and 0 @unsafe; `decode(encode(s)) == s`; ALL PROOFS CHECK and a green tick per package. |
| 31–44 s | pan right | GPT-2: loading (7 s, shown 3.5× faster), then the prompt and the 8 tokens at their real pace (1.2 s), each with its id; the ids against PyTorch; "same tokens as PyTorch". |
| 44–54 s | pan down | Bend-style bars: v1 hatched (off the chart), v2 violet, PyTorch grey; "PyTorch is still ~35× faster on MNIST, ~2–5× on GPT-2." |
| 54–60 s | pull back over the whole film, then settle | the cells fly back into the glyph; `github.com/nuxyel/bend-ml`, "5 packages on BendHub · 24 laws · MIT", 36/36 checks. |

## Posting

On X the video autoplays muted, so the captions carry the message; attach `bend-ml.mp4` to the first post of `x-thread.md`. A manual screen recording is still possible (`wf-recorder -f bend-ml.mp4`, then run the commands of the storyboard), but the generated video already shows the real outputs.
