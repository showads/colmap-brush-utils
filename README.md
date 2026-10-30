# COLMAP + Brush Utils

Standalone macOS video → COLMAP → Gaussian splat workflow, independent of ComfyUI.

Requires `ffmpeg`, `ffprobe`, `colmap`, Python 3, and an optional locally compiled Brush binary. No pip dependencies.

```bash
python3 splat.py ~/Movies/orbit.mp4 --frames 96 --resolution 1280 \
  --train brush --brush-bin ~/Projects/brush/target/release/brush --brush-viewer
```

The CLI extracts up to 96 evenly sampled frames, runs CPU COLMAP, analyzes `sparse/0`, and optionally trains Brush. Default training settings: 10,000 iterations, maximum image dimension 1280, 1M maximum splats, SH degree 2, growth stop at 6000, refinement every 200 iterations, exports every 5000 iterations, and 2GiB image cache.

Outputs are stored in `splats/<video-name>/`, including `images/`, `database.db`, `sparse/0/`, `manifest.json`, and Brush exports in `exports/`.

To retrain without repeating reconstruction:

```bash
python3 splat.py ~/Movies/orbit.mp4 --skip-extract --skip-colmap \
  --train brush --brush-bin ~/Projects/brush/target/release/brush --brush-viewer
```

Use `--brush-iters`, `--brush-max-splats`, `--brush-sh-degree`, `--brush-export-every` to tune training. Pass additional arguments with repeated `--brush-option=--flag` and `--brush-option=value` or append `--brush-args ...` as the last option.

`--frames` controls the target number of sampled still images, not the video's original total frames. `--resolution` caps the longest extracted image dimension; `--brush-resolution` independently caps Brush's input. Start with 64–96 frames at 1280px.

**Important:** The CLI has not been verified on an Intel Mac / AMD GPU. Brush mask options do not generate masks. COLMAP reconstruction accuracy determines final training quality. The script protects against overwriting existing frames or the COLMAP database.
