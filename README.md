# JellyViz Scientific (Beginner Friendly)

Turn an Excel survey about women in IT into a scientific-looking **jellyfish** visualization.
- Always saves a PNG; tries to save an MP4 if FFmpeg is available.
- Your data stays on your machine. **Do not upload real Excel files** to this repo.

## Quick Start
```bash
conda env create -f environment.yml
conda activate jellyviz
python -m jellyviz.cli --excel "C:\path\to\your.xlsx" --outdir outputs
