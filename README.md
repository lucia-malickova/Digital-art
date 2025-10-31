# JellyViz Scientific (Beginner Friendly)

Turn an Excel survey about women in IT into a scientific-looking **jellyfish** visualization.
- Always saves a PNG; tries to save an MP4 if FFmpeg is available.
- Your data stays on your machine. **Do not upload real Excel files** to this repo.

## Quick Start
```bash
conda env create -f environment.yml
conda activate jellyviz
python -m jellyviz.cli --excel "C:\path\to\your.xlsx" --outdir outputs

## `LICENSE`
(If you want MIT; otherwise use your preferred license.)
```text
MIT License

Copyright (c) 2025 Lucia Malickova

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
