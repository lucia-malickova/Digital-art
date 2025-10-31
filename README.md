JellyViz Scientific (Beginner Friendly)

Turn an Excel survey about women in IT into a scientific-looking jellyfish image (PNG) and an optional video (MP4).

🔒 Keep your real Excel private. Do not upload it to GitHub.
Use the included synthetic example or your own file stored on your computer.

1) What’s inside this repo
   Digital-art/
├─ src/jellyviz/               # the package code (for the CLI command)
│  ├─ __init__.py
│  ├─ excel_params.py          # reads Excel → (amplitude, frequency, inclusion, hue)
│  └─ render_scientific.py     # renders PNG/MP4
├─ meduza_scientific.py        # single-file script (even simpler)
├─ environment.yml             # creates the conda environment
├─ .gitignore                  # ignores outputs and local data
└─ README.md

Optional (if you add it later):
examples/sample.xlsx          # synthetic demo Excel (safe to share)
outputs/                      # generated PNG/MP4 (ignored by git)

2) Install tools (one time)

Install Anaconda (Windows/Mac/Linux):
https://www.anaconda.com/download

Open Anaconda Prompt (Windows) or Terminal (Mac/Linux).

Create and activate the environment:
conda env create -f environment.yml
conda activate jellyviz
If you already created the environment earlier, just run conda activate jellyviz.

3) Easiest way to run (single file)

This is the simplest path for beginners.

Open your Excel (.xlsx) with the survey data (or use the synthetic sample if provided).

In the file meduza_scientific.py, find this line at the top:
EXCEL_PATH = r"C:\path\to\your.xlsx"

Replace it with the full path to your Excel.
Tip (Windows): Right-click the file → Copy as path → paste it after r.

In Anaconda Prompt, run:
conda activate jellyviz
python meduza_scientific.py

You will get:

a PNG (always), e.g. meduza_scientific_YYYYMMDD-HHMMSS.png

an MP4 (if FFmpeg is available), e.g. meduza_scientific_YYYYMMDD-HHMMSS.mp4

Files are saved to the current folder.

4) Pro way (command line interface)

This uses the packaged code in src/jellyviz.

With your own Excel:
python -m jellyviz.cli --excel "C:\path\to\your.xlsx" --outdir outputs

With the synthetic example (if examples/sample.xlsx exists):
python -m jellyviz.cli --excel "examples/sample.xlsx" --outdir outputs

Optional arguments:

--frames 600 (video length; higher = longer)

--fps 30 (frames per second)

Outputs appear in the outputs/ folder.

5) What the visualization means (plain English)

The script reads four indicators from your Excel:

Overall representation of women in the company

Women in leadership roles

Women in tech teams (and whether a tech area exists)

Sector (only influences hue/color family)

These are blended into parameters that shape the jellyfish:

amplitude / inclusion → number of points & transparency (fuller or lighter shape)

frequency → how the pattern “pulses” over time

hue → color tint (sector-based)

6) Typical beginner mistakes & quick fixes

Error: Excel not found
→ Check EXCEL_PATH. Use a full path. On Windows prefer the r"..." raw string.

Excel opens but values look odd
→ Use .xlsx format and keep column headers meaningful (Spanish/Portuguese terms are supported).
The parser looks for words like “proporción de mujeres”, “roles de liderazgo”, “área de tecnología”.

No MP4 generated
→ Install FFmpeg in this conda env:
conda install -c conda-forge imageio-ffmpeg ffmpeg

Then re-run.

Macro-block warning (video resized)
→ It’s OK. We render at 912×912 to avoid this; if you change sizes, prefer multiples of 16.

7) Reuse with a different survey

Keep your Excel headers descriptive. The script searches by substrings (ES/PT supported).

You don’t need to change the code—just point EXCEL_PATH or the --excel argument to a new file.

8) Origin & credits

Originally inspired by a MATLAB generative artwork; reworked in Python within the WHPC mentoring context.
This repo demonstrates a transparent, reproducible mapping from survey data to visual storytelling.

9) License

MIT (see LICENSE).


Need a tiny checklist for your mentee?

Install Anaconda

conda env create -f environment.yml

conda activate jellyviz

Edit EXCEL_PATH in meduza_scientific.py

python meduza_scientific.py

Find your PNG (and MP4) next to the script (or in outputs/ when using the CLI)

That’s it.


