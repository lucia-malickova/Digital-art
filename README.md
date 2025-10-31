# 🪼 JellyViz Scientific (Beginner Friendly)

Turn an Excel survey about women in IT into a **scientific-looking jellyfish visualization** — as an image (PNG) and optionally a video (MP4).  
This project is designed for **absolute beginners** who have never coded before.  
You only need to **follow the steps exactly as written** — no coding skills required.

---

## 💡 What is JellyViz?

JellyViz is a beginner-friendly Python project that transforms simple Excel data (like survey responses) into a smooth scientific visualization that looks like a living jellyfish.  
The visual shape and motion are generated based on **four main parameters** extracted from your Excel file:

- `amplitude` → size of movements  
- `frequency` → rhythm and pulse  
- `inclusion` → density and brightness  
- `hue` → color tone

The result is a clean, reproducible visualization that could be used in research or creative data storytelling.

---

## ⚠️ Before You Start

- **Keep your real Excel private.** Do **not** upload it to GitHub or share it publicly.  
- Use the included `synthetic_women_IT_survey_*.xlsx` file for testing.  
- Later, replace it with your own Excel stored **locally on your computer**.

---

## 🧩 Quick Checklist for Mentee

✅ Step 1 — Install [Anaconda](https://www.anaconda.com/products/distribution) (if not installed).  
✅ Step 2 — Download this GitHub repo as a ZIP → unzip it anywhere on your computer.  
✅ Step 3 — Open **Anaconda Prompt** → navigate to the project folder, e.g.:
```bash
cd Desktop/Digital-art
```
✅ Step 4 — Create the environment (this installs everything automatically):
```bash
conda env create -f environment.yml
conda activate jellyviz
```
✅ Step 5 — Run the scientific jellyfish:
```bash
python meduza_scientific.py
```

You should see:
```
Parametre z dát: {'amplitude': 0.45, 'frequency': 0.35, 'inclusion': 0.50, 'hue': 0.60}
✓ Hotovo → meduza_scientific_YYYYMMDD-HHMMSS.mp4
✓ Náhľad PNG → meduza_scientific_YYYYMMDD-HHMMSS.png
```

Your **outputs** will appear automatically in the same folder.

---

## 📁 Project Layout

```
Digital-art/
├─ src/
│  └─ jellyviz/
│     ├─ __init__.py
│     ├─ excel_params.py        # reads Excel → (amplitude, frequency, inclusion, hue)
│     ├─ render_scientific.py   # generates the MP4/PNG scientific visualization
│     └─ cli.py                 # command-line interface
├─ synthetic_women_IT_survey_20251031.xlsx  # synthetic demo dataset (safe to use)
├─ meduza_scientific.py         # single-file version (simplified for beginners)
├─ environment.yml              # Conda setup file
├─ .gitignore                   # ignores output files and local Excel data
├─ LICENSE
└─ README.md
```

---

## 🧠 How It Works (Simplified)

1. The script reads data from Excel using **pandas**.  
2. It converts text answers (like “Sí”, “No”, “Alta”) into numbers.  
3. These numbers control **movement, color, and density** of points.  
4. The result is plotted frame-by-frame using **matplotlib** and saved as PNG + MP4 via **imageio**.

---

## 🧬 Educational Origin

This visualization was adapted from a **MATLAB scientific figure** and rewritten in Python as part of the **Women in High Performance Computing (WHPC) mentoring program**.  
Originally based on a study about **women in the IT industry**, but it can be easily repurposed for any other dataset — for example, women in science, leadership, or education.

---

## 🪸 Example Idea for Reuse

You can reuse JellyViz for other research:
- Gender equality in STEM fields  
- Representation of minorities in tech  
- Inclusion metrics in AI or data science  

Just prepare a similar Excel table (questions, percentages, or yes/no values).

---
![JellyViz Example](./meduza_scientific_20251031-191606)

🎞️ Watch the animation here:  
[![Watch the animation](https://img.youtube.com/vi/snEl3e49-to/0.jpg)](https://www.youtube.com/watch?v=snEl3e49-to)



## 🧰 Credits

Created by **Lucia Malíčková**  
as part of the *Women in High Performance Computing (WHPC) Mentoring Program*.  
Designed for mentees to **learn by doing**, step by step.

---

## 📜 License

This project is shared under the **MIT License** — feel free to use, modify, and share for educational and non-commercial purposes.

---
