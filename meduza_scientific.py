# meduza_scientific.py
# Scientific, clean-look “jellyfish” visualization from Excel data.
# - Reads parameters from an Excel survey (Spanish/Portuguese friendly parsing)
# - Maps them to geometry/color
# - Always saves a PNG; tries to save an MP4 if ffmpeg is available

import os, re, math, time, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imageio import v2 as imageio
from matplotlib.colors import hsv_to_rgb

# ---------- 1) SET YOUR EXCEL PATH HERE ----------
# Example (Windows): r"C:\Users\YourName\Documents\your_survey.xlsx"
EXCEL_PATH = r"Base completa - Encuesta sobre TD e Igualdad de Género en la Industria Argentina.xlsx"

# ---------- 2) EXCEL → PARAMETERS ----------
def yn_to_num(val):
    """Map yes/no in ES/PT/EN to 1/0; return NaN if unknown."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return np.nan
    s = str(val).strip().lower()
    yes = {'si','sí','sim','yes','y','true','verdadero','verdadeiro'}
    no  = {'no','nao','não','false','falso'}
    if s in yes: return 1.0
    if s in no:  return 0.0
    if 'sí' in s or s.startswith('si') or 'sim' in s: return 1.0
    if s == 'no' or 'não' in s or 'nao' in s: return 0.0
    return np.nan

def proportion_from_phrase(val):
    """Extract proportion from phrases (e.g., 'alta', 'media') or 30 / 30%."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return np.nan
    s = str(val).strip().lower()
    cues = [
        ('ninguna',0.0),('nula',0.0),('muy baja',0.15),('baja',0.3),
        ('media',0.5),('moderada',0.5),('alta',0.8),('muy alta',0.95),
        ('nenhuma',0.0),('muito baixa',0.15),('baixa',0.3),('média',0.5),('muito alta',0.95)
    ]
    for k,v in cues:
        if k in s: return v
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*%?', s)
    if m:
        x = float(m.group(1).replace(',', '.'))
        return np.clip(x/100.0 if '%' in s else x, 0, 1)
    return np.nan

def find_col(columns, patterns):
    """Find first column whose lowercase name contains any of the patterns."""
    for col in columns:
        s = str(col).lower()
        if any(p in s for p in patterns):
            return col
    return None

def extract_params_from_excel(path):
    """Return dict(amplitude, frequency, inclusion, hue) derived from Excel."""
    xl = pd.ExcelFile(path, engine="openpyxl")
    df = xl.parse(xl.sheet_names[0]).copy()
    cols = list(df.columns)

    c_has_women     = find_col(cols, ['¿cuenta con mujeres', 'mujeres dentro', 'mulheres', 'mujeres'])
    c_prop_total    = find_col(cols, ['proporción de mujeres','proporcao','proporção','total del personal'])
    c_prop_lider    = find_col(cols, ['liderazgo','liderança','roles de liderazgo','directoras','gerentes'])
    c_has_tech      = find_col(cols, ['área de tecnología','area de tecnologia','tecnolog'])
    c_prop_techteam = find_col(cols, ['equipo del área de tecnología','equipe de tecnologia','equipo de tecnologia'])
    c_sector        = find_col(cols, ['sector industrial','rama de actividad','setor','ramo'])

    def safe(c): return df[c] if (c is not None and c in df.columns) else pd.Series([np.nan]*len(df))
    s_has_women     = safe(c_has_women).map(yn_to_num)
    s_prop_total    = safe(c_prop_total).map(proportion_from_phrase)
    s_prop_lider    = safe(c_prop_lider).map(proportion_from_phrase)
    s_has_tech      = safe(c_has_tech).map(yn_to_num)
    s_prop_techteam = safe(c_prop_techteam).map(proportion_from_phrase)

    def fill(series, val):
        if not isinstance(series, pd.Series) or not series.notna().any():
            return pd.Series([val]*len(df))
        return series

    # Fallbacks (so the script always runs even if some columns are missing)
    s_has_women     = fill(s_has_women, 1.0)
    s_prop_total    = fill(s_prop_total, 0.45)
    s_prop_lider    = fill(s_prop_lider, 0.25)
    s_has_tech      = fill(s_has_tech, 0.60)
    s_prop_techteam = fill(s_prop_techteam, 0.30)

    amplitude = float(np.nanmean(0.6*s_prop_total + 0.4*s_prop_techteam))
    frequency = float(np.nanmean(0.7*s_prop_lider + 0.3*s_has_tech))
    inclusion = float(np.nanmean(0.3*s_has_women + 0.2*s_has_tech + 0.25*s_prop_total + 0.25*s_prop_techteam))

    if c_sector is not None and c_sector in df.columns:
        hue = (abs(hash(' | '.join(df[c_sector].astype(str).fillna('NA').tolist()))) % 360) / 360.0
    else:
        hue = 0.60

    def safe_val(x, default):
        try:
            if math.isnan(x): return default
        except:
            pass
        return float(np.clip(x, 0, 1))

    return dict(
        amplitude=safe_val(amplitude, 0.5),
        frequency=safe_val(frequency, 0.5),
        inclusion=safe_val(inclusion, 0.5),
        hue=safe_val(hue, 0.60),
    )

# ---------- 3) DATA → LOOK ----------
# More points = smoother shape (scientific look = no trails)
def compute_points(i_vals, x_vals, y_vals, t):
    """Vectorized geometry (same math as the MATLAB pattern)."""
    k = 5.0*np.cos(x_vals/14.0)*np.cos(y_vals/30.0)
    e = y_vals/8.0 - 13.0
    d = (np.sqrt(k*k + e*e)**2)/59.0 + 4.0
    q = 60.0 - 3.0*np.sin(np.arctan2(k,e)*e) + k*(3.0 + (4.0/d)*np.sin(d*d - 2.0*t))
    c = d/2.0 + e/99.0 - t/18.0
    px = q*np.sin(c) + 200.0
    py = (q + d*9.0)*np.cos(c) + 200.0
    return px, py

def fig_to_rgb(fig):
    """Convert current Matplotlib figure to an RGB numpy array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape((h, w, 4))[..., :3]

# ---------- 4) MAIN ----------
def main():
    # Validate Excel path early for beginners
    if not os.path.exists(EXCEL_PATH):
        print("\n[ERROR] Excel file not found. Edit EXCEL_PATH at the top of this script.\n"
              f"Current value:\n  {EXCEL_PATH}\n")
        sys.exit(1)

    # Read parameters from Excel
    P = extract_params_from_excel(EXCEL_PATH)
    print("Parameters:", P)

    # Appearance from data
    N_POINTS = int(6000 + round(P['inclusion'] * 24000))  # ~6k..30k
    SIZE  = 1.0 + 2.5*P['amplitude']                      # 1..3.5
    ALPHA = 0.45 + 0.35*P['amplitude']                    # 0.45..0.80
    FPS   = 30                                            # smooth video
    FRAMES= 600                                           # ~20 s
    DT    = np.pi/20                                      # same as MATLAB step

    # Colors: low saturation points (readability), dark neutral background
    FG = hsv_to_rgb([P['hue'], 0.22, 1.0])                 # almost white with a tint
    BG = tuple(hsv_to_rgb([ (P['hue']+0.55)%1.0, 0.35, 0.10 ]))  # dark complementary

    # Precompute vectors
    i_vals = np.arange(1, N_POINTS+1, dtype=float)
    x_vals = np.mod(i_vals, 200.0)
    y_vals = i_vals / 43.0

    # Canvas (912×912 is codec-friendly; avoids macro-block resize warnings)
    W, H = 912, 912
    dpi = 100
    fig = plt.figure(figsize=(W/dpi, H/dpi), dpi=dpi)
    ax = plt.gca()
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 400); ax.set_ylim(0, 400); ax.set_aspect('equal'); ax.axis('off')

    # Output files with timestamp
    ts = time.strftime("%Y%m%d-%H%M%S")
    OUT_MP4 = os.path.abspath(f"meduza_scientific_{ts}.mp4")
    OUT_PNG = os.path.abspath(f"meduza_scientific_{ts}.png")

    print("Saving to:", OUT_MP4)

    # --- Render loop (video) + save first frame as PNG ---
    # Try ffmpeg writer; if unavailable, skip video gracefully.
    writer = None
    try:
        writer = imageio.get_writer(
            OUT_MP4, format="FFMPEG", fps=FPS,
            codec="libx264", pixelformat="yuv420p", macro_block_size=16
        )
    except Exception:
        try:
            writer = imageio.get_writer(
                OUT_MP4, format="FFMPEG", fps=FPS,
                codec="mpeg4", pixelformat="yuv420p", macro_block_size=16
            )
        except Exception as e:
            print(f"[INFO] MP4 video will be skipped (ffmpeg/codec issue): {e}")
            writer = None

    t = 0.0
    first_frame_saved = False
    for _ in range(FRAMES):
        t += DT
        ax.cla()
        ax.set_facecolor(BG)
        ax.set_xlim(0, 400); ax.set_ylim(0, 400); ax.set_aspect('equal'); ax.axis('off')

        px, py = compute_points(i_vals, x_vals, y_vals, t)
        ax.scatter(px, py, s=SIZE, c=[FG], alpha=ALPHA, marker='o', linewidths=0)

        # Small legend (scientific look)
        txt = (f"amplitude={P['amplitude']:.2f} | frequency={P['frequency']:.2f} | "
               f"inclusion={P['inclusion']:.2f} | hue={P['hue']:.2f}\n"
               f"N={N_POINTS}  FPS={FPS}")
        ax.text(8, 392, txt, ha='left', va='top', fontsize=8, color=(0.9,0.9,0.9), family='DejaVu Sans')

        frame = fig_to_rgb(fig)

        # Save first frame as PNG (always)
        if not first_frame_saved:
            plt.imsave(OUT_PNG, frame)
            first_frame_saved = True

        # Append to video if writer is available
        if writer is not None:
            writer.append_data(np.ascontiguousarray(frame))

    if writer is not None:
        writer.close()

    plt.close(fig)
    print("✓ PNG:", OUT_PNG)
    if writer is not None:
        print("✓ MP4:", OUT_MP4)

if __name__ == "__main__":
    main()
