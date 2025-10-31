import os, time
import numpy as np
import matplotlib.pyplot as plt
from imageio import v2 as imageio
from matplotlib.colors import hsv_to_rgb
from .excel_params import params_from_excel

def _fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape((h, w, 4))[..., :3]

def _open_ffmpeg_writer(path, fps, prefer_mpeg4=False):
    codec = "mpeg4" if prefer_mpeg4 else "libx264"
    return imageio.get_writer(
        path, format="FFMPEG", fps=fps, codec=codec,
        pixelformat="yuv420p", macro_block_size=16
    )

def render(output_dir="outputs", excel_path=None, frames=600, fps=30):
    assert excel_path and os.path.exists(excel_path), f"Excel not found: {excel_path}"
    P = params_from_excel(excel_path)

    N_POINTS = int(6000 + round(P['inclusion'] * 24000))
    SIZE  = 1.0 + 2.5*P['amplitude']
    ALPHA = 0.45 + 0.35*P['amplitude']
    DT    = np.pi/20

    FG = hsv_to_rgb([P['hue'], 0.22, 1.0])
    BG = tuple(hsv_to_rgb([ (P['hue']+0.55)%1.0, 0.35, 0.10 ]))

    i_vals = np.arange(1, N_POINTS+1, dtype=float)
    x_vals = np.mod(i_vals, 200.0)
    y_vals = i_vals / 43.0

    def compute_points(t):
        k = 5.0*np.cos(x_vals/14.0)*np.cos(y_vals/30.0)
        e = y_vals/8.0 - 13.0
        d = (np.sqrt(k*k + e*e)**2)/59.0 + 4.0
        q = 60.0 - 3.0*np.sin(np.arctan2(k,e)*e) + k*(3.0 + (4.0/d)*np.sin(d*d - 2.0*t))
        c = d/2.0 + e/99.0 - t/18.0
        px = q*np.sin(c) + 200.0
        py = (q + d*9.0)*np.cos(c) + 200.0
        return px, py

    W, H = 912, 912  # codec-friendly
    dpi = 100
    fig = plt.figure(figsize=(W/dpi, H/dpi), dpi=dpi)
    ax = plt.gca()
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 400); ax.set_ylim(0, 400); ax.set_aspect('equal'); ax.axis('off')

    ts = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    out_mp4 = os.path.join(output_dir, f"meduza_scientific_{ts}.mp4")
    out_png = os.path.join(output_dir, f"meduza_scientific_{ts}.png")

    # Try ffmpeg writer with fallback
    try:
        writer = _open_ffmpeg_writer(out_mp4, fps, prefer_mpeg4=False)
    except Exception:
        writer = _open_ffmpeg_writer(out_mp4, fps, prefer_mpeg4=True)

    # Save a clean PNG first (t=0)
    t = 0.0
    px, py = compute_points(t)
    ax.scatter(px, py, s=SIZE, c=[FG], alpha=ALPHA, marker='o', linewidths=0)
    plt.savefig(out_png, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())

    # Video frames
    for _ in range(frames):
        t += DT
        ax.cla()
        ax.set_facecolor(BG)
        ax.set_xlim(0, 400); ax.set_ylim(0, 400); ax.set_aspect('equal'); ax.axis('off')

        px, py = compute_points(t)
        ax.scatter(px, py, s=SIZE, c=[FG], alpha=ALPHA, marker='o', linewidths=0)

        txt = (f"amplitude={P['amplitude']:.2f} | frequency={P['frequency']:.2f} | "
               f"inclusion={P['inclusion']:.2f} | hue={P['hue']:.2f}\n"
               f"N={N_POINTS}  FPS={fps}")
        ax.text(8, 392, txt, ha='left', va='top', fontsize=8, color=(0.9,0.9,0.9), family='DejaVu Sans')

        frame = _fig_to_rgb(fig)
        writer.append_data(np.ascontiguousarray(frame))

    writer.close(); plt.close(fig)
    print("✓ PNG:", os.path.abspath(out_png))
    print("✓ MP4:", os.path.abspath(out_mp4))
