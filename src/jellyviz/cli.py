import argparse
from .render_scientific import render

def main():
    p = argparse.ArgumentParser(description="JellyViz — scientific jellyfish")
    p.add_argument("--excel", required=True, help="Path to your Excel (.xlsx) — keep private")
    p.add_argument("--outdir", default="outputs", help="Folder for outputs (PNG/MP4)")
    p.add_argument("--frames", type=int, default=600, help="Number of frames (video length)")
    p.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = p.parse_args()

    render(output_dir=args.outdir, excel_path=args.excel, frames=args.frames, fps=args.fps)

if __name__ == "__main__":
    main()
