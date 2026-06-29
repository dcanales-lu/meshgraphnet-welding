"""Spiral comparison poster / animation: FEM vs GENERIC surrogate vs |error|.

Reads the spiral FEM ground truth and the GENERIC-ON rollout prediction
(same mesh, 1526 steps) and renders a 3-panel top-down thermal view:
``FEM | surrogate | |error|``. With ``--poster`` it writes a single
representative frame; otherwise it writes an animated GIF (subsampled).

Run::

    uv run python scripts/make_spiral_video.py --poster      # -> docs/paper/spiral_video_poster.png
    uv run python scripts/make_spiral_video.py --stride 4    # -> docs/paper/spiral_compare.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs" / "paper"
FEM = REPO / "data" / "output" / "spiral_fem.npz"
SUR = REPO / "data" / "output" / "spiral_genon_ep60.npz"


def load():
    f = np.load(FEM, allow_pickle=True)
    s = np.load(SUR, allow_pickle=True)
    coords, cells = f["coords"], f["cells"]
    tri = Triangulation(coords[:, 0], coords[:, 1], cells)
    return tri, f["temperature"], s["temperature"], f["source_position"]


def _panel(ax, tri, field, vmin, vmax, cmap, title):
    im = ax.tripcolor(tri, field, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=13)
    return im


def make_figure(tri, Tf, Ts, t):
    vmax = float(max(Tf.max(), Ts.max())); vmin = 290.0
    emax = float(np.abs(Ts - Tf).max())
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    imT = _panel(axes[0], tri, Tf[t], vmin, vmax, "jet", "FEM (ground truth)")
    _panel(axes[1], tri, Ts[t], vmin, vmax, "jet", "GENERIC surrogate")
    imE = _panel(axes[2], tri, np.abs(Ts[t] - Tf[t]), 0, emax, "magma", "$|$error$|$")
    cb1 = fig.colorbar(imT, ax=axes[:2], fraction=0.025, pad=0.02); cb1.set_label("T [K]")
    cb2 = fig.colorbar(imE, ax=axes[2], fraction=0.05, pad=0.04); cb2.set_label("|error| [K]")
    return fig, axes, imT, imE, vmin, vmax, emax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poster", action="store_true", help="single frame instead of GIF")
    ap.add_argument("--frame", type=int, default=-1, help="poster frame index (-1 = auto)")
    ap.add_argument("--stride", type=int, default=3, help="frame stride")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gif", action="store_true", help="write GIF instead of mp4")
    ap.add_argument("--frames", action="store_true",
                    help="write a PNG sequence for LaTeX \\animategraphics (Flash-free)")
    ap.add_argument("--dpi", type=int, default=85, help="dpi for the PNG sequence")
    args = ap.parse_args()

    tri, Tf, Ts, spos = load()
    S = Tf.shape[0]

    if args.frames:
        fdir = OUT_DIR / "spiral_frames"
        fdir.mkdir(parents=True, exist_ok=True)
        for old in fdir.glob("frame-*.png"):
            old.unlink()
        idx = list(range(0, S, args.stride))
        fig, axes, *_ = make_figure(tri, Tf, Ts, idx[0])
        ttl = fig.suptitle("", fontsize=12)
        for k, t in enumerate(idx):
            axes[0].collections[0].set_array(Tf[t])
            axes[1].collections[0].set_array(Ts[t])
            axes[2].collections[0].set_array(np.abs(Ts[t] - Tf[t]))
            ttl.set_text(f"Archimedean spiral weld  |  step {t}/{S-1}")
            fig.savefig(fdir / f"frame-{k}.png", dpi=args.dpi)
        plt.close(fig)
        print(f"Wrote {len(idx)} frames to {fdir}  (frame-0 .. frame-{len(idx)-1})")
        return

    if args.poster:
        t = args.frame if args.frame >= 0 else int(np.argmax(Tf.mean(axis=1)))
        fig, *_ = make_figure(tri, Tf, Ts, t)
        out = OUT_DIR / "spiral_video_poster.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Wrote {out}  (frame {t}/{S})")
        return

    # animation (subsampled) -> mp4 (ffmpeg via imageio-ffmpeg) or gif (pillow)
    import matplotlib.animation as animation
    frames = list(range(0, S, args.stride))
    fig, axes, imT, imE, vmin, vmax, emax = make_figure(tri, Tf, Ts, frames[0])
    ttl = fig.suptitle("", fontsize=12)

    def update(t):
        axes[0].collections[0].set_array(Tf[t])
        axes[1].collections[0].set_array(Ts[t])
        axes[2].collections[0].set_array(np.abs(Ts[t] - Tf[t]))
        ttl.set_text(f"Archimedean spiral weld  |  step {t}/{S-1}")
        return []

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    if args.gif:
        out = OUT_DIR / "spiral_compare.gif"
        ani.save(out, writer="pillow", fps=args.fps)
    else:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        out = OUT_DIR / "spiral_compare.mp4"
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=3000,
                                        extra_args=["-pix_fmt", "yuv420p"])
        ani.save(out, writer=writer, dpi=110)
    plt.close(fig)
    print(f"Wrote {out}  ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
