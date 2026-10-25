"""
plot_accuracy_vs_speed.py - wykres dokladnosc (mAP@50-95) vs szybkosc (FPS) dla 18 modeli.

Uruchomienie (z katalogu glownego repozytorium):
    python scripts/plot_accuracy_vs_speed.py
Wynik: results/accuracy_vs_speed.png
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "models_summary.csv"
OUT = ROOT / "results" / "accuracy_vs_speed.png"

HIGHLIGHT = {
    "YOLOv9m": "najdokładniejszy",
    "YOLOv8n": "najszybszy",
    "YOLO11s": "najlepszy kompromis",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
ACCENT, OTHER = "#2a78d6", "#a9a8a2"
# przesuniecia podpisow tam, gdzie punkty leza blisko siebie
LABEL_OFFSET = {"YOLO12s": (-14, 6), "YOLO11m": (5, 6)}


def main():
    with open(SRC, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for r in rows:
        name, fps, m = r["Model"], float(r["FPS"]), float(r["mAP@50-95 (%)"])
        hi = name in HIGHLIGHT
        ax.scatter(fps, m, s=70 if hi else 40, color=ACCENT if hi else OTHER,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if hi:
            ax.annotate(f"{name}\n{HIGHLIGHT[name]}", (fps, m), xytext=(8, 4),
                        textcoords="offset points", fontsize=9, color=INK, va="bottom")
        else:
            ax.annotate(name, (fps, m), xytext=LABEL_OFFSET.get(name, (5, -9)),
                        textcoords="offset points", fontsize=7, color=MUTED)

    ax.set_xlabel("Szybkość [FPS, model.val(), RTX 5060 Ti]", color=MUTED, fontsize=9)
    ax.set_ylabel("mAP@50-95 [%]", color=MUTED, fontsize=9)
    ax.set_title("Dokładność a szybkość - 18 modeli YOLO (v5-v12, n/s/m)",
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlim(45, 135)
    ax.set_ylim(75.8, 81.3)
    fig.tight_layout()
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"Zapisano: {OUT}")


if __name__ == "__main__":
    main()
