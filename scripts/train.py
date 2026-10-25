"""
train.py - trening 18 modeli YOLO jeden po drugim przy jednakowych hiperparametrach.

Uruchomienie (z katalogu glownego repozytorium):
    python scripts/train.py                     # wszystkie 18 modeli
    python scripts/train.py --models yolov8n.pt yolo11s.pt

Powiadomienia (opcjonalnie): ustaw zmienna srodowiskowa DISCORD_WEBHOOK_URL,
a skrypt wysle wiadomosc na Discorda po kazdym modelu. Bez niej wypisuje
informacje tylko w konsoli. Adresu webhooka nie wpisuje sie do kodu.
"""

import argparse
import multiprocessing
import os
import traceback
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "configs" / "data.yaml"

# 6 generacji x 3 warianty. W YOLOv9 najmniejszy wariant nazywa sie "t" (odpowiednik "n").
MODELS = [
    "yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt",
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
    "yolov9t.pt", "yolov9s.pt", "yolov9m.pt",
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
    "yolo12n.pt", "yolo12s.pt", "yolo12m.pt",
]

# Hiperparametry wspolne dla wszystkich modeli (porownanie w rownych warunkach).
TRAIN_ARGS = dict(
    epochs=100,
    patience=20,     # early stopping po 20 epokach bez poprawy
    imgsz=640,
    batch=-1,        # AutoBatch - Ultralytics dobiera batch do pamieci GPU
    workers=4,
    amp=True,        # trening w precyzji mieszanej (FP16)
    cache=False,
    seed=0,
    device=0,
    project="Wyniki_Magisterka",
)


def notify(message: str) -> None:
    """Wypisuje komunikat i - jesli skonfigurowano - wysyla go na Discorda."""
    print(f"\n[INFO] {message}\n")
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        return
    try:
        import requests
        requests.post(webhook, json={"content": message}, timeout=10)
    except Exception as e:  # powiadomienie nie moze przerwac treningu
        print(f"[UWAGA] Nie udalo sie wyslac powiadomienia: {e}")


def run_name(weights: str) -> str:
    """'yolov5nu.pt' -> 'yolov5n_run' (taka nazwa katalogu jak w pracy)."""
    stem = Path(weights).stem
    if stem.startswith("yolov5") and stem.endswith("u"):
        stem = stem[:-1]
    return f"{stem}_run"


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening serii modeli YOLO")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="lista wag startowych (domyslnie wszystkie 18)")
    args = parser.parse_args()

    notify(f"Start treningu. Modeli do przetrenowania: {len(args.models)}")

    for weights in args.models:
        notify(f"Trening modelu {weights}...")
        try:
            model = YOLO(weights)
            model.train(data=str(DATA_YAML), name=run_name(weights), **TRAIN_ARGS)
            notify(f"OK: {weights} zakonczony")
        except Exception as e:
            # blad jednego modelu nie zatrzymuje calej serii
            notify(f"BLAD: {weights} - {str(e)[:200]}")
            print(traceback.format_exc())

    notify("Koniec serii treningow.")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # wymagane przy wielu procesach na Windowsie
    main()
