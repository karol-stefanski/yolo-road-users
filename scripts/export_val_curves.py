"""
export_val_curves.py - szczegolowa ewaluacja jednego modelu i eksport danych do wykresow.

Zapisuje do CSV: metryki per klasa, macierz pomylek, krzywe P/R/F1 w funkcji
progu pewnosci, krzywa PR oraz podsumowanie (m.in. F1 przy progu 0,50
i prog maksymalizujacy F1).

Uwagi:
  - iou=0.7 jak w treningu (args.yaml); inna wartosc daje inne tlumienie ramek
    w NMS i rozjazd wynikow wzgledem logu treningu,
  - krzywe rozpoznawane po naglowkach osi, nie po kolejnosci
    (w Ultralytics 8.4.42 kolejnosc to [PR, F1(conf), P(conf), R(conf)]).

Uruchomienie (z katalogu glownego repozytorium):
    python scripts/export_val_curves.py --weights runs/detect/Wyniki_Magisterka/yolov9m_run/weights/best.pt
"""

from pathlib import Path
import argparse
import csv
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WAGI = "best.pt"
DATA_YAML = str(ROOT / "configs" / "data.yaml")


def main():
    global WAGI
    parser = argparse.ArgumentParser(description="Eksport metryk i krzywych modelu")
    parser.add_argument("--weights", default=WAGI, help="sciezka do best.pt")
    parser.add_argument("--out", default="dane_do_wykresow", help="katalog wyjsciowy")
    args = parser.parse_args()
    WAGI = args.weights
    wyjscie = Path(args.out)
    wyjscie.mkdir(exist_ok=True)

    print(f"Wagi:      {WAGI}")
    print(f"data.yaml: {DATA_YAML}")
    print("iou:       0.7  (zgodnie z args.yaml z treningu)\n")

    model = YOLO(WAGI)
    m = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=640,
        conf=0.001,
        iou=0.7,          # <<< kluczowa poprawka
        max_det=300,
        workers=0,
        plots=True,
        save_json=False,
        verbose=True,
    )

    nazwy = [m.names[i] for i in sorted(m.names)]
    box = m.box

    # 1. METRYKI PER KLASA -> rysunek 8.5
    p = wyjscie / "01_metryki_per_klasa.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Klasa", "Precyzja", "Czulosc", "mAP50", "mAP50_95"])
        for i, idx in enumerate(box.ap_class_index):
            w.writerow([m.names[int(idx)], f"{box.p[i]:.6f}", f"{box.r[i]:.6f}",
                        f"{box.ap50[i]:.6f}", f"{box.ap[i]:.6f}"])
        w.writerow(["WSZYSTKIE", f"{box.mp:.6f}", f"{box.mr:.6f}",
                    f"{box.map50:.6f}", f"{box.map:.6f}"])
    print(f"[OK] {p}")

    # 2. MACIERZ POMYLEK -> rysunek 8.6
    mac = m.confusion_matrix.matrix
    etykiety = nazwy + ["tlo"]
    p = wyjscie / "02_macierz_pomylek.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["przewidziana_rzeczywista"] + etykiety)
        for i, et in enumerate(etykiety):
            w.writerow([et] + [int(v) for v in mac[i]])
    print(f"[OK] {p}")

    # 3. KRZYWE — rozpoznawane po naglowkach
    krzywe = {}
    for x, y, xl, yl in m.curves_results:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[None, :]
        klucz = f"{xl}_{yl}".lower()
        krzywe[klucz] = (x, y, xl, yl)
        nazwa = {"recall_precision": "PR_precyzja_od_czulosci",
                 "confidence_f1": "F1_od_progu",
                 "confidence_precision": "precyzja_od_progu",
                 "confidence_recall": "czulosc_od_progu"}.get(klucz, klucz)
        p = wyjscie / f"03_{nazwa}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([xl] + [f"{yl}__{n}" for n in nazwy[:y.shape[0]]])
            for j in range(len(x)):
                w.writerow([f"{x[j]:.6f}"] + [f"{y[k][j]:.6f}" for k in range(y.shape[0])])
        print(f"[OK] {p}   ({xl} -> {yl})")

    # 4. PODSUMOWANIE — F1 z wlasciwej krzywej
    x_conf, y_f1, _, _ = krzywe["confidence_f1"]
    f1_sr = y_f1.mean(axis=0)
    i50 = int(np.argmin(np.abs(x_conf - 0.50)))
    imax = int(np.argmax(f1_sr))

    p = wyjscie / "04_podsumowanie.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Wielkosc", "Wartosc"])
        w.writerow(["wagi", WAGI])
        w.writerow(["iou (NMS)", "0.7"])
        w.writerow(["mAP50-95 wszystkie klasy", f"{box.map:.6f}"])
        w.writerow(["mAP50 wszystkie klasy", f"{box.map50:.6f}"])
        w.writerow(["precyzja srednia", f"{box.mp:.6f}"])
        w.writerow(["czulosc srednia", f"{box.mr:.6f}"])
        w.writerow(["F1 srednie przy progu 0,50", f"{f1_sr[i50]:.6f}"])
        w.writerow(["F1 maksymalne", f"{f1_sr[imax]:.6f}"])
        w.writerow(["prog dla maksimum F1", f"{x_conf[imax]:.6f}"])
    print(f"[OK] {p}")
    print(f"\nmAP50-95 = {box.map*100:.2f}%")
    print(f"          F1 przy progu 0,50 = {f1_sr[i50]:.4f}")
    print("\nGotowe. Wyniki w:", wyjscie.resolve())


if __name__ == "__main__":
    main()