"""
collect_results.py — zbiera wyniki 18 wytrenowanych modeli YOLO do jednej tabeli.

Dla kazdego katalogu treningu czyta results.csv, uruchamia model.val()
na zbiorze walidacyjnym (metryki + czasy preprocess/inference/postprocess)
i zapisuje:
  - summary_table.csv      (metryki, rozmiar, parametry, FPS)
  - per_class_metrics.csv  (mAP@50-95 dla kazdej z 5 klas)

Uruchomienie (z katalogu glownego repozytorium):
    python scripts/collect_results.py
    python scripts/collect_results.py --results-dir runs/detect/Wyniki_Magisterka --no-fps
"""

import argparse
import os
import csv
import sys
from pathlib import Path

# Spróbuj zaimportować ultralytics — potrzebne do pomiaru FPS i info o modelu
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    print("[UWAGA] Brak biblioteki ultralytics — pomiar FPS i info o modelu pominięty.")
    print("        Zainstaluj: pip install ultralytics")

# ==========================================
# KONFIGURACJA — dostosuj do swoich ścieżek
# ==========================================

# Katalog glowny repozytorium (nadrzedny wobec scripts/)
PROJECT_DIR = str(Path(__file__).resolve().parents[1])

# Domyslny katalog z wynikami treningu (mozna nadpisac --results-dir)
RESULTS_DIR = os.path.join(PROJECT_DIR, "runs", "detect", "Wyniki_Magisterka")

# Sciezka do data.yaml (potrzebna do model.val())
DATA_YAML = os.path.join(PROJECT_DIR, "configs", "data.yaml")

# Lista modeli — dokładnie jak w trenuj.py
# Format: (nazwa_wyświetlana, nazwa_katalogu_run)
MODELS = [
    ("YOLOv5n",  "yolov5n_run"),
    ("YOLOv5s",  "yolov5s_run"),
    ("YOLOv5m",  "yolov5m_run"),
    ("YOLOv8n",  "yolov8n_run"),
    ("YOLOv8s",  "yolov8s_run"),
    ("YOLOv8m",  "yolov8m_run"),
    ("YOLOv9t",  "yolov9t_run"),
    ("YOLOv9s",  "yolov9s_run"),
    ("YOLOv9m",  "yolov9m_run"),
    ("YOLOv10n", "yolov10n_run"),
    ("YOLOv10s", "yolov10s_run"),
    ("YOLOv10m", "yolov10m_run"),
    ("YOLOv11n", "yolo11n_run"),
    ("YOLOv11s", "yolo11s_run"),
    ("YOLOv11m", "yolo11m_run"),
    ("YOLOv12n", "yolo12n_run"),
    ("YOLOv12s", "yolo12s_run"),
    ("YOLOv12m", "yolo12m_run"),
]

# Plik wyjsciowy
OUTPUT_CSV = os.path.join(PROJECT_DIR, "results", "summary_table.csv")

# Czy mierzyć FPS przez model.val()? (trwa kilka minut per model)
# Ustaw False jeśli chcesz tylko metryki z results.csv
MEASURE_FPS = True


def read_results_csv(run_dir):
    """
    Czyta results.csv z katalogu treningu.
    Zwraca dane z OSTATNIEGO wiersza (najlepsza epoka lub ostatnia).
    """
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"  [BRAK] {csv_path}")
        return None

    with open(csv_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        print(f"  [PUSTY] {csv_path}")
        return None

    # Nagłówki — usuwamy spacje
    headers = [h.strip() for h in lines[0].strip().split(',')]
    # Ostatni wiersz z danymi
    last_line = lines[-1].strip().split(',')
    last_values = [v.strip() for v in last_line]

    data = {}
    for h, v in zip(headers, last_values):
        try:
            data[h] = float(v)
        except ValueError:
            data[h] = v

    return data


def get_best_pt_size_mb(run_dir):
    """Zwraca rozmiar pliku best.pt w MB."""
    best_path = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best_path):
        size_bytes = os.path.getsize(best_path)
        return round(size_bytes / (1024 * 1024), 2)
    return None


def get_training_time_hours(results_data):
    """
    Wyciąga czas treningu z results.csv.
    Ultralytics zapisuje czas kumulatywnie w sekundach lub godzinach
    w zależności od wersji.
    """
    # Szukamy kolumny z czasem
    for key in ['time', 'train/time']:
        if key in results_data:
            val = results_data[key]
            if isinstance(val, (int, float)):
                # Jeśli wartość > 100, prawdopodobnie sekundy
                if val > 100:
                    return round(val / 3600, 2)
                else:
                    return round(val, 2)
    return None


def measure_speed_and_metrics(run_dir, data_yaml):
    """
    Uruchamia model.val() na zbiorze walidacyjnym.
    Zwraca: speed dict, per-class metrics, overall metrics.
    """
    best_path = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best_path):
        print(f"  [BRAK] {best_path}")
        return None, None

    try:
        model = YOLO(best_path)

        # Uruchom walidację — to daje nam speed + metryki
        metrics = model.val(
            data=data_yaml,
            imgsz=640,
            batch=1,       # batch=1 dla dokładnego pomiaru latency
            device=0,      # GPU
            verbose=False,
        )

        # Speed: dict z kluczami 'preprocess', 'inference', 'postprocess' (ms)
        speed = metrics.speed

        # Metryki ogólne
        overall = {
            'precision': round(metrics.box.mp * 100, 2),      # mean precision (%)
            'recall': round(metrics.box.mr * 100, 2),          # mean recall (%)
            'mAP50': round(metrics.box.map50 * 100, 2),        # mAP@0.5 (%)
            'mAP50_95': round(metrics.box.map * 100, 2),       # mAP@0.5:0.95 (%)
        }

        # Metryki per klasa
        per_class = {}
        class_names = metrics.names  # dict: {0: 'Hulajnoga', 1: 'Motocyklista', ...}
        if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
            for i, class_map in enumerate(metrics.box.maps):
                if i in class_names:
                    per_class[class_names[i]] = round(class_map * 100, 2)

        # Info o modelu
        model_info = model.info(verbose=False)

        return {
            'speed': speed,
            'overall': overall,
            'per_class': per_class,
            'parameters_M': round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
        }, metrics

    except Exception as e:
        print(f"  [BŁĄD model.val()] {e}")
        return None, None


def main():
    global RESULTS_DIR, MEASURE_FPS
    parser = argparse.ArgumentParser(description="Zbieranie wynikow 18 modeli YOLO")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="katalog z podkatalogami <model>_run")
    parser.add_argument("--no-fps", action="store_true",
                        help="pomin model.val() - tylko dane z results.csv")
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir
    MEASURE_FPS = not args.no_fps
    os.makedirs(os.path.join(PROJECT_DIR, "results"), exist_ok=True)

    print("=" * 70)
    print("  ZBIERANIE WYNIKÓW Z 18 MODELI YOLO")
    print("=" * 70)

    if not os.path.exists(RESULTS_DIR):
        print(f"\n[BŁĄD] Nie znaleziono katalogu: {RESULTS_DIR}")
        print(f"       Podaj wlasciwy katalog: --results-dir <sciezka>")
        sys.exit(1)

    # Sprawdź dostępne katalogi
    available = os.listdir(RESULTS_DIR)
    print(f"\nKatalogi w {RESULTS_DIR}:")
    for d in sorted(available):
        print(f"  {d}")

    results_rows = []
    per_class_rows = []

    for display_name, run_folder in MODELS:
        run_dir = os.path.join(RESULTS_DIR, run_folder)

        print(f"\n{'─' * 50}")
        print(f"  Model: {display_name} ({run_folder})")
        print(f"{'─' * 50}")

        if not os.path.exists(run_dir):
            print(f"  [POMINIĘTY] Katalog nie istnieje: {run_dir}")
            results_rows.append({
                'Model': display_name,
                'Rozmiar (MB)': 'N/A',
                'Czas treningu (h)': 'N/A',
                'Preprocess (ms)': 'N/A',
                'Inference (ms)': 'N/A',
                'Postprocess (ms)': 'N/A',
                'FPS': 'N/A',
                'Precision (%)': 'N/A',
                'Recall (%)': 'N/A',
                'mAP@50 (%)': 'N/A',
                'mAP@50-95 (%)': 'N/A',
                'Parametry (M)': 'N/A',
            })
            continue

        # 1. Rozmiar best.pt
        size_mb = get_best_pt_size_mb(run_dir)
        print(f"  Rozmiar best.pt: {size_mb} MB")

        # 2. Dane z results.csv
        csv_data = read_results_csv(run_dir)
        train_time = None
        csv_precision = None
        csv_recall = None
        csv_map50 = None
        csv_map50_95 = None

        if csv_data:
            train_time = get_training_time_hours(csv_data)

            # Metryki z results.csv (jako backup jeśli model.val() nie zadziała)
            for key in csv_data:
                if 'precision' in key.lower():
                    csv_precision = round(csv_data[key] * 100, 2) if isinstance(csv_data[key], float) and csv_data[key] <= 1 else csv_data[key]
                if 'recall' in key.lower():
                    csv_recall = round(csv_data[key] * 100, 2) if isinstance(csv_data[key], float) and csv_data[key] <= 1 else csv_data[key]
                if 'map50-95' in key.lower() or 'map50_95' in key.lower() or key.strip() == 'metrics/mAP50-95(B)':
                    csv_map50_95 = round(csv_data[key] * 100, 2) if isinstance(csv_data[key], float) and csv_data[key] <= 1 else csv_data[key]
                if ('map50' in key.lower() and '95' not in key.lower()) or key.strip() == 'metrics/mAP50(B)':
                    csv_map50 = round(csv_data[key] * 100, 2) if isinstance(csv_data[key], float) and csv_data[key] <= 1 else csv_data[key]

            print(f"  Czas treningu: {train_time} h")
            print(f"  Epoki: {int(csv_data.get('epoch', -1)) + 1}")
            print(f"  mAP@50 (z CSV): {csv_map50}%")
            print(f"  mAP@50-95 (z CSV): {csv_map50_95}%")

        # 3. Pomiar FPS i dokładne metryki przez model.val()
        speed_data = None
        val_precision = None
        val_recall = None
        val_map50 = None
        val_map50_95 = None
        parameters_m = None
        per_class = {}

        if MEASURE_FPS and HAS_ULTRALYTICS and os.path.exists(DATA_YAML):
            print(f"  Uruchamiam model.val() dla pomiaru FPS...")
            val_result, raw_metrics = measure_speed_and_metrics(run_dir, DATA_YAML)
            if val_result:
                speed_data = val_result['speed']
                val_precision = val_result['overall']['precision']
                val_recall = val_result['overall']['recall']
                val_map50 = val_result['overall']['mAP50']
                val_map50_95 = val_result['overall']['mAP50_95']
                parameters_m = val_result['parameters_M']
                per_class = val_result.get('per_class', {})

                print(f"  Speed: pre={speed_data.get('preprocess', '?'):.2f}ms, "
                      f"inf={speed_data.get('inference', '?'):.2f}ms, "
                      f"post={speed_data.get('postprocess', '?'):.2f}ms")
                total_ms = (speed_data.get('preprocess', 0) +
                           speed_data.get('inference', 0) +
                           speed_data.get('postprocess', 0))
                fps = round(1000 / total_ms, 1) if total_ms > 0 else 0
                print(f"  FPS: {fps}")
                print(f"  mAP@50-95 (z val): {val_map50_95}%")
                print(f"  Parametry: {parameters_m}M")

                if per_class:
                    print(f"  Per-class mAP@50-95:")
                    for cls_name, cls_map in per_class.items():
                        print(f"    {cls_name}: {cls_map}%")
        else:
            if not MEASURE_FPS:
                print(f"  [POMINIĘTO] Pomiar FPS wyłączony (MEASURE_FPS=False)")
            elif not HAS_ULTRALYTICS:
                print(f"  [POMINIĘTO] Brak ultralytics")
            elif not os.path.exists(DATA_YAML):
                print(f"  [POMINIĘTO] Brak data.yaml pod: {DATA_YAML}")

        # Użyj danych z model.val() jeśli dostępne, w przeciwnym razie z CSV
        final_precision = val_precision if val_precision is not None else csv_precision
        final_recall = val_recall if val_recall is not None else csv_recall
        final_map50 = val_map50 if val_map50 is not None else csv_map50
        final_map50_95 = val_map50_95 if val_map50_95 is not None else csv_map50_95

        # Oblicz FPS
        if speed_data:
            pre = speed_data.get('preprocess', 0)
            inf = speed_data.get('inference', 0)
            post = speed_data.get('postprocess', 0)
            total = pre + inf + post
            fps = round(1000 / total, 1) if total > 0 else 0
        else:
            pre = inf = post = fps = 'N/A'

        row = {
            'Model': display_name,
            'Rozmiar (MB)': size_mb if size_mb else 'N/A',
            'Parametry (M)': parameters_m if parameters_m else 'N/A',
            'Czas treningu (h)': train_time if train_time else 'N/A',
            'Preprocess (ms)': round(pre, 2) if isinstance(pre, (int, float)) else pre,
            'Inference (ms)': round(inf, 2) if isinstance(inf, (int, float)) else inf,
            'Postprocess (ms)': round(post, 2) if isinstance(post, (int, float)) else post,
            'FPS': fps,
            'Precision (%)': final_precision if final_precision else 'N/A',
            'Recall (%)': final_recall if final_recall else 'N/A',
            'mAP@50 (%)': final_map50 if final_map50 else 'N/A',
            'mAP@50-95 (%)': final_map50_95 if final_map50_95 else 'N/A',
        }
        results_rows.append(row)

        # Per-class row
        if per_class:
            pc_row = {'Model': display_name}
            pc_row.update(per_class)
            per_class_rows.append(pc_row)

    # ==========================================
    # ZAPIS DO CSV
    # ==========================================
    print(f"\n{'=' * 70}")
    print(f"  ZAPISUJĘ WYNIKI")
    print(f"{'=' * 70}")

    # 1. Tabela zbiorcza
    fieldnames = ['Model', 'Rozmiar (MB)', 'Parametry (M)', 'Czas treningu (h)',
                  'Preprocess (ms)', 'Inference (ms)', 'Postprocess (ms)', 'FPS',
                  'Precision (%)', 'Recall (%)', 'mAP@50 (%)', 'mAP@50-95 (%)']

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)

    print(f"\n  ✓ Zapisano: {OUTPUT_CSV}")
    print(f"    ({len(results_rows)} modeli)")

    # 2. Tabela per-class
    if per_class_rows:
        pc_output = os.path.join(PROJECT_DIR, "results", "per_class_metrics.csv")
        # Zbierz wszystkie nazwy klas
        all_classes = set()
        for row in per_class_rows:
            all_classes.update(k for k in row.keys() if k != 'Model')
        all_classes = sorted(all_classes)

        pc_fieldnames = ['Model'] + all_classes
        with open(pc_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=pc_fieldnames, delimiter=';')
            writer.writeheader()
            for row in per_class_rows:
                writer.writerow(row)

        print(f"  ✓ Zapisano: {pc_output}")
        print(f"    ({len(per_class_rows)} modeli × {len(all_classes)} klas)")

    # 3. Podgląd tabeli w konsoli
    print(f"\n{'=' * 70}")
    print(f"  PODGLĄD WYNIKÓW")
    print(f"{'=' * 70}\n")

    # Nagłówek
    header = f"{'Model':<12} {'MB':>8} {'Param(M)':>9} {'Czas(h)':>8} {'Pre(ms)':>8} {'Inf(ms)':>8} {'Post(ms)':>9} {'FPS':>7} {'P(%)':>7} {'R(%)':>7} {'mAP50':>7} {'mAP50-95':>9}"
    print(header)
    print("─" * len(header))

    for row in results_rows:
        line = (f"{row['Model']:<12} "
                f"{str(row['Rozmiar (MB)']):>8} "
                f"{str(row['Parametry (M)']):>9} "
                f"{str(row['Czas treningu (h)']):>8} "
                f"{str(row['Preprocess (ms)']):>8} "
                f"{str(row['Inference (ms)']):>8} "
                f"{str(row['Postprocess (ms)']):>9} "
                f"{str(row['FPS']):>7} "
                f"{str(row['Precision (%)']):>7} "
                f"{str(row['Recall (%)']):>7} "
                f"{str(row['mAP@50 (%)']):>7} "
                f"{str(row['mAP@50-95 (%)']):>9}")
        print(line)

    print(f"\n{'=' * 70}")
    print(f"  GOTOWE! Wyniki: {OUTPUT_CSV}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()