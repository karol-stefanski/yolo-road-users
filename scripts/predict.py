# -*- coding: utf-8 -*-
"""
predict.py - detekcja na wlasnych zdjeciach i filmach + pomiar rzeczywistego FPS.

Co robi:
  * detekcja z progiem pewnosci 0,50 (optimum z krzywej F1) i agnostic NMS
    (tlumienie nakladajacych sie ramek niezaleznie od klasy - jeden obiekt
    nie dostaje dwoch etykiet, np. "Motocyklista" + "Samochod"),
  * dodatkowy filtr konfliktow klas po stronie skryptu,
  * wlasne rysowanie ramek i podpisow w PIL (polskie znaki, podpisy bez nachodzenia),
  * pomiar RZECZYWISTEGO FPS calego potoku: dekodowanie -> detekcja ->
    rysowanie -> zapis klatki,
  * statystyki wykryc per klasa, raport diagnostyczny (duplikaty, konflikty klas),
    wyciecie klatek z najwieksza liczba wykryc.

Przy porownywaniu modeli zmieniaj tylko --model. Pozostale parametry musza byc
identyczne, bo FPS obejmuje caly potok i kazda zmiana psuje porownywalnosc.

Uruchomienie (z katalogu glownego repozytorium):
    python scripts/predict.py --model yolov9m
    python scripts/predict.py --model yolov8n --frames 3

Material wejsciowy:  materialy/zdjecia/  oraz  materialy/wideo/
Wyniki:              WYNIKI_PREDYKCJI_<model>/
"""

import argparse
import os
import csv
import glob
import time
from collections import Counter

# ==================== USTAWIENIA ====================
MODEL_BEST = "yolov9m"      # nadpisywane przez --model
FOLDER_WYNIKOW_TRENINGU = os.path.join("runs", "detect", "Wyniki_Magisterka")

FOLDER_ZDJEC = os.path.join("materialy", "zdjecia")
FOLDER_WIDEO = os.path.join("materialy", "wideo")
FOLDER_WYJ   = f"WYNIKI_PREDYKCJI_{MODEL_BEST}"   # osobny folder dla kazdego modelu

CONF = 0.50          # prog pewnosci (optimum wg krzywej F1 z rozdz. 8)
IOU  = 0.50          # prog IoU dla NMS
MAX_DET = 100
AGNOSTIC_NMS = True  # NMS niezalezny od klasy

# Dodatkowy filtr po stronie skryptu (zabezpieczenie, gdyby NMS cos przepuscil)
USUWAJ_KONFLIKTY_KLAS = True
PROG_KONFLIKTU = 0.55    # IoU powyzej ktorego dwie ramki ROZNYCH klas
                         # uznaje sie za ten sam obiekt
PROG_DUPLIKATU = 0.60    # IoU dla duplikatow TEJ SAMEJ klasy (diagnostyka)

TRYB_ETYKIET = "pelne"   # "pelne" | "krotkie" | "brak"
POKAZ_LEGENDE = True
GRUBOSC_RAMKI = 3
ROZMIAR_CZCIONKI = 20
KLATKI_DO_WYCIECIA = 6   # ile klatek z kazdego filmu zapisac (--frames)
# ====================================================

ROZSZ_OBRAZ = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ROZSZ_WIDEO = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

KOLORY = {
    "Hulajnoga":    (255, 87, 34),
    "Motocyklista": (0, 150, 255),
    "Pieszy":       (76, 175, 80),
    "Rowerzysta":   (233, 30, 99),
    "Samochód":     (156, 39, 176),
    "Samochod":     (156, 39, 176),
}
KOLOR_DOMYSLNY = (255, 193, 7)
NAZWY_PL = {"Samochod": "Samochód"}


def znajdz_font(rozmiar):
    from PIL import ImageFont
    for sc in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf",
               r"C:\Windows\Fonts\calibri.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        if os.path.isfile(sc):
            try:
                return ImageFont.truetype(sc, rozmiar)
            except Exception:
                pass
    return ImageFont.load_default()


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    w = (x2 - x1) * (y2 - y1)
    pa = (a[2] - a[0]) * (a[3] - a[1])
    pb = (b[2] - b[0]) * (b[3] - b[1])
    return w / (pa + pb - w + 1e-9)


def usun_konflikty(det):
    """
    Usuwa ramki ROZNYCH klas opisujace ten sam obiekt.
    Zostawia te o wyzszej pewnosci. Zwraca (nowa_lista, ile_usunieto).
    """
    if len(det) < 2:
        return det, 0
    posortowane = sorted(det, key=lambda d: -d[5])   # malejaco wg pewnosci
    zachowane = []
    usuniete = 0
    for kand in posortowane:
        konflikt = False
        for zach in zachowane:
            if kand[4] != zach[4] and iou(kand[:4], zach[:4]) > PROG_KONFLIKTU:
                konflikt = True
                break
        if konflikt:
            usuniete += 1
        else:
            zachowane.append(kand)
    return zachowane, usuniete


def policz_problemy(det):
    """Zwraca (duplikaty_tej_samej_klasy, konflikty_roznych_klas)."""
    dup = kon = 0
    for i in range(len(det)):
        for j in range(i + 1, len(det)):
            p = iou(det[i][:4], det[j][:4])
            if det[i][4] == det[j][4]:
                if p > PROG_DUPLIKATU:
                    dup += 1
            else:
                if p > PROG_KONFLIKTU:
                    kon += 1
    return dup, kon


def rysuj_detekcje(obraz_bgr, detekcje, font, klasy_obecne):
    """Rysuje ramki i podpisy bez wzajemnego nachodzenia."""
    import numpy as np
    from PIL import Image, ImageDraw

    im = Image.fromarray(obraz_bgr[:, :, ::-1])
    d = ImageDraw.Draw(im)
    W, H = im.size

    for (x1, y1, x2, y2, klasa, pew) in detekcje:
        d.rectangle([x1, y1, x2, y2],
                    outline=KOLORY.get(klasa, KOLOR_DOMYSLNY), width=GRUBOSC_RAMKI)

    if TRYB_ETYKIET != "brak":
        zajete = []
        for (x1, y1, x2, y2, klasa, pew) in sorted(
                detekcje, key=lambda t: -((t[2] - t[0]) * (t[3] - t[1]))):
            kolor = KOLORY.get(klasa, KOLOR_DOMYSLNY)
            tekst = klasa if TRYB_ETYKIET == "krotkie" else f"{klasa} {pew:.2f}"
            try:
                bb = d.textbbox((0, 0), tekst, font=font)
                tw, th = bb[2] - bb[0], bb[3] - bb[1] + 6
            except Exception:
                tw, th = len(tekst) * 8, ROZMIAR_CZCIONKI + 6
            wybrana = None
            for (px, py) in [(x1, y1 - th - 2), (x1, y2 + 2),
                             (x1 + 2, y1 + 2), (x2 - tw - 2, y1 - th - 2)]:
                px = max(0, min(px, W - tw - 1)); py = max(0, min(py, H - th - 1))
                pr = (px, py, px + tw + 6, py + th)
                if all(iou(pr, z) < 0.02 for z in zajete):
                    wybrana = pr; break
            if wybrana is None:
                continue
            zajete.append(wybrana)
            d.rectangle([wybrana[0], wybrana[1], wybrana[2], wybrana[3]], fill=kolor)
            d.text((wybrana[0] + 3, wybrana[1] + 2), tekst, fill=(255, 255, 255), font=font)

    if POKAZ_LEGENDE and klasy_obecne:
        pad = 8; wys_w = ROZMIAR_CZCIONKI + 6
        try:
            szer = max(int(font.getlength(k)) for k in klasy_obecne) + 40
        except Exception:
            szer = 180
        wysokosc = wys_w * len(klasy_obecne) + pad
        d.rectangle([W - szer - pad, pad, W - pad, pad + wysokosc],
                    fill=(255, 255, 255), outline=(60, 60, 60), width=1)
        for i, k in enumerate(sorted(klasy_obecne)):
            yy = pad + 5 + i * wys_w
            d.rectangle([W - szer, yy, W - szer + 18, yy + 14],
                        fill=KOLORY.get(k, KOLOR_DOMYSLNY))
            d.text((W - szer + 24, yy - 2), k, fill=(0, 0, 0), font=font)

    return np.array(im)[:, :, ::-1]


def wyciagnij_detekcje(r, names):
    det = []
    if r.boxes is None or len(r.boxes) == 0:
        return det
    for b in r.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        nazwa = names[int(b.cls)]
        det.append((x1, y1, x2, y2, NAZWY_PL.get(nazwa, nazwa), float(b.conf)))
    return det


def przetworz(model, font, statystyki, wiersze_csv, raport):
    """Przetwarza zdjecia."""
    pliki = [p for p in glob.glob(os.path.join(FOLDER_ZDJEC, "*"))
             if p.lower().endswith(ROZSZ_OBRAZ)]
    if not pliki:
        print(f"   (brak zdjec w {FOLDER_ZDJEC})")
        return
    import cv2
    print(f"   Znaleziono zdjec: {len(pliki)}")
    for i, sc in enumerate(pliki, 1):
        r = model.predict(sc, conf=CONF, iou=IOU, max_det=MAX_DET,
                          agnostic_nms=AGNOSTIC_NMS, verbose=False)[0]
        det = wyciagnij_detekcje(r, model.names)
        dup, kon = policz_problemy(det)
        raport["duplikaty"] += dup
        raport["konflikty_przed"] += kon
        if USUWAJ_KONFLIKTY_KLAS:
            det, usun = usun_konflikty(det)
            raport["usuniete"] += usun
        obraz = rysuj_detekcje(cv2.imread(sc), det, font, sorted({d[4] for d in det}))
        nazwa = os.path.basename(sc)
        cv2.imwrite(os.path.join(FOLDER_WYJ, "zdjecia", f"det_{nazwa}"), obraz)
        for (x1, y1, x2, y2, klasa, pew) in det:
            statystyki[klasa] += 1
            wiersze_csv.append([nazwa, "-", klasa, round(pew, 3),
                                int(x1), int(y1), int(x2), int(y2)])
        info = f"   [{i}/{len(pliki)}] {nazwa}: wykryc={len(det)}"
        if kon:
            info += f"  (konfliktow klas: {kon})"
        print(info)


def przetworz_wideo(model, font, statystyki, wiersze_csv, pomiary_fps, raport):
    pliki = [p for p in glob.glob(os.path.join(FOLDER_WIDEO, "*"))
             if p.lower().endswith(ROZSZ_WIDEO)]
    if not pliki:
        print(f"   (brak filmow w {FOLDER_WIDEO})")
        return
    import cv2
    print(f"   Znaleziono filmow: {len(pliki)}")

    for sc in pliki:
        nazwa = os.path.splitext(os.path.basename(sc))[0]
        print(f"\n   --- Film: {os.path.basename(sc)} ---")
        cap = cv2.VideoCapture(sc)
        if not cap.isOpened():
            print("   !! Nie mozna otworzyc pliku"); continue

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        szer = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        wys = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ile_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(os.path.join(FOLDER_WYJ, "wideo", f"det_{nazwa}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (szer, wys))
        kandydaci = []; nr = 0
        dup_f = kon_f = usun_f = 0
        t0 = time.time()

        while True:
            ok, klatka = cap.read()
            if not ok:
                break
            nr += 1
            r = model.predict(klatka, conf=CONF, iou=IOU, max_det=MAX_DET,
                              agnostic_nms=AGNOSTIC_NMS, verbose=False)[0]
            det = wyciagnij_detekcje(r, model.names)
            d_, k_ = policz_problemy(det)
            dup_f += d_; kon_f += k_
            if USUWAJ_KONFLIKTY_KLAS:
                det, u_ = usun_konflikty(det); usun_f += u_
            adn = rysuj_detekcje(klatka, det, font, sorted({d[4] for d in det}))
            writer.write(adn)
            for (x1, y1, x2, y2, klasa, pew) in det:
                statystyki[klasa] += 1
            if det:
                kandydaci.append((len(det), nr, adn.copy()))
                if nr % 10 == 0:
                    for (x1, y1, x2, y2, klasa, pew) in det:
                        wiersze_csv.append([os.path.basename(sc), nr, klasa,
                                            round(pew, 3), int(x1), int(y1), int(x2), int(y2)])
            if nr % 50 == 0:
                print(f"      klatek: {nr}/{ile_all}")

        czas = time.time() - t0
        cap.release(); writer.release()
        realny = nr / czas if czas > 0 else 0
        raport["duplikaty"] += dup_f
        raport["konflikty_przed"] += kon_f
        raport["usuniete"] += usun_f
        pomiary_fps.append((os.path.basename(sc), nr, round(czas, 2), round(realny, 1)))
        print(f"   Klatek: {nr} | czas: {czas:.2f} s | RZECZYWISTY FPS: {realny:.1f}")
        print(f"   Duplikaty: {dup_f} | konflikty klas: {kon_f} | usunieto: {usun_f}")

        kandydaci.sort(key=lambda x: (-x[0], x[1]))
        wybrane, uzyte = [], []
        odstep = max(5, nr // (KLATKI_DO_WYCIECIA * 2))
        for ile, nrk, obr in kandydaci:
            if all(abs(nrk - u) > odstep for u in uzyte):
                wybrane.append((ile, nrk, obr)); uzyte.append(nrk)
            if len(wybrane) >= KLATKI_DO_WYCIECIA:
                break
        for ile, nrk, obr in wybrane:
            cv2.imwrite(os.path.join(FOLDER_WYJ, "klatki",
                        f"{nazwa}_klatka{nrk:05d}_wykryc{ile}.jpg"), obr)
        print(f"   Wyciete klatki: {len(wybrane)}")


def main():
    global MODEL_BEST, FOLDER_WYJ, KLATKI_DO_WYCIECIA
    parser = argparse.ArgumentParser(description="Detekcja na zdjeciach i wideo + pomiar FPS")
    parser.add_argument("--model", default=MODEL_BEST,
                        help="nazwa modelu, np. yolov9m, yolov8n, yolo11s")
    parser.add_argument("--frames", type=int, default=KLATKI_DO_WYCIECIA,
                        help="liczba klatek do wyciecia z kazdego filmu")
    args = parser.parse_args()
    MODEL_BEST = args.model
    FOLDER_WYJ = f"WYNIKI_PREDYKCJI_{MODEL_BEST}"
    KLATKI_DO_WYCIECIA = args.frames

    print("\n" + "=" * 62)
    print("  PREDYKCJA NA NOWYM MATERIALE")
    print("=" * 62 + "\n")

    for f in (FOLDER_ZDJEC, FOLDER_WIDEO, FOLDER_WYJ,
              os.path.join(FOLDER_WYJ, "zdjecia"),
              os.path.join(FOLDER_WYJ, "wideo"),
              os.path.join(FOLDER_WYJ, "klatki")):
        os.makedirs(f, exist_ok=True)

    best = os.path.join(FOLDER_WYNIKOW_TRENINGU, f"{MODEL_BEST}_run", "weights", "best.pt")
    if not os.path.isfile(best):
        t = [p for p in glob.glob(os.path.join("**", "best.pt"), recursive=True)
             if MODEL_BEST in p] or glob.glob(os.path.join("**", "best.pt"), recursive=True)
        if not t:
            print("!! Nie znaleziono best.pt"); return
        best = t[0]

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"!! Brak ultralytics: {e}"); return

    print(f"Model         : {best}")
    print(f"conf={CONF}  iou={IOU}  agnostic_nms={AGNOSTIC_NMS}")
    print(f"filtr konfliktow klas: {USUWAJ_KONFLIKTY_KLAS} (prog IoU={PROG_KONFLIKTU})\n")

    model = YOLO(best)
    font = znajdz_font(ROZMIAR_CZCIONKI)

    statystyki = Counter(); wiersze_csv = []; pomiary_fps = []
    raport = {"duplikaty": 0, "konflikty_przed": 0, "usuniete": 0}

    print("[1/2] ZDJECIA")
    przetworz(model, font, statystyki, wiersze_csv, raport)
    print("\n[2/2] WIDEO")
    przetworz_wideo(model, font, statystyki, wiersze_csv, pomiary_fps, raport)

    print("\n" + "-" * 62)
    print("  ZLICZONE WYKRYCIA W PODZIALE NA KLASY")
    print("-" * 62)
    if statystyki:
        for k, v in sorted(statystyki.items()):
            print(f"   {k:<16}{v:>8}")
        print(f"   {'RAZEM':<16}{sum(statystyki.values()):>8}")
    else:
        print("   (brak wykryc)")

    with open(os.path.join(FOLDER_WYJ, "statystyki_wykryc.csv"), "w",
              newline="", encoding="utf-8") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["Klasa", "Liczba wykryc"])
        for k, v in sorted(statystyki.items()):
            wr.writerow([k, v])
        wr.writerow(["RAZEM", sum(statystyki.values())])

    with open(os.path.join(FOLDER_WYJ, "detekcje_szczegolowe.csv"), "w",
              newline="", encoding="utf-8") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["Plik", "Klatka", "Klasa", "Pewnosc", "x1", "y1", "x2", "y2"])
        wr.writerows(wiersze_csv)

    if pomiary_fps:
        print("\n" + "-" * 62)
        print("  RZECZYWISTY FPS PRZETWARZANIA WIDEO")
        print("-" * 62)
        for n, kl, cz, fp in pomiary_fps:
            print(f"   {n}: {kl} klatek / {cz} s = {fp} FPS")
        with open(os.path.join(FOLDER_WYJ, "fps_wideo.csv"), "w",
                  newline="", encoding="utf-8") as f:
            wr = csv.writer(f, delimiter=";")
            wr.writerow(["Plik", "Liczba klatek", "Czas [s]", "FPS"])
            wr.writerows(pomiary_fps)

    # ---------- RAPORT DIAGNOSTYCZNY ----------
    linie = [
        "RAPORT DIAGNOSTYCZNY PREDYKCJI",
        "=" * 40,
        f"Model                        : {best}",
        f"conf / iou / agnostic_nms    : {CONF} / {IOU} / {AGNOSTIC_NMS}",
        f"Filtr konfliktow klas        : {USUWAJ_KONFLIKTY_KLAS} (prog {PROG_KONFLIKTU})",
        "",
        f"Duplikaty (ta sama klasa)    : {raport['duplikaty']}",
        f"Konflikty klas (przed filtr.): {raport['konflikty_przed']}",
        f"Ramek usunietych przez filtr : {raport['usuniete']}",
        "",
        "Wykrycia per klasa:",
    ]
    for k, v in sorted(statystyki.items()):
        linie.append(f"   {k:<16}{v:>8}")
    linie.append(f"   {'RAZEM':<16}{sum(statystyki.values()):>8}")
    tresc = "\n".join(linie)
    with open(os.path.join(FOLDER_WYJ, "raport_diagnostyczny.txt"), "w",
              encoding="utf-8") as f:
        f.write(tresc)

    print("\n" + "-" * 62)
    print("  DIAGNOSTYKA")
    print("-" * 62)
    print(f"   Duplikaty (ta sama klasa)     : {raport['duplikaty']}")
    print(f"   Konflikty klas (przed filtrem): {raport['konflikty_przed']}")
    print(f"   Ramek usunietych przez filtr  : {raport['usuniete']}")
    print("-" * 62)

    print("\n" + "=" * 62)
    print(f"  GOTOWE. Wyniki: {os.path.abspath(FOLDER_WYJ)}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()