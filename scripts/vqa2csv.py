import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from settings import CURRENT_CONFIGURATION

# =========================
# USTAWIENIA
# =========================

FPS = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VQA event phases to per-frame CSV and plot."
    )

    parser.add_argument(
        "--config",
        default=CURRENT_CONFIGURATION,
        help="Path to YAML config with a 'vqa' section.",
    )

    return parser.parse_args()


def load_vqa_paths(config_path: str) -> tuple[str, str, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    vqa = config.get("vqa")

    if not isinstance(vqa, dict):
        raise ValueError(
            f"Missing 'vqa' section in {config_path}"
        )

    missing = [
        key
        for key in ("vqa_input", "vqa_csv", "vqa_plot")
        if not vqa.get(key)
    ]

    if missing:
        raise ValueError(
            f"Missing keys in 'vqa' of {config_path}: "
            f"{', '.join(missing)}"
        )

    return (
        vqa["vqa_input"],
        vqa["vqa_csv"],
        vqa["vqa_plot"],
    )


args = parse_args()

vqa_input, vqa_csv, vqa_plot = load_vqa_paths(args.config)

# Tworzymy katalogi wyjściowe, jeśli nie istnieją
Path(vqa_csv).parent.mkdir(parents=True, exist_ok=True)
Path(vqa_plot).parent.mkdir(parents=True, exist_ok=True)

print(f"Config: {args.config}")
print(f"Wczytywanie: {vqa_input}")


# =========================
# WCZYTANIE JSON
# =========================

with open(vqa_input, "r", encoding="utf-8") as f:
    data = json.load(f)

# W tym zbiorze główny obiekt jest listą
scenario = data[0]

event_phases = scenario["event_phase"]


# =========================
# KOLEJNOŚĆ FAZ
# =========================

phase_order = {
    "prerecognition": 0,
    "recognition": 1,
    "judgement": 2,
    "action": 3,
    "avoidance": 4
}


# =========================
# WYZNACZENIE ZAKRESU KLATEK
# =========================

min_time = min(float(p["start_time"]) for p in event_phases)
max_time = max(float(p["end_time"]) for p in event_phases)

first_frame = int(min_time * FPS)
last_frame = int(max_time * FPS)

# Domyślnie klatka nie należy do żadnej fazy
frames = {
    frame: "none"
    for frame in range(first_frame, last_frame + 1)
}


# =========================
# PRZYPISANIE FAZ DO KLATEK
# =========================

# Sortujemy chronologicznie.
# Jeśli fazy lekko na siebie nachodzą,
# późniejsza faza nadpisze wcześniejszą.
event_phases = sorted(
    event_phases,
    key=lambda x: float(x["start_time"])
)

for phase in event_phases:

    start_time = float(phase["start_time"])
    end_time = float(phase["end_time"])

    phase_name = phase["labels"][0]

    start_frame = int(start_time * FPS)
    end_frame = int(end_time * FPS)

    for frame in range(start_frame, end_frame + 1):
        frames[frame] = phase_name


# =========================
# DATAFRAME
# =========================

df = pd.DataFrame({
    "frame": list(frames.keys()),
    "phase": list(frames.values())
})

df.to_csv(vqa_csv, index=False)

print(f"Zapisano CSV: {vqa_csv}")
print()
print(df.head(20))


# =========================
# WYKRES
# =========================

# Zamieniamy nazwę fazy na liczbę tylko na potrzeby wykresu
df["phase_number"] = df["phase"].map(phase_order)

plt.figure(figsize=(14, 5))

plt.step(
    df["frame"],
    df["phase_number"],
    where="post"
)

plt.xlabel("Numer klatki")
plt.ylabel("Faza")

plt.yticks(
    list(phase_order.values()),
    list(phase_order.keys())
)

plt.title("Faza zdarzenia w zależności od numeru klatki")

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(vqa_plot, dpi=300)
plt.close()

print(f"Zapisano wykres: {vqa_plot}")