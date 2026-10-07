import json
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USTAWIENIA
# =========================

JSON_PATH = "data/vqa/20230707_19_CN8_T1.json"
FPS = 30

CSV_OUTPUT = "outputs/vqa_timelines/phases_per_frame.csv"
PLOT_OUTPUT = "outputs/vqa_timelines/phases_plot.png"


# =========================
# WCZYTANIE JSON
# =========================

with open(JSON_PATH, "r", encoding="utf-8") as f:
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

df.to_csv(CSV_OUTPUT, index=False)

print(f"Zapisano CSV: {CSV_OUTPUT}")
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
plt.savefig(PLOT_OUTPUT, dpi=300)
plt.show()

print(f"Zapisano wykres: {PLOT_OUTPUT}")