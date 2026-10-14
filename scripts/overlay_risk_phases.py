import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from settings import CURRENT_CONFIGURATION

OVERLAY_FILENAME = "risk_vs_phases_overlay.png"

RISK_TO_NUMERIC = {
    "ok": 0,
    "warning": 1,
    "danger": 2,
}

PHASE_ORDER = {
    "prerecognition": 0,
    "recognition": 1,
    "judgement": 2,
    "action": 3,
    "avoidance": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nakłada poziom ryzyka i fazy VQA na jeden wykres."
    )
    parser.add_argument(
        "--config",
        default=CURRENT_CONFIGURATION,
        help="Ścieżka do pliku YAML z sekcjami 'data_output' i 'vqa'.",
    )
    parser.add_argument(
        "--risk-csv",
        default=None,
        help="Nadpisuje data_output.risk_timeline_csv.",
    )
    parser.add_argument(
        "--phases-csv",
        default=None,
        help="Nadpisuje vqa.vqa_csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Nadpisuje domyślne data_output.plots/{OVERLAY_FILENAME}.",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Przesunięcie dodawane do numerów klatek faz VQA, "
             "jeśli oba pliki liczą klatki od innego punktu.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Pokaż wykres w oknie po zapisaniu.",
    )
    return parser.parse_args()


def load_config_paths(config_path: str) -> tuple[str, str, Path]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_output = config.get("data_output")
    vqa = config.get("vqa")

    if not isinstance(data_output, dict):
        raise ValueError(f"Missing 'data_output' section in {config_path}")
    if not isinstance(vqa, dict):
        raise ValueError(f"Missing 'vqa' section in {config_path}")

    missing = [
        f"data_output.{key}"
        for key in ("risk_timeline_csv", "plots")
        if not data_output.get(key)
    ] + [
        f"vqa.{key}"
        for key in ("vqa_csv",)
        if not vqa.get(key)
    ]

    if missing:
        raise ValueError(
            f"Missing keys in {config_path}: {', '.join(missing)}"
        )

    return (
        data_output["risk_timeline_csv"],
        vqa["vqa_csv"],
        Path(data_output["plots"]) / OVERLAY_FILENAME,
    )


def load_risk(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")

    if "risk_numeric" in df.columns:
        df["risk_numeric"] = pd.to_numeric(df["risk_numeric"], errors="coerce")
    else:
        df["risk_numeric"] = df["risk_level"].map(RISK_TO_NUMERIC)

    return df.dropna(subset=["frame_idx"]).sort_values("frame_idx")


def load_phases(path: str, frame_offset: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce") + frame_offset
    # "none" -> NaN, czyli przerwa na wykresie (tak jak w vqa2csv.py)
    df["phase_number"] = df["phase"].map(PHASE_ORDER)
    return df.dropna(subset=["frame"]).sort_values("frame")


def plot_overlay(risk: pd.DataFrame, phases: pd.DataFrame, output: Path):
    fig, ax_risk = plt.subplots(figsize=(14, 5))
    ax_phase = ax_risk.twinx()

    ax_risk.step(
        risk["frame_idx"],
        risk["risk_numeric"],
        where="post",
        color="tab:red",
        linewidth=1.8,
        label="Poziom ryzyka",
    )
    ax_risk.set_yticks([0, 1, 2])
    ax_risk.set_yticklabels(["OK", "WARNING", "DANGER"])
    ax_risk.set_ylabel("Poziom ryzyka", color="tab:red")
    ax_risk.tick_params(axis="y", colors="tab:red")
    ax_risk.set_ylim(-0.3, 2.3)

    ax_phase.step(
        phases["frame"],
        phases["phase_number"],
        where="post",
        color="tab:blue",
        linewidth=1.8,
        alpha=0.8,
        label="Faza zdarzenia (VQA)",
    )
    ax_phase.set_yticks(list(PHASE_ORDER.values()))
    ax_phase.set_yticklabels(list(PHASE_ORDER.keys()))
    ax_phase.set_ylabel("Faza zdarzenia", color="tab:blue")
    ax_phase.tick_params(axis="y", colors="tab:blue")
    ax_phase.set_ylim(-0.5, 4.5)

    ax_risk.set_xlabel("Numer klatki")
    ax_risk.set_title("Poziom ryzyka i faza zdarzenia w funkcji numeru klatki")
    ax_risk.grid(axis="x", alpha=0.3)

    # Wspólna legenda z obu osi
    h1, l1 = ax_risk.get_legend_handles_labels()
    h2, l2 = ax_phase.get_legend_handles_labels()
    ax_risk.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Zapisano wykres: {output}")
    return fig


def main():
    args = parse_args()

    config_risk_csv, config_phases_csv, config_output = load_config_paths(
        args.config
    )

    risk_csv = args.risk_csv or config_risk_csv
    phases_csv = args.phases_csv or config_phases_csv
    output = Path(args.output) if args.output else config_output

    print(f"Config: {args.config}")
    print(f"Ryzyko: {risk_csv}")
    print(f"Fazy VQA: {phases_csv}")

    risk = load_risk(risk_csv)
    phases = load_phases(phases_csv, args.frame_offset)

    print(
        f"Ryzyko: klatki {int(risk['frame_idx'].min())}–{int(risk['frame_idx'].max())}"
    )
    print(
        f"Fazy VQA: klatki {int(phases['frame'].min())}–{int(phases['frame'].max())}"
    )

    plot_overlay(risk, phases, output)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()