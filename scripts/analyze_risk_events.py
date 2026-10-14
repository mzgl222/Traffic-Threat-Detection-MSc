import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from settings import CURRENT_CONFIGURATION

RISK_NUMERIC = {
    "safe": 0,
    "warning": 1,
    "danger": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize traffic safety risk events."
        )
    )

    parser.add_argument(
        "--config",
        # required=True,
        default=CURRENT_CONFIGURATION,
        help="Path to YAML config with a 'data_output' section.",
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="Override: path to risk events CSV (default: data_output.risk_csv).",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override: directory for generated plots (default: data_output.plots).",
    )

    return parser.parse_args()


def load_config_paths(
    config_path: str,
) -> tuple[str, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_output = config.get("data_output")

    if not isinstance(data_output, dict):
        raise ValueError(
            f"Missing 'data_output' section in {config_path}"
        )

    missing = [
        key
        for key in ("risk_csv", "plots")
        if not data_output.get(key)
    ]

    if missing:
        raise ValueError(
            f"Missing keys in 'data_output' of {config_path}: "
            f"{', '.join(missing)}"
        )

    return (
        data_output["risk_csv"],
        data_output["plots"],
    )


def load_data(
    path: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_columns = [
        "frame_idx",
        "timestamp_s",
        "pedestrian_id",
        "vehicle_id",
        "ttc_s",
        "distance_m",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["risk_numeric"] = (
        df["risk_level"]
        .map(RISK_NUMERIC)
    )

    return df


def plot_risk_over_time(
    df: pd.DataFrame,
    output_dir: Path,
):
    if df.empty:
        print(
            "No risk events found."
        )
        return

    plt.figure(
        figsize=(12, 5)
    )

    for (
        pedestrian_id,
        vehicle_id,
    ), pair in df.groupby(
        [
            "pedestrian_id",
            "vehicle_id",
        ]
    ):
        pair = pair.sort_values(
            "timestamp_s"
        )

        plt.step(
            pair["timestamp_s"],
            pair["risk_numeric"],
            where="post",
            label=(
                f"P#{int(pedestrian_id)} "
                f"- V#{int(vehicle_id)}"
            ),
        )

    plt.yticks(
        [0, 1, 2],
        [
            "SAFE",
            "WARNING",
            "DANGER",
        ],
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Risk level"
    )

    plt.title(
        "Risk level over time"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        output_dir
        / "risk_over_time.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_ttc_over_time(
    df: pd.DataFrame,
    output_dir: Path,
):
    data = df.dropna(
        subset=["ttc_s"]
    )

    if data.empty:
        print(
            "No TTC data found in risk events."
        )
        return

    plt.figure(
        figsize=(12, 5)
    )

    for (
        pedestrian_id,
        vehicle_id,
    ), pair in data.groupby(
        [
            "pedestrian_id",
            "vehicle_id",
        ]
    ):
        pair = pair.sort_values(
            "timestamp_s"
        )

        plt.plot(
            pair["timestamp_s"],
            pair["ttc_s"],
            label=(
                f"P#{int(pedestrian_id)} "
                f"- V#{int(vehicle_id)}"
            ),
        )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "TTC [s]"
    )

    plt.title(
        "TTC during risk events"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        output_dir
        / "risk_ttc_over_time.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_rule_counts(
    df: pd.DataFrame,
    output_dir: Path,
):
    if df.empty:
        return

    counts = (
        df["rule"]
        .value_counts()
        .sort_values(
            ascending=True
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    counts.plot(
        kind="barh"
    )

    plt.xlabel(
        "Number of frames"
    )

    plt.ylabel(
        "Rule"
    )

    plt.title(
        "Risk rule occurrences"
    )

    plt.tight_layout()

    output_path = (
        output_dir
        / "risk_rule_counts.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def print_summary(
    df: pd.DataFrame,
):
    print()
    print(
        "----- RISK SUMMARY -----"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Unique pedestrian-vehicle pairs: "
        f"{df[['pedestrian_id', 'vehicle_id']].drop_duplicates().shape[0]}"
    )

    print()

    print(
        "Risk levels:"
    )

    print(
        df["risk_level"]
        .value_counts()
    )

    print()

    print(
        "Rules:"
    )

    print(
        df["rule"]
        .value_counts()
    )

    print(
        "------------------------"
    )

    print()


def main():
    args = parse_args()

    config_csv, config_output_dir = load_config_paths(
        args.config
    )

    csv_path = args.csv or config_csv

    output_dir = Path(
        args.output_dir or config_output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Config: {args.config}"
    )

    print(
        f"Loading: {csv_path}"
    )

    df = load_data(
        csv_path
    )

    print_summary(
        df
    )

    plot_risk_over_time(
        df,
        output_dir,
    )

    plot_ttc_over_time(
        df,
        output_dir,
    )

    plot_rule_counts(
        df,
        output_dir,
    )

    print()

    print(
        f"Analysis finished. "
        f"Plots saved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()