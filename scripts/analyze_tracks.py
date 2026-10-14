import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


from settings import CURRENT_CONFIGURATION


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize traffic safety CSV results."
    )

    parser.add_argument(
        "--config",
        default=CURRENT_CONFIGURATION,
        help="Path to YAML config with a 'data_output' section.",
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="Override: path to tracks CSV (default: data_output.csv).",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override: directory where plots will be saved (default: data_output.plots).",
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
        for key in ("csv", "plots")
        if not data_output.get(key)
    ]

    if missing:
        raise ValueError(
            f"Missing keys in 'data_output' of {config_path}: "
            f"{', '.join(missing)}"
        )

    return (
        data_output["csv"],
        data_output["plots"],
    )


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_columns = [
        "timestamp_s",
        "track_id",
        "class_id",
        "confidence",
        "image_x",
        "image_y",
        "world_x_m",
        "world_y_m",
        "speed_kmh",
        "distance_to_crosswalk_m",
        "ttc_crosswalk_s",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def get_vehicles(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["category"] == "vehicle"
    ].copy()


def plot_vehicle_speed(
    df: pd.DataFrame,
    output_dir: Path,
):
    vehicles = get_vehicles(df)

    vehicles = vehicles.dropna(
        subset=["speed_kmh"]
    )

    if vehicles.empty:
        print("No vehicle speed data found.")
        return

    plt.figure(figsize=(12, 6))

    for track_id, track in vehicles.groupby("track_id"):
        track = track.sort_values(
            "timestamp_s"
        )

        class_name = track[
            "class_name"
        ].iloc[0]

        plt.plot(
            track["timestamp_s"],
            track["speed_kmh"],
            label=f"{class_name} #{int(track_id)}",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Speed [km/h]")
    plt.title("Vehicle speed over time")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    output_path = (
        output_dir
        / "vehicle_speed_over_time.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_vehicle_distance(
    df: pd.DataFrame,
    output_dir: Path,
):
    vehicles = get_vehicles(df)

    vehicles = vehicles.dropna(
        subset=["distance_to_crosswalk_m"]
    )

    if vehicles.empty:
        print(
            "No distance-to-crosswalk data found."
        )
        return

    plt.figure(figsize=(12, 6))

    for track_id, track in vehicles.groupby("track_id"):
        track = track.sort_values(
            "timestamp_s"
        )

        class_name = track[
            "class_name"
        ].iloc[0]

        plt.plot(
            track["timestamp_s"],
            track["distance_to_crosswalk_m"],
            label=f"{class_name} #{int(track_id)}",
        )

    plt.xlabel("Time [s]")
    plt.ylabel(
        "Distance to crosswalk [m]"
    )

    plt.title(
        "Vehicle distance to crosswalk"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    output_path = (
        output_dir
        / "vehicle_distance_to_crosswalk.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_vehicle_ttc(
    df: pd.DataFrame,
    output_dir: Path,
):
    vehicles = get_vehicles(df)

    vehicles = vehicles.dropna(
        subset=["ttc_crosswalk_s"]
    )

    if vehicles.empty:
        print("No TTC data found.")
        return

    plt.figure(figsize=(12, 6))

    for track_id, track in vehicles.groupby("track_id"):
        track = track.sort_values(
            "timestamp_s"
        )

        class_name = track[
            "class_name"
        ].iloc[0]

        plt.plot(
            track["timestamp_s"],
            track["ttc_crosswalk_s"],
            label=f"{class_name} #{int(track_id)}",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("TTC [s]")
    plt.title(
        "Time to crosswalk"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    output_path = (
        output_dir
        / "vehicle_ttc.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_world_trajectories(
    df: pd.DataFrame,
    output_dir: Path,
):
    tracks = df.dropna(
        subset=[
            "world_x_m",
            "world_y_m",
        ]
    )

    if tracks.empty:
        print(
            "No world-coordinate data found."
        )
        return

    plt.figure(figsize=(8, 10))

    for track_id, track in tracks.groupby(
        "track_id"
    ):
        track = track.sort_values(
            "timestamp_s"
        )

        class_name = track[
            "class_name"
        ].iloc[0]

        plt.plot(
            track["world_x_m"],
            track["world_y_m"],
            label=(
                f"{class_name} "
                f"#{int(track_id)}"
            ),
        )

    plt.xlabel("World X [m]")
    plt.ylabel("World Y [m]")

    plt.title(
        "Object trajectories in world coordinates"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    # Useful because this is a spatial plot
    plt.axis("equal")

    plt.tight_layout()

    output_path = (
        output_dir
        / "world_trajectories.png"
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
    print("----- DATA SUMMARY -----")

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Frames: "
        f"{df['frame_idx'].nunique()}"
    )

    print(
        f"Tracks: "
        f"{df['track_id'].nunique()}"
    )

    print()

    print("Objects by category:")

    print(
        df.groupby(
            "category"
        )["track_id"]
        .nunique()
    )

    print()

    if "state" in df.columns:
        print("States:")

        print(
            df["state"]
            .value_counts(
                dropna=False
            )
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

    print_summary(df)

    plot_vehicle_speed(
        df,
        output_dir,
    )

    plot_vehicle_distance(
        df,
        output_dir,
    )

    plot_vehicle_ttc(
        df,
        output_dir,
    )

    plot_world_trajectories(
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