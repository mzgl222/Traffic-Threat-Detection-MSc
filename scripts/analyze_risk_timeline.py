import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV = "outputs/scenario_risk_timeline.csv"

RISK_TO_NUMERIC = {
    "ok": 0,
    "warning": 1,
    "danger": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize frame-level risk timeline."
    )

    parser.add_argument(
        "csv",
        nargs="?",
        default=DEFAULT_CSV,
        help="Path to risk timeline CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/risk_timeline_analysis",
        help="Directory for generated plots.",
    )

    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_columns = [
        "frame_idx",
        "timestamp_s",
        "risk_numeric",
        "num_risk_events",
        "num_warning_events",
        "num_danger_events",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    if "risk_numeric" not in df.columns:
        df["risk_numeric"] = (
            df["risk_level"]
            .map(RISK_TO_NUMERIC)
        )

    return df


def plot_risk_timeline(
    df: pd.DataFrame,
    output_dir: Path,
):
    if df.empty:
        print("No data found.")
        return

    plt.figure(figsize=(14, 5))

    plt.step(
        df["frame_idx"],
        df["risk_numeric"],
        where="post",
    )

    plt.yticks(
        [0, 1, 2],
        ["OK", "WARNING", "DANGER"],
    )

    plt.xlabel("Frame index")
    plt.ylabel("Risk level")
    plt.title("Risk level for each frame")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = (
        output_dir / "risk_timeline.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_event_counts(
    df: pd.DataFrame,
    output_dir: Path,
):
    if df.empty:
        return

    plt.figure(figsize=(14, 5))

    plt.plot(
        df["frame_idx"],
        df["num_risk_events"],
        label="All risk events",
    )

    plt.plot(
        df["frame_idx"],
        df["num_warning_events"],
        label="Warning events",
    )

    plt.plot(
        df["frame_idx"],
        df["num_danger_events"],
        label="Danger events",
    )

    plt.xlabel("Frame index")
    plt.ylabel("Number of events")
    plt.title("Number of risk events per frame")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    output_path = (
        output_dir / "risk_event_counts_per_frame.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}")


def print_summary(df: pd.DataFrame):
    print()
    print("----- RISK TIMELINE SUMMARY -----")

    print(f"Frames: {len(df)}")

    print()
    print("Frame counts by risk level:")
    print(
        df["risk_level"]
        .value_counts()
        .reindex(["ok", "warning", "danger"], fill_value=0)
    )

    print()
    print("Max number of risk events in a frame:")
    print(df["num_risk_events"].max())

    print("---------------------------------")
    print()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading: {args.csv}")

    df = load_data(args.csv)
    df = df.sort_values("frame_idx")

    print_summary(df)

    plot_risk_timeline(df, output_dir)
    plot_event_counts(df, output_dir)

    print()
    print(
        f"Analysis finished. Plots saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()