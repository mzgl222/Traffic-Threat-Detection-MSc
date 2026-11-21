"""Output path helpers for run.py (redirecting outputs, overwrite guard)."""

from __future__ import annotations

import copy
from pathlib import Path


BINARY_TIMELINE_KEY = "binary_timeline_csv"
BINARY_TIMELINE_DEFAULT_NAME = "scenario_binary_timeline.csv"


def binary_timeline_path(config: dict) -> str:
    """data_output.binary_timeline_csv, or scenario_binary_timeline.csv next to data_output.csv."""
    data_output = config["data_output"]
    explicit = data_output.get(BINARY_TIMELINE_KEY)
    if explicit:
        return str(explicit)
    return str(Path(data_output["csv"]).with_name(BINARY_TIMELINE_DEFAULT_NAME))


def redirect_outputs(config: dict, output_dir: str | Path) -> dict:
    """Return a copy of config with video.output and all data_output paths
    moved under output_dir.

    The structure relative to the scene directory (parent of video.output) is
    kept, e.g. outputs/<scene>/csv/scenario_tracks.csv -> <output_dir>/csv/scenario_tracks.csv.
    Paths outside the scene directory keep only their file name.
    The input config is not modified.
    """
    result = copy.deepcopy(config)
    output_dir = Path(output_dir)
    scene_dir = Path(config["video"]["output"]).parent

    def move(path: str) -> str:
        path = Path(path)
        try:
            relative = path.relative_to(scene_dir)
        except ValueError:
            relative = Path(path.name)
        return str(output_dir / relative)

    result["video"]["output"] = move(config["video"]["output"])

    data_output = result["data_output"]
    data_output[BINARY_TIMELINE_KEY] = binary_timeline_path(config)
    for key, value in list(data_output.items()):
        if isinstance(value, str) and value:
            data_output[key] = move(value)

    return result


def output_files(config: dict, engine_name: str) -> list[Path]:
    """Files run.py writes for the given engine."""
    data_output = config["data_output"]
    video_output = Path(config["video"]["output"])
    paths = [
        video_output,
        video_output.with_name(video_output.stem + "_temp.mp4"),
        Path(data_output["csv"]),
        Path(data_output["interactions_csv"]),
        Path(data_output["interaction_frames_csv"]),
    ]
    if engine_name == "legacy":
        paths += [
            Path(data_output["risk_csv"]),
            Path(data_output["risk_timeline_csv"]),
        ]
    else:
        paths.append(Path(binary_timeline_path(config)))
    return paths


def ensure_no_existing_outputs(paths) -> None:
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs "
            "(use --overwrite or choose another --output-dir):\n  "
            + "\n  ".join(existing)
        )
