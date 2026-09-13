from pathlib import Path
import subprocess

import cv2
import numpy as np
import yaml

from traffic_safety.detector import ObjectTracker
from traffic_safety.homography import HomographyTransformer
from traffic_safety.pipeline import TrafficSafetyPipeline
from traffic_safety.speed import SpeedEstimator
from traffic_safety.trajectory import TrajectoryStore


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_color(category: str) -> tuple[int, int, int]:
    """
    OpenCV uses BGR colors.
    """

    if category == "pedestrian":
        return (0, 255, 255)

    if category == "bicycle":
        return (255, 255, 0)

    if category == "vehicle":
        return (0, 255, 0)

    return (255, 255, 255)


def draw_trajectory(frame, obj, color):
    """
    Draw the image-space trajectory of a tracked object.
    """

    points = [
        (
            int(point.image_x),
            int(point.image_y),
        )
        for point in obj.trajectory
    ]

    if len(points) < 2:
        return

    for i in range(1, len(points)):
        cv2.line(
            frame,
            points[i - 1],
            points[i],
            color,
            2,
        )


def main():
    config = load_config("configs/extended.yaml")

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    video_path = config["video"]["input"]
    output_path = Path(config["video"]["output"])

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # OpenCV creates a temporary MP4 file.
    # It will later be converted to H.264 with ffmpeg.
    temp_output_path = output_path.with_name(
        output_path.stem + "_temp.mp4"
    )

    tracked_classes = set(
        config["classes"]["tracked"]
    )

    # ---------------------------------------------------------
    # Components
    # ---------------------------------------------------------

    tracker = ObjectTracker(
        model_path=config["model"]["path"],
        tracker=config["model"]["tracker"],
        confidence=config["model"]["confidence"],
        allowed_classes=tracked_classes,
    )

    homography = HomographyTransformer(
        image_points=config["calibration"]["image_points"],
        world_points=config["calibration"]["world_points"],
    )

    speed_estimator = SpeedEstimator(
        history_length=config["tracking"]["history_length"]
    )

    trajectory_store = TrajectoryStore(
        history_length=config["tracking"]["trajectory_length"]
    )

    pipeline = TrafficSafetyPipeline(
        tracker=tracker,
        homography=homography,
        speed_estimator=speed_estimator,
        trajectory_store=trajectory_store,
    )

    # ---------------------------------------------------------
    # Video
    # ---------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError(
            "Could not determine video FPS."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(
        f"Video: {width}x{height} @ {fps:.2f} FPS"
    )

    # ---------------------------------------------------------
    # Video writer
    # ---------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(temp_output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video: "
            f"{temp_output_path}"
        )

    # ---------------------------------------------------------
    # Calibration polygon
    # ---------------------------------------------------------

    calibration_polygon = np.array(
        config["calibration"]["image_points"],
        dtype=np.int32,
    )

    frame_idx = 0

    # ---------------------------------------------------------
    # Processing loop
    # ---------------------------------------------------------

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            timestamp = frame_idx / fps
            frame_idx += 1

            objects = pipeline.process_frame(
                frame,
                timestamp,
            )

            # -------------------------------------------------
            # Visualization
            # -------------------------------------------------

            for obj in objects:
                x1, y1, x2, y2 = obj.bbox

                color = get_color(
                    obj.category
                )

                # ---------------------------------------------
                # Trajectory
                # ---------------------------------------------

                if config["visualization"].get(
                    "draw_trajectory",
                    True,
                ):
                    draw_trajectory(
                        frame,
                        obj,
                        color,
                    )

                # ---------------------------------------------
                # Bounding box
                # ---------------------------------------------

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,
                )

                # ---------------------------------------------
                # Contact point
                # ---------------------------------------------

                if config["visualization"].get(
                    "draw_contact_point",
                    True,
                ):
                    cv2.circle(
                        frame,
                        (
                            int(obj.image_x),
                            int(obj.image_y),
                        ),
                        4,
                        (0, 0, 255),
                        -1,
                    )

                # ---------------------------------------------
                # Label
                # ---------------------------------------------

                if obj.speed_kmh is not None:
                    label = (
                        f"{obj.class_name} "
                        f"#{obj.track_id} | "
                        f"{obj.speed_kmh:.1f} km/h"
                    )
                else:
                    label = (
                        f"{obj.class_name} "
                        f"#{obj.track_id}"
                    )

                cv2.putText(
                    frame,
                    label,
                    (
                        int(x1),
                        max(
                            int(y1) - 10,
                            20,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # -------------------------------------------------
            # Object counters
            # -------------------------------------------------

            vehicle_count = sum(
                obj.category == "vehicle"
                for obj in objects
            )

            pedestrian_count = sum(
                obj.category == "pedestrian"
                for obj in objects
            )

            bicycle_count = sum(
                obj.category == "bicycle"
                for obj in objects
            )

            status_text = (
                f"Vehicles: {vehicle_count} | "
                f"Pedestrians: {pedestrian_count} | "
                f"Bicycles: {bicycle_count}"
            )

            cv2.putText(
                frame,
                status_text,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

            # -------------------------------------------------
            # Calibration area
            # -------------------------------------------------

            if config["visualization"].get(
                "draw_calibration_area",
                True,
            ):
                cv2.polylines(
                    frame,
                    [calibration_polygon],
                    isClosed=True,
                    color=(255, 0, 0),
                    thickness=2,
                )

            # -------------------------------------------------
            # Write frame
            # -------------------------------------------------

            writer.write(frame)

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # Convert MP4V -> H.264
    # ---------------------------------------------------------

    print("Converting video to H.264...")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_output_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )

    # Remove temporary OpenCV file
    temp_output_path.unlink()

    print(
        f"Saved processed video to: {output_path}"
    )


if __name__ == "__main__":
    main()