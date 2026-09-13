from pathlib import Path
import subprocess
import logging
from tqdm import tqdm
import cv2
import numpy as np
import yaml

from traffic_safety.detector import ObjectTracker
from traffic_safety.homography import HomographyTransformer
from traffic_safety.pipeline import TrafficSafetyPipeline
from traffic_safety.speed import SpeedEstimator
from traffic_safety.trajectory import TrajectoryStore
from traffic_safety.zones import ZoneManager

CURRENT_CONFIGURATION = "configs/wts.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

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
    logger.info("Starting traffic safety pipeline")

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    logger.info("Loading configuration...")

    config = load_config(CURRENT_CONFIGURATION)

    logger.info("Configuration loaded")

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    video_path = config["video"]["input"]
    output_path = Path(
        config["video"]["output"]
    )

    logger.info(
        f"Input video: {video_path}"
    )

    logger.info(
        f"Output video: {output_path}"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_output_path = output_path.with_name(
        output_path.stem + "_temp.mp4"
    )

    tracked_classes = set(
        config["classes"]["tracked"]
    )

    logger.info(
        f"Tracked classes: "
        f"{sorted(tracked_classes)}"
    )

    # ---------------------------------------------------------
    # Initialize components
    # ---------------------------------------------------------

    logger.info(
        "Loading object detector and tracker..."
    )

    tracker = ObjectTracker(
        model_path=config["model"]["path"],
        tracker=config["model"]["tracker"],
        confidence=config["model"]["confidence"],
        allowed_classes=tracked_classes,
    )

    logger.info(
        "Object detector and tracker ready"
    )

    logger.info(
        "Creating homography transformer..."
    )

    homography = HomographyTransformer(
        image_points=config["calibration"]["image_points"],
        world_points=config["calibration"]["world_points"],
    )

    logger.info(
        "Homography transformer ready"
    )

    logger.info(
        "Creating speed estimator..."
    )

    speed_estimator = SpeedEstimator(
        history_length=config["tracking"]["history_length"]
    )

    logger.info(
        "Speed estimator ready"
    )

    logger.info(
        "Creating trajectory store..."
    )

    trajectory_store = TrajectoryStore(
        history_length=config["tracking"]["trajectory_length"]
    )

    logger.info(
        "Trajectory store ready"
    )

    logger.info("Creating zone manager...")

    zone_manager = ZoneManager(
        config["zones"]
    )

    logger.info("Zone manager ready")

    logger.info(
        "Creating processing pipeline..."
    )

    pipeline = TrafficSafetyPipeline(
        tracker=tracker,
        homography=homography,
        speed_estimator=speed_estimator,
        trajectory_store=trajectory_store,
        zone_manager=zone_manager,
    )

    logger.info(
        "Pipeline initialized successfully"
    )

    # ---------------------------------------------------------
    # Open input video
    # ---------------------------------------------------------

    logger.info(
        "Opening input video..."
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        cap.release()

        raise RuntimeError(
            "Could not determine video FPS"
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration_seconds = (
        total_frames / fps
        if fps > 0
        else 0
    )

    logger.info(
        "Video opened successfully"
    )

    logger.info(
        f"Resolution: {width}x{height}"
    )

    logger.info(
        f"FPS: {fps:.2f}"
    )

    logger.info(
        f"Frames: {total_frames}"
    )

    logger.info(
        f"Duration: "
        f"{duration_seconds:.1f} s"
    )

    # ---------------------------------------------------------
    # Initialize video writer
    # ---------------------------------------------------------

    logger.info(
        "Creating output video writer..."
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

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

    logger.info(
        "Output video writer ready"
    )

    # ---------------------------------------------------------
    # Calibration polygon
    # ---------------------------------------------------------

    calibration_polygon = np.array(
        config["calibration"]["image_points"],
        dtype=np.int32,
    )

    frame_idx = 0

    logger.info(
        "Starting frame processing..."
    )

    # ---------------------------------------------------------
    # Processing loop
    # ---------------------------------------------------------

    try:
        with tqdm(
            total=total_frames,
            desc="Processing video",
            unit="frame",
            dynamic_ncols=True,
        ) as progress_bar:

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                timestamp = (
                    frame_idx / fps
                )

                frame_idx += 1

                # ---------------------------------------------
                # Pipeline
                # ---------------------------------------------

                objects = (
                    pipeline.process_frame(
                        frame,
                        timestamp,
                    )
                )

                # ---------------------------------------------
                # Visualization
                # ---------------------------------------------

                for zone in zone_manager.get_all():
                    cv2.polylines(
                        frame,
                        [zone.polygon],
                        isClosed=True,
                        color=(255, 0, 255),
                        thickness=2,
                    )
                

                for obj in objects:
                    x1, y1, x2, y2 = (
                        obj.bbox
                    )

                    color = get_color(
                        obj.category
                    )

                    # Trajectory
                    if (
                        config[
                            "visualization"
                        ].get(
                            "draw_trajectory",
                            True,
                        )
                    ):
                        draw_trajectory(
                            frame,
                            obj,
                            color,
                        )

                    # Bounding box
                    cv2.rectangle(
                        frame,
                        (
                            int(x1),
                            int(y1),
                        ),
                        (
                            int(x2),
                            int(y2),
                        ),
                        color,
                        2,
                    )

                    # Contact point
                    if (
                        config[
                            "visualization"
                        ].get(
                            "draw_contact_point",
                            True,
                        )
                    ):
                        cv2.circle(
                            frame,
                            (
                                int(
                                    obj.image_x
                                ),
                                int(
                                    obj.image_y
                                ),
                            ),
                            4,
                            (0, 0, 255),
                            -1,
                        )

                    # Label
                    zone_text = (
                        obj.zone
                        if obj.zone is not None
                        else "outside"
                    )
                    if (
                        obj.speed_kmh
                        is not None
                    ):
                        label = (
                            f"{obj.class_name} "
                            f"#{obj.track_id} | "
                            f"{obj.speed_kmh:.1f} "
                            f"km/h | "
                            f"{zone_text}"
                        )

                    else:
                        label = (
                            f"{obj.class_name} "
                            f"#{obj.track_id} | "
                            f"{zone_text}"
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

                # ---------------------------------------------
                # Counters
                # ---------------------------------------------

                vehicle_count = sum(
                    obj.category
                    == "vehicle"
                    for obj
                    in objects
                )

                pedestrian_count = sum(
                    obj.category
                    == "pedestrian"
                    for obj
                    in objects
                )

                bicycle_count = sum(
                    obj.category
                    == "bicycle"
                    for obj
                    in objects
                )

                status_text = (
                    f"Vehicles: "
                    f"{vehicle_count} | "
                    f"Pedestrians: "
                    f"{pedestrian_count} | "
                    f"Bicycles: "
                    f"{bicycle_count}"
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

                # ---------------------------------------------
                # Calibration area
                # ---------------------------------------------

                if (
                    config[
                        "visualization"
                    ].get(
                        "draw_calibration_area",
                        True,
                    )
                ):
                    cv2.polylines(
                        frame,
                        [
                            calibration_polygon
                        ],
                        isClosed=True,
                        color=(255, 0, 0),
                        thickness=2,
                    )

                # ---------------------------------------------
                # Write output frame
                # ---------------------------------------------

                writer.write(
                    frame
                )

                progress_bar.update(1)

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    logger.info(
        "Frame processing finished"
    )

    logger.info(
        f"Processed "
        f"{frame_idx} frames"
    )

    # ---------------------------------------------------------
    # Convert MP4V -> H.264
    # ---------------------------------------------------------

    logger.info(
        "Converting video to H.264..."
    )

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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    logger.info(
        "H.264 conversion finished"
    )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    if temp_output_path.exists():
        temp_output_path.unlink()

        logger.info(
            "Temporary video removed"
        )

    logger.info(
        f"Finished successfully. "
        f"Output saved to: "
        f"{output_path}"
    )

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        logger.warning(
            "Processing interrupted by user"
        )

    except Exception:
        logger.exception(
            "Pipeline failed"
        )

        raise