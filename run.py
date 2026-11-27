from pathlib import Path
import argparse
import subprocess
import logging
from tqdm import tqdm
import cv2
import numpy as np
import yaml

from traffic_safety.detector import ObjectTracker
from traffic_safety.homography import HomographyTransformer
from traffic_safety.pipeline import TrafficSafetyPipeline
from traffic_safety.speed import METHODS as SPEED_METHODS, resolve_speed_settings, speed_estimator_from_config
from traffic_safety.trajectory import TrajectoryStore
from traffic_safety.zones import ZoneManager
from traffic_safety.state import StateEstimator
from traffic_safety.risk_metrics import RiskMetricsEstimator
from traffic_safety.csv_writer import CsvResultWriter
from traffic_safety.risk_event_writer import RiskEventCsvWriter
from traffic_safety.risk_timeline_writer import RiskTimelineCsvWriter
from traffic_safety.interaction_writer import InteractionCsvWriter
from traffic_safety.binary_timeline_writer import BinaryTimelineCsvWriter
from traffic_safety.binary_rules import DANGER
from traffic_safety.rule_engines import (
    BINARY,
    ENGINE_NAMES,
    LEGACY,
    binary_events_as_rule_results,
    create_rule_engine,
    resolve_engine_name,
)
from traffic_safety.output_paths import (
    binary_timeline_path,
    ensure_no_existing_outputs,
    output_files,
    redirect_outputs,
)


from scripts.settings import CURRENT_CONFIGURATION


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

def get_frame_risk_level(risk_events) -> str:
    if any(
        event.level.value == "danger"
        for event in risk_events
    ):
        return "danger"

    if any(
        event.level.value == "warning"
        for event in risk_events
    ):
        return "warning"

    return "ok"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Traffic safety pipeline (detection, tracking, rules, CSV export)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Scenario YAML (default: CURRENT_CONFIGURATION from scripts/settings.py).",
    )
    parser.add_argument(
        "--rule-engine",
        choices=ENGINE_NAMES,
        default=None,
        help="Rule engine: 'legacy' (SAFE/WARNING/DANGER) or 'binary' (OK/DANGER). "
             "Default: config rules.engine, otherwise 'legacy'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write all outputs under this directory instead of the paths in the YAML.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files (binary mode / --output-dir).",
    )
    parser.add_argument(
        "--binary-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the binary OK/DANGER decision on the output video (binary mode only).",
    )
    parser.add_argument(
        "--speed-method",
        choices=SPEED_METHODS,
        default=None,
        help="Vehicle speed estimation: 'median_step' (original) or 'regression' "
             "(least squares over a causal time window). "
             "Default: config tracking.speed_method, otherwise 'median_step'.",
    )
    parser.add_argument(
        "--speed-window-s",
        type=float,
        default=None,
        help="Time window [s] for --speed-method regression "
             "(default: config tracking.speed_window_s, otherwise 0.5).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames (smoke tests).",
    )
    return parser.parse_args(argv)


def draw_binary_decision(frame, risk_binary: int, events) -> None:
    if risk_binary == DANGER:
        rules = sorted({event.rule.value for event in events})
        text = "DANGER | " + ", ".join(rules)
        color = (0, 0, 255)
    else:
        text = "OK"
        color = (0, 200, 0)

    cv2.putText(
        frame,
        text,
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )


def main(argv=None):
    args = parse_args(argv)

    logger.info("Starting traffic safety pipeline")

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    logger.info("Loading configuration...")

    config_path = args.config or CURRENT_CONFIGURATION
    config = load_config(config_path)

    logger.info(f"Configuration loaded: {config_path}")

    engine_name = resolve_engine_name(args.rule_engine, config)

    if args.output_dir is not None:
        config = redirect_outputs(config, args.output_dir)
        logger.info(f"Outputs redirected to: {args.output_dir}")

    # Legacy mode without --output-dir keeps the original behaviour
    # (outputs are overwritten). Otherwise refuse to overwrite.
    if (engine_name == BINARY or args.output_dir is not None) and not args.overwrite:
        ensure_no_existing_outputs(output_files(config, engine_name))

    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    logger.info(
        f"Creating rule engine: {engine_name}"
    )

    rule_engine = create_rule_engine(engine_name, config)

    logger.info(
        "Rule engine ready"
    )

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

    speed_settings = resolve_speed_settings(
        config["tracking"], args.speed_method, args.speed_window_s
    )
    speed_estimator = speed_estimator_from_config(
        config["tracking"], args.speed_method, args.speed_window_s
    )

    logger.info(
        f"Speed estimator ready (method={speed_settings['method']}, "
        f"window_s={speed_settings['window_s']})"
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
    logger.info(
    "Creating state estimator..."
    )

    state_estimator = StateEstimator(
        stopped_speed_threshold=2.0,
        slowing_threshold=-2.0,
        state_confirm_frames=3,
        stopped_confirm_frames=5,
        speed_history_length=6,
    )

    logger.info(
        "State estimator ready"
    )

    crosswalk_reference_line_image = [
        tuple(point)
        for point in config[
            "crosswalk"
        ]["reference_line_image"]
    ]

    logger.info(
        "Creating risk metrics estimator..."
    )

    risk_metrics_estimator = RiskMetricsEstimator(
        history_window_s=0.7,
        min_samples=5,
        min_closing_speed_m_s=0.5,
    )

    logger.info(
        "Risk metrics estimator ready"
    )

    pipeline = TrafficSafetyPipeline(
        tracker=tracker,
        homography=homography,
        speed_estimator=speed_estimator,
        trajectory_store=trajectory_store,
        zone_manager=zone_manager,
        state_estimator=state_estimator,
        crosswalk_reference_line_image=crosswalk_reference_line_image,
        risk_metrics_estimator=risk_metrics_estimator,

    )

    logger.info(
        "Pipeline initialized successfully"
    )

    logger.info(
        "Creating CSV result writer..."
    )

    csv_writer = CsvResultWriter(
        config["data_output"]["csv"]
    )

    logger.info(
        "CSV result writer ready"
    )

    risk_event_writer = None
    risk_timeline_writer = None
    binary_timeline_writer = None

    if engine_name == LEGACY:
        logger.info(
            "Creating risk event CSV writer..."
        )

        risk_event_writer = RiskEventCsvWriter(
            config["data_output"]["risk_csv"]
        )

        logger.info(
            "Risk event CSV writer ready"
        )

        logger.info(
            "Creating risk timeline CSV writer..."
        )

        risk_timeline_writer = RiskTimelineCsvWriter(
            config["data_output"]["risk_timeline_csv"]
        )

        logger.info(
            "Risk timeline CSV writer ready"
        )
    else:
        binary_csv = binary_timeline_path(config)

        binary_timeline_writer = BinaryTimelineCsvWriter(
            binary_csv
        )

        logger.info(
            f"Binary timeline CSV writer ready: {binary_csv}"
        )

    study_metadata = config["study_metadata"]

    interaction_writer = InteractionCsvWriter(
        output_path=config["data_output"]["interactions_csv"],
        frame_metadata_path=config["data_output"][
            "interaction_frames_csv"
        ],
        video_id=study_metadata["video_id"],
        scene_id=study_metadata["scene_id"],
        source_event_id=study_metadata["source_event_id"],
        camera_id=study_metadata["camera_id"],
    )

    logger.info(
        "Interaction CSV writer ready"
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
            total=(
                min(total_frames, args.max_frames)
                if args.max_frames is not None
                else total_frames
            ),
            desc="Processing video",
            unit="frame",
            dynamic_ncols=True,
        ) as progress_bar:

            while True:
                if (
                    args.max_frames is not None
                    and frame_idx >= args.max_frames
                ):
                    break

                ret, frame = cap.read()

                if not ret:
                    break

                timestamp = (
                    frame_idx / fps
                )

                

                # ---------------------------------------------
                # Pipeline
                # ---------------------------------------------

                objects = (
                    pipeline.process_frame(
                        frame,
                        timestamp,
                    )
                )
                if engine_name == LEGACY:
                    risk_events = rule_engine.evaluate(
                        objects
                    )
                    interaction_rule_results = risk_events
                else:
                    binary_events = rule_engine.evaluate(
                        objects
                    )
                    risk_binary = rule_engine.frame_label(
                        binary_events
                    )
                    interaction_rule_results = (
                        binary_events_as_rule_results(binary_events)
                    )
                    # Legacy event overlay below is not used in binary mode.
                    risk_events = []

                interaction_writer.write_frame(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp,
                    objects=objects,
                    risk_events=interaction_rule_results,
                    pipeline_ok=True,
                )
                csv_writer.write_frame(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    objects=objects,
                )

                if engine_name == LEGACY:
                    risk_event_writer.write_frame(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        events=risk_events,
                    )
                    frame_risk_level = get_frame_risk_level(
                        risk_events
                    )

                    risk_timeline_writer.write_frame(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        risk_level=frame_risk_level,
                        risk_events=risk_events,
                    )
                else:
                    binary_timeline_writer.write_frame(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        risk_binary=risk_binary,
                        events=binary_events,
                    )

                frame_idx += 1

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

                    state_text = (
                        obj.state
                        if obj.state is not None
                        else "unknown"
                    )
                    parts = [
                        f"{obj.class_name} #{obj.track_id}",
                        zone_text,
                        state_text,
                    ]

                    if obj.category == "vehicle":
                        if obj.speed_kmh is not None:
                            parts.append(
                                f"{obj.speed_kmh:.1f} km/h"
                            )

                        if obj.distance_to_crosswalk_m is not None:
                            parts.append(
                                f"{obj.distance_to_crosswalk_m:.1f} m"
                            )

                        if obj.ttc_crosswalk_s is not None:
                            parts.append(
                                f"TTC {obj.ttc_crosswalk_s:.1f} s"
                            )

                    label = " | ".join(parts)

                    # if (
                    #     obj.speed_kmh
                    #     is not None
                    # ):
                    #     label = (
                    #         f"{obj.class_name} "
                    #         f"#{obj.track_id} | "
                    #         f"{obj.speed_kmh:.1f} "
                    #         f"km/h | "
                    #         f"{zone_text} | "
                    #         f"{state_text}"
                    #     )

                    # else:
                    #     label = (
                    #         f"{obj.class_name} "
                    #         f"#{obj.track_id} | "
                    #         f"{zone_text} | "
                    #         f"{state_text}"
                    #     )
                    

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

                for event in risk_events:
                    text = (
                        f"{event.level.value.upper()} | "
                        f"person #{event.pedestrian_id} "
                        f"<-> car #{event.vehicle_id}"
                    )

                    cv2.putText(
                        frame,
                        text,
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (
                            (0, 0, 255)
                            if event.level.value == "danger"
                            else (0, 165, 255)
                        ),
                        2,
                    )

                if engine_name == BINARY and args.binary_overlay:
                    draw_binary_decision(
                        frame,
                        risk_binary,
                        binary_events,
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
        csv_writer.close()
        for optional_writer in (
            risk_event_writer,
            risk_timeline_writer,
            binary_timeline_writer,
        ):
            if optional_writer is not None:
                optional_writer.close()
        interaction_writer.close()
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