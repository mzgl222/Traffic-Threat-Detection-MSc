from dataclasses import dataclass

from traffic_safety.detector import ObjectTracker
from traffic_safety.homography import HomographyTransformer
from traffic_safety.speed import SpeedEstimator
from traffic_safety.trajectory import (
    TrajectoryPoint,
    TrajectoryStore,
)
from traffic_safety.zones import ZoneManager
from traffic_safety.state import StateEstimator
from traffic_safety.risk_metrics import RiskMetricsEstimator, distance_to_line

@dataclass
class ProcessedObject:
    track_id: int

    class_id: int
    class_name: str
    category: str

    confidence: float

    bbox: tuple[float, float, float, float]

    image_x: float
    image_y: float

    world_x: float
    world_y: float

    speed_kmh: float | None

    trajectory: list[TrajectoryPoint]

    zone: str | None
    state: str | None
    distance_to_crosswalk_m: float | None
    ttc_crosswalk_s: float | None


class TrafficSafetyPipeline:
    def __init__(
        self,
        tracker: ObjectTracker,
        homography: HomographyTransformer,
        speed_estimator: SpeedEstimator,
        trajectory_store: TrajectoryStore,
        zone_manager: ZoneManager,
        state_estimator: StateEstimator,
        crosswalk_reference_line_image,
        risk_metrics_estimator: RiskMetricsEstimator,
        
        
    ):
        self.tracker = tracker
        self.homography = homography
        self.speed_estimator = speed_estimator
        self.trajectory_store = trajectory_store
        self.zone_manager = zone_manager
        self.state_estimator = state_estimator
        self.crosswalk_reference_line_image = crosswalk_reference_line_image
        self.risk_metrics_estimator = risk_metrics_estimator
        start_image, end_image = (
            self.crosswalk_reference_line_image
        )

        start_world = self.homography.image_to_world(
            start_image[0],
            start_image[1],
        )

        end_world = self.homography.image_to_world(
            end_image[0],
            end_image[1],
        )

        self.crosswalk_reference_line_world = (
            start_world,
            end_world,
        )

    def process_frame(
        self,
        frame,
        timestamp: float,
    ) -> list[ProcessedObject]:

        tracked_objects = self.tracker.process(frame)

        processed_objects = []

        for obj in tracked_objects:
            # ---------------------------------------------
            # Contact point
            # ---------------------------------------------

            image_x, image_y = obj.contact_point

            # ---------------------------------------------
            # Homography
            # ---------------------------------------------

            world_x, world_y = self.homography.image_to_world(
                image_x,
                image_y,
            )

            # ---------------------------------------------
            # Zone
            # ---------------------------------------------

            zone = self.zone_manager.get_zone(
                image_x,
                image_y,
            )

            # ---------------------------------------------
            # Trajectory
            # ---------------------------------------------

            self.trajectory_store.update(
                track_id=obj.track_id,
                timestamp=timestamp,
                image_x=image_x,
                image_y=image_y,
                world_x=world_x,
                world_y=world_y,
            )

            trajectory = self.trajectory_store.get(
                obj.track_id
            )

            # ---------------------------------------------
            # Speed
            # ---------------------------------------------

            speed_kmh = None
            distance_to_crosswalk_m = None
            ttc_crosswalk_s = None

            if obj.category == "vehicle":
                speed_kmh = self.speed_estimator.update(
                    track_id=obj.track_id,
                    timestamp=timestamp,
                    world_x=world_x,
                    world_y=world_y,
                )
                if zone == "vehicle_approach":

                    line_start, line_end = self.crosswalk_reference_line_world
                    distance_to_crosswalk_m = distance_to_line(world_x, world_y,line_start, line_end)
                    ttc_crosswalk_s = self.risk_metrics_estimator.update_ttc(
                        track_id=obj.track_id,
                        timestamp=timestamp,
                        distance_to_crosswalk_m=distance_to_crosswalk_m,
                    )


            # ---------------------------------------------
            # Create processed object
            # ---------------------------------------------

            processed_obj = ProcessedObject(
                track_id=obj.track_id,
                class_id=obj.class_id,
                class_name=obj.class_name,
                category=obj.category,
                confidence=obj.confidence,
                bbox=obj.bbox,
                image_x=image_x,
                image_y=image_y,
                world_x=world_x,
                world_y=world_y,
                speed_kmh=speed_kmh,
                trajectory=trajectory,
                zone=zone,
                state=None,
                distance_to_crosswalk_m=distance_to_crosswalk_m,
                ttc_crosswalk_s=ttc_crosswalk_s,
            )

            # ---------------------------------------------
            # State estimation
            # ---------------------------------------------

            processed_obj.state = (
                self.state_estimator.update(
                    processed_obj
                )
            )

            # ---------------------------------------------
            # Add result
            # ---------------------------------------------

            processed_objects.append(
                processed_obj
            )

        return processed_objects