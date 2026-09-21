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


class TrafficSafetyPipeline:
    def __init__(
        self,
        tracker: ObjectTracker,
        homography: HomographyTransformer,
        speed_estimator: SpeedEstimator,
        trajectory_store: TrajectoryStore,
        zone_manager: ZoneManager,
        state_estimator: StateEstimator
    ):
        self.tracker = tracker
        self.homography = homography
        self.speed_estimator = speed_estimator
        self.trajectory_store = trajectory_store
        self.zone_manager = zone_manager
        self.state_estimator = state_estimator

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

            if obj.category == "vehicle":
                speed_kmh = self.speed_estimator.update(
                    track_id=obj.track_id,
                    timestamp=timestamp,
                    world_x=world_x,
                    world_y=world_y,
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