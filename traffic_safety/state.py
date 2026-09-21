from collections import defaultdict, deque
from enum import Enum


class PedestrianState(str, Enum):
    OUTSIDE = "outside"
    WAITING = "waiting"
    ENTERING = "entering"
    CROSSING = "crossing"
    LEAVING = "leaving"


class VehicleState(str, Enum):
    OUTSIDE = "outside"
    APPROACHING = "approaching"
    SLOWING = "slowing"
    STOPPED = "stopped"
    PASSING = "passing"


class StateEstimator:
    def __init__(
        self,
        stopped_speed_threshold: float = 2.0,
        slowing_threshold: float = -2.0,
        state_confirm_frames: int = 3,
        stopped_confirm_frames: int = 5,
        speed_history_length: int = 6,
    ):
        self.stopped_speed_threshold = stopped_speed_threshold
        self.slowing_threshold = slowing_threshold

        self.state_confirm_frames = state_confirm_frames
        self.stopped_confirm_frames = stopped_confirm_frames

        # Previous information about each tracked object
        self.previous_zones = {}
        self.previous_states = {}

        # Candidate states waiting for confirmation
        self.pending_states = {}
        self.pending_counts = defaultdict(int)

        # Speed history for vehicles
        self.speed_history = defaultdict(
            lambda: deque(
                maxlen=speed_history_length
            )
        )

    def update(self, obj):
        """
        Estimate and stabilize the semantic state of an object.
        """

        if obj.category == "pedestrian":
            return self._update_pedestrian(obj)

        if obj.category == "vehicle":
            return self._update_vehicle(obj)

        return None

    # ---------------------------------------------------------
    # Pedestrian
    # ---------------------------------------------------------

    def _update_pedestrian(self, obj):
        track_id = obj.track_id

        current_zone = obj.zone
        previous_zone = self.previous_zones.get(
            track_id
        )

        # ---------------------------------------------
        # Candidate state
        # ---------------------------------------------

        if current_zone is None:
            candidate_state = (
                PedestrianState.OUTSIDE
            )

        elif current_zone in {
            "pedestrian_waiting_left",
            "pedestrian_waiting_right",
        }:
            if previous_zone == "crosswalk":
                candidate_state = (
                    PedestrianState.LEAVING
                )
            else:
                candidate_state = (
                    PedestrianState.WAITING
                )

        elif current_zone == "crosswalk":
            if previous_zone in {
                "pedestrian_waiting_left",
                "pedestrian_waiting_right",
            }:
                candidate_state = (
                    PedestrianState.ENTERING
                )
            else:
                candidate_state = (
                    PedestrianState.CROSSING
                )

        else:
            candidate_state = (
                PedestrianState.OUTSIDE
            )

        # ---------------------------------------------
        # Store current zone
        # ---------------------------------------------

        self.previous_zones[
            track_id
        ] = current_zone

        # ---------------------------------------------
        # Stabilize state
        # ---------------------------------------------

        return self._stabilize_state(
            track_id=track_id,
            candidate_state=candidate_state,
        )

    # ---------------------------------------------------------
    # Vehicle
    # ---------------------------------------------------------

    def _update_vehicle(self, obj):
        track_id = obj.track_id
        current_zone = obj.zone
        current_speed = obj.speed_kmh

        # ---------------------------------------------
        # Update speed history
        # ---------------------------------------------

        if current_speed is not None:
            self.speed_history[
                track_id
            ].append(current_speed)

        # ---------------------------------------------
        # Candidate state
        # ---------------------------------------------

        if current_zone == "crosswalk":
            candidate_state = (
                VehicleState.PASSING
            )

        elif current_zone != "vehicle_approach":
            candidate_state = (
                VehicleState.OUTSIDE
            )

        elif current_speed is None:
            candidate_state = (
                VehicleState.APPROACHING
            )

        elif (
            current_speed
            <= self.stopped_speed_threshold
        ):
            candidate_state = (
                VehicleState.STOPPED
            )

        else:
            speed_delta = (
                self._calculate_speed_trend(
                    track_id
                )
            )

            if (
                speed_delta
                <= self.slowing_threshold
            ):
                candidate_state = (
                    VehicleState.SLOWING
                )
            else:
                candidate_state = (
                    VehicleState.APPROACHING
                )

        # ---------------------------------------------
        # Stabilize state
        # ---------------------------------------------

        confirm_frames = (
            self.stopped_confirm_frames
            if candidate_state
            == VehicleState.STOPPED
            else self.state_confirm_frames
        )

        return self._stabilize_state(
            track_id=track_id,
            candidate_state=candidate_state,
            confirm_frames=confirm_frames,
        )

    # ---------------------------------------------------------
    # Speed trend
    # ---------------------------------------------------------

    def _calculate_speed_trend(
        self,
        track_id: int,
    ) -> float:
        """
        Compare average speed from the first half
        of the history with the second half.

        Negative value means the vehicle is slowing down.
        """

        history = list(
            self.speed_history[track_id]
        )

        if len(history) < 4:
            return 0.0

        midpoint = len(history) // 2

        old_values = history[:midpoint]
        new_values = history[midpoint:]

        old_speed = (
            sum(old_values)
            / len(old_values)
        )

        new_speed = (
            sum(new_values)
            / len(new_values)
        )

        return new_speed - old_speed

    # ---------------------------------------------------------
    # State stabilization
    # ---------------------------------------------------------

    def _stabilize_state(
        self,
        track_id: int,
        candidate_state,
        confirm_frames: int | None = None,
    ):
        if confirm_frames is None:
            confirm_frames = (
                self.state_confirm_frames
            )

        current_state = (
            self.previous_states.get(
                track_id
            )
        )

        # ---------------------------------------------
        # First state
        # ---------------------------------------------

        if current_state is None:
            self.previous_states[
                track_id
            ] = candidate_state

            return candidate_state

        # ---------------------------------------------
        # Candidate is already current state
        # ---------------------------------------------

        if candidate_state == current_state:
            self.pending_states.pop(
                track_id,
                None,
            )

            self.pending_counts[
                track_id
            ] = 0

            return current_state

        # ---------------------------------------------
        # New candidate
        # ---------------------------------------------

        pending_state = (
            self.pending_states.get(
                track_id
            )
        )

        if pending_state != candidate_state:
            self.pending_states[
                track_id
            ] = candidate_state

            self.pending_counts[
                track_id
            ] = 1

            return current_state

        # ---------------------------------------------
        # Same candidate again
        # ---------------------------------------------

        self.pending_counts[
            track_id
        ] += 1

        # ---------------------------------------------
        # Candidate confirmed
        # ---------------------------------------------

        if (
            self.pending_counts[
                track_id
            ]
            >= confirm_frames
        ):
            self.previous_states[
                track_id
            ] = candidate_state

            self.pending_states.pop(
                track_id,
                None,
            )

            self.pending_counts[
                track_id
            ] = 0

            return candidate_state

        # ---------------------------------------------
        # Keep old state until candidate is confirmed
        # ---------------------------------------------

        return current_state

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def remove_track(
        self,
        track_id: int,
    ):
        """
        Remove all stored state for a track.
        Useful later when old tracks are cleaned up.
        """

        self.previous_zones.pop(
            track_id,
            None,
        )

        self.previous_states.pop(
            track_id,
            None,
        )

        self.pending_states.pop(
            track_id,
            None,
        )

        self.pending_counts.pop(
            track_id,
            None,
        )

        self.speed_history.pop(
            track_id,
            None,
        )