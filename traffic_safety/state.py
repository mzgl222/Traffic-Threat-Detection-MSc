from collections import defaultdict, deque
from enum import Enum


class PedestrianState(str, Enum):
    # ENTERING / LEAVING were removed (Phase II, B1): they were unreachable
    # under the 3-frame confirmation. Temporal information about entering
    # and leaving the crosswalk is provided by traffic_safety/motion_features.py.
    OUTSIDE = "outside"
    WAITING = "waiting"
    CROSSING = "crossing"


PEDESTRIAN_WAITING_ZONES = frozenset(
    {
        "pedestrian_waiting_left",
        "pedestrian_waiting_right",
    }
)


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

        # Previous (stabilized) state of each tracked object
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

    @staticmethod
    def _state_key(obj) -> tuple[str, int]:
        """
        Key for all per-object history.

        The tracker may keep the same track_id when the detector changes
        the class (e.g. person <-> motorcycle). Keying by (category, track_id)
        prevents a vehicle from inheriting a pedestrian state and vice versa.
        """
        return (obj.category, obj.track_id)

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
        track_id = self._state_key(obj)

        current_zone = obj.zone

        # ---------------------------------------------
        # Candidate state (depends only on the current zone)
        # ---------------------------------------------

        if current_zone in PEDESTRIAN_WAITING_ZONES:
            candidate_state = (
                PedestrianState.WAITING
            )

        elif current_zone == "crosswalk":
            candidate_state = (
                PedestrianState.CROSSING
            )

        else:
            candidate_state = (
                PedestrianState.OUTSIDE
            )

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
        track_id = self._state_key(obj)
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
        track_id: tuple[str, int],
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
        track_id: tuple[str, int],
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
        category: str | None = None,
    ):
        """
        Remove all stored state for a track.
        Useful later when old tracks are cleaned up.

        History is keyed by (category, track_id); with category=None
        the track is removed for every category.
        """

        stores = (
            self.previous_states,
            self.pending_states,
            self.pending_counts,
            self.speed_history,
        )

        for store in stores:
            keys = [
                key
                for key in list(store)
                if key[1] == track_id
                and (category is None or key[0] == category)
            ]

            for key in keys:
                store.pop(key, None)
