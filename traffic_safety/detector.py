from dataclasses import dataclass

from ultralytics import YOLO


CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def get_category(class_id: int) -> str:
    if class_id == 0:
        return "pedestrian"

    if class_id == 1:
        return "bicycle"

    if class_id in {2, 3, 5, 7}:
        return "vehicle"

    return "other"


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    category: str
    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def contact_point(self) -> tuple[float, float]:
        """
        Approximate point where the object touches the ground.
        """
        x = (self.x1 + self.x2) / 2
        y = self.y2

        return x, y


class ObjectTracker:
    def __init__(
        self,
        model_path: str,
        tracker: str = "bytetrack.yaml",
        confidence: float = 0.15,
        allowed_classes: set[int] | None = None,
    ):
        self.model = YOLO(model_path)

        self.tracker = tracker
        self.confidence = confidence
        self.allowed_classes = allowed_classes

    def process(self, frame) -> list[TrackedObject]:
        results = self.model.track(
            source=frame,
            persist=True,
            verbose=False,
            tracker=self.tracker,
            conf=self.confidence,
        )

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return []

        objects = []

        for box in result.boxes:
            class_id = int(box.cls[0])

            if (
                self.allowed_classes is not None
                and class_id not in self.allowed_classes
            ):
                continue

            track_id = int(box.id[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            objects.append(
                TrackedObject(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=CLASS_NAMES.get(
                        class_id,
                        f"class_{class_id}",
                    ),
                    category=get_category(class_id),
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return objects