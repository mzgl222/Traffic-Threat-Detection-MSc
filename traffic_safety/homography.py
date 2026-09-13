import cv2
import numpy as np


class HomographyTransformer:
    def __init__(self, image_points, world_points):
        self.image_points = np.array(image_points, dtype=np.float32)
        self.world_points = np.array(world_points, dtype=np.float32)

        if self.image_points.shape != (4, 2):
            raise ValueError("image_points must contain exactly four 2D points")

        if self.world_points.shape != (4, 2):
            raise ValueError("world_points must contain exactly four 2D points")

        self.matrix = cv2.getPerspectiveTransform(
            self.image_points,
            self.world_points
        )

    def image_to_world(self, x: float, y: float) -> tuple[float, float]:
        point = np.array([[[x, y]]], dtype=np.float32)

        transformed = cv2.perspectiveTransform(
            point,
            self.matrix
        )

        world_x, world_y = transformed[0][0]

        return float(world_x), float(world_y)