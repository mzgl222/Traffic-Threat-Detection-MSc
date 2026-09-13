import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


WINDOW_NAME = "Homography point selector"

# Punkty w oryginalnej rozdzielczości obrazu
points: list[tuple[int, int]] = []

# Skala użyta tylko do wyświetlania obrazu
display_scale = 1.0

original_frame = None
display_frame = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive homography point selector."
    )

    parser.add_argument(
        "input",
        help="Path to an image or video file."
    )

    parser.add_argument(
        "--output",
        "-o",
        default="scripts/homography_points.yaml",
        help="Output YAML file."
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame number to use when input is a video."
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=1920,
        help="Maximum width of displayed image."
    )

    parser.add_argument(
        "--max-height",
        type=int,
        default=1080,
        help="Maximum height of displayed image."
    )

    parser.add_argument(
        "--width-m",
        type=float,
        default=None,
        help="Real-world width of selected area in meters."
    )

    parser.add_argument(
        "--length-m",
        type=float,
        default=None,
        help="Real-world length of selected area in meters."
    )

    return parser.parse_args()


def load_frame(path: str, frame_number: int):
    """
    Loads either an image or one frame from a video.
    """

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".webp",
    }

    suffix = Path(path).suffix.lower()

    if suffix in image_extensions:
        frame = cv2.imread(path)

        if frame is None:
            raise RuntimeError(f"Could not load image: {path}")

        return frame

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            f"Could not read frame {frame_number} from {path}"
        )

    return frame


def calculate_display_scale(
    frame,
    max_width: int,
    max_height: int
):
    height, width = frame.shape[:2]

    width_scale = max_width / width
    height_scale = max_height / height

    return min(
        width_scale,
        height_scale,
        1.0
    )


def create_display_image(frame, scale):
    if scale == 1.0:
        return frame.copy()

    return cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )


def redraw():
    """
    Draws selected points and polygon.
    """

    global display_frame

    display_frame = create_display_image(
        original_frame,
        display_scale
    )

    # Convert original coordinates into display coordinates.
    display_points = [
        (
            int(x * display_scale),
            int(y * display_scale)
        )
        for x, y in points
    ]

    # Draw connecting lines.
    if len(display_points) >= 2:
        for i in range(len(display_points) - 1):
            cv2.line(
                display_frame,
                display_points[i],
                display_points[i + 1],
                (0, 255, 255),
                2
            )

    # Close polygon if all 4 points exist.
    if len(display_points) == 4:
        cv2.line(
            display_frame,
            display_points[-1],
            display_points[0],
            (0, 255, 255),
            2
        )

    # Draw points.
    for index, (x, y) in enumerate(display_points):
        cv2.circle(
            display_frame,
            (x, y),
            7,
            (0, 0, 255),
            -1
        )

        original_x, original_y = points[index]

        label = (
            f"{index + 1}: "
            f"({original_x}, {original_y})"
        )

        cv2.putText(
            display_frame,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Instructions.
    instructions = [
        "LEFT CLICK: add point",
        "RIGHT CLICK / U: undo",
        "R: reset",
        "S: save",
        "Q / ESC: quit",
        f"Selected: {len(points)}/4",
    ]

    y = 30

    for text in instructions:
        cv2.putText(
            display_frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        y += 28

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 4:
            print(
                "Already selected 4 points. "
                "Undo or reset before adding another."
            )
            return

        # Convert display coordinates back into
        # original image coordinates.
        original_x = round(x / display_scale)
        original_y = round(y / display_scale)

        points.append(
            (original_x, original_y)
        )

        print(
            f"Point {len(points)}: "
            f"({original_x}, {original_y})"
        )

        redraw()

    elif event == cv2.EVENT_RBUTTONDOWN:
        undo_last_point()


def undo_last_point():
    if not points:
        return

    removed = points.pop()

    print(
        f"Removed point: {removed}"
    )

    redraw()


def reset_points():
    points.clear()

    print("Points reset.")

    redraw()


def save_config(
    output_path: str,
    width_m: float | None,
    length_m: float | None
):
    if len(points) != 4:
        print(
            "Exactly 4 points must be selected before saving."
        )
        return

    config = {
        "calibration": {
            "image_points": [
                [int(x), int(y)]
                for x, y in points
            ]
        }
    }

    # If real dimensions were given,
    # generate corresponding world coordinates.
    if width_m is not None and length_m is not None:
        config["calibration"]["world_points"] = [
            [0.0, 0.0],
            [float(width_m), 0.0],
            [float(width_m), float(length_m)],
            [0.0, float(length_m)],
        ]

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output.open(
        "w",
        encoding="utf-8"
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True
        )

    print()
    print(f"Saved configuration to: {output}")
    print()

    print(
        yaml.safe_dump(
            config,
            sort_keys=False,
            allow_unicode=True
        )
    )


def main():
    global original_frame
    global display_scale

    args = parse_args()
    original_frame = load_frame(
        args.input,
        args.frame
    )

    height, width = original_frame.shape[:2]

    print(
        f"Original image resolution: "
        f"{width} x {height}"
    )

    display_scale = calculate_display_scale(
        original_frame,
        args.max_width,
        args.max_height
    )

    print(
        f"Display scale: {display_scale:.3f}"
    )

    print()
    print("Select points in this order:")
    print("1. top-left")
    print("2. top-right")
    print("3. bottom-right")
    print("4. bottom-left")
    print()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_AUTOSIZE
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key == ord("u"):
            undo_last_point()

        elif key == ord("r"):
            reset_points()

        elif key == ord("s"):
            save_config(
                args.output,
                args.width_m,
                args.length_m
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()