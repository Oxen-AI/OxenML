from __future__ import annotations
import sys
import math
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from oxen.annotations import Annotation


class OxenBoundingBox(Annotation):
    def __init__(self, min_x, min_y, width, height, label="Unknown"):
        self.min_x = min_x
        self.min_y = min_y
        self.width = width
        self.height = height
        self.label = label

    def __repr__(self):
        return f"<BoundingBox x: {self.min_x}, y: {self.min_y} w: {self.width} h: {self.height} label: {self.label}>"

    def tsv_header(self) -> str:
        return "file\tlabel\tmin_x\tmin_y\twidth\theight"

    def csv_header(self) -> str:
        return "file,label,min_x,min_y,width,height"

    def to_tsv(self) -> str:
        return f"{self.label}\t{self.min_x:.2f}\t{self.min_y:.2f}\t{self.width:.2f}\t{self.height:.2f}"

    def to_csv(self) -> str:
        return f"{self.label},{self.min_x:.2f},{self.min_y:.2f},{self.width:.2f},{self.height:.2f}"

    def from_csv(self, line: str) -> OxenBoundingBox:
        return [float(item.strip()) for item in line.split(",")]

    def from_arr(label: str, arr: list[float]) -> OxenBoundingBox:
        return OxenBoundingBox(arr[0], arr[1], arr[2], arr[3], label)

    def diagonal(self) -> float:
        return math.sqrt((self.width * self.width) + (self.height * self.height))

    def from_keypoints(annotation) -> OxenBoundingBox:
        min_x = sys.float_info.max
        min_y = sys.float_info.max

        max_x = 0
        max_y = 0

        for kp in annotation.keypoints:
            if kp.x < min_x and kp.confidence > 0.5:
                min_x = kp.x

            if kp.y < min_y and kp.confidence > 0.5:
                min_y = kp.y

            if kp.x > max_x and kp.confidence > 0.5:
                max_x = kp.x

            if kp.y > max_y and kp.confidence > 0.5:
                max_y = kp.y

        width = max_x - min_x
        height = max_y - min_y
        return OxenBoundingBox(min_x=min_x, min_y=min_y, width=width, height=height)

    def plot_image_file(self, image_file):
        frame = plt.imread(image_file)

        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(frame)

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (self.min_x, self.min_y),
            self.width,
            self.height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()
