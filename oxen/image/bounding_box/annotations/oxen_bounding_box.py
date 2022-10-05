
import sys
import math

from oxen.annotations import Annotation

class OxenBoundingBox(Annotation):
    def __init__(self, min_x, min_y, width, height, label="Unknown"):
        self.min_x = min_x
        self.min_y = min_y
        self.width = width
        self.height = height
        self.label = label

    def __repr__(self):
        return f"<BoundingBox x: {self.min_x}, y: {self.min_y} w: {self.width} h: {self.height}>"

    def tsv_header(self):
        return "file\tmin_x\tmin_y\twidth\theight"

    def csv_header(self):
        return "file,min_x,min_y,width,height"

    def to_tsv(self):
        return f"{self.min_x}\t{self.min_y}\t{self.width}\t{self.height}"
    
    def to_csv(self):
        return f"{self.min_x},{self.min_y},{self.width},{self.height}"

    def diagonal(self):
        return math.sqrt((self.width * self.width) + (self.height * self.height))

    def from_keypoints(annotation):
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
