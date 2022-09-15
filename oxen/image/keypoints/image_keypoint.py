
import json

class ImageKeypoint:
    def __init__(self, x, y, confidence=0.0):
        self.x = x
        self.y = y
        self.confidence = confidence

    def __repr__(self):
        return f"<OxenImageKeypoint x: {self.x}, y: {self.y} confidence: {self.confidence}>"

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def within_threshold(self, other, threshold):
        return abs(other.x - self.x) < threshold and abs(other.y - self.y) < threshold
