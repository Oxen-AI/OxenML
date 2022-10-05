import json


class ImageKeypoint:
    def __init__(self, x, y, confidence=0.0):
        self.x = x
        self.y = y
        self.confidence = confidence

    def __repr__(self):
        return f"<ImageKeypoint x: {self.x}, y: {self.y} confidence: {self.confidence}>"

    def to_tsv(self):
        return f"{self.x}\t{self.y}\t{self.confidence}"

    def to_csv(self):
        return f"{self.x},{self.y},{self.confidence}"

    @classmethod
    def average(cls, keypoints: list):
        if len(keypoints) == 0:
            # Don't divide by zero
            return cls(x=0, y=0, confidence=0)
        x = 0.0
        y = 0.0
        c = 0.0
        for kp in keypoints:
            x += kp.x
            y += kp.y
            c += kp.confidence
        total = float(len(keypoints))
        return cls(x=x / total, y=y / total, confidence=c / total)

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def within_threshold(self, other, threshold):
        return abs(other.x - self.x) < threshold and abs(other.y - self.y) < threshold
