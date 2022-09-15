

from . import HumanPoseKeypointAnnotation

class CocoHumanKeypointsAnnotation(HumanPoseKeypointAnnotation):
    """
    17 keypoints from the MSCoco dataset
    """

    joints = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(self):
        super().__init__(joints=CocoHumanKeypointsAnnotation.joints)

    def from_tsv(line):
        annotation = CocoHumanKeypointsAnnotation()
        annotation.parse_tsv(line)
        return annotation
