

from . import HumanPoseKeypointAnnotation
from . import Joint

class CocoHumanKeypointsAnnotation(HumanPoseKeypointAnnotation):
    """
    17 keypoints from the MSCoco dataset
    """

    joints = [
        Joint.NOSE,
        Joint.LEFT_EYE,
        Joint.RIGHT_EYE,
        Joint.LEFT_EAR,
        Joint.RIGHT_EAR,
        Joint.LEFT_SHOULDER,
        Joint.RIGHT_SHOULDER,
        Joint.LEFT_ELBOW,
        Joint.RIGHT_ELBOW,
        Joint.LEFT_WRIST,
        Joint.RIGHT_WRIST,
        Joint.LEFT_HIP,
        Joint.RIGHT_HIP,
        Joint.LEFT_KNEE,
        Joint.RIGHT_KNEE,
        Joint.LEFT_ANKLE,
        Joint.RIGHT_ANKLE
    ]

    def __init__(self):
        super().__init__(joints=CocoHumanKeypointsAnnotation.joints)

    def from_tsv(line):
        annotation = CocoHumanKeypointsAnnotation()
        annotation.parse_tsv(line)
        return annotation
