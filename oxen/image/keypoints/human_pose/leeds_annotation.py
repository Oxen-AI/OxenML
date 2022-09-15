

from . import HumanPoseKeypointAnnotation
from . import Joint

class LeedsHumanKeypointsAnnotation(HumanPoseKeypointAnnotation):
    """
    14 keypoints from the Leeds Sports Dataset
    
    http://sam.johnson.io/research/lsp.html
    """

    joints = [
        Joint.RIGHT_ANKLE,
        Joint.RIGHT_KNEE,
        Joint.RIGHT_HIP,
        Joint.LEFT_HIP,
        Joint.LEFT_KNEE,
        Joint.LEFT_ANKLE,
        Joint.RIGHT_WRIST,
        Joint.RIGHT_ELBOW,
        Joint.RIGHT_SHOULDER,
        Joint.LEFT_SHOULDER,
        Joint.LEFT_ELBOW,
        Joint.LEFT_WRIST,
        Joint.NECK,
        Joint.TOP_HEAD
    ]

    def __init__(self):
        super().__init__(joints=LeedsHumanKeypointsAnnotation.joints)

    def from_tsv(line):
        annotation = LeedsHumanKeypointsAnnotation()
        annotation.parse_tsv(line)
        return annotation
