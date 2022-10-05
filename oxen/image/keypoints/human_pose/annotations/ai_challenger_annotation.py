

from oxen.image.keypoints.human_pose import HumanPoseKeypointAnnotation
from .. import Joint

class AIChallengerHumanKeypointsAnnotation(HumanPoseKeypointAnnotation):
    """
    14 keypoints from the AI Challenger dataset
    """

    joints = [
        Joint.TOP_HEAD,
        Joint.NECK,
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
        super().__init__(joints=AIChallengerHumanKeypointsAnnotation.joints)
