

from oxen.image.keypoints.human_pose import HumanPoseKeypointAnnotation

class AIChallengerKeypointsAnnotation(HumanPoseKeypointAnnotation):
    """
    14 keypoints from the AI Challenger dataset
    """

    joints = [
        "top_head",
        "neck",
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
        super().__init__(joints=AIChallengerKeypointsAnnotation.joints)
