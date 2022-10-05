from .. import HumanPoseKeypointAnnotation
from .. import Joint


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
        Joint.RIGHT_ANKLE,
    ]

    def __init__(self):
        super().__init__(joints=CocoHumanKeypointsAnnotation.joints)

    def csv_header(self):
        d = ","  # delimiter
        joint_data = d.join(
            [
                f"{joint.value}_x{d}{joint.value}_y{d}{joint.value}_confidence"
                for joint in self.joints
            ]
        )
        return f"file,{joint_data}"

    def to_csv(self):
        return ",".join([kp.to_csv() for kp in self.keypoints])

    def tsv_header(self):
        d = "\t"  # delimiter
        joint_data = d.join(
            [
                f"{joint.value}_x{d}{joint.value}_y{d}{joint.value}_confidence"
                for joint in self.joints
            ]
        )
        return f"file{d}{joint_data}"

    def to_tsv(self):
        return "\t".join([kp.to_tsv() for kp in self.keypoints])

    def from_tsv(line):
        annotation = CocoHumanKeypointsAnnotation()
        annotation.parse_tsv(line)
        return annotation
