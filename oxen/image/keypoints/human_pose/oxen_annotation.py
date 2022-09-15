
from oxen.annotations.annotation import Annotation
from oxen.image.keypoints.human_pose import CocoHumanKeypointsAnnotation
from oxen.image.keypoints.human_pose import AIChallengerHumanKeypointsAnnotation
from . import HumanPoseKeypointAnnotation
from . import Joint
from oxen.image.keypoints.image_keypoint import ImageKeypoint

class OxenHumanKeypointsAnnotation(HumanPoseKeypointAnnotation):
    """
    OxenSkeleton has 13 keypoints as a subset of other pose keypoint systems
    """

    joints = [
        Joint.HEAD,
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
        super().__init__(joints=OxenHumanKeypointsAnnotation.joints)

    def from_nd_array(output):
        annotation = OxenHumanKeypointsAnnotation()
        annotation.parse_heatmap_output(output)
        return annotation

    def from_tsv(line):
        annotation = OxenHumanKeypointsAnnotation()
        annotation.parse_tsv(line)
        return annotation

    def from_keypoints(kps):
        annotation = OxenHumanKeypointsAnnotation()
        annotation.parse_keypoints(kps)
        return annotation

    def from_annotation(ann: HumanPoseKeypointAnnotation):
        if isinstance(ann, CocoHumanKeypointsAnnotation):
            return OxenHumanKeypointsAnnotation.from_coco(ann)
        if isinstance(ann, AIChallengerHumanKeypointsAnnotation):
            return OxenHumanKeypointsAnnotation.from_ai_challenger(ann)
        raise NotImplementedError(f"Cannot convert unknown type {type(ann)}")

    def from_coco(coco_kps):
        # first five joints (nose, left_eye, left_ear, right_eye, right_ear) in mscoco collapse down to head
        return OxenHumanKeypointsAnnotation._collapse_top_n(coco_kps, 5)

    def from_ai_challenger(coco_kps):
        # first 2 joints in mscoco (head, neck) collapse down to head
        return OxenHumanKeypointsAnnotation._collapse_top_n(coco_kps, 2)

    def _collapse_top_n(ann: HumanPoseKeypointAnnotation, n: int):
        is_visible = False
        sum_x = 0.0
        sum_y = 0.0
        total = 0.0
        for i in range(n):
            kp = ann.keypoints[i]
            sum_x += kp.x
            sum_y += kp.y
            if kp.confidence > 0.5:
                total += 1
                is_visible = True

        avg_x = sum_x / float(total) if total > 0 else 0
        avg_y = sum_y / float(total) if total > 0 else 0
        confidence = 1 if is_visible else 0
        oxen_kps = OxenHumanKeypointsAnnotation()
        oxen_kps.keypoints.append(
            ImageKeypoint(x=avg_x, y=avg_y, confidence=confidence)
        )
        for kp in ann.keypoints[n:]:
            oxen_kps.keypoints.append(kp)

        assert len(oxen_kps.keypoints) == len(OxenHumanKeypointsAnnotation.joints)

        return oxen_kps

