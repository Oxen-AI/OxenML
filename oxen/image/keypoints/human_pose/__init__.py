# from oxen.image.keypoints.human_pose.annotate import annotate
# from oxen.image.keypoints.human_pose.resize import resize
# from oxen.image.keypoints.human_pose.plot import plot

# from oxen.image.keypoints.human_pose.keypoint_annotation import HumanPoseKeypointAnnotation
# from oxen.image.keypoints.human_pose.oxen_keypoint_annotation import OxenHumanKeypointsAnnotation
# from oxen.image.keypoints.human_pose.coco_keypoint_annotation import CocoHumanKeypointsAnnotation
# from oxen.image.keypoints.human_pose.ai_challenger_keypoint_annotation import AIChallengerKeypointsAnnotation

from .human_pose_keypoint_annotation import HumanPoseKeypointAnnotation
from .coco_keypoint_annotation import CocoHumanKeypointsAnnotation
from .ai_challenger_keypoint_annotation import AIChallengerKeypointsAnnotation
from .oxen_keypoint_annotation import OxenHumanKeypointsAnnotation
