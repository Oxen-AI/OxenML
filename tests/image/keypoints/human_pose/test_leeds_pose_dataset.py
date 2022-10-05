import sys
import pathlib

# Need to add the oxen dir to test paths
oxen_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()
sys.path.append(str(oxen_dir))

from oxen.image.keypoints.human_pose import LeedsKeypointsDataset, Joint


def test_load_leeds():
    dataset = LeedsKeypointsDataset(annotation_file="tests/data/leeds_joints.mat")
    assert dataset.num_inputs() == 2000

    file = "im0001.jpg"
    file_annotations = dataset.get_annotations(file)
    assert len(file_annotations) == 1

    annotation = file_annotations.annotations[0]

    assert len(annotation.joints) == 14

    keypoint = annotation.get_joint_keypoint(Joint.LEFT_SHOULDER)
    assert keypoint.x == 513
    assert keypoint.y == 365
    assert keypoint.confidence == 1.0
