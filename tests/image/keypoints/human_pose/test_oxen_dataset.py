import sys, os
import pathlib
import uuid

# Need to add the oxen dir to test paths
oxen_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()
sys.path.append(str(oxen_dir))

from oxen.image.keypoints.image_keypoint import ImageKeypoint
from oxen.image.keypoints.human_pose import Joint
from oxen.image.keypoints.human_pose import (
    CocoHumanKeypointsDataset,
    OxenHumanKeypointsDataset,
)


def test_convert_coco():
    coco_dataset = CocoHumanKeypointsDataset(
        annotation_file="tests/data/coco_person_keypoints_small.json"
    )

    dataset = OxenHumanKeypointsDataset.from_dataset(coco_dataset)
    assert dataset.num_inputs() == 3

    file = "000000101172.jpg"
    file_annotations = dataset.get_annotations(file)
    assert len(file_annotations) == 1

    annotation = file_annotations.annotations[0]

    assert len(annotation.joints) == 13

    keypoint = annotation.get_joint_keypoint(Joint.LEFT_SHOULDER)
    assert keypoint.x == 269
    assert keypoint.y == 144
    assert keypoint.confidence == 1.0

    face_keypoints = [
        coco_dataset.get_annotations(file)
        .annotations[0]
        .get_joint_keypoint(Joint.NOSE),
        coco_dataset.get_annotations(file)
        .annotations[0]
        .get_joint_keypoint(Joint.LEFT_EYE),
        coco_dataset.get_annotations(file)
        .annotations[0]
        .get_joint_keypoint(Joint.RIGHT_EYE),
        coco_dataset.get_annotations(file)
        .annotations[0]
        .get_joint_keypoint(Joint.LEFT_EAR),
        coco_dataset.get_annotations(file)
        .annotations[0]
        .get_joint_keypoint(Joint.RIGHT_EAR),
    ]

    # Head should be average of all the face joints
    keypoint = annotation.get_joint_keypoint(Joint.HEAD)
    face_avg = ImageKeypoint.average(face_keypoints)
    assert keypoint.x == face_avg.x
    assert keypoint.y == face_avg.y
    assert keypoint.confidence == 1.0


def test_write_converted_dataset():
    coco_dataset = CocoHumanKeypointsDataset(
        annotation_file="tests/data/coco_person_keypoints_small.json"
    )

    filename = f"tests/data/{uuid.uuid4().hex}.ndjson"

    dataset = OxenHumanKeypointsDataset.from_dataset(coco_dataset)
    dataset.write_output("tests/data/", filename)

    # cleanup
    os.remove(filename)
