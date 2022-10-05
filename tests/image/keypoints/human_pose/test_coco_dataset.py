import sys, os
import pathlib

# Need to add the oxen dir to test paths
oxen_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()
sys.path.append(str(oxen_dir))

from oxen.image.keypoints.human_pose import CocoHumanKeypointsDataset, Joint
from oxen.image.bounding_box import CocoBoundingBoxDataset


def test_load_coco_person_keypoints():
    dataset = CocoHumanKeypointsDataset(
        annotation_file="tests/data/coco_person_keypoints_small.json"
    )
    assert dataset.num_inputs() == 3

    file = "000000101172.jpg"
    file_annotations = dataset.get_annotations(file)
    assert len(file_annotations) == 1

    annotation = file_annotations.annotations[0]

    assert len(annotation.joints) == 17

    keypoint = annotation.get_joint_keypoint(Joint.LEFT_SHOULDER)
    assert keypoint.x == 269
    assert keypoint.y == 144
    assert keypoint.confidence == 1.0


def test_convert_coco_to_tsv():
    dataset = CocoHumanKeypointsDataset(
        annotation_file="tests/data/coco_person_keypoints_small.json"
    )

    base_dir = "test"
    outfile = "/tmp/test_coco_kp_output.tsv"
    dataset.write_tsv(base_dir, outfile=outfile)

    with open(outfile) as f:
        lines = f.readlines()
        assert lines[0].startswith("file\tnose_x\tnose_y\tnose_confidence\tleft_eye_x")
        assert lines[1].startswith("test/000000101172.jpg\t206\t79\t1.0")

    os.remove(outfile)
