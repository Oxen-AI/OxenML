
import sys
import pathlib

# Need to add the oxen dir to test paths
oxen_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()
sys.path.append(str(oxen_dir))

from oxen.image.keypoints.human_pose.ms_coco_dataset import MSCocoKeypointsDataset

def test_load_coco():
    dataset = MSCocoKeypointsDataset(
        annotation_file="tests/data/coco_small.json"
    )
    assert dataset.num_inputs() == 3
    
    file = "000000101172.jpg"
    file_annotations = dataset.get_annotations(file)
    assert len(file_annotations) == 1
    
    annotation = file_annotations.annotations[0]
    keypoint = annotation.get_joint_keypoint("left_shoulder")
    assert keypoint.x == 269
    assert keypoint.y == 144
    assert keypoint.confidence == 1.0
    
