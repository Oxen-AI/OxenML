import sys, os
import pathlib

# Need to add the oxen dir to test paths
oxen_dir = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(oxen_dir))

from oxen.image.bounding_box.annotations.oxen_bounding_box import OxenBoundingBox
from oxen.image.bounding_box.datasets import LabelStudioCSVBoundingBoxDataset


def test_load_label_studio_bbox():
    dataset = LabelStudioCSVBoundingBoxDataset(
        path="tests/data/label_studio_bbox.csv"
    )
    assert dataset.num_inputs() == 3

    file = "upload/1/14adee8b-000000000109.jpg"
    file_annotations = dataset.get_annotations(file)
    assert len(file_annotations) == 1

    first_annotation = file_annotations[0]
    assert first_annotation.min_x == 189.10315789473688
    assert first_annotation.min_y == 158.4505263157895
    assert first_annotation.width == 8.016842105263157
    assert first_annotation.height == 8.488421052631578

