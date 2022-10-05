import os
import json

from oxen.image.keypoints.human_pose import OxenHumanKeypointsAnnotation
from oxen.annotations.annotations_dataset import AnnotationsDataset
from oxen.annotations.file_annotations import FileAnnotations


class OxenHumanKeypointsDataset(AnnotationsDataset):
    def __init__(self):
        super().__init__()

    def from_file(path: str):
        dataset = OxenHumanKeypointsDataset()
        dataset.annotations = OxenHumanKeypointsDataset.load_annotations_from_file(path)
        return dataset

    def from_dataset(dataset: AnnotationsDataset):
        annotations = {}
        for (id, file_ann) in dataset.annotations.items():
            converted_ann = FileAnnotations(file_ann.file)
            for ann in file_ann.annotations:
                converted_ann.add_annotation(
                    OxenHumanKeypointsAnnotation.from_annotation(ann)
                )
            annotations[id] = converted_ann
        dataset = OxenHumanKeypointsDataset()
        dataset.annotations = annotations
        return dataset

    def load_annotations_from_file(path: str) -> dict[str, FileAnnotations]:
        annotations = {}
        print(f"TODO: implement...")
        return annotations

    def convert_dataset_annotations(
        dataset: AnnotationsDataset,
    ) -> dict[str, FileAnnotations]:
        print(dataset.annotations)
