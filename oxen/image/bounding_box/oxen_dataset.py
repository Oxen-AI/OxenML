import os
import json

from oxen.image.bounding_box.annotations import CocoBoundingBoxDataset
from oxen.annotations.annotations_dataset import AnnotationsDataset
from oxen.annotations.file_annotations import FileAnnotations


class OxenBoundingBoxDataset(AnnotationsDataset):
    def __init__(self):
        super().__init__()

    def from_file(path: str):
        dataset = OxenBoundingBoxDataset()
        dataset.annotations = OxenBoundingBoxDataset.load_annotations_from_file(path)
        return dataset

    def load_annotations_from_file(path: str) -> dict[str, FileAnnotations]:
        annotations = {}
        print(f"TODO: implement...")
        return annotations

    def convert_dataset_annotations(dataset: AnnotationsDataset) -> dict[str, FileAnnotations]:
        print(dataset.annotations)
        
