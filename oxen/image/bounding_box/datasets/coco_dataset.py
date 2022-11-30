import os
import json

from oxen.image.bounding_box.annotations import OxenBoundingBox
from oxen.annotations.annotations_dataset import AnnotationsDataset
from oxen.annotations.id_annotations import IDAnnotations


class CocoBoundingBoxDataset(AnnotationsDataset):
    def __init__(self, path: str):
        super().__init__()
        self.annotations = self._load_dataset(path)

    def _load_dataset(self, annotation_file):
        if not os.path.exists(annotation_file):
            raise ValueError("Annotation file not found")

        print(f"Loading dataset from {annotation_file}")
        with open(annotation_file) as json_file:
            data = json.load(json_file)

        # Find all image files and ids
        file_annotations = {}
        for item in data["images"]:
            id = str(item["id"])
            file_annotations[id] = IDAnnotations(id=item["file_name"])

        print(f"Got {len(file_annotations)} image files")

        categories = {}
        for item in data["categories"]:
            categories[int(item["id"])] = item["name"]

        # Grab all the annotations and organize them by image
        print(f"Parsing {len(data['annotations'])} annotations")
        for item in data["annotations"]:
            category_num = int(item["category_id"])
            if not category_num in categories:
                raise Exception(f"Unknown category id {category_num}")

            label_name = categories[category_num]
            image_id = str(item["image_id"])

            raw_bbox = item["bbox"]
            bb = OxenBoundingBox(
                min_x=raw_bbox[0],
                min_y=raw_bbox[1],
                width=raw_bbox[2],
                height=raw_bbox[3],
                label=label_name,
            )

            file_annotations[image_id].add_annotation(bb)

        annotations = {}
        for (_id, annotation) in file_annotations.items():
            # Filter out annotations with all zeros..
            if len(annotation.annotations) > 0:
                annotations[annotation.file] = annotation

        return annotations
