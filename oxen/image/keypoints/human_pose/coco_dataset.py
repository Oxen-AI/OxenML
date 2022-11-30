import os
import json

from oxen.image.keypoints.human_pose import CocoHumanKeypointsAnnotation
from oxen.annotations.annotations_dataset import AnnotationsDataset
from oxen.annotations.id_annotations import IDAnnotations


class CocoHumanKeypointsDataset(AnnotationsDataset):
    def __init__(self, annotation_file: str):
        super().__init__()
        self.annotations = self._load_dataset(annotation_file)

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

        # Grab all the annotations and organize them by image
        print(f"Parsing {len(data['annotations'])} annotations")
        for item in data["annotations"]:
            category_num = int(item["category_id"])
            iscrowd = "iscrowd" in item and item["iscrowd"] == 1
            # --- Check if category is person not in a crowd
            if category_num == 1 and not iscrowd:
                id = str(item["image_id"])
                raw_kps = item["keypoints"]

                kp = CocoHumanKeypointsAnnotation()
                kp.parse_array(raw_kps)

                # Filter out annotations with all zeros..
                if kp.is_all_zeros():
                    continue

                file_annotations[id].add_annotation(kp)

        annotations = {}
        for (_id, annotation) in file_annotations.items():
            # Filter out annotations with all zeros..
            if len(annotation.annotations) > 0:
                annotations[annotation.file] = annotation

        return annotations
