import os
import json

from oxen.image.keypoints.human_pose import AIChallengerKeypointsAnnotation
from oxen.image.keypoints.human_pose import CocoHumanKeypointsAnnotation
from oxen.image.keypoints.human_pose import OxenHumanKeypointsAnnotation
from oxen.annotations.annotations_dataset import AnnotationsDataset
from oxen.annotations.file_annotations import FileAnnotations


class MSCocoKeypointsDataset(AnnotationsDataset):
    def __init__(self, annotation_file: str, input_type: str = "mscoco"):
        super().__init__()
        self.annotations = self._load_dataset(annotation_file, input_type)

    def _parse_raw_keypoints(self, raw_kps, input_type: str):
        if "mscoco" == input_type:
            coco_kp = CocoHumanKeypointsAnnotation()
            coco_kp.parse_array(raw_kps)
            kp = OxenHumanKeypointsAnnotation.from_coco(coco_kp)
            return kp
        elif "ai_challenger" == input_type:
            ai_challenger_kp = AIChallengerKeypointsAnnotation()
            ai_challenger_kp.parse_array(raw_kps)
            kp = OxenHumanKeypointsAnnotation.from_ai_challenger(ai_challenger_kp)
            return kp
        coco_kp = CocoHumanKeypointsAnnotation()
        coco_kp.parse_array(raw_kps)
        return coco_kp

    def _load_dataset(self, annotation_file, input_type: str):
        if not os.path.exists(annotation_file):
            raise ValueError("Annotation file not found")

        print(f"Loading dataset from {annotation_file}")
        with open(annotation_file) as json_file:
            data = json.load(json_file)

        # Find all image files and ids
        file_annotations = {}
        for item in data["images"]:
            id = str(item["id"])
            file_annotations[id] = FileAnnotations(file=item["file_name"])

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

                kp = self._parse_raw_keypoints(raw_kps, input_type)

                if kp.is_all_zeros():
                    continue

                file_annotations[id].add_annotation(kp)

        annotations = {}
        for (_id, annotation) in file_annotations.items():
            annotations[annotation.file] = annotation

        return annotations
