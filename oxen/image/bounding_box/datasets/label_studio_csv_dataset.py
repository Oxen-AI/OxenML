from __future__ import annotations

import csv
import json

from oxen.annotations.annotations_dataset import AnnotationsDataset
from oxen.annotations.file_annotations import FileAnnotations
from oxen.image.bounding_box.annotations.oxen_bounding_box import OxenBoundingBox


class LabelStudioCSVBoundingBoxDataset(AnnotationsDataset):
    def __init__(self, path: str):
        super().__init__()
        self._load_annotations_from_file(path)

    def _load_annotations_from_file(self, path: str) -> dict[str, FileAnnotations]:
        annotations = {}
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for (i, row) in enumerate(reader):
                # skip header
                if i == 0:
                    continue

                # Label studio exports with /data/upload/ prefix
                filename = row[0].replace("/data/", "")
                labels_val = row[2]
                
                print(labels_val)
                
                labels_json = json.loads(labels_val)
                for json_val in labels_json:
                    # Label studio exports are in percentage of width
                    x = json_val["x"] * json_val["original_width"] / 100.0
                    y = json_val["y"] * json_val["original_height"] / 100.0
                    width = json_val["width"] * json_val["original_width"] / 100.0
                    height = json_val["height"] * json_val["original_height"] / 100.0
                    bounding_box = OxenBoundingBox(
                        min_x=x,
                        min_y=y,
                        width=width,
                        height=height,
                        label=json_val["rectanglelabels"][0]
                    )

                    if not filename in annotations:
                        annotations[filename] = FileAnnotations(filename)

                    annotations[filename].add_annotation(bounding_box)
        self.annotations = annotations
