from __future__ import annotations

import csv

from oxen.annotations.annotations_dataset import AnnotationsDataset
from oxen.annotations.id_annotations import IDAnnotations
from oxen.image.bounding_box.annotations.oxen_bounding_box import OxenBoundingBox


class OxenCSVBoundingBoxDataset(AnnotationsDataset):
    def __init__(self, delimiter=",", has_header: bool = True):
        super().__init__()
        self.delimiter = delimiter
        self.has_header = has_header

    def from_file(
        path: str, delimiter=",", has_header: bool = True
    ) -> OxenCSVBoundingBoxDataset:
        dataset = OxenCSVBoundingBoxDataset(delimiter=delimiter, has_header=has_header)
        dataset._load_annotations_from_file(path)
        return dataset

    def _load_annotations_from_file(self, path: str) -> dict[str, IDAnnotations]:
        print(f"Skip header? {self.has_header}")
        annotations = {}
        with open(path) as csvfile:
            reader = csv.reader(csvfile, delimiter=self.delimiter)
            for (i, row) in enumerate(reader):
                if i == 0 and self.has_header:
                    continue

                filename = row[0]
                label = row[1]
                bounding_box = OxenBoundingBox.from_arr(
                    label, [float(item) for item in row[2:]]
                )

                if not filename in annotations:
                    annotations[filename] = IDAnnotations(filename)

                annotations[filename].add_annotation(bounding_box)
        self.annotations = annotations
