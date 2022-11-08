
from oxen.annotations.annotations_dataset import AnnotationsDataset
from oxen.image.bounding_box.datasets.coco_dataset import CocoBoundingBoxDataset
from oxen.image.bounding_box.datasets.oxen_csv_dataset import OxenCSVBoundingBoxDataset
from oxen.image.bounding_box.datasets.label_studio_csv_dataset import LabelStudioCSVBoundingBoxDataset

def create_dataset(file: str, format: str) -> AnnotationsDataset:
    match format:
        case "coco":
            return CocoBoundingBoxDataset(path=file)
        case "oxen_csv":
            return OxenCSVBoundingBoxDataset.from_file(path=file)
        case "label_studio_csv":
            return LabelStudioCSVBoundingBoxDataset(path=file)
        case _:
            raise Exception(f"Unknown format: {format}")