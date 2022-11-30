
from oxen.annotations.annotations_dataset import AnnotationsDataset
from oxen.text.classification.datasets import Clinic150UCIDataset

def create_dataset(file: str, format: str) -> AnnotationsDataset:
    match format:
        case "clinic_150_uci":
            return Clinic150UCIDataset(path=file)
        case _:
            raise Exception(f"Unknown format: {format}")