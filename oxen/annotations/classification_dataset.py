
from oxen.annotations.annotations_dataset import AnnotationsDataset

from typing import List

class ClassificationDataset(AnnotationsDataset):
    
    def filter_labels(labels: List[str]):
        annotations = {}

