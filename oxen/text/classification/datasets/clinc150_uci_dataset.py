
from oxen.annotations import AnnotationsDataset
from oxen.annotations import IDAnnotations
from oxen.text.classification.annotations import TextLabel

import json
import uuid

class Clinic150UCIDataset(AnnotationsDataset):
    def __init__(self, path: str):
        super().__init__()
        self.annotations = self._load_dataset(path)
        
    def _load_dataset(self, path: str):
        annotations = {}
        
        with open(path) as json_file:
            data = json.load(json_file)
        
        for key in data.keys():
            for example in data[key]:
                id = str(uuid.uuid4())
                text = example[0]
                label = example[1]
                
                annotation = TextLabel(id=id, text=text, label=label)
                a = IDAnnotations(id=id, annotations=[annotation])
                annotations[id] = a
            
        return annotations
