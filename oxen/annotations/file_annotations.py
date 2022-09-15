
import jsonpickle

from oxen.annotations.annotation import Annotation

class FileAnnotations:
    def __init__(self, file):
        self.file = file
        self.annotations: list[Annotation] = []

    def __repr__(self):
        return f"<FileAnnotations file: {self.file}, len(annotations): {len(self.annotations)}>"

    def __len__(self) -> int:
        return len(self.annotations)

    def add_annotation(self, annotation):
        self.annotations.append(annotation)

    def num_annotations(self):
        return len(self.annotations)

    def to_tsv(self):
        return "\n".join([f"{self.file}\t{a.to_tsv()}" for a in self.annotations])

    def to_json(self):
        return jsonpickle.encode(
            {"input": self.file, "outputs": self.annotations},
            unpicklable=False,
        )