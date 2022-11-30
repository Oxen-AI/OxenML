import jsonpickle

from oxen.annotations.annotation import Annotation


class IDAnnotations:
    """
    Class that represents a set of annotations that reference some ID.
    
    Examples:
    * The ID could be an image filename, and annotations could be many bounding box annotations per file
    * The ID could be an external UUID for a document, and annotations could be a list of annotations on that document
    """
    def __init__(self, id, annotations = []):
        self.id = id
        self.annotations: list[Annotation] = annotations

    def __repr__(self):
        return f"<IDAnnotations id: {self.id}, len(annotations): {len(self.annotations)}>"

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, i: int):
        return self.annotations[i]

    def add_annotation(self, annotation):
        self.annotations.append(annotation)

    def num_annotations(self):
        return len(self.annotations)

    def to_tsv(self):
        return "\n".join([f"{self.id}\t{a.to_tsv()}" for a in self.annotations])

    def to_csv(self):
        return "\n".join([f"{self.id},{a.to_csv()}" for a in self.annotations])

    def csv_header(self):
        return self.annotations[0].csv_header()

    def tsv_header(self):
        return self.annotations[0].tsv_header()

    def to_json(self):
        return jsonpickle.encode(
            {"input": self.id, "outputs": self.annotations},
            unpicklable=False,
        )
