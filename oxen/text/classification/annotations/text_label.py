
from oxen.annotations import Annotation

class TextLabel(Annotation):
    def __init__(self, id: str, label: str, text: str):
        self.id = id
        self.label = label
        self.text = text
    
    def tsv_header(self) -> str:
        return "id\tlabel\ttext"

    def csv_header(self) -> str:
        return "id,label,text"
    
    def to_tsv(self) -> str:
        return f"{self.label}\t{self.text}"

    def to_csv(self) -> str:
        return f"{self.label},{self.text}"