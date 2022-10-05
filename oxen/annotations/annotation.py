

class Annotation:
    def __repr__(self):
        return f"<Annotation json: {self.to_json()}>"

    def tsv_header(self):
        raise NotImplementedError()
    
    def csv_header(self):
        raise NotImplementedError()

    def to_tsv(self):
        raise NotImplementedError()

    def to_csv(self):
        raise NotImplementedError()

    def to_json(self):
        raise NotImplementedError()