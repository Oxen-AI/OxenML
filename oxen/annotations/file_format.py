
from enum import Enum

class FileFormat(Enum):
    TSV = "tsv"
    CSV = "csv"
    ND_JSON = "ndjson"


def ext_equals(a: str, b: FileFormat):
    return ((a == ".csv" and b == FileFormat.CSV) or
           (a == ".tsv" and b == FileFormat.TSV) or
           (a == ".ndjson" and b == FileFormat.ND_JSON))