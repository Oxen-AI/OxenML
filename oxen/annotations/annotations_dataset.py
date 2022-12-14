from argparse import FileType
import os

from . import Annotation
from . import FileFormat


class AnnotationsDataset:
    def __init__(self):
        self.annotations: dict[str, list[Annotation]] = {}

    def list_annotations(self):
        annotations = []
        for (_, a) in self.annotations.items():
            annotations.append(a)
        return annotations

    def list_inputs(self):
        files = []
        for (file, _) in self.annotations.items():
            files.append(file)
        return files

    def num_inputs(self):
        return len(self.annotations)

    def get_annotations(self, key: str) -> list[Annotation]:
        return self.annotations[key]

    def write_tsv(self, outfile):
        self.write_output(outfile, output_type=FileFormat.TSV)

    def write_csv(self, outfile):
        self.write_output(outfile, output_type=FileFormat.CSV)

    def write_tsv_with_base_dir(self, base_img_dir, outfile):
        self.write_output(outfile, base_img_dir, output_type=FileFormat.TSV)

    def write_csv_with_base_dir(self, base_img_dir, outfile):
        self.write_output(outfile, base_img_dir, output_type=FileFormat.CSV)

    def write_ndjson(self, base_img_dir: str, outfile: str):
        self.write_output(outfile, base_img_dir, output_type=FileFormat.ND_JSON)

    def write_output(
        self,
        outfile: str,
        base_img_dir: str = None,
        one_example_per_file: bool = False,
        output_type: FileFormat = FileFormat.CSV,
    ):
        print(f"Writing annotations to {outfile}")
        num_outputted = 0
        with open(outfile, "w") as f:
            for id in self.annotations.keys():
                annotations = self.annotations[id]

                if one_example_per_file and len(annotations.annotations) != 1:
                    continue

                # Set the proper filepath to not just be filename
                if base_img_dir != None:
                    file = os.path.join(base_img_dir, annotations.file)
                    annotations.file = file

                if FileFormat.TSV == output_type:
                    if num_outputted == 0:
                        f.write(f"{annotations[0].tsv_header()}\n")
                    f.write(f"{annotations.to_tsv()}\n")
                elif FileFormat.CSV == output_type:
                    if num_outputted == 0:
                        f.write(f"{annotations[0].csv_header()}\n")
                    f.write(f"{annotations.to_csv()}\n")
                elif FileFormat.ND_JSON == output_type:
                    f.write(f"{annotations.to_json()}\n")
                else:
                    raise ValueError(f"Unknown argument: {output_type}")
                num_outputted += 1
        print(f"Wrote {num_outputted} annotations to {outfile}")
