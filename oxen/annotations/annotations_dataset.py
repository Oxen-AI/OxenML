import os

from . import Annotation

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

    def write_tsv(self, base_img_dir, outfile):
        self.write_output(base_img_dir, outfile, output_type="tsv")

    def write_tsv(self, base_img_dir: str, outfile: str):
        self.write_output(base_img_dir, outfile, output_type="json")

    def write_output(
        self, base_img_dir, outfile, one_example_per_file=False, output_type="tsv"
    ):
        print(f"Writing annotations to {outfile}")
        num_outputted = 0
        with open(outfile, "w") as f:
            for id in self.annotations.keys():
                file_annotations = self.annotations[id]
                # print(f"{file_annotations.file} has {len(file_annotations.annotations)} annotations")
                if len(file_annotations.annotations) == 0:
                    # we filtered before it got to here
                    continue

                if one_example_per_file and len(file_annotations.annotations) != 1:
                    continue

                # Set the proper filepath to not just be filename
                file = os.path.join(base_img_dir, file_annotations.file)
                file_annotations.file = file

                if "tsv" == output_type:
                    f.write(f"{file_annotations.to_tsv()}\n")
                elif "json" == output_type:
                    f.write(f"{file_annotations.to_json()}\n")
                else:
                    raise ValueError(f"Unknown argument: {output_type}")
                num_outputted += 1
        print(f"Wrote {num_outputted} annotations to {outfile}")
