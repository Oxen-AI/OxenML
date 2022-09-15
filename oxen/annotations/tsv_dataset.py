
from oxen.annotations import AnnotationsDataset
from oxen.annotations import FileAnnotations

from oxen.image.keypoints.human_pose import OxenHumanKeypointsAnnotation, CocoHumanKeypointsAnnotation

class TSVKeypointsDataset(AnnotationsDataset):
    def __init__(self, annotation_file: str, type:str='oxen'):
        super().__init__()
        self.type = type
        self.annotations = self._load_dataset(annotation_file)

    def _load_dataset(self, annotation_file: str):
        with open(annotation_file) as f:
            file_annotations = {}
            delimiter = "\t"
            for line in f:

                line = line.strip()
                split_line = line.split(delimiter)
                filename = split_line[0]

                if not filename in file_annotations:
                    file_annotations[filename] = FileAnnotations(file=filename)

                try:
                    if 'keypoint_oxen' == self.type:
                        a = OxenHumanKeypointsAnnotation.from_tsv(
                            delimiter.join(split_line[1:])
                        )
                    elif 'keypoint_coco' == self.type:
                        a = CocoHumanKeypointsAnnotation.from_tsv(
                            delimiter.join(split_line[1:])
                        )
                    else:
                        raise Exception(f"TSVKeypointsDataset unknown data type: {self.type}")
                    file_annotations[filename].annotations.append(a)
                except Exception as e:
                    print(f"Err: {e}")
                    print(line)
                    exit()

            return file_annotations
