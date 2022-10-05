import scipy.io

from oxen.annotations.annotations_dataset import AnnotationsDataset
from oxen.annotations.file_annotations import FileAnnotations
from oxen.image.keypoints.human_pose import LeedsHumanKeypointsAnnotation


class LeedsKeypointsDataset(AnnotationsDataset):
    def __init__(self, annotation_file):
        super().__init__()
        self.annotations = self._load_dataset(annotation_file)

    def _load_dataset(self, annotation_file):
        # m is 3x14x2000

        # This is the order of the joints
        # 0) Right ankle -> 13
        # 1) Right knee -> 11
        # 2) Right hip -> 9
        # 3) Left hip -> 8
        # 4) Left knee -> 10
        # 5) Left ankle -> 12
        # 6) Right wrist -> 7
        # 7) Right elbow -> 5
        # 8) Right shoulder -> 3
        # 9) Left shoulder -> 2
        # 10) Left elbow -> 4
        # 11) Left wrist -> 6
        # 12) Neck -> 1
        # 13) Head top -> 0

        data = scipy.io.loadmat(annotation_file)["joints"]
        num_joints = len(data[0])
        num_examples = len(data[0][0])

        # Map to same indicies as AIChallenger so we can convert after
        # idx_mapping = [13, 11, 9, 8, 10, 12, 7, 5, 3, 2, 4, 6, 1, 0]

        file_annotations = {}
        for e in range(num_examples):
            kps = []
            for j in range(num_joints):
                # i = idx_mapping[j]
                kps.append(data[0][j][e])
                kps.append(data[1][j][e])
                kps.append(1.0 if data[2][j][e] == 0.0 else 0.0)
            filename = f"im{str(e+1).zfill(4)}.jpg"
            file_annotation = FileAnnotations(file=filename)
            ann = LeedsHumanKeypointsAnnotation()
            ann.parse_array(kps)
            file_annotation.add_annotation(ann)

            # ai_challenge_kps = AIChallengerKeypointsAnnotation()
            # ai_challenge_kps.parse_array(kps)
            # oxen_kps = OxenHumanKeypointsAnnotation.from_ai_challenger(ai_challenge_kps)
            # file_annotation.add_annotation(oxen_kps)

            file_annotations[filename] = file_annotation
        return file_annotations
