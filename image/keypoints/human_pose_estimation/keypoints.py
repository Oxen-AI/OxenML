import os, sys
import json
import jsonpickle
import math
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum
import scipy.io


class PredictionOutcome(Enum):
    TRUE_POSITIVE = 1
    FALSE_POSITIVE = 2
    TRUE_NEGATIVE = 3
    FALSE_NEGATIVE = 4


class OxenImageKeypoint:
    def __init__(self, x, y, confidence=0.0):
        self.x = x
        self.y = y
        self.confidence = confidence

    def __repr__(self):
        return f"<OxenImageKeypoint x: {self.x}, y: {self.y} confidence: {self.confidence}>"

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def within_threshold(self, other, threshold):
        return abs(other.x - self.x) < threshold and abs(other.y - self.y) < threshold


class FileAnnotations:
    def __init__(self, file):
        self.file = file
        self.annotations = []

    def __repr__(self):
        return f"<FileAnnotations file: {self.file}, len(annotations): {len(self.annotations)}>"

    def add_annotation(self, annotation):
        self.annotations.append(annotation)

    def to_tsv(self):
        return "\n".join([f"{self.file}\t{a.to_tsv()}" for a in self.annotations])

    def to_json(self):
        return jsonpickle.encode(
            {"input": self.file, "outputs": [a.keypoints for a in self.annotations]},
            unpicklable=False,
        )


class BoundingBox:
    def __init__(self, min_x, min_y, width, height):
        self.min_x = min_x
        self.min_y = min_y
        self.width = width
        self.height = height

    def __repr__(self):
        return f"<BoundingBox x: {self.min_x}, y: {self.min_y} w: {self.width} h: {self.height}>"

    def diagonal(self):
        return math.sqrt((self.width * self.width) + (self.height * self.height))

    def from_human_kp(annotation):
        min_x = sys.float_info.max
        min_y = sys.float_info.max

        max_x = 0
        max_y = 0

        for kp in annotation.keypoints:
            if kp.x < min_x and kp.confidence > 0.5:
                min_x = kp.x

            if kp.y < min_y and kp.confidence > 0.5:
                min_y = kp.y

            if kp.x > max_x and kp.confidence > 0.5:
                max_x = kp.x

            if kp.y > max_y and kp.confidence > 0.5:
                max_y = kp.y

        width = max_x - min_x
        height = max_y - min_y
        return BoundingBox(min_x=min_x, min_y=min_y, width=width, height=height)


class HumanPoseKeypointAnnotation:
    def __init__(self, joints):
        self.joints = joints
        self.keypoints = []

    def __repr__(self):
        return f"<HumanPoseKeypointAnnotation n: {len(self.keypoints)}>"

    def compute_outcomes(self, prediction, confidence_thresh=0.5, fract_torso=0.2):
        outcomes = []

        bounding_box = BoundingBox.from_human_kp(self)
        # print(f"Got BB {bounding_box}")
        diag = bounding_box.diagonal()
        # print(f"Got diag {diag}")
        thresh = fract_torso * diag
        for (i, joint) in enumerate(self.joints):
            gt_kp = self.keypoints[i]
            pred_kp = prediction.keypoints[i]
            # print(f"Comparing {joint} {gt_kp} -> {pred_kp} thresh {thresh}")
            if (
                gt_kp.within_threshold(pred_kp, thresh)
                and pred_kp.confidence > confidence_thresh
            ):
                outcomes.append(PredictionOutcome.TRUE_POSITIVE)

            if (
                not gt_kp.within_threshold(pred_kp, thresh)
                and pred_kp.confidence > confidence_thresh
            ):
                outcomes.append(PredictionOutcome.FALSE_POSITIVE)

            if (
                gt_kp.within_threshold(pred_kp, thresh)
                and pred_kp.confidence < confidence_thresh
            ):
                outcomes.append(PredictionOutcome.FALSE_NEGATIVE)

            if (
                not gt_kp.within_threshold(pred_kp, thresh)
                and pred_kp.confidence < confidence_thresh
            ):
                outcomes.append(PredictionOutcome.TRUE_NEGATIVE)

        return outcomes

    def plot_image_file(self, image_file, color="#FF0000", threshold=0.5):
        frame = plt.imread(image_file)
        self.plot_image_frame(frame, color=color, threshold=threshold)

    def plot_image_frame(self, frame, color="#FF0000", threshold=0.5):
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
        [ax.axis("off") for ax in np.ravel(axes)]

        ax_orig = axes[0]
        ax_all = axes[1]

        ax_orig.imshow(frame)
        ax_all.imshow(frame)

        for keypoint in self.keypoints:
            print(keypoint)
            if keypoint.confidence > threshold:
                ax_all.scatter(
                    [keypoint.x], [keypoint.y], c=color, marker="x", s=50, linewidths=5
                )

        plt.tight_layout(pad=2.0)
        plt.show()

    def is_all_zeros(self):
        # Check if it is all zeros, then there is no one in the image...
        all_zeros = True
        for k in self.keypoints:
            if k.x != 0 or k.y != 0:
                all_zeros = False
        return all_zeros

    def parse_tsv(self, data):
        self.keypoints = []  # clear / initialize keypoints
        self.parse_array(data.split("\t"))

    def parse_keypoints(self, kps, confidence=1.0):
        self.keypoints = []  # clear / initialize keypoints
        for kp in kps:
            self.keypoints.append(
                OxenImageKeypoint(x=kp.x, y=kp.y, confidence=confidence)
            )

    def parse_array(self, data):
        step = 3
        n_keypoints = int(len(data) / step)
        # Since the last entry is the visibility flag, we discard it.
        for i in range(0, n_keypoints * step, step):
            x = float(data[i])
            y = float(data[i + 1])
            is_vis_row = data[i + 2]
            confidence = (
                1.0 if is_vis_row == 2.0 or is_vis_row == "True" else float(is_vis_row)
            )
            self.keypoints.append(OxenImageKeypoint(x=x, y=y, confidence=confidence))

        # Sanity check
        if len(self.keypoints) != len(self.joints):
            print(self.joints)
            raise Exception(
                f"{type(self)} Could not parse array into keypoints {len(self.keypoints)} != {len(self.joints)}"
            )

    def parse_heatmap_output(self, outputs):
        n_keypoints = outputs.shape[2]
        for i in range(n_keypoints):
            heatmap = np.array(outputs[:, :, i])

            x = np.argmax(np.amax(heatmap, axis=0))
            y = np.argmax(np.amax(heatmap, axis=1))
            confidence = heatmap[y][x]

            self.keypoints.append(OxenImageKeypoint(x=x, y=y, confidence=confidence))

    def parse_json(self, data):
        raise NotImplementedError()

    def to_tsv(self):
        return "\t".join(
            [f"{k.x}\t{k.y}\t{'{:.2f}'.format(k.confidence)}" for k in self.keypoints]
        )


class CocoHumanKeypointsAnnotation(HumanPoseKeypointAnnotation):
    """
    17 keypoints from the MSCoco dataset
    """

    joints = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(self):
        super().__init__(joints=CocoHumanKeypointsAnnotation.joints)

    def from_tsv(line):
        annotation = CocoHumanKeypointsAnnotation()
        annotation.parse_tsv(line)
        return annotation


class AIChallengerKeypointsAnnotation(HumanPoseKeypointAnnotation):
    """
    14 keypoints from the AI Challenger dataset
    """

    joints = [
        "top_head",
        "neck",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(self):
        super().__init__(joints=AIChallengerKeypointsAnnotation.joints)


class OxenHumanKeypointsAnnotation(HumanPoseKeypointAnnotation):
    """
    OxenSkeleton has 13 keypoints as a subset of other pose keypoint systems
    """

    joints = [
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(self):
        super().__init__(joints=OxenHumanKeypointsAnnotation.joints)

    def from_nd_array(output):
        annotation = OxenHumanKeypointsAnnotation()
        annotation.parse_heatmap_output(output)
        return annotation

    def from_tsv(line):
        annotation = OxenHumanKeypointsAnnotation()
        annotation.parse_tsv(line)
        return annotation

    def from_keypoints(kps):
        annotation = OxenHumanKeypointsAnnotation()
        annotation.parse_keypoints(kps)
        return annotation

    def from_coco(coco_kps):
        # first five joints (nose, left_eye, left_ear, right_eye, right_ear) in mscoco collapse down to head
        return OxenHumanKeypointsAnnotation._collapse_top_n(coco_kps, 5)

    def from_ai_challenger(coco_kps):
        # first 2 joints in mscoco (head, neck) collapse down to head
        return OxenHumanKeypointsAnnotation._collapse_top_n(coco_kps, 2)

    def _collapse_top_n(coco_kps, n: int):
        is_visible = False
        sum_x = 0.0
        sum_y = 0.0
        total = 0.0
        for i in range(n):
            kp = coco_kps.keypoints[i]
            sum_x += kp.x
            sum_y += kp.y
            if kp.confidence > 0.5:
                total += 1
                is_visible = True

        avg_x = sum_x / float(total) if total > 0 else 0
        avg_y = sum_y / float(total) if total > 0 else 0
        confidence = 1 if is_visible else 0
        oxen_kps = OxenHumanKeypointsAnnotation()
        oxen_kps.keypoints.append(
            OxenImageKeypoint(x=avg_x, y=avg_y, confidence=confidence)
        )
        for kp in coco_kps.keypoints[n:]:
            oxen_kps.keypoints.append(kp)

        assert len(oxen_kps.keypoints) == len(OxenHumanKeypointsAnnotation.joints)

        return oxen_kps


class PersonKeypointsDataset:
    def __init__(self):
        self.annotations = {}

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

    def get_annotations(self, key):
        return self.annotations[key]

    def write_tsv(self, base_img_dir, outfile):
        self.write_output(base_img_dir, outfile, output_type="tsv")

    def write_tsv(self, base_img_dir, outfile):
        self.write_output(base_img_dir, outfile, output_type="json")

    def write_output(
        self, base_img_dir, outfile, one_person_per_image=False, output_type="tsv"
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

                if one_person_per_image and len(file_annotations.annotations) != 1:
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


class TSVKeypointsDataset(PersonKeypointsDataset):
    def __init__(self, annotation_file):
        super().__init__()
        self.annotations = self._load_dataset(annotation_file)

    def _load_dataset(self, annotation_file):
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
                    # a = OxenHumanKeypointsAnnotation.from_tsv(
                    #     delimiter.join(split_line[1:])
                    # )
                    a = CocoHumanKeypointsAnnotation.from_tsv(
                        delimiter.join(split_line[1:])
                    )
                    file_annotations[filename].annotations.append(a)
                except Exception as e:
                    print(f"Err: {e}")
                    print(line)
                    exit()

            return file_annotations


class LeedsKeypointsDataset(PersonKeypointsDataset):
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
        idx_mapping = [13, 11, 9, 8, 10, 12, 7, 5, 3, 2, 4, 6, 1, 0]

        file_annotations = {}
        for e in range(num_examples):
            kps = []
            for j in range(num_joints):
                i = idx_mapping[j]
                kps.append(data[0][i][e])
                kps.append(data[1][i][e])
                kps.append(1.0 if data[2][j][e] == 0.0 else 0.0)
            filename = f"im{str(e+1).zfill(4)}.jpg"
            file_annotation = FileAnnotations(file=filename)
            ai_challenge_kps = AIChallengerKeypointsAnnotation()
            ai_challenge_kps.parse_array(kps)
            oxen_kps = OxenHumanKeypointsAnnotation.from_ai_challenger(ai_challenge_kps)
            file_annotation.add_annotation(oxen_kps)
            file_annotations[filename] = file_annotation
        return file_annotations


class MSCocoKeypointsDataset(PersonKeypointsDataset):
    def __init__(self, annotation_file: str, input_type: str = "mscoco"):
        super().__init__()
        self.annotations = self._load_dataset(annotation_file, input_type)

    def _parse_raw_keypoints(self, raw_kps, input_type: str):
        if "mscoco" == input_type:
            coco_kp = CocoHumanKeypointsAnnotation()
            coco_kp.parse_array(raw_kps)
            kp = OxenHumanKeypointsAnnotation.from_coco(coco_kp)
            return kp
        elif "ai_challenger" == input_type:
            ai_challenger_kp = AIChallengerKeypointsAnnotation()
            ai_challenger_kp.parse_array(raw_kps)
            kp = OxenHumanKeypointsAnnotation.from_ai_challenger(ai_challenger_kp)
            return kp
        coco_kp = CocoHumanKeypointsAnnotation()
        coco_kp.parse_array(raw_kps)
        return coco_kp

    def _load_dataset(self, annotation_file, input_type: str):
        if not os.path.exists(annotation_file):
            raise ValueError("Annotation file not found")

        print(f"Loading dataset from {annotation_file}")
        with open(annotation_file) as json_file:
            data = json.load(json_file)

        # Find all image files and ids
        file_annotations = {}
        for item in data["images"]:
            id = str(item["id"])
            file_annotations[id] = FileAnnotations(file=item["file_name"])

        print(f"Got {len(file_annotations)} image files")

        # Grab all the annotations and organize them by image
        print(f"Parsing {len(data['annotations'])} annotations")
        for item in data["annotations"]:
            category_num = int(item["category_id"])
            iscrowd = "iscrowd" in item and item["iscrowd"] == 1
            # --- Check if category is person not in a crowd
            if category_num == 1 and not iscrowd:
                id = str(item["image_id"])
                raw_kps = item["keypoints"]

                kp = self._parse_raw_keypoints(raw_kps, input_type)

                if kp.is_all_zeros():
                    continue

                file_annotations[id].add_annotation(kp)

        annotations = {}
        for (_id, annotation) in file_annotations.items():
            annotations[annotation.file] = annotation

        return annotations
