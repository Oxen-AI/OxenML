import os
import json
import jsonpickle
import numpy as np
from matplotlib import pyplot as plt


class OxenImageKeypoint:
    def __init__(self, x, y, confidence=0.0):
        self.x = x
        self.y = y
        self.confidence = confidence

    def __repr__(self):
        return f"<OxenImageKeypoint x: {self.x}, y: {self.y} confidence: {self.confidence}>"

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


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


class HumanPoseKeypoints:
    def __init__(self, joints):
        self.joints = joints
        self.keypoints = []

    def __repr__(self):
        return f"<HumanPoseKeypoints n: {len(self.keypoints)}>"

    def plot_image_file(self, image_file, color="#FF0000", threshold=0.5):
        frame = plt.imread(image_file)
        self.plot_frame(frame, color=color, threshold=threshold)

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

    def parse_array(self, data):
        step = 3
        n_keypoints = int(len(data) / step)
        # Since the last entry is the visibility flag, we discard it.
        for i in range(0, n_keypoints * step, step):
            x = float(data[i])
            y = float(data[i + 1])
            is_vis_row = data[i + 2]
            confidence = 1.0 if is_vis_row == 2.0 or is_vis_row == "True" else 0.0
            self.keypoints.append(OxenImageKeypoint(x=x, y=y, confidence=confidence))

    def parse_heatmap_output(self, outputs):
        n_keypoints = outputs.shape[2]
        for i in range(n_keypoints):
            heatmap = np.array(outputs[:, :, i])

            x = np.argmax(np.amax(heatmap, axis=1))
            y = np.argmax(np.amax(heatmap, axis=0))
            confidence = heatmap[x][y]

            self.keypoints.append(OxenImageKeypoint(x=x, y=y, confidence=confidence))

    def parse_json(self, data):
        raise NotImplementedError()

    def to_tsv(self):
        return "\t".join(
            [f"{k.x}\t{k.y}\t{'{:.2f}'.format(k.confidence)}" for k in self.keypoints]
        )


class CocoHumanKeypoints(HumanPoseKeypoints):
    """
    CocoSkeleton has the 17 keypoints from the MSCoco dataset
    """

    def __init__(self):
        super().__init__(
            joints=[
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
        )


class OxenHumanKeypointsAnnotation(HumanPoseKeypoints):
    """
    OxenSkeleton has 13 keypoints as a subset of other pose keypoint systems
    """

    def __init__(self):
        super().__init__(
            joints=[
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
        )

    def from_nd_array(output):
        annotation = OxenHumanKeypointsAnnotation()
        annotation.parse_heatmap_output(output)
        return annotation

    def from_tsv(line):
        annotation = OxenHumanKeypointsAnnotation()
        annotation.parse_tsv(line)
        return annotation

    def from_coco(coco_kps):
        is_visible = False
        sum_x = 0.0
        sum_y = 0.0
        total = 0.0
        # first five joints in mscoco collapse down to head
        num_to_sum = 5
        for i in range(num_to_sum):
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
        for kp in coco_kps.keypoints[num_to_sum:]:
            oxen_kps.keypoints.append(kp)

        return oxen_kps


class PersonKeypointsDataset:
    def __init__(self):
        self.annotations = []

    def list_annotations(self):
        return self.annotations

    def write_tsv(self, base_img_dir, outfile):
        self.write_output(base_img_dir, outfile, output_type="tsv")

    def write_tsv(self, base_img_dir, outfile):
        self.write_output(base_img_dir, outfile, output_type="json")

    def write_output(
        self, base_img_dir, outfile, one_person_per_image=False, output_type="tsv"
    ):
        print(f"Writing {len(self.annotations)} annotations to {outfile}")
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

                a = OxenHumanKeypointsAnnotation.from_tsv(
                    delimiter.join(split_line[1:])
                )
                file_annotations[filename].annotations.append(a)

            annotations = []
            for (_id, annotation) in file_annotations.items():
                annotations.append(annotation)
            return annotations


class MSCocoKeypointsDataset(PersonKeypointsDataset):
    def __init__(self, annotation_file, collapse_head=False):
        super().__init__()
        self.annotations = self._load_dataset(annotation_file, collapse_head)

    def _load_dataset(self, annotation_file, collapse_head):
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
            iscrowd = item["iscrowd"] == 1
            # --- Check if category is person not in a crowd
            if category_num == 1 and not iscrowd:
                id = str(item["image_id"])
                raw_kps = item["keypoints"]

                coco_kp = CocoHumanKeypoints()
                coco_kp.parse_array(raw_kps)

                if coco_kp.is_all_zeros():
                    continue

                if collapse_head:
                    file_annotations[id].add_annotation(
                        OxenHumanKeypointsAnnotation.from_coco(coco_kp)
                    )
                else:
                    file_annotations[id].add_annotation(coco_kp)

        annotations = []
        for (_id, annotation) in file_annotations.iter():
            annotations.append(annotation)

        return annotations
