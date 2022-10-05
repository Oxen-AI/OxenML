from matplotlib import pyplot as plt
from typing import Optional
import numpy as np

from oxen.image.bounding_box.annotations import OxenBoundingBox
from oxen.image.keypoints.image_keypoint import ImageKeypoint
from oxen.metrics.outcome import PredictionOutcome


class HumanPoseKeypointAnnotation:
    def __init__(self, joints):
        self.joints = joints
        self.keypoints = []

    def __repr__(self):
        return f"<HumanPoseKeypointAnnotation n: {len(self.keypoints)}>"

    def get_joint_keypoint(self, name: str) -> Optional[ImageKeypoint]:
        for (i, joint) in enumerate(self.joints):
            if joint == name:
                return self.keypoints[i]
        return None

    def compute_outcomes(self, prediction, confidence_thresh=0.5, fract_torso=0.2):
        outcomes = []

        bounding_box = OxenBoundingBox.from_human_kp(self)
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
            self.keypoints.append(ImageKeypoint(x=kp.x, y=kp.y, confidence=confidence))

    def parse_array(self, data):
        step = 3
        n_keypoints = int(len(data) / step)
        # Since the last entry is the visibility flag, we discard it.
        for i in range(0, n_keypoints * step, step):
            x = int(data[i])
            y = int(data[i + 1])
            is_vis_row = data[i + 2]
            confidence = (
                1.0 if is_vis_row == 2.0 or is_vis_row == "True" else float(is_vis_row)
            )
            self.keypoints.append(ImageKeypoint(x=x, y=y, confidence=confidence))

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

            self.keypoints.append(ImageKeypoint(x=x, y=y, confidence=confidence))

    def parse_json(self, data):
        raise NotImplementedError()

    def to_tsv(self):
        return "\t".join(
            [f"{k.x}\t{k.y}\t{'{:.2f}'.format(k.confidence)}" for k in self.keypoints]
        )
