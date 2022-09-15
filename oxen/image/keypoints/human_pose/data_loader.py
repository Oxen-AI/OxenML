import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
from matplotlib import pyplot as plt
import random
import logging
import math

from oxen.image.keypoints.human_pose.ms_coco_dataset import TSVKeypointsDataset

class Dataloader:
    def __init__(
        self, image_dir, aug, image_size, num_keypoints, should_load_into_memory=False
    ):
        self.image_dir = image_dir
        self.aug = aug
        self.n_channels = 3
        self.image_shape = (image_size, image_size, self.n_channels)
        self.n_keypoints = num_keypoints
        self.should_load_into_memory = should_load_into_memory
        self.inputs = []
        self.outputs = []
        self._random_indices = []
        self._example_idx = 0

    def input_shape(self):
        return (self.image_size[0], self.image_size[1], self.n_channels)

    def input_length(self):
        return self.image_size[0] * self.image_size[1] * self.n_channels

    def num_outputs(self):
        return self.n_keypoints

    def num_examples(self):
        return len(self.inputs)

    def reset(self):
        self._example_idx = 0
        self._random_indices = list(range(len(self.inputs)))

    def shuffle(self):
        self.reset()
        random.shuffle(self._random_indices)

    def load_annotations(self, filename, total=-1, shuffle=True):
        if not os.path.exists(filename):
            logging.error(f"Annotation file does not exist {filename}")
            return False

        logging.info(f"Reading annotation file: {filename}")
        paths = []
        keypoints = []
        # First gather filenames and keypoints, so we can do a nice progress bar
        dataset = TSVKeypointsDataset(annotation_file=filename)
        for filename in dataset.list_inputs():
            path = os.path.join(self.image_dir, filename)
            if not os.path.exists():
                raise Exception(f"Training image not found: {path}")
            
            for annotation in dataset.get_annotations(filename):
                paths.append(path)
                keypoints.append(annotation.keypoints)

        # Then load the data in correct format, either into memory or just file pointers
        logging.info(f"Loading {len(paths)} annotations")
        for i in tqdm(range(len(paths))):
            filename = paths[i]
            outputs = np.array(keypoints[i])

            if self.should_load_into_memory:
                frame = self.read_image_from_disk(filename)
                self.inputs.append(frame)
            else:
                self.inputs.append(filename)

            self.outputs.append(outputs)

        logging.info(f"Done loading {len(self.inputs)}")
        if shuffle:
            # Shuffle at the start to make sure we are ready to rock
            self.shuffle()
        else:
            self.reset()
        return True

    def save_inputs_outputs(self, inputs, outputs, predictions, filename="output.png"):
        num_rows = inputs.shape[0]
        _, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(8, 6))
        [ax.axis("off") for ax in np.ravel(axes)]

        for i in range(num_rows):
            input = inputs[i]
            output = outputs[i]
            pred = predictions[i]
            num_channels = float(output.shape[2])
            input_heatmap = np.sum(output, axis=2) / num_channels
            output_heatmap = (np.sum(pred, axis=2) / num_channels) * 255.0

            axes[i, 0].imshow(input)
            axes[i, 1].imshow(input_heatmap, interpolation="nearest")
            axes[i, 2].imshow(output_heatmap, interpolation="nearest")

        plt.savefig(filename)

    def get_batch(self, size):
        input_batch = np.zeros(
            (size, self.image_shape[0], self.image_shape[1], self.image_shape[2]),
            dtype="uint8",
        )
        output_batch = np.empty(
            (size, self.image_shape[0], self.image_shape[1], self.n_keypoints),
            dtype="float32",
        )
        for batch_i in range(size):
            index = self._random_indices[self._example_idx]

            # Read the current image not resized
            frame = self.current_image(index)

            kps = []
            visible = []
            # To apply our data augmentation pipeline, we first need to
            # form Keypoint objects with the original coordinates.
            for k in self.outputs[index]:
                kps.append(Keypoint(x=k.x, y=k.y))
                visible.append(k.confidence > 0.5)

            # We then project the original image and its keypoint coordinates.
            kps_obj = KeypointsOnImage(kps, shape=frame.shape)
            (new_image, new_kps_obj) = self.aug(image=frame, keypoints=kps_obj)

            # logging.info(f"Frame [{index}] shape {frame.shape}")
            # logging.info(f"Image [{index}] shape {new_image.shape}")
            # logging.info(f"Index [{batch_i}] Image batch shape {input_batch.shape}")
            input_batch[batch_i] = new_image

            # Parse the coordinates from the new keypoint object.
            output = self._generate_output_heatmap(new_kps_obj, visible)

            # Reshape to be (1, 1, n_keypoints * 2) for x,y
            output_batch[
                batch_i,
            ] = output

            self._example_idx += 1

        return (input_batch, output_batch)
    
    def _generate_output_heatmap(
        self,
        kps_obj,
        visible: list[bool]
    ):
        width = self.image_shape[0]
        height = self.image_shape[1]
        sigma = 6.0
        output = np.zeros((width, height, self.n_keypoints), dtype=np.float32)
        for i, keypoint in enumerate(kps_obj):
            if not visible[i]:
                continue

            center_x = int(keypoint.x)
            center_y = int(keypoint.y)
            # print(f"center is {center_x},{center_y}")

            if center_x < output.shape[0] and center_y < output.shape[1]:
                output[center_x][center_y][i] = 1

            th = 1.6052
            delta = math.sqrt(th * 2)

            x0 = int(max(0, center_x - delta * sigma))
            y0 = int(max(0, center_y - delta * sigma))

            x1 = int(min(width, center_x + delta * sigma))
            y1 = int(min(height, center_y + delta * sigma))

            # gaussian filter
            for y in range(y0, y1):
                for x in range(x0, x1):
                    d = (x - center_x) ** 2 + (y - center_y) ** 2
                    exp = d / 2.0 / sigma / sigma
                    if exp > th:
                        continue

                    if x < output.shape[0] and y < output.shape[1]:
                        output[y][x][i] = max(output[y][x][i], math.exp(-exp))
                        output[y][x][i] = min(output[y][x][i], 1.0)
        return output

    def current_image(self, index):
        if self.should_load_into_memory:
            return self.inputs[index]
        else:
            filename = self.inputs[index]
            return self.read_image_from_disk(filename)

    def read_image_from_disk(self, filename):
        frame = plt.imread(filename)
        # If the image is RGBA convert it to RGB.
        if frame.shape[-1] == 4:
            frame = frame.astype(np.uint8)
            frame = Image.fromarray(frame)
            frame = np.array(frame.convert("RGB"))
        return frame
