import os
from signal import valid_signals
from tqdm import tqdm
from PIL import Image
import numpy as np
from skimage.transform import resize
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
from matplotlib import pyplot as plt
import random
import logging
import math
from util import visualize_keypoints

class Dataloader():
  def __init__(self, image_dir, aug, image_size, num_keypoints, should_load_into_memory=False):
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
    filenames = []
    keypoints = []
    # First gather filenames and keypoints, so we can do a nice progress bar
    with open(filename, 'r') as f:
      for line in f:
        split_line = line.strip().split('\t')
        filename = split_line[0]
        
        # skip top row in tsv
        if filename == "filename":
          continue

        output = split_line[1:]

        fullpath = os.path.join(self.image_dir, filename)
        if not os.path.exists(fullpath):
          logging.error(f"Could not find file {fullpath}")
          return False

        filenames.append(fullpath)
        keypoints.append(output)

        if total > 0 and len(filenames) > total:
          break

    # Then load the data in correct format, either into memory or just file pointers
    logging.info(f"Loading {len(filenames)} annotations")
    for i in tqdm(range(len(filenames))):
      filename = filenames[i]
      label_idx = np.array(keypoints[i])

      if self.should_load_into_memory:
        frame = self.read_image_from_disk(filename)
        self.inputs.append(frame)
      else:
        self.inputs.append(filename)

      self.outputs.append(label_idx)

    logging.info(f"Done loading {len(self.inputs)}")
    if shuffle:
      # Shuffle at the start to make sure we are ready to rock
      self.shuffle()
    else:
      self.reset()
    return True

  def get_batch(self, size, show_images=False):
    input_batch = np.zeros((size, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
    output_batch = np.empty(
      (size, self.image_shape[0], self.image_shape[1], self.n_keypoints), dtype="float32"
    )

    debug_images = []
    debug_kps = []

    for batch_i in range(size):
      index = self._random_indices[self._example_idx]
      
      # Read the current image not resized
      frame = self.current_image(index)

      # line is formatted with x,y,visible for each joint, so need to skip by 3s
      line = self.outputs[index]

      kps = []
      visible = []
      step = 3
      # To apply our data augmentation pipeline, we first need to
      # form Keypoint objects with the original coordinates.
      for i in range(0, self.n_keypoints*step, step):
        kps.append(Keypoint(x=float(line[i]), y=float(line[i+1])))
        visible.append(line[i+2] == '2' or line[i+2] == 'True')

      # TODO use dataloader from OxenDatasets
      # We then project the original image and its keypoint coordinates.
      kps_obj = KeypointsOnImage(kps, shape=frame.shape)
      (new_image, new_kps_obj) = self.aug(image=frame, keypoints=kps_obj)

      if show_images:
        debug_images.append(new_image)
        debug_kps.append(new_kps_obj)

      # logging.info(f"Frame [{index}] shape {frame.shape}")
      # logging.info(f"Image [{index}] shape {new_image.shape}")
      # logging.info(f"Index [{batch_i}] Image batch shape {input_batch.shape}")
      input_batch[batch_i] = new_image

      # Parse the coordinates from the new keypoint object.
      width = self.image_shape[0]
      height = self.image_shape[1]
      sigma = 6.0
      output = np.zeros((width, height, self.n_keypoints), dtype=np.float32)
      for i, keypoint in enumerate(new_kps_obj):
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
              output[x][y][i] = max(output[x][y][i], math.exp(-exp))
              output[x][y][i] = min(output[x][y][i], 1.0)
        # heatmap = output[:,:,i]
        # print(f"showing heatmap [{i}]")
        # plt.imshow(heatmap, interpolation='nearest')
        # plt.show()


      # Reshape to be (1, 1, n_keypoints * 2) for x,y
      output_batch[batch_i,] = output

      self._example_idx += 1

    if len(debug_images) > 0:
      visualize_keypoints(debug_images, debug_kps)

    return (input_batch, output_batch)

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

