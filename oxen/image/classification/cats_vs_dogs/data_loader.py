import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from skimage.transform import resize
import random

class Dataloader():
  def __init__(self, image_size=(256, 256), should_load_into_memory=False):
    self.image_size = image_size
    self.n_channels = 3
    self.should_load_into_memory = should_load_into_memory
    self.labels = []
    self.inputs = []
    self.outputs = []
    self._random_indices = []
    self._example_idx = 0

  def input_shape(self):
    return (self.image_size[0], self.image_size[1], self.n_channels)

  def input_length(self):
    return self.image_size[0] * self.image_size[1] * self.n_channels

  def num_outputs(self):
    return len(self.labels)

  def num_examples(self):
    return len(self.inputs)

  def label_from_idx(self, idx):
    return self.labels[idx]

  def reset(self):
    self._example_idx = 0
    self._random_indices = list(range(len(self.inputs)))

  def shuffle(self):
    self.reset()
    random.shuffle(self._random_indices)
    
  def load_labels(self, filename):
    if not os.path.exists(filename):
      print(f"Label file does not exist {filename}")
      return False

    print("Reading labels...")
    labels = set()
    with open(filename, 'r') as f:
      for line in f:
        label = line.strip()
        labels.add(label)
    
    for label in labels:
      self.labels.append(label)
    
    # Since they are in a set, we will have to sort them to get consistent behavior
    self.labels.sort()
    print(f"Got {len(self.labels)} labels")
    for label in self.labels:
      print(label)

    return True

  def load_annotations(self, filename, total=-1, shuffle=True):
    if not os.path.exists(filename):
      print(f"Annotation file does not exist {filename}")
      return False

    if len(self.labels) == 0:
      print(f"Must call dataloader.load_labels() before dataloader.load_annotations()")
      return False

    print(f"Reading annotation file: {filename}")
    filenames = []
    labels = []
    # First gather labels and filenames, so we can do a nice progress bar
    with open(filename, 'r') as f:
      for line in f:
        split_line = line.strip().split('\t')
        filename = split_line[0]
        label = split_line[1]
        if label not in self.labels:
          print(f"Label not known: {label}")
          return False

        if not os.path.exists(filename):
          print(f"Could not find file {filename}")
          return False

        filenames.append(filename)
        labels.append(label)

        if total > 0 and len(filenames) > total:
          break

    # Then load the data in correct format, either into memory or just file pointers
    print(f"Loading {len(filenames)} annotations")
    for i in tqdm(range(len(filenames))):
      filename = filenames[i]
      label_idx = self.labels.index(labels[i])

      if self.should_load_into_memory:
        frame = self.read_image_from_disk(filename)
        self.inputs.append(frame)
      else:
        self.inputs.append(filename)

      self.outputs.append(label_idx)

    print(f"Done loading {len(self.inputs)}")
    if shuffle:
      # Shuffle at the start to make sure we are ready to rock
      self.shuffle()
    else:
      self.reset()
    return True

  def get_batch(self, size):
    input_batch = np.zeros((size, self.image_size[0], self.image_size[1], self.n_channels))
    output_batch = np.zeros((size, len(self.labels)))
    for i in range(size):
      index = self._random_indices[self._example_idx]
      if self.should_load_into_memory:
        input_batch[i] = self.inputs[index]
      else:
        filename = self.inputs[index]
        frame = self.read_image_from_disk(filename)
        input_batch[i] = frame

      output_batch[i][self.outputs[index]] = 1
      self._example_idx += 1

    return (input_batch, output_batch)

  def read_image_from_disk(self, filename):
    image = Image.open(filename)
    frame = np.asarray(image)
    return resize(frame, (self.image_size[0], self.image_size[1]))

