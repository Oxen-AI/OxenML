import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from skimage.transform import resize

class Dataloader():
  def __init__(self, data_dir, annotation_file, label_file, image_size=(256, 256), should_load_into_memory=False):
    self.data_dir = data_dir
    self.annotation_file = annotation_file
    self.label_file = label_file
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

  def load(self):
    print(f"Loading data from annotation file {self.annotation_file} and {self.label_file}")

    if not self._load_labels():
      print("Error: Could not load labels")
      return

    if not self._load_annotations():
      print("Error: Could not load annotations")
      return

    self.shuffle()

  def shuffle(self):
    self._example_idx = 0
    self._random_indices = list(range(len(self.inputs)-1))

  def _load_labels(self):
    if not os.path.exists(self.label_file):
      print(f"Label file does not exist {self.label_file}")
      return False

    print("Reading labels...")
    labels = set()
    with open(self.label_file, 'r') as f:
      for line in f:
        label = line.strip()
        labels.add(label)
    
    for label in labels:
      self.labels.append(label)
    
    # since they are in a set, we will have to sort them to get consistent behavior
    self.labels.sort()
    print(f"Got {len(self.labels)} labels")
    for label in self.labels:
      print(label)

    return True

  def _load_annotations(self):
    if not os.path.exists(self.annotation_file):
      print(f"Annotation file does not exist {self.annotation_file}")
      return False

    print(f"Reading annotation file: {self.annotation_file}")
    filenames = []
    labels = []
    with open(self.annotation_file, 'r') as f:
      for line in f:
        split_line = line.strip().split('\t')
        filename = split_line[0]
        label = split_line[1]
        if label not in self.labels:
          print(f"Label not known: {label}")
          return False

        filename = os.path.join(self.data_dir, filename)
        if not os.path.exists(filename):
          print(f"Could not find file {filename}")
          return False

        filenames.append(filename)
        labels.append(label)

    print(f"Loading {len(filenames)} annotations")
    for i in tqdm(range(len(filenames))):
      filename = filenames[i]
      label_idx = self.labels.index(labels[i])

      if self.should_load_into_memory:
        image = Image.open(filename)
        frame = np.asarray(image)
        frame = resize(frame, (self.image_size[0], self.image_size[1]))
        frame = frame.flatten().reshape(1, self.input_length())
        self.inputs.append(frame)
      else:
        self.inputs.append(filename)

      self.outputs.append(label_idx)
    print(f"Done loading {len(self.inputs)}")

    return True

  def get_batch(self, size):
    if self._example_idx + size > len(self.inputs):
      self.shuffle()

    input_batch = np.zeros((size, self.input_length()))
    output_batch = np.zeros((size, len(self.labels)))
    for i in range(size):
      index = self._random_indices[self._example_idx]
      if self.should_load_into_memory:
        input_batch[i] = self.inputs[index]
      else:
        filename = self.inputs[index]
        image = Image.open(filename)
        frame = np.asarray(image)
        frame = resize(frame, (self.image_size[0], self.image_size[1]))
        frame = frame.flatten().reshape(1, self.input_length())
        input_batch[i] = frame

      output_batch[i][self.outputs[index]] = 1
      self._example_idx += 1

    return (input_batch, output_batch)

