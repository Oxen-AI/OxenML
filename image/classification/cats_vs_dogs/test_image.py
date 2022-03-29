
import sys, os
from tensorflow import keras
from PIL import Image
import numpy as np
from skimage.transform import resize
from data_loader import Dataloader

if len(sys.argv) != 4:
  print(f"Usage {sys.argv[0]} <model> <data-dir> <img-file>")
  exit()

model_file = sys.argv[1]
data_dir = sys.argv[2]
image_file = sys.argv[3]

print(f"Loading model... {model_file}")
model = keras.models.load_model(model_file)

# TODO: Save hyperparameter file for stuff like this...
img_size = 256
image = Image.open(image_file)
frame = np.asarray(image)
frame = resize(frame, (256, 256))
frame = frame.reshape((1, 256, 256, 3))

outputs = model(frame)
index = np.argmax(outputs[0])
print(f"outputs {outputs}\nindex {index}")


annotations_file = os.path.join(os.path.join(data_dir, "annotations"), "annotations.txt")
labels_file = os.path.join(os.path.join(data_dir, "labels"), "labels.txt")

dataloader = Dataloader(
  data_dir=data_dir,
  annotation_file=annotations_file,
  label_file=labels_file,
  should_load_into_memory=False
)
dataloader.load()

prob = outputs[0][index] * 100
label = dataloader.label_from_idx(index)
print(f"({label}) {prob}%")
