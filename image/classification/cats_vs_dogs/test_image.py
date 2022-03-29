
import sys, os
from tensorflow import keras
from PIL import Image
import numpy as np
from skimage.transform import resize
from data_loader import Dataloader
import simplejson as json

if len(sys.argv) != 4:
  print(f"Usage {sys.argv[0]} <model> <labels-file> <img-file>")
  exit()

model_dir = sys.argv[1]
labels_file = sys.argv[2]
image_file = sys.argv[3]

print(f"Loading model... {model_dir}")
model = keras.models.load_model(model_dir)

hyper_param_file = os.path.join(model_dir, "params.json")
params = {}
with open(hyper_param_file, 'r') as f:
  params = json.load(f)

img_size = params['image_size']
image = Image.open(image_file)
frame = np.asarray(image)
frame = resize(frame, (img_size, img_size))
frame = frame.reshape((1, img_size, img_size, 3))

outputs = model(frame)
index = np.argmax(outputs[0])
print(f"outputs {outputs}\nindex {index}")

dataloader = Dataloader()
dataloader.load_labels(labels_file)

prob = outputs[0][index] * 100
label = dataloader.label_from_idx(index)
print(f"({label}) {prob}%")
