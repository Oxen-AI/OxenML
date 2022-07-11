
import sys, os
from tensorflow import keras
from PIL import Image
import numpy as np
from skimage.transform import resize
from data_loader import Dataloader
import simplejson as json
from tqdm import tqdm

if len(sys.argv) != 4:
  print(f"Usage {sys.argv[0]} <model> <annotation-file> <label-file>")
  exit()

model_dir = sys.argv[1]
annotations_file = sys.argv[2]
labels_file = sys.argv[3]

print(f"Loading model... {model_dir}")
model = keras.models.load_model(model_dir)

hyper_param_file = os.path.join(model_dir, "params.json")
params = {}
with open(hyper_param_file, 'r') as f:
  params = json.load(f)

img_size = params['image_size']
dataloader = Dataloader(image_size=(img_size, img_size))
dataloader.load_labels(labels_file)
dataloader.load_annotations(annotations_file, shuffle=False)

threshold = 0.5
total = dataloader.num_examples()
tps = {} # true positive, correct, and confidence is over threshold
fps = {} # false positive, incorrect, and above confidence threshold
tns = {} # true negative, incorrect, but below confidence threshold
fns = {} # false negative, correct, but below confidence threshol
print(f"Validating {total} examples")
for index in tqdm(range(total)):
  (i, o) = dataloader.get_batch(1)
  outputs = model(i)
  predicted_index = np.argmax(outputs[0])
  print(f"[{index}/{total}] outputs {outputs}\npredicted_index {predicted_index}")
  prob = outputs[0][predicted_index]
  correct_index = np.argmax(o)

  guessed_label = dataloader.label_from_idx(predicted_index)
  correct_label = dataloader.label_from_idx(correct_index)
  print(f"{guessed_label} == {correct_label} {prob}%")

  # true positive
  if correct_index == predicted_index and prob > threshold:
    print(f"TP {correct_label}")
    if correct_label not in tps:
      tps[correct_label] = 0
    tps[correct_label] += 1

  # false positive
  if correct_index != predicted_index and prob > threshold:
    print(f"FP {correct_label}")
    if correct_label not in fps:
      fps[correct_label] = 0
    fps[correct_label] += 1

  # true negative
  if correct_index != predicted_index and prob < threshold:
    print(f"TN {correct_label}")
    if correct_label not in tns:
      tns[correct_label] = 0
    tns[correct_label] += 1

  # false negative
  if correct_index == predicted_index and prob < threshold:
    print(f"FN {correct_label}")
    if correct_label not in fns:
      fns[correct_label] = 0
    fns[correct_label] += 1
  
  print(f"tps: {tps}")
  print(f"fps: {fps}")
  print(f"tns: {tns}")
  print(f"fns: {fns}")
  
  
print("Categories")
for label in dataloader.labels:
  tp = tps[label] if label in tps else 0.0
  fp = fps[label] if label in fps else 0.0
  tn = tns[label] if label in tns else 0.0
  fn = fns[label] if label in fns else 0.0

  accuracy = 0.0 if (tp == 0.0 and fp == 0.0 and tn == 0.0 and fn == 0.0) else (tp + tn) / (tp + fp + tn + fn)
  precision = 0.0 if (tp == 0.0 and fp == 0.0) else tp / (tp + fp)
  recall =  0.0 if (tp == 0.0 and fn == 0.0) else tp / (tp + fn)
  
  print(f"---- {label} ----")
  print(f"Accuracy = ({tp} + {tn}) / ({tp} + {fp} + {tn} + {fn}) = {accuracy}")
  print(f"Precision = {tp} / ({tp} + {fp}) = {precision}")
  print(f"Recall = {tp} / ({tp} + {fn}) = {recall}")



