
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
dataloader.load_annotations(annotations_file)

num_correct = 0
total = dataloader.num_examples()
category_counts = {}
guess_counts = {}
correct_counts = {}
print(f"Validating {total} examples")
for index in tqdm(range(total)):
  (i, o) = dataloader.get_batch(1)
  outputs = model(i)
  predicted_index = np.argmax(outputs[0])
  # print(f"[{index}/{total}] outputs {outputs}\npredicted_index {predicted_index}")
  prob = outputs[0][predicted_index] * 100
  guessed_label = dataloader.label_from_idx(predicted_index)
  # print(f"({label}) {prob}%")
  # print(f"{o}")

  correct_index = np.argmax(o)
  if correct_index == predicted_index:
    if guessed_label not in correct_counts:
      correct_counts[guessed_label] = 0
    correct_counts[guessed_label] += 1
    num_correct += 1

  # Sum up counts
  correct_label = dataloader.label_from_idx(correct_index)
  if correct_label not in category_counts:
    category_counts[correct_label] = 0
  category_counts[correct_label] += 1
  
  if guessed_label not in guess_counts:
    guess_counts[guessed_label] = 0
  guess_counts[guessed_label] += 1
  
  
print("Category Counts:")
for (k,v) in category_counts.items():
  guess_count = 0
  if k in guess_counts:
    guess_count = guess_counts[k]

  correct_count = 0
  if k in correct_counts:
    correct_count = correct_counts[k]

  precision = float(correct_count) / float(v)
  # recall = float(guess_count) / float(v)
  
  print(f"{k} Precision = {correct_count} / {v} = {precision}")
  # print(f"{k} Recall = {guess_count} / {v} = {recall}")

accuracy = float(num_correct) / float(total)
print(f"Accuracy = {num_correct} / {total} = {accuracy}")


