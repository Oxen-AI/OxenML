


import sys, os
import shutil
import json
from alive_progress import alive_bar

if len(sys.argv) != 3:
  print(f"Usage: {sys.argv[0]} <data-dir> <output-dir>")
  exit()

data_dir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

annotations_file = os.path.join(data_dir, "annotation_dict.json")
labels_file = os.path.join(data_dir, "labels_subset.json")
output_annotations_file = os.path.join(output_dir, "annotations.txt")
output_labels_file = os.path.join(output_dir, "labels.txt")

print(f"Reading annotations: {annotations_file}")
annotations = {}
with open(annotations_file) as f:
  annotations = json.load(f)

print(f"Got {len(annotations)} annotations")

print(f"Reading labels: {labels_file}")
labels = {}
with open(labels_file) as f:
  labels = json.load(f)
print(f"got labels: {labels}")

reverse_labels = {}
for key in labels.keys():
  val = labels[key]
  reverse_labels[val] = key

video_files = []
num_videos = len(annotations)
filenames = []
key_counts = {}
with alive_bar(num_videos, title=f'Processing videos') as bar:
  for key in annotations.keys():
    value = f"{annotations[key]}"
    filename = f"examples/{key}.mp4"
    src_file = os.path.join(data_dir, filename)
    bar()

    if not value in labels:
        continue

    # print(f"looking up value: {value}")
    classification = labels[value]
    # print(f"got class: {classification}")

    if not classification in key_counts:
        key_counts[classification] = 0
    key_counts[classification] += 1

    count = key_counts[classification]

    filename = f"{classification}_{count}.mp4"
    dst_file = os.path.join(output_dir, filename)
    shutil.copyfile(src_file, dst_file)

print("Done.")
