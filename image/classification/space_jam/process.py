
import sys, os
import cv2
import json
from alive_progress import alive_bar


if len(sys.argv) != 3:
  print(f"Usage: {sys.argv[0]} <data-dir> <output-dir>")
  exit()

data_dir = sys.argv[1]
output_dir = sys.argv[2]
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

annotations_file = os.path.join(data_dir, "annotation_dict.json")
labels_file = os.path.join(data_dir, "labels_dict.json")
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

key_counts = {}
with alive_bar(num_videos, title=f'Processing videos') as bar:
    for key in annotations.keys():
        value = annotations[key]
        filename = f"examples/{key}.mp4"
        f = os.path.join(data_dir, filename)

        # print(f"looking up value: {value}")
        classification = labels[f"{value}"]
        # print(f"got class: {classification}")

        # print(f"Got video file: {f}")
        video = cv2.VideoCapture(f)
        success = 1

        while success:
            success, image = video.read()
            
            if success:
                if not classification in key_counts:
                    key_counts[classification] = 0
                key_counts[classification] += 1

                count = key_counts[classification]
                output_name = f"{classification}_{count}.jpg"
                output_file = os.path.join(train_dir, output_name)

                # print(f"Saving image: {output_file}")

                # Saves the frames with frame-count
                cv2.imwrite(output_file, image)
                break

        bar()
        # if len(key_counts) == 10:
        #     break

print(f"Saving annotations")

train_annotations = []

with open(output_annotations_file, 'w') as f:
    for key in key_counts.keys():
        num_examples = key_counts[key]

        for i in range(num_examples):
            filename = f"{key}_{i+1}.jpg"
            train_filename = os.path.join(train_dir, filename)

            f.write(f"{train_filename}\t{key}")
            f.write("\n")


with open(output_labels_file, 'w') as f:
    for key in reverse_labels.keys():
        f.write(key)
        f.write("\n")


print("Done.")
