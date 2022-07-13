
import sys, os
import cv2
from alive_progress import alive_bar

def load_files(directory):
  files = []
  for filename in os.listdir(directory):
    src_file = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(src_file) and "_" in filename and ".mp4" in filename:
      files.append(src_file)
  return files

def take_first_frame(file, output_dir, label, video_id):
  filenames = []
  # print(f"Got video file: {f}")
  video = cv2.VideoCapture(file)
  success = 1
  frame = 0

  while success:
    success, image = video.read()
    if success:
      output_name = f"{label}_{video_id}_{frame}.jpg"
      output_file = os.path.join(output_dir, output_name)
      
      # print(f"Saving image: {output_file}")

      # Saves the frames with frame-count
      cv2.imwrite(output_file, image)
      filenames.append(output_file)
      frame += 1

      break
  return filenames

def take_all_frames(file, output_dir, label, video_id):
  filenames = []
  # print(f"Got video file: {f}")
  video = cv2.VideoCapture(file)
  success = 1
  frame = 0

  while success:
    success, image = video.read()
    if success:
      output_name = f"{label}_{video_id}_{frame}.jpg"
      output_file = os.path.join(output_dir, output_name)
      
      # print(f"Saving image: {output_file}")

      # Saves the frames with frame-count
      cv2.imwrite(output_file, image)
      filenames.append(output_file)
      frame += 1

  return filenames

def take_first_middle_last_frames(file, output_dir, label, video_id):
  filenames = []
  # print(f"Got video file: {f}")
  video = cv2.VideoCapture(file)
  success = 1
  frame = 0

  frames = []
  while success:
    success, image = video.read()
    if success:
      output_name = f"{label}_{video_id}_{frame}.jpg"
      output_file = os.path.join(output_dir, output_name)

      filenames.append(output_file)
      frames.append(image)
      frame += 1

  total_frames = len(frames)
  halfway = int(total_frames / 2)
  indices = [0, halfway, total_frames - 1]
  out_filenames = []
  for i in indices:
    output_file = filenames[i]
    image = frames[i]
    cv2.imwrite(output_file, image)
    out_filenames = [output_file]
  
  return output_file

if len(sys.argv) != 3:
  print(f"Usage: {sys.argv[0]} <input_dir> <first,first_mid_last,all>")
  exit()

input_dir = sys.argv[1]
which_frames = sys.argv[2]

video_train_dir = os.path.join(input_dir, "train")
video_test_dir = os.path.join(input_dir, "test")

images_dir = os.path.join(input_dir, "images")

image_train_dir = os.path.join(images_dir, "train")
image_test_dir = os.path.join(images_dir, "test")

annotations_dir = os.path.join(images_dir, "annotations")
labels_dir = os.path.join(images_dir, "labels")

if not os.path.exists(image_train_dir):
  os.makedirs(image_train_dir)
  
if not os.path.exists(image_test_dir):
  os.makedirs(image_test_dir)

if not os.path.exists(annotations_dir):
  os.makedirs(annotations_dir)
  
if not os.path.exists(labels_dir):
  os.makedirs(labels_dir)

train_annotations_file = os.path.join(annotations_dir, "train_annotations.txt")
test_annotations_file = os.path.join(annotations_dir, "test_annotations.txt")
output_labels_file = os.path.join(labels_dir, "labels.txt")

train_files = load_files(video_train_dir)
test_files = load_files(video_test_dir)

num_videos = len(train_files) + len(test_files)
print(f"Got {num_videos} videos")

labels = set()
video_files = [train_files, test_files]
output_dirs = [image_train_dir, image_test_dir]
filenames = []
key_counts = {}
with alive_bar(num_videos, title=f'Processing videos') as bar:
  for (i, files) in enumerate(video_files):
    output_dir = output_dirs[i]
    filenames.append([])
    for file in files:
      bar()

      basename = os.path.basename(file)
      label = basename.split("_")[0]
      video_id = basename.split(".")[0].split("_")[1]

      labels.add(label)

      if "first" == which_frames:
        filenames[i].extend(take_first_frame(file, output_dir, label, video_id))
      elif "first_mid_last" == which_frames:
        filenames[i].extend(take_first_middle_last_frames(file, output_dir, label, video_id))
      elif "all" == which_frames:
        filenames[i].extend(take_all_frames(file, output_dir, label, video_id))

print(f"Saving annotations")

annotation_files = [train_annotations_file, test_annotations_file]

for (i, files) in enumerate(filenames):
  with open(annotation_files[i], 'w') as f:
    for filename in files:
      basename = os.path.basename(filename)
      key = basename.split("_")[0]
      line = f"{filename}\t{key}"
      print(line)
      f.write(line)
      f.write("\n")
      # break


with open(output_labels_file, 'w') as f:
  for key in labels:
    f.write(key)
    f.write("\n")

print("Done.")
