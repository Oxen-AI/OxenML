
import os
import sys
import shutil

if len(sys.argv) != 4:
  print(f"Usage: {sys.argv[0]} <input_dir> <train_dir> <num_train>")
  exit()

input_dir = sys.argv[1]
train_dir = sys.argv[2]
num_train = int(sys.argv[3])

if not os.path.exists(input_dir):
  print(f"Directory does not exist {input_dir}")
  exit()

if not os.path.exists(train_dir):
  os.makedirs(train_dir)

src_files = []
for filename in os.listdir(input_dir):
  src_file = os.path.join(input_dir, filename)
  # checking if it is a file
  if os.path.isfile(src_file) and "_" in filename and ".mp4" in filename:
    src_files.append(src_file)

# Sort so we always copy in the same order
src_files.sort()

train_counts = {}
for src_file in src_files:
  filename = os.path.basename(src_file)
  category = filename.split("_")[0]

  if not category in train_counts:
    train_counts[category] = 0

  train_count = train_counts[category]

  if train_count < num_train:
    dst_file = os.path.join(train_dir, filename)
    shutil.copyfile(src_file, dst_file)
    train_counts[category] += 1

print(f"Train counts: {train_counts}")
print("Done.")
