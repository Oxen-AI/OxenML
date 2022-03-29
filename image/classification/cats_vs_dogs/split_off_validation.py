
import os
import sys
import shutil
import random

if len(sys.argv) != 4:
  print(f"Usage: {sys.argv[0]} <input_dir> <output_dir> <num_entries>")
  exit()

input_dir = sys.argv[1]
output_dir = sys.argv[2]
num_entries = int(sys.argv[3])

if not os.path.exists(input_dir):
  print(f"Directory does not exist {input_dir}")
  exit()

if not os.path.exists(output_dir):
  os.mkdir(output_dir)


count = 0
files = []
for file in os.listdir(input_dir):
  files.append(file)
random.shuffle(files)

for file in files:
  if count >= num_entries:
    break
  print(file)
  train_file = os.path.join(input_dir, file)
  valid_file = os.path.join(output_dir, file)
  print(f"Moving {train_file} -> {valid_file}")
  shutil.move(train_file, valid_file)
  count += 1

print(f"Moved {count} files")

