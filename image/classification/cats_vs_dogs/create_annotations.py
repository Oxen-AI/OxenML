
import os
import sys

if len(sys.argv) != 4:
  print(f"Usage: {sys.argv[0]} <data_dir> <file_prefix> <output.txt>")
  exit()

input_dir = sys.argv[1]
prefix = sys.argv[2]
output_file = sys.argv[3]

if not os.path.exists(input_dir):
  print(f"Directory does not exist {input_dir}")
  exit()

count = 0
with open(output_file, 'w') as output:
  print(f"Writing to {output_file}")
  for file in os.listdir(input_dir):
    label = file.split(".")[0]
    line = f"{prefix}/{file}\t{label}"
    output.write(line)
    output.write("\n")
    count += 1

print(f"Wrote {count} entries")

