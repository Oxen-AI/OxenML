
import os
import sys

if len(sys.argv) != 3:
  print(f"Usage: {sys.argv[0]} <data_dir> <output.txt>")
  exit()

input_dir = sys.argv[1]
output_file = sys.argv[2]

if not os.path.exists(input_dir):
  print(f"Directory does not exist {input_dir}")
  exit()

count = 0
with open(output_file, 'w') as output:
  print(f"Writing to {output_file}")
  for file in os.listdir(input_dir):
    label = file.split(".")[0].split("_")[0]
    path = os.path.join(input_dir, file)  
    line = f"{path}\t{label}"
    output.write(line)
    output.write("\n")
    count += 1

print(f"Wrote {count} entries")

