import os
import sys
import shutil

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <annotations.txt> <output_dir>")
    exit()

annotations_file = sys.argv[1]
output_dir = sys.argv[2]
out_image_dir = os.path.join(output_dir, "images")
out_annotations_file = os.path.join(output_dir, "annotations.txt")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(out_image_dir):
    os.makedirs(out_image_dir)

annotations = []
category_counts = {}
with open(annotations_file) as f:
    for line in f:
        line = line.strip()
        split_line = line.split("\t")
        filename = split_line[0]
        category = split_line[1]
        if not category in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1
        annotations.append((filename, category))

min_count = sys.maxsize
for cat in category_counts.keys():
    count = category_counts[cat]
    if count < min_count:
        min_count = count

with open(out_annotations_file, "w") as f:
    category_counts = {}
    for (filename, category) in annotations:
        if not category in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1

        if category_counts[category] >= min_count:
            continue

        basename = os.path.basename(filename)
        cp_path = os.path.join(out_image_dir, basename)
        shutil.copyfile(filename, cp_path)
        f.write(f"{cp_path}\t{category}\n")
