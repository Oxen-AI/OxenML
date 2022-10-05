import sys, os
import shutil
import csv
from typing import NamedTuple
from alive_progress import alive_bar


class Example(NamedTuple):
    filename: str
    label: str


if len(sys.argv) != 6:
    print(
        f"Usage: {sys.argv[0]} <data-dir> <output-dir> <column-of-interest> <pos-label> <neg-label>"
    )
    exit()

data_dir = sys.argv[1]
output_dir = sys.argv[2]
column_of_interest = int(sys.argv[3])
pos_label = sys.argv[4]
neg_label = sys.argv[5]

image_dir = os.path.join(data_dir, "img_align_celeba/img_align_celeba/")
annotations_file = os.path.join(data_dir, "list_attr_celeba.csv")

output_images_dir = os.path.join(output_dir, "images")
output_annotations_dir = os.path.join(output_dir, "annotations")
output_annotations_file = os.path.join(output_annotations_dir, "annotations.tsv")

if not os.path.exists(output_images_dir):
    os.makedirs(output_images_dir)

if not os.path.exists(output_annotations_dir):
    os.makedirs(output_annotations_dir)

print(f"Reading annotations: {annotations_file}")
examples = []
with open(annotations_file) as f:
    csv_reader = csv.reader(f, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f"Extracting feature {row[column_of_interest]}")
        else:
            filename = os.path.join(row[0])
            label = pos_label if row[column_of_interest] == "1" else neg_label
            examples.append(Example(filename, label))

        line_count += 1

num_examples = len(examples)
print(f"Got {len(examples)} annotations")

with alive_bar(num_examples, title=f"Processing images") as bar:
    with open(output_annotations_file, "w") as f:

        for i, example in enumerate(examples):
            src_file = os.path.join(image_dir, example.filename)
            basename = os.path.basename(src_file)
            filename = f"{example.label}_{basename}"
            dst_file = os.path.join(output_images_dir, filename)
            # print(f"{src_file} -> {dst_file}")
            shutil.copyfile(src_file, dst_file)
            f.write(f"{dst_file}\t{example.label}\n")

            bar()


print("Done.")
