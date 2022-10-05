import sys, os
import shutil
from typing import NamedTuple
from alive_progress import alive_bar


class Example(NamedTuple):
    filename: str
    label: str


if len(sys.argv) != 5:
    print(
        f"Usage: {sys.argv[0]} <base-dir> <input-annotations.tsv> <output-dir> <total-per-cat>"
    )
    exit()

base_dir = sys.argv[1]
annotations_file = sys.argv[2]
output_dir = sys.argv[3]
total_per_cat = int(sys.argv[4])
output_examples_dir = os.path.join(output_dir, "examples")
output_annotations_dir = os.path.join(output_dir, "annotations")
output_annotations_file = os.path.join(
    output_annotations_dir, os.path.basename(annotations_file)
)

if not os.path.exists(output_examples_dir):
    os.makedirs(output_examples_dir)

if not os.path.exists(output_annotations_dir):
    os.makedirs(output_annotations_dir)

print(f"Reading annotations: {annotations_file}")
examples = []
total_label_counts = {}
with open(annotations_file) as f:
    for line in f:
        line = line.strip()
        split = line.split("\t")
        filename = split[0]
        label = split[1]

        if not label in total_label_counts:
            total_label_counts[label] = 0
        total_label_counts[label] += 1

        examples.append(Example(filename, label))

for key in total_label_counts.keys():
    count = total_label_counts[key]
    if count < total_per_cat:
        print(
            f"Category `{key}` does not have enough examples {count} < {total_per_cat}"
        )
        exit()

num_examples = total_per_cat * len(total_label_counts)
print(f"Got {len(examples)} annotations copying {num_examples}")

label_counts = {}
with alive_bar(num_examples, title=f"Processing videos") as bar:
    with open(output_annotations_file, "w") as f:
        for example in examples:
            label = example.label
            src_file = os.path.join(base_dir, example.filename)

            if not label in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

            count = label_counts[label]

            if count > total_per_cat:
                continue

            _, extension = os.path.splitext(src_file)

            filename = f"{label}_{count}{extension}"
            dst_file = os.path.join(output_examples_dir, filename)
            shutil.copyfile(src_file, dst_file)
            f.write(f"{dst_file}\t{example.label}\n")
            bar()

print("Done.")
