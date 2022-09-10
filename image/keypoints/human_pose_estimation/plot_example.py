from matplotlib import pyplot as plt
import argparse
import os
import numpy as np

from keypoints import TSVKeypointsDataset
from keypoints import OxenHumanKeypointsAnnotation

def main():
    parser = argparse.ArgumentParser(
        description="Command line tool to resize images based on an annotation file"
    )

    parser.add_argument(
        "-d", "--data", type=str, required=True, help="Base directory for images"
    )
    parser.add_argument(
        "-a",
        "--annotations",
        type=str,
        required=True,
        help="The input file of annotations",
    )
    parser.add_argument(
        "-n",
        "--line_num",
        type=int,
        required=True,
        help="Line number of the image you want to plot",
    )

    args = parser.parse_args()

    base_dir = args.data
    annotations_file = args.annotations

    delimiter = "\t"
    filenames = []
    print(f"Reading {annotations_file}")
    with open(annotations_file) as f:
        for line in f:
            split_line = line.strip().split(delimiter)
            filenames.append(split_line[0])

    dataset = TSVKeypointsDataset(annotations_file)

    line_num = args.line_num
    if line_num >= len(filenames):
        print(f"Index out of range {line_num} >= {len(filenames)}")
        exit()

    filename = filenames[line_num]
    print(f"Displaying file: {filename}")
    annotation = dataset.get_annotations(filename).annotations[0]
    fullpath = os.path.join(args.data, filename)
    annotation.plot_image_file(fullpath)


if __name__ == "__main__":
    main()
