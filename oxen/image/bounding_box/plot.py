from matplotlib import pyplot as plt
import argparse
import os
import numpy as np

from oxen.image.bounding_box import CSVBoundingBoxDataset


def plot(raw_args):
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

    args = parser.parse_args(raw_args)

    base_dir = args.data
    annotations_file = args.annotations

    dataset = CSVBoundingBoxDataset.from_file(annotations_file, has_header=True)

    fullpaths = []
    filenames = []
    annotations = []
    for filename in dataset.list_inputs():
        fullpath = os.path.join(args.data, filename)
        if not os.path.exists(fullpath):
            print(f"Could not find file: {fullpath}")
            exit()

        for annotation in dataset.get_annotations(filename):
            filenames.append(filename)
            fullpaths.append(fullpath)
            annotations.append(annotation)

    line_num = args.line_num
    if line_num >= len(filenames):
        print(f"Index out of range {line_num} >= {len(filenames)}")
        exit()

    filename = filenames[line_num]

    fullpath = fullpaths[line_num]
    annotation = annotations[line_num]

    print(f"Displaying file: {filename}")
    print(f"Annotation: {annotation}")
    annotation.plot_image_file(fullpath)
