import argparse
import os
from tqdm import tqdm

import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint

from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

from oxen.image.bounding_box import OxenBoundingBox
from oxen.image.bounding_box.datasets import OxenCSVBoundingBoxDataset

from oxen.annotations import IDAnnotations


def resize_annotation(outdir, file_type, aug, dataset, filenames, fullpaths, i):
    filename = filenames[i]
    fullpath = fullpaths[i]

    annotations = dataset.get_annotations(filename)
    file_annotation = IDAnnotations(id=filename)
    for (i, annotation) in enumerate(annotations.annotations):
        # Need to convert to imgaug Keypoint objects
        img_aug_kps = [
            Keypoint(x=annotation.min_x, y=annotation.min_y),
            Keypoint(x=annotation.width, y=annotation.height),
        ]

        # We then project the original image to resize the keypoint coordinates.
        frame = plt.imread(fullpath)
        kps_obj = KeypointsOnImage(img_aug_kps, shape=frame.shape)
        (image, new_kps_obj) = aug(image=frame, keypoints=kps_obj)

        # Create a new OxenHumanKeypointsAnnotation so that we can write to tsv
        origin = new_kps_obj[0]
        size = new_kps_obj[1]
        new_ann = OxenBoundingBox(
            min_x=origin.x,
            min_y=origin.y,
            width=size.x,
            height=size.y,
            label=annotation.label,
        )

        try:
            out_filename = os.path.join(outdir, os.path.basename(filename))
            parent = os.path.dirname(out_filename)
            if not os.path.exists(parent):
                os.makedirs(parent)
            plt.imsave(out_filename, image, format=file_type)

            # Only add the annotation if we successfully converted the image
            file_annotation.annotations.append(new_ann)
        except:
            print(f"Could not save file {fullpath}")
    return file_annotation


def resize(raw_args):

    parser = argparse.ArgumentParser(
        description="Command line tool to resize bounding box images based on an annotation file"
    )

    parser.add_argument(
        "-a",
        "--annotations",
        type=str,
        required=True,
        help="The input file of annotations",
    )
    parser.add_argument(
        "-p",
        "--output_prefix",
        type=str,
        required=True,
        help="Base path to images that the output annotations file references",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Base path to images that the annotations file references",
    )
    parser.add_argument("--width", type=int, default=224, help="Width of output images")
    parser.add_argument(
        "--height", type=int, default=224, help="Height of output images"
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="jpg",
        help="The type of image file you want to save.",
    )
    parser.add_argument(
        "--output_images",
        type=str,
        required=True,
        help="The output directory you want to put the images",
    )
    parser.add_argument(
        "--output_annotations",
        type=str,
        required=True,
        help="The output file for resized annotations",
    )
    parser.add_argument("--with_header", action="store_true")
    args = parser.parse_args(raw_args)

    annotations_file = args.annotations
    outdir = args.output_images
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    dataset = OxenCSVBoundingBoxDataset.from_file(
        annotations_file, has_header=args.with_header
    )
    aug = iaa.Sequential([iaa.Resize({"height": args.height, "width": args.width})])

    filenames = dataset.list_inputs()
    fullpaths = []
    for filename in filenames:
        fullpath = os.path.join(args.data, filename)
        if not os.path.exists(fullpath):
            print(f"Could not find file: {fullpath}")
            exit()
        fullpaths.append(fullpath)

    print(f"Resizing {len(filenames)} images to {args.width}x{args.height}")

    n_jobs = multiprocessing.cpu_count()
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(resize_annotation)(
            outdir, args.output_type, aug, dataset, filenames, fullpaths, i
        )
        for i in tqdm(range(len(filenames)))
    )

    out_filename = args.output_annotations
    parent = os.path.dirname(out_filename)
    if not os.path.exists(parent):
        os.makedirs(parent)

    print(f"Writing {len(results)} to output {out_filename}")
    with open(out_filename, "w") as outfile:
        for (i, result) in enumerate(results):
            if args.with_header and i == 0:
                print(result)
                outfile.write(f"{result.csv_header()}\n")
            basename = os.path.basename(result.file)
            result_file = os.path.join(args.output_prefix, basename)
            result.file = result_file

            lines = result.to_csv()
            outfile.write(f"{lines}\n")

    print("Done.")
