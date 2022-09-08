import argparse
import os
from tqdm import tqdm

import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint

from matplotlib import pyplot as plt

from keypoints import FileAnnotations
from keypoints import OxenHumanKeypointsAnnotation
from keypoints import TSVKeypointsDataset

def main():

    parser = argparse.ArgumentParser(
        description="Command line tool to resize images based on an annotation file"
    )

    parser.add_argument(
        "-a",
        "--annotations",
        type=str,
        required=True,
        help="The input file of annotations",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Base path to images that the annotations file references",
    )
    parser.add_argument(
        "-s", "--size", type=int, default=224, help="Size of output images"
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
    args = parser.parse_args()

    annotations_file = args.annotations
    outdir = args.output_images
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    aug = iaa.Sequential([iaa.Resize({"height": args.size, "width": args.size})])

    dataset = TSVKeypointsDataset(annotation_file=annotations_file)

    filenames = dataset.list_inputs()
    fullpaths = []
    for filename in filenames:
        fullpath = os.path.join(args.data, filename)
        if not os.path.exists(fullpath):
            print(f"Could not find file: {fullpath}")
            exit()
        fullpaths.append(fullpath)
    
    print(f"Resizing {len(filenames)} images to {args.size}x{args.size}")
    with open(args.output_annotations, "w") as outfile:
        for i in tqdm(range(len(filenames))):
            filename = filenames[i]
            fullpath = fullpaths[i]

            annotations = dataset.get_annotations(filename)
            file_annotation = FileAnnotations(file=filename)
            for (i, annotation) in enumerate(annotations.annotations):
                # Need to convert to imgaug Keypoint objects
                img_aug_kps = [Keypoint(x=point.x, y=point.y) for point in annotation.keypoints]

                # We then project the original image to resize the keypoint coordinates.
                frame = plt.imread(fullpath)
                kps_obj = KeypointsOnImage(img_aug_kps, shape=frame.shape)
                (image, new_kps_obj) = aug(image=frame, keypoints=kps_obj)

                # Create a new OxenHumanKeypointsAnnotation so that we can write to tsv
                new_ann = OxenHumanKeypointsAnnotation.from_keypoints(new_kps_obj)

                try:
                    out_filename = os.path.join(outdir, filename)
                    parent = os.path.dirname(out_filename)
                    if not os.path.exists(parent):
                        os.makedirs(parent)
                    plt.imsave(out_filename, image, format="jpg")
                    
                    # Only add the annotation if we successfully converted the image
                    file_annotation.annotations.append(new_ann)
                except:
                    print(f"Could not save file {fullpath}")

            lines = file_annotation.to_tsv()
            outfile.write(f"{lines}\n")

    print("Done.")


if __name__ == "__main__":
    main()
