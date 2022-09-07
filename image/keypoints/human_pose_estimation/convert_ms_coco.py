import argparse
from keypoints import MSCocoKeypointsDataset


def main():

    parser = argparse.ArgumentParser(
        description="Command line tool to process COCO dataset"
    )

    parser.add_argument(
        "-i", "--input", type=str, required=True, help="The input directory of images"
    )
    parser.add_argument(
        "-a",
        "--annotations",
        type=str,
        required=True,
        help="The input coco annotations json file",
    )
    parser.add_argument(
        "--output_images",
        type=str,
        default="",
        help="The output directory you want to put the annotations and images",
    )
    parser.add_argument(
        "--output_annotations",
        type=str,
        required=True,
        help="The output directory you want to put the annotations and images",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        required=True,
        help="The prefix you want to use for annotations and outputs",
    )
    parser.add_argument(
        "-t",
        "--output_type",
        type=str,
        default="tsv",
        help="The type of output you want to write [tsv, json]",
    )
    parser.add_argument(
        "--one_person_per_image",
        action="store_true",
        help="Filter out images with multiple people",
    )
    parser.add_argument(
        "--collapse_head",
        action="store_true",
        help="Convert annotations to a collapse the nose,left_eye,right_eye,left_ear,right_ear into one head keypoint",
    )

    args = parser.parse_args()

    annotations_file = args.annotations
    output_prefix = args.prefix
    one_person_per_image = args.one_person_per_image
    output_annotations = args.output_annotations

    dataset = MSCocoKeypointsDataset(annotations_file, collapse_head=args.collapse_head)
    dataset.write_output(
        base_img_dir=output_prefix,
        one_person_per_image=one_person_per_image,
        outfile=output_annotations,
        output_type=args.output_type,
    )


if __name__ == "__main__":
    main()
