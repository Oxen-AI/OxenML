import argparse
import pathlib

from . import CocoHumanKeypointsDataset
from oxen.annotations.file_format import FileFormat, ext_equals


def convert(raw_args):
    parser = argparse.ArgumentParser(
        description="Command line tool to process COCO dataset"
    )
    parser.add_argument(
        "-b",
        "--base_dir",
        type=str,
        required=True,
        help="The base input directory of images",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="The input coco annotations json file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="The output directory you want to put the annotations",
    )
    parser.add_argument(
        "--from_type",
        type=str,
        default="mscoco",
        help="Convert annotations from type to our oxen representation with 13 keypoints [mscoco, ai_challenger]",
    )

    args = parser.parse_args(raw_args)

    dataset = CocoHumanKeypointsDataset(args.input)

    ext = pathlib.Path(args.output).suffix
    if ext_equals(ext, FileFormat.ND_JSON):
        dataset.write_ndjson(base_img_dir=args.base_dir, outfile=args.output)
    elif ext_equals(ext, FileFormat.TSV):
        dataset.write_tsv(base_img_dir=args.base_dir, outfile=args.output)
    elif ext_equals(ext, FileFormat.CSV):
        dataset.write_csv(base_img_dir=args.base_dir, outfile=args.output)
    else:
        raise Exception(f"Invalid extension {ext}")
