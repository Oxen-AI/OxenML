import argparse
import os
import pathlib

from oxen.annotations.file_format import FileFormat
from oxen.annotations.file_format import ext_equals

from . import CocoBoundingBoxDataset


def convert(raw_args):

    parser = argparse.ArgumentParser(
        description="Command line tool to convert a set of bounding box to another type"
    )

    parser.add_argument(
        "-b",
        "--base_dir",
        type=str,
        default="",
        help="The base directory of all the images",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="The input file of annotations",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="The output file of annotations",
    )
    args = parser.parse_args(raw_args)

    if not os.path.exists(args.input_file):
        raise Exception(f"File does not exist: {args.input_file}")

    dataset = CocoBoundingBoxDataset(annotation_file=args.input_file)

    ext = pathlib.Path(args.output_file).suffix
    if ext_equals(ext, FileFormat.ND_JSON):
        dataset.write_ndjson(base_img_dir=args.base_dir, outfile=args.output_file)
    elif ext_equals(ext, FileFormat.TSV):
        dataset.write_tsv(base_img_dir=args.base_dir, outfile=args.output_file)
    elif ext_equals(ext, FileFormat.CSV):
        dataset.write_csv(base_img_dir=args.base_dir, outfile=args.output_file)
    else:
        raise Exception(f"Invalid extension {ext}")
