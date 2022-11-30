import argparse
import os
import pathlib

from oxen.annotations.file_format import FileFormat
from oxen.annotations.file_format import ext_equals

from oxen.text.classification.datasets.factory import create_dataset


def convert(raw_args):

    parser = argparse.ArgumentParser(
        description="Command line tool to convert classification datasets to a more manageable format"
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
    parser.add_argument(
        "--input_format",
        type=str,
        required=True,
        help="Format of the input data",
    )
    args = parser.parse_args(raw_args)

    if not os.path.exists(args.input_file):
        raise Exception(f"File does not exist: {args.input_file}")

    dataset = create_dataset(file=args.input_file, format=args.input_format)

    ext = pathlib.Path(args.output_file).suffix
    if ext_equals(ext, FileFormat.TSV):
        dataset.write_tsv(outfile=args.output_file)
    elif ext_equals(ext, FileFormat.CSV):
        dataset.write_csv(outfile=args.output_file)
    else:
        raise Exception(f"Invalid extension {ext}")
