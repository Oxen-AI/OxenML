import argparse

from .models import YoloV3


def predict(raw_args):
    parser = argparse.ArgumentParser(
        description="Command line tool to resize images based on an annotation file"
    )

    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to the model weights"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the config file for the model",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="Path to the image you want to predict on",
    )

    args = parser.parse_args(raw_args)

    print(f"Loading model {args.model}...")
    model = YoloV3(args.model, args.config)
    print(f"Predict {args.image}")
    predictions = model.predict_file(args.image)
    print(predictions)
