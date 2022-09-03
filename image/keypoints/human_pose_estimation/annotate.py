import argparse
import os
import json

from keypoints import TSVKeypointsDataset, OxenHumanKeypointsAnnotation, FileAnnotations
from tensorflow import keras
from matplotlib import pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    dest="model_dir",
    required=True,
    type=str,
    help="The directory of the model",
)
parser.add_argument(
    "-d",
    dest="data_dir",
    required=True,
    type=str,
    help="The base directory to look for the image data",
)
parser.add_argument(
    "-a",
    dest="annotations_file",
    required=True,
    type=str,
    help="The annotations file we are validating against",
)
parser.add_argument(
    "-o",
    dest="output",
    required=True,
    type=str,
    help="Where to store the output annotations",
)

args = parser.parse_args()

model_dir = args.model_dir
annotations_file = args.annotations_file

# TODO: wrap this all in a model class who can load the model, predict, resize, etc
print(f"Loading model... {model_dir}")
model = keras.models.load_model(model_dir)

hyper_param_file = os.path.join(model_dir, "params.json")
params = {}
with open(hyper_param_file, "r") as f:
    params = json.load(f)

img_size = params["image_size"]

print(f"Loading dataset... {annotations_file}")
dataset = TSVKeypointsDataset(annotation_file=annotations_file)
print(f"Running model on {len(dataset.annotations)} files")
model_annotations = []
for annotation in tqdm(dataset.annotations):
    fullpath = os.path.join(args.data_dir, annotation.file)
    frame = plt.imread(fullpath)
    frame = frame.reshape(1, img_size, img_size, 3)
    outputs = model.predict(frame)

    frame = frame.reshape(img_size, img_size, 3)
    outputs = outputs.reshape((img_size, img_size, params["num_keypoints"]))

    a = FileAnnotations(file=annotation.file)
    a.add_annotation(OxenHumanKeypointsAnnotation.from_nd_array(outputs))
    model_annotations.append(a)

print(f"Writing output to {args.output}")
with open(args.output, "w") as f:
    for annotation in model_annotations:
        f.write(annotation.to_tsv())
        f.write("\n")
