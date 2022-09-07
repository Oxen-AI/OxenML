import sys, os
from tensorflow import keras
import numpy as np
import simplejson as json
import imgaug.augmenters as iaa
import argparse
from matplotlib import pyplot as plt
from data_loader import Dataloader
from keypoints import OxenHumanKeypointsAnnotation


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    dest="model_dir",
    required=True,
    type=str,
    help="Directory of model you want to test",
)
parser.add_argument(
    "-i",
    dest="image_file",
    required=True,
    type=str,
    help="The predictions from the model",
)
parser.add_argument(
    "-t",
    dest="threshold",
    default=0.5,
    type=float,
    help="Confidence threshold",
)

args = parser.parse_args()

model_dir = args.model_dir
image_file = args.image_file

print(f"Loading model... {model_dir}")
model = keras.models.load_model(model_dir)

hyper_param_file = os.path.join(model_dir, "params.json")
params = {}
with open(hyper_param_file, "r") as f:
    params = json.load(f)

print(f"Got hyperparams {params}")

num_keypoints = params["num_keypoints"] if "num_keypoints" in params else 13

img_size = params["image_size"]
test_aug = iaa.Sequential([iaa.Resize(img_size, interpolation="linear")])
dataloader = Dataloader(
    image_dir=None, num_keypoints=num_keypoints, aug=test_aug, image_size=img_size
)

frame = dataloader.read_image_from_disk(image_file)
(frame) = test_aug(image=frame)
frame = frame.reshape(1, img_size, img_size, 3)
print(f"frame {frame.shape}")

outputs = model.predict(frame)
frame = frame.reshape(img_size, img_size, 3)
outputs = outputs.reshape((img_size, img_size, dataloader.num_outputs()))
print(outputs.shape)

annotation = OxenHumanKeypointsAnnotation.from_nd_array(outputs)
annotation.plot_image_frame(frame, threshold=args.threshold)
