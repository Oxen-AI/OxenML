import sys, os
import tensorflow as tf
from tensorflow import keras
import simplejson as json
from data_loader import Dataloader
from model import ImageKeypointsModel
import imgaug.augmenters as iaa

import argparse

import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


def save_model(epoch, step, model, params):
    # Save model
    name = f"epoch_{epoch}_step_{step}"
    print(f"Saving model {name}")
    model_dir = os.path.join(output_dir, name)
    model.save(model_dir)

    # Save hyper params
    hyper_param_file = os.path.join(model_dir, "params.json")
    with open(hyper_param_file, "w") as f:
        f.write(json.dumps(params))


parser = argparse.ArgumentParser()
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
    help="The annotations file we are training on",
)
parser.add_argument(
    "-o",
    dest="output_dir",
    required=True,
    type=str,
    help="Where to store the models you have trained",
)
parser.add_argument(
    "-l", dest="load_model", default="", type=str, help="Which model to load"
)
parser.add_argument(
    "--n_keypoints", default=13, type=int, help="Which epoch to start on"
)
parser.add_argument(
    "--batch_size", default=32, type=int, help="Which epoch to start on"
)
parser.add_argument(
    "--start_epoch", default=0, type=int, help="Which epoch to start on"
)
parser.add_argument(
    "--num_epochs", default=10000, type=int, help="How long to train for"
)
parser.add_argument(
    "--save_every", default=-1, type=int, help="How often to save a model"
)
parser.add_argument(
    "--save_images_every",
    default=1000,
    type=int,
    help="How often to save some sample predictions",
)
parser.add_argument(
    "--save_on_epoch",
    action="store_true",
    help="If you want to save on the epoch marker",
)
# TODO: implement the keras dataloader
parser.add_argument(
    "--load_into_memory",
    action="store_true",
    help="If you want to load all the data into memory",
)

# Parse and print the results
args = parser.parse_args()

print("Training with args")
print(args)

data_dir = args.data_dir
annotations_file = args.annotations_file
output_dir = args.output_dir
start_epoch = args.start_epoch

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_size = 224
num_epochs = args.num_epochs
batch_size = args.batch_size

learning_rate = 1e-4
num_keypoints = args.n_keypoints

hyper_params = {
    "image_size": image_size,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_keypoints": num_keypoints,
}

train_aug = iaa.Sequential(
    [
        iaa.Resize(image_size, interpolation="linear"),
        iaa.Fliplr(0.3),
        # `Sometimes()` applies a function randomly to the inputs with
        # a given probability (0.3, in this case).
        iaa.Sometimes(0.3, iaa.Affine(rotate=10, scale=(0.5, 0.7))),
    ]
)

dataloader = Dataloader(
    image_dir=data_dir,
    should_load_into_memory=args.load_into_memory,
    aug=train_aug,
    num_keypoints=num_keypoints,
    image_size=image_size,
)

logging.info(f"Loading annotations from {annotations_file}")
if not dataloader.load_annotations(annotations_file):
    print("Unable to load annotations file")
    exit()

logging.info(f"Building model")
num_outputs = dataloader.num_outputs()
model = ImageKeypointsModel(
    image_size=image_size, num_outputs=num_outputs, learning_rate=learning_rate
)
(model, loss_fn, optimizer) = model.build()

model_dir = args.load_model
if model_dir != "":
    print(f"Loading saved model from {model_dir}")
    model = keras.models.load_model(model_dir)

print(model.summary())
keras.backend.clear_session()

save_every = args.save_every
num_batches = int(dataloader.num_examples() / batch_size)
total_epochs = num_epochs + start_epoch
logging.info(
    f"Training for {num_epochs} epochs starting on {start_epoch} on {num_batches} batches"
)
total_step = 0
for epoch in range(start_epoch, total_epochs):
    if args.save_on_epoch:
        save_model(epoch, 0, model, hyper_params)

    for step in range(num_batches):
        total_step += 1
        (x, y) = dataloader.get_batch(batch_size, show_images=False)
        with tf.GradientTape() as tape:
            model_outputs = model(x, training=True)
            loss_value = loss_fn(y, model_outputs)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        if step % 10 == 0:
            logging.info(f"Epoch {epoch} Batch {step} Loss {loss_value}")

        if save_every > 0 and step % save_every == 0:
            save_model(epoch, step, model, hyper_params)

        if total_step % args.save_images_every == 0:
            # Save an image at the end of every epoch
            path = os.path.join(output_dir, f"predictions_epoch_{epoch}_{step}.png")
            dataloader.save_inputs_outputs(x, y, model_outputs, path)
    # if epoch % 100 == 0:
    #   save_model(epoch, step, model, hyper_params)

    logging.info(f"---- End Epoch {epoch} ----")
    dataloader.shuffle()
