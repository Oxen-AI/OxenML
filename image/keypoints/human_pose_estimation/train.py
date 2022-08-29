
import sys, os
import tensorflow as tf
from tensorflow import keras
import simplejson as json
from data_loader import Dataloader
from model import ImageKeypointsModel
import imgaug.augmenters as iaa

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

if len(sys.argv) < 3:
  print(f"Usage: {sys.argv[0]} <data-dir> <output-dir>")
  exit()

def save_model(epoch, step, model, params):
  # Save model
  name = f"epoch_{epoch}_step_{step}"
  model_dir = os.path.join(output_dir, name)
  model.save(model_dir)

  # Save hyper params
  hyper_param_file = os.path.join(model_dir, "params.json")
  with open(hyper_param_file, 'w') as f:
    f.write(json.dumps(params))

data_dir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

annotations_dir = os.path.join(data_dir, "annotations")
annotations_file = os.path.join(annotations_dir, "keypoints_annotations.tsv")

image_size = 224
num_epochs = 10000
batch_size = 32
learning_rate = 1e-4

hyper_params = {
  'image_size': image_size,
  'num_epochs': num_epochs,
  'batch_size': batch_size,
  'learning_rate': learning_rate
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
  should_load_into_memory=False,
  aug=train_aug,
  image_size=image_size
)

logging.info(f"Loading annotations from {annotations_file}")
if not dataloader.load_annotations(annotations_file):
  print("Unable to load annotations file")
  exit()

logging.info(f"Building model")
num_outputs = dataloader.num_outputs()
model = ImageKeypointsModel(
  image_size=image_size,
  num_outputs=num_outputs,
  learning_rate=learning_rate
)
(model, loss_fn, optimizer) = model.build()
print(model.summary())

num_batches = int(dataloader.num_examples() / batch_size)
logging.info(f"Training for {num_epochs} epochs on {num_batches} batches")
for epoch in range(num_epochs):
  save_model(epoch, 0, model, hyper_params)
  for step in range(num_batches):
    (x, y) = dataloader.get_batch(batch_size, show_images=False)
    with tf.GradientTape() as tape:
      logits = model(x, training=True)
      loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # if step % 10 == 0:
    logging.info(f"Epoch {epoch} Batch {step} Loss {loss_value}")
    if step % 500 == 0:
      save_model(epoch, step, model, hyper_params)
  # if epoch % 100 == 0:
  #   save_model(epoch, step, model, hyper_params)


  logging.info(f"---- End Epoch {epoch} ----")
  dataloader.shuffle()
