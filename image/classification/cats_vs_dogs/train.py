
import sys, os
import tensorflow as tf
from tensorflow import keras
import simplejson as json
from data_loader import Dataloader
from model import ImageClassifierModel

if len(sys.argv) != 3:
  print(f"Usage: {sys.argv[0]} <data-dir> <output-dir>")
  exit()

def save_model(epoch, model, params):
  # Save model
  model_dir = os.path.join(output_dir, f"epoch_{epoch}")
  model.save(model_dir)

  # Save image of architecture
  out_plot = os.path.join(os.path.join(output_dir, f"epoch_{epoch}"), "plot.png")
  keras.utils.plot_model(model, out_plot, show_shapes=True)
  
  # Save hyper params
  hyper_param_file = os.path.join(model_dir, "params.json")
  with open(hyper_param_file, 'w') as f:
    f.write(json.dumps(params))

data_dir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

annotations_file = os.path.join(os.path.join(data_dir, "annotations"), "train_annotations.txt")
labels_file = os.path.join(os.path.join(data_dir, "labels"), "labels.txt")

image_size = 150
num_epochs = 100
batch_size = 32
learning_rate = 1e-5

hyper_params = {
  'image_size': image_size,
  'num_epochs': num_epochs,
  'batch_size': batch_size,
  'learning_rate': learning_rate
}

dataloader = Dataloader(
  should_load_into_memory=True,
  image_size=(image_size, image_size)
)
if not dataloader.load_labels(labels_file):
  print("Unable to load labels file")
  exit()

if not dataloader.load_annotations(annotations_file):
  print("Unable to load annotations file")
  exit()

num_outputs = dataloader.num_outputs()
model = ImageClassifierModel(
  type="mobile_net_v2",
  input_size=image_size,
  output_size=num_outputs,
  learning_rate=learning_rate
)
(model, loss_fn, optimizer) = model.build()
print(model.summary())

num_batches = int(dataloader.num_examples() / batch_size) - 1
print(f"Training for {num_epochs} epochs on {num_batches} batches")
save_model(0, model, hyper_params)
for epoch in range(num_epochs):
  for step in range(num_batches):
    (x, y) = dataloader.get_batch(batch_size)
    with tf.GradientTape() as tape:
      logits = model(x, training=True)
      loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    if step % 10 == 0:
      print(f"Epoch {epoch} Batch {step} Loss {loss_value}")
  save_model(epoch, model, hyper_params)
  print(f"---- End Epoch {epoch} ----")
  dataloader.shuffle()
