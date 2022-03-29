
import sys, os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

from data_loader import Dataloader

if len(sys.argv) != 3:
  print(f"Usage: {sys.argv[0]} <data-dir> <output-dir>")
  exit()

data_dir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
  os.mkdir(output_dir)

annotations_file = os.path.join(os.path.join(data_dir, "annotations"), "annotations.txt")
labels_file = os.path.join(os.path.join(data_dir, "labels"), "labels.txt")

# TODO: separate loading labels from loading annotations, and set in memory flag then
dataloader = Dataloader(
  data_dir=data_dir,
  annotation_file=annotations_file,
  label_file=labels_file,
  should_load_into_memory=True
)
dataloader.load()

num_epochs = 10
batch_size = 32
num_batches = int(dataloader.num_examples() / batch_size) - 1

num_outputs = dataloader.num_outputs()
inputs = keras.Input(shape=(dataloader.input_length(),), name="image_inputs")
features_1 = layers.Dense(128, activation="relu")(inputs)
features_2 = layers.Dense(64, activation="relu")(features_1)
outputs = layers.Dense(num_outputs, activation="softmax")(features_2)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(metrics=["accuracy"])

print(model.summary())
loss_fn = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

print(f"Training for {num_epochs} epochs on {num_batches} batches")
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
  
  model_dir = os.path.join(output_dir, f"epoch_{epoch}")
  print(f"---- End Epoch {epoch} Saving model to {model_dir} ----")
  model.save(model_dir)

  out_plot = os.path.join(os.path.join(output_dir, f"epoch_{epoch}"), "plot.png")
  plot_model(model, out_plot, show_shapes=True)
  dataloader.shuffle()
