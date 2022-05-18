
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ImageClassifierModel():
  def __init__(self, type, input_size, output_size, learning_rate):
    self.type = type
    self.input_size = input_size
    self.output_size = output_size
    self.learning_rate = learning_rate

  def build(self):
    if "simple_conv_net" == self.type:
      return self._build_conv_net()
    elif "mobile_net_v2" == self.type:
      return self._build_mobile_net_v2()
    else:
      print("Unknown model type")
      exit()

  def _build_conv_net(self):
    self.model = keras.Sequential(
      [
          keras.Input(shape=(self.input_size, self.input_size, 3)),
          layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
          layers.MaxPooling2D(pool_size=(2, 2)),
          layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
          layers.MaxPooling2D(pool_size=(2, 2)),
          layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
          layers.MaxPooling2D(pool_size=(2, 2)),
          layers.Flatten(),
          layers.Dropout(0.5),
          layers.Dense(self.output_size, activation="softmax"),
      ]
    )
    self.model.compile(metrics=["accuracy"])
    loss_fn = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
    return (self.model, loss_fn, optimizer)

  def _build_mobile_net_v2(self):
    # Create input and pre-processing layers for MobileNetV2
    inputs = layers.Input(shape=(self.input_size, self.input_size, 3))
    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
    inputs = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)
    base_model = keras.applications.Xception(
      weights="imagenet",  # Load weights pre-trained on ImageNet.
      input_shape=(self.input_size, self.input_size, 3),
      include_top=False,
    )(inputs)

    # Freeze the pretrained weights
    base_model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(self.output_size, activation="softmax")(x)
    self.model = keras.Model(inputs, outputs)
    self.model.compile(metrics=["accuracy"])

    # Unfreeze model from block 10 onwards
    self.model = self.unfreeze(self.model, "block_10")

    loss_fn = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
    return (self.model, loss_fn, optimizer)

  def unfreeze(self, model, block_name, verbose=0):
    """Unfreezes Keras model layers.

    Arguments:
        model: Keras model.
        block_name: Str, layer name for example block_name = 'block4'.
                    Checks if supplied string is in the layer name.
        verbose: Int, 0 means silent, 1 prints out layers trainability status.

    Returns:
        Keras model with all layers after (and including) the specified
        block_name to trainable, excluding BatchNormalization layers.
    """

    # Unfreeze from block_name onwards
    set_trainable = False

    for layer in model.layers:
        if block_name in layer.name:
            set_trainable = True
        if set_trainable and not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
            if verbose == 1:
                print(layer.name, "trainable")
        else:
            if verbose == 1:
                print(layer.name, "NOT trainable")
    print("Trainable weights:", len(model.trainable_weights))
    print("Non-trainable weights:", len(model.non_trainable_weights))
    return model