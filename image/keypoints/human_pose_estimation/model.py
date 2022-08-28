from tensorflow import keras
import logging

class ImageKeypointsModel():
  def __init__(self, image_size, num_outputs, learning_rate=1.e-4, type='basic'):
    self.type = type
    self.image_size = image_size
    self.num_outputs = num_outputs
    self.learning_rate = learning_rate

  def build(self):
    logging.info(f"Building model type {self.type}")
    if 'basic' == self.type:
      return self._build_basic()
    else:
      print("Unknown model type")
      exit()

  def _build_basic(self):
    inputs = keras.layers.Input(shape=(self.image_size, self.image_size, 3))

    inputs = keras.applications.mobilenet_v2.preprocess_input(inputs)
    base_model = keras.applications.MobileNetV2(
      weights="imagenet",  # Load weights pre-trained on ImageNet.
      input_shape=(self.image_size, self.image_size, 3),
      include_top=False,
    )(inputs)

    # Entry block
    x = keras.layers.Conv2D(1280, 3, padding="same")(base_model)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("sigmoid")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    # for filters in [64, 128]:
    #     x = keras.layers.Activation("sigmoid")(x)
    #     x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
    #     x = keras.layers.BatchNormalization()(x)

    #     x = keras.layers.Activation("sigmoid")(x)
    #     x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
    #     x = keras.layers.BatchNormalization()(x)

    #     x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

    #     # Project residual
    #     residual = keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
    #         previous_block_activation
    #     )
    #     x = keras.layers.add([x, residual])  # Add back residual
    #     previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [512, 256, 128, 64, 32]:
        x = keras.layers.Activation("sigmoid")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("sigmoid")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = keras.layers.Conv2D(17, 3, activation="sigmoid", padding="same")(x)

    self.model = keras.Model(inputs, outputs)
    self.model.compile(metrics=["accuracy"])

    optimizer = keras.optimizers.Adam(self.learning_rate)
    loss_fn = keras.losses.MeanSquaredError()

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
        if set_trainable and not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True
            if verbose == 1:
                print(layer.name, "trainable")
        else:
            if verbose == 1:
                print(layer.name, "NOT trainable")
    print("Trainable weights:", len(model.trainable_weights))
    print("Non-trainable weights:", len(model.non_trainable_weights))
    return model
