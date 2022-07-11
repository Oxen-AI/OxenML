
# Get the data

`oxen clone http://hub.oxen.ai/g/Cats-vs-Dogs`

`cd Cats-vs-Dogs`

`oxen pull`


# Experiments

## You may not be lucky enough to have 25000 images in your dataset, yet, lets increasingly add data

## Let's start with something more reasonable, like 100, overfit on it, go up to 1000, see how it does

## Use oxen to manage adding more data and facilitating all these experiments from simple model, complex model, etc

## Add "none" category for more real world example

## Model iteration
- Simple feed forward NN (change to 150x150 input, make sure we can see the images)
- ConvNet (start with RMSProp or whatever I started with)
- ConvNet more layers
- ConvNet longer training loop
- ConvNet Adam Optimizer, lower lr
  - https://karpathy.github.io/2019/04/25/recipe/
- MobileNet pretrain with image net, then fine tuned
  - https://keras.io/guides/transfer_learning/
  - https://keras.io/examples/keras_recipes/sample_size_estimate/

