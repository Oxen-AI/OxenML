Here's my experimental setup for the demo Wednesday:

Training a XceptionNet architecture that was pretrained on the ImageNet dataset on the SpaceJam dataset which has 10 categories for Basketball Action Recognition:

* block
* pass
* run
* dribble
* shoot
* ballinhand
* defense
* pick
* noaction
* walk

There are 37085 video clips of the actions which break down into these categories.

11749 walk
6490 noaction
5924 run
3866 defense
3490 dribble
2362 ballinhand
1070 pass
 996 block
 712 pick
 426 shoot

Each video clip is a few seconds long, if we split the videos into images we end up with around 500,000 total images.

It takes awhile to train on all the data. For this reason I am starting by using the just the first frame of each video for 37085 images total. I plan on running 3 experiments, with 10,000 training images, 20,000 training images, and 30,000 training images, leaving the last 7,085 out for the test and validation sets.

It takes around an hour for a full iteration (epoch) through the 20k dataset. Usually you need many epochs to fit the data.

TODO: How many epochs to train until full convergence?

Demo will be Oxen commands for adding data, then a set of precision / recall on each category with the new data. 

If training takes too long to get decent numbers...We will just show the Oxen commands and assume he knows that the numbers should increase.