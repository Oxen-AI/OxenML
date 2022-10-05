# OxenML

Tools and scripts to enable demos on OxenData

# Running Tool

`python runner.py bbox_convert`

# Running tests

To run an individual test file

`pytest -s tests/image/bbox/test_coco_dataset.py`

To run full test suite (will pickup all files prefixed with test_)

`pytest -s tests/`


## Datasets

Natural Language Processing

1) IMDB movie review sentiment classification 
  - https://keras.io/examples/nlp/text_classification_from_scratch/
  - https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
2) Named Entity Recognition (NER) CoNLL-2003
  - https://www.clips.uantwerpen.be/conll2003/ner/
  - https://www.clips.uantwerpen.be/conll2003/ner.tgz
3) arXiv Paper Abstracts
  - https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts
4) Multimodal entailment (text+image, does this piece of information imply or contradict the image?)
  - https://keras.io/examples/nlp/multimodal_entailment/
  - https://github.com/sayakpaul/Multimodal-Entailment-Baseline/releases/download/v1.0.0/tweet_images.tar.gz
5) Machine Language Translation
  - There are a ton of datasets here:
    - https://www.manythings.org/anki/
  - https://keras.io/examples/nlp/neural_machine_translation_with_keras_nlp/
6) MSCOCO (Natural language image search)
  - https://keras.io/examples/nlp/nl_image_search/
  - https://cocodataset.org/#home
7) Newsgroup20 (20 different news categories)
  - http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz
8) SQUAD question answering dataset
  - https://rajpurkar.github.io/SQuAD-explorer/
  - https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
9) SNLI (Stanford Natural Language Inference) Corpus
  - https://keras.io/examples/nlp/semantic_similarity_with_bert/
  - https://nlp.stanford.edu/projects/snli/
  - https://nlp.stanford.edu/projects/snli/snli_1.0.zip
10) Cornell Movie Dialog corpus
  - https://keras.io/examples/nlp/text_generation_fnet/
  - http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
11) SimpleBooks
  - https://arxiv.org/abs/1911.12391
  - https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip
12) Disaster Tweets
  - You are predicting whether a given tweet is about a real disaster or not.
  - https://www.kaggle.com/competitions/nlp-getting-started/data?select=train.csv
13) Wikipedia Text
  - The Wikipedia corpus contains about 2 billion words of text from a 2014 dump of the Wikipedia (about 4.4 million pages)
  - https://www.corpusdata.org/wikipedia.asp

Computer Vision

13) Cats vs Dogs
  - https://keras.io/examples/vision/image_classification_from_scratch/
  - https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
14) MNIST (Hand Written Digit Classification)
  - http://yann.lecun.com/exdb/mnist/
15) Image segmentation (pets)
  - https://keras.io/examples/vision/oxford_pets_image_segmentation/
  - https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
  - https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
16) 3D image classification from CT scans
  - https://keras.io/examples/vision/3D_image_classification/
  - https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1
  - https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip
  - https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip
  - We will be using the associated radiological findings of the CT scans as labels to build a classifier to predict presence of viral pneumonia. 
17) The Street View House Numbers (SVHN) Dataset
  - https://keras.io/examples/vision/adamatch/
  - http://ufldl.stanford.edu/housenumbers/
  - http://ufldl.stanford.edu/housenumbers/train.tar.gz
18) CIFAR-10 dataset
  - https://www.cs.toronto.edu/~kriz/cifar.html
19) CIFAR-100 dataset
  - https://www.cs.toronto.edu/~kriz/cifar.html
20) Tensorflow Flowers
  - https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=tf_flowers
  - http://download.tensorflow.org/example_images/flower_photos.tgz
21) Captcha Images
  - https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
  - The dataset contains 1040 captcha files as png images. 
22) The Crowd Instance-level Human Parsing (CIHP) dataset
  - Each image in CIHP is labeled with pixel-wise annotations for 20 categories, as well as instance-level identification. This dataset can be used for the "human part segmentation" task.
  - https://keras.io/examples/vision/deeplabv3_plus/
  - https://drive.google.com/drive/folders/1OLBd23ufm6CU8CZmLEYMdF-x2b8mRgxV
23) DIODE: A Dense Indoor and Outdoor Depth Dataset
  - The goal in monocular depth estimation is to predict the depth value of each pixel or inferring depth information, given only a single RGB image as input.
  - http://diode-dataset.s3.amazonaws.com/val.tar.gz
24) DIV2K Dataset
  - single-image super-resolution dataset
  - https://data.vision.ee.ethz.ch/cvl/DIV2K/
25) IAM Handwriting Database
  - https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
26) Flickr8K dataset
  - This dataset comprises over 8,000 images, that are each paired with five different captions.
  - https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
  - https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
27) Stanford Dogs
  - http://vision.stanford.edu/aditya86/ImageNetDogs/main.html
28) StanfordExtra Dog Keypoints
  - https://github.com/benjiebob/StanfordExtra
  - 12k labelled instances of dogs in-the-wild with 2D keypoint and segmentations.
29) LoL Dataset for low-light image enhancement
  - https://daooshee.github.io/BMVC2018website/
30) FashionMNIST
  - https://github.com/zalandoresearch/fashion-mnist/
31) NERF Novel view synthesis
  - https://www.matthewtancik.com/nerf
  - https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz
  - https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
32) stl10 image classification
  - https://ai.stanford.edu/~acoates/stl10/
33) Caltech 101
  - https://keras.io/examples/vision/object_detection_using_vision_transformer/
  - https://data.caltech.edu/records/20086
  - https://data.caltech.edu/tindfiles/serve/e41f5188-0b32-41fa-801b-d1e840915e80/
34) ModelNet40 dataset
  - Point cloud classification
  - https://keras.io/examples/vision/pointnet/
  - http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
35) ShapeNet dataset
  - https://shapenet.org/
  - ShapeNet is an ongoing effort to establish a richly-annotated, large-scale dataset of 3D shapes
36) Omniglot data set for one-shot learning
  - The Omniglot data set is designed for developing more human-like learning algorithms. It contains 1623 different handwritten characters from 50 different alphabets. Each of the 1623 characters was drawn online via Amazon's Mechanical Turk by 20 different people.
  - https://github.com/brendenlake/omniglot/
  - https://keras.io/examples/vision/reptile/
37) Totally-Looks-Like
  - https://sites.google.com/view/totally-looks-like-dataset
  - https://keras.io/examples/vision/siamese_network/
38) BSDS500 image segmentation and boundary detection
  - https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
  - https://keras.io/examples/vision/super_resolution_sub_pixel/
39) UCF101 - Action Recognition Data Set
  - https://keras.io/examples/vision/video_classification/
  - https://www.crcv.ucf.edu/data/UCF101.php
40) MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification
  - https://medmnist.com/
  - https://keras.io/examples/vision/vivit/
41) CelebA
  - https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Audio

42) LibriVox
  - Free public domain audiobooks
  - https://librivox.org/
  - https://keras.io/examples/audio/ctc_asr/
43) The LJ Speech Dataset
  - https://keithito.com/LJ-Speech-Dataset/
  - https://keras.io/examples/audio/melgan_spectrogram_inversion/
44) Speaker Recognition
  - https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
  - https://keras.io/examples/audio/speaker_recognition_using_cnn/
45) Classify the English accent spoken
  - https://www.openslr.org/resources/83/
  - https://keras.io/examples/audio/uk_ireland_accent_recognition/
46) Common Voice
  - https://voice.mozilla.org/en/datasets
47) LibriTTS
  -  multi-speaker English corpus of approximately 585 hours of read English speech at 24kHz sampling rate
  -  http://www.openslr.org/60
48) voxceleb
  - An large scale dataset for speaker identification. This data is collected from over 1,251 speakers, with over 150k samples in total. This release contains the audio part of the voxceleb1.1 dataset.
  - http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
49) speech_commands
  - An audio dataset of spoken words designed to help train and evaluate keyword spotting systems.
  - https://arxiv.org/abs/1804.03209
  - https://www.tensorflow.org/datasets/catalog/speech_commands

Video

50) youtube_vis
  - https://youtube-vos.org/dataset/vis/
51) DAVIS: Densely Annotated VIdeo Segmentation
  - https://davischallenge.org/

Tablular

-  Kaggle Credit Card Fraud data set
  - https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
  - https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

More:

https://www.tensorflow.org/datasets