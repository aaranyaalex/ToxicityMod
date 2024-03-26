"""
Data preprocessing functions for Kaggle Dataset
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pandas as pd

class Dataset:
    def __init__(self, max_features=200000, max_text_length=1800) -> None:

        # store the data from dataset path
        self.fp = os.path.join('jigsaw-toxic-comment-classification-challenge','train.csv', 'train.csv')
        self.raw = pd.read_csv(self.fp)
        self.max_features = max_features

        # generate the tokenizer (Text to Vec)
        self.tokenize(max_features, max_text_length)

    def tokenize(self, n_features, n_length):
        self.tokenizer = TextVectorization(max_tokens=n_features, output_sequence_length=n_length, output_mode='int')
        # add vocabulary to the tokenizer
        self.tokenizer.adapt(self.raw["comment_text"].values)

    def prepare(self, shuffle_seed=160000, num_batches=16, prefetch_seed=8, split=[0.7, 0.2, 0.1]):
        self.flags = self.raw.columns[2:].values
        flag_data = self.raw[self.raw.columns[2:]].values
        comments = self.raw["comment_text"].values

        # generate the TF dataset, and prepare the batches
        dataset = tf.data.Dataset.from_tensor_slices((self.tokenizer(comments), flag_data))
        dataset = dataset.cache()
        dataset = dataset.shuffle(shuffle_seed)
        dataset = dataset.batch(num_batches)
        self.dataset = dataset.prefetch(prefetch_seed) 

        # split the data into train, validation and test
        self.train = dataset.take(int(len(dataset)*split[0]))
        self.val = dataset.skip(int(len(dataset)*split[0])).take(int(len(dataset)*split[1]))
        self.test = dataset.skip(int(len(dataset)*(split[0] + split[1]))).take(int(len(dataset)*split[2]))    