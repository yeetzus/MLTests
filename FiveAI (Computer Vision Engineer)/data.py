#!/usr/bin/env Python3

import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np

class LoadMNIST:
	def __init__(self):
		self._mnist_train = tfds.load(name='mnist', split='train')
		self._mnist_test = tfds.load(name='mnist', split='test')

	def _create_trains_set(self):
		def preprocess(example):
			image, label = example['image'], example['label']
			image = tf.cast(image, tf.float32)
			image = tf.reshape(image, shape=(28 * 28, 1))
			image = image / 255.0
			label = tf.one_hot(label, depth=10)
			return image, label

		self._mnist_train = self._mnist_train.map(preprocess)
		self._mnist_train = self._mnist_train.repeat(20).shuffle(1024).batch(32)
		
	def create_data_fetcher(self):
		iterator = self._mnist_train.make_initializable_iterator()
		return iterator
