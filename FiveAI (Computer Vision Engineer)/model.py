#!/usr/bin/env Python3

import tensorflow as tf

class MNISTClassifier:
	def __init__(self):
		self._inputs = tf.placeholder(tf.float32, shape=[None, 28 * 28, 1])
		self._targets = tf.placeholder(tf.uint8, shape=[None, 10])
		self._create_network()

	def _create_network(self):
		net = tf.layers.dense(self._inputs, 
							  units=500, 
							  kernel_initializer='glorot_uniform',
							  bias_initializer='zeros',
							  activation=tf.nn.relu)
		net = tf.layers.dense(net, 
							  units=500, 
							  kernel_initializer='glorot_uniform',
							  bias_initializer='zeros',
							  activation=tf.nn.relu)
		net = tf.layers.dense(net, 
							  units=500, 
							  kernel_initializer='glorot_uniform',
							  bias_initializer='zeros',
							  activation=tf.nn.relu)
		net = tf.layers.dense(net, 
							  units=200, 
							  kernel_initializer='glorot_uniform',
							  bias_initializer='zeros',
							  activation=tf.nn.relu)
		net = tf.layers.dense(net, 
							  units=200, 
							  kernel_initializer='glorot_uniform',
							  bias_initializer='zeros',
							  activation=tf.nn.relu)
		net = tf.layers.dense(net, 
							  units=50, 
							  kernel_initializer='glorot_uniform',
							  bias_initializer='zeros',
							  activation=tf.nn.relu)
		net = tf.layers.dense(net, 
							  units=100, 
							  kernel_initializer='glorot_uniform',
							  bias_initializer='zeros',
							  activation=tf.nn.relu)
		self._net = tf.layers.dense(net, 
							  units=10, 
							  kernel_initializer='glorot_uniform',
							  bias_initializer='zeros')

	def _create_loss(self):
		self._loss = tf.nn.softmax_cross_entropy_with_logits(labels=self._targets, logits=self._net)

	def _create_optimizer(self):
		self._train = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(self._loss)




		

