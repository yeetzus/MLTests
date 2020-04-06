# #!/usr/bin/env python
# """This file contains all the model information: the training steps, the batch size and the model iself."""
# import numpy as np
# import tensorflow as tf
# tf.enable_eager_execution()

# def get_training_steps():
# 	"""Returns the number of batches that will be used to train your solution.
# 	It is recommended to change this value."""
# 	return 2000

# def get_batch_size():
# 	"""Returns the batch size that will be used by your solution.
# 	It is recommended to change this value."""
# 	return 64

# def solution(features, labels, mode):
# 	"""Returns an EstimatorSpec that is constructed using the solution that you have to write below."""
# 	# Input Layer (a batch of images that have 64x64 pixels and are RGB colored (3)
# 	# learning_rate = tf.Variable(1e-4, name='learning_rate:0')
# 	input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])
# 	input_layer = tf.image.adjust_contrast(input_layer, 5)
# 	input_layer = tf.image.adjust_saturation(input_layer, 5)

# 	# TODO: Code of your solution
# 	regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)
# 	net = tf.image.central_crop(input_layer, 0.40)
# 	net = tf.layers.conv2d(input_layer, filters=8, kernel_size=(4, 4), strides=(2,2), padding='VALID', kernel_regularizer=regularizer)
# 	net = tf.layers.max_pooling2d(net, pool_size=(2,2), strides=(1,1))
# 	net = tf.layers.conv2d(net, filters=12, kernel_size=(4, 4), strides=(2, 2), padding='VALID', kernel_regularizer=regularizer)
# 	net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(1,1))
# 	net = tf.nn.dropout(net, rate=0.50)
# 	net = tf.contrib.layers.flatten(net)
# 	net = tf.layers.dense(net, units=256, kernel_regularizer=regularizer, activation=tf.nn.relu)
# 	net = tf.nn.dropout(net, rate=0.5)
# 	net = tf.layers.dense(net, units=256, kernel_regularizer=regularizer, activation=tf.nn.relu)
# 	net = tf.nn.dropout(net, rate=0.5)
# 	net = tf.layers.dense(net, units=64, kernel_regularizer=regularizer, activation=tf.nn.relu)
# 	net = tf.nn.dropout(net, rate=0.5)
# 	out = tf.layers.dense(net, units=4)

# 	if mode == tf.estimator.ModeKeys.PREDICT:
# 		# TODO: return tf.estimator.EstimatorSpec with prediction values of all classes
# 		# predictions = {'top_1': tf.argmax(out, -1),
# 		# 			   'logits':out}
# 		predictions = {'CLASSES': tf.argmax(out, -1), 'PROBABILITIES':tf.nn.softmax(out)}
# 		return tf.estimator.EstimatorSpec(mode, predictions=predictions)
# 	else:
# 		labels = tf.one_hot(labels, depth=4)
# 		reg_loss = tf.losses.get_regularization_loss()
# 		loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=out)
# 		loss = tf.reduce_mean(loss)
# 		loss += reg_loss
# 		eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=tf.argmax(labels, axis=-1), predictions=tf.argmax(out, axis=-1))}
		
# 		if mode == tf.estimator.ModeKeys.TRAIN:
# 			# TODO: Let the model train here
# 			# TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
# 			global_step = tf.train.get_or_create_global_step()
# 			boundaries = [1000]
# 			values = [1e-4, 8e-5]
# 			learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
# 			train_op = tf.compat.v1.train.RMSPropOptimizer(1e-4).minimize(loss, global_step = global_step)
# 			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

# 		elif mode == tf.estimator.ModeKeys.EVAL:
# 			# The classes variable below exists of an tensor that contains all the predicted classes in a batch
# 			# TODO: eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=classes)}
# 			# TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
# 			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
# 		else:
# 			raise NotImplementedError()

#!/usr/bin/env python
"""This file contains all the model information: the training steps, the batch size and the model iself."""
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

def get_training_steps():
	"""Returns the number of batches that will be used to train your solution.
	It is recommended to change this value."""
	return 1000

def get_batch_size():
	"""Returns the batch size that will be used by your solution.
	It is recommended to change this value."""
	return 64

def solution(features, labels, mode):
	"""Returns an EstimatorSpec that is constructed using the solution that you have to write below."""
	# Input Layer (a batch of images that have 64x64 pixels and are RGB colored (3)
	# learning_rate = tf.Variable(1e-4, name='learning_rate:0')
	input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])
	# input_layer = tf.image.adjust_brightness(input_layer, -50)
	input_layer = tf.image.adjust_contrast(input_layer, 5)	

	# TODO: Code of your solution
	regularizer = tf.contrib.layers.l2_regularizer(scale=0.085)
	net = tf.image.central_crop(input_layer, 0.40)
	net = tf.layers.conv2d(input_layer, filters=12, kernel_size=(4, 4), strides=(2,2), padding='VALID', kernel_regularizer=regularizer)
	net = tf.layers.max_pooling2d(net, pool_size=(2,2), strides=(1,1))
	net = tf.layers.conv2d(net, filters=12, kernel_size=(4, 4), strides=(2, 2), padding='VALID', kernel_regularizer=regularizer)
	net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(1,1))
	net = tf.nn.dropout(net, rate=0.50)
	net = tf.contrib.layers.flatten(net)
	net = tf.layers.dense(net, units=256, kernel_regularizer=regularizer)
	net = tf.nn.dropout(net, rate=0.5)
	net = tf.layers.dense(net, units=256, kernel_regularizer=regularizer)
	net = tf.nn.dropout(net, rate=0.5)
	net = tf.layers.dense(net, units=64, kernel_regularizer=regularizer)
	net = tf.nn.dropout(net, rate=0.5)
	out = tf.layers.dense(net, units=4)

	if mode == tf.estimator.ModeKeys.PREDICT:
		# TODO: return tf.estimator.EstimatorSpec with prediction values of all classes
		# predictions = {'top_1': tf.argmax(out, -1),
		# 			   'logits':out}
		predictions = {'CLASSES': tf.argmax(out, -1), 'PROBABILITIES':tf.nn.softmax(out)}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)
	else:
		labels = tf.one_hot(labels, depth=4)
		reg_loss = tf.losses.get_regularization_loss()
		loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=out)
		loss = tf.reduce_mean(loss)
		loss += reg_loss
		eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=tf.argmax(labels, axis=-1), predictions=tf.argmax(out, axis=-1))}
		
		if mode == tf.estimator.ModeKeys.TRAIN:
			# TODO: Let the model train here
			# TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
			global_step = tf.train.get_or_create_global_step()
			boundaries = [1000]
			values = [1e-4, 8e-5]
			learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
			train_op = tf.compat.v1.train.RMSPropOptimizer(1e-4).minimize(loss, global_step = global_step)
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

		elif mode == tf.estimator.ModeKeys.EVAL:
			# The classes variable below exists of an tensor that contains all the predicted classes in a batch
			# TODO: eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=classes)}
			# TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
		else:
			raise NotImplementedError()