#!/usr/bin/env python
"""This file trains the model upon the training data and evaluates it with the test data.
It uses the arguments it got via the gcloud command."""

import argparse
import os
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

import trainer.data as data
import trainer.model as model

def train_model(params):
	"""The function gets the training data from the training folder,
	the evaluation data from the test folder and trains your solution from the model.py file with it."""
	(train_data, train_labels) = data.create_data_with_labels("data/train/") # Create training data

	# Training data -- I CAN DO DATA AUGMENTATION HERE
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=model.get_batch_size(),
		num_epochs=None,
		shuffle=True)

	(eval_data, eval_labels) = data.create_data_with_labels("data/test/")

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": eval_data},
		y=eval_labels,
		num_epochs=1,
		shuffle=False)

	estimator = tf.estimator.Estimator(model_fn=model.solution)
	steps_per_eval = int(model.get_training_steps() / params.eval_steps)

	for _ in range(params.eval_steps):
		out = estimator.train(train_input_fn, steps=steps_per_eval)
		estimator.evaluate(eval_input_fn)

	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": eval_data},
		batch_size=1000,
		num_epochs=1,
		shuffle=False)
	out = estimator.predict(predict_input_fn)
	preds = []
	for a in out:
		preds.append(a['CLASSES'])
	print(preds)		
	print(list(eval_labels))
		


if __name__ == "__main__":
	PARSER = argparse.ArgumentParser()
	PARSER.add_argument(
		'--eval-steps',
		help='Number of steps to run evaluation for at each checkpoint',
		default=10,
		type=int
	)

	ARGS = PARSER.parse_args()
	tf.compat.v1.logging.set_verbosity('INFO')
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__['INFO'] / 10)


	HPARAMS = hparam.HParams(**ARGS.__dict__)
	train_model(HPARAMS)