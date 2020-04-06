import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# Network flags
flags.DEFINE_integer('input_dimensions', 784, 'Dimensions of input observation')
flags.DEFINE_spaceseplist('units_per_layer', [500, 500, 500, 200, 200, 100, 50, 10], 'Dimensions of input observation')

# Experiment flags
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_integer('train_batch_size', 150, 'Batch size of train')
flags.DEFINE_integer('test_batch_size', 500, 'Batch size of test')
flags.DEFINE_integer('epochs', 5, 'Number of epochs to run')

def load_mnist_train_test_data():
	train_loader = t.utils.data.DataLoader(
			datasets.MNIST('./data/', train=True, download=True, 
							transform=transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.1307,), (0.3081,))])),
			batch_size=FLAGS.train_batch_size, shuffle=True)
	test_loader = t.utils.data.DataLoader(
			datasets.MNIST('./data/', train=False, download=True,
							transform=transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.1307,), (0.3081,))])),
			batch_size=FLAGS.test_batch_size, shuffle=True)

	return train_loader, test_loader

class MNISTClassifier(nn.Module):
	def __init__(self):
		super(MNISTClassifier, self).__init__()

		self._create_net()
		self._create_optimizer()

	def _create_net(self):
		self._fc1 = nn.Linear(in_features=FLAGS.input_dimensions, out_features=FLAGS.units_per_layer[0]) 				  
		self._fc2 = nn.Linear(in_features=FLAGS.units_per_layer[0], out_features=FLAGS.units_per_layer[1]) 
		self._fc3 = nn.Linear(in_features=FLAGS.units_per_layer[1], out_features=FLAGS.units_per_layer[2]) 
		self._fc4 = nn.Linear(in_features=FLAGS.units_per_layer[2], out_features=FLAGS.units_per_layer[3]) 
		self._fc5 = nn.Linear(in_features=FLAGS.units_per_layer[3], out_features=FLAGS.units_per_layer[4]) 
		self._fc6 = nn.Linear(in_features=FLAGS.units_per_layer[4], out_features=FLAGS.units_per_layer[5]) 
		self._fc7 = nn.Linear(in_features=FLAGS.units_per_layer[5], out_features=FLAGS.units_per_layer[6]) 
		self._fc8 = nn.Linear(in_features=FLAGS.units_per_layer[6], out_features=FLAGS.units_per_layer[7])
		
	def forward(self, X):
		X = self._fc1(X)
		X = F.relu(X)
		X = self._fc2(X)
		X = F.relu(X)
		X = self._fc3(X)
		X = F.relu(X)
		X = self._fc4(X)
		X = F.relu(X)
		X = self._fc5(X)
		X = F.relu(X)
		X = self._fc6(X)
		X = F.relu(X)
		X = self._fc7(X)
		X = F.relu(X)
		X = self._fc8(X)
		output = F.log_softmax(X, dim=1)
		return output

	def _create_optimizer(self):
		self.loss = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

def train(argv):
	train_data_fetcher, test_data_fetcher = load_mnist_train_test_data()
	test_data_fetcher = iter(test_data_fetcher)
	net = MNISTClassifier()
	
	for epoch in range(FLAGS.epochs):
		for i, data in enumerate(train_data_fetcher, 0):
			X, y = data
			X = X.view(X.shape[0], -1)
			y_hat = net(X)
			net.optimizer.zero_grad()
			loss = net.loss(y_hat, y)
			loss.backward()
			net.optimizer.step()	
		X_test, y_test = test_data_fetcher.next()
		X_test = X_test.view(X_test.shape[0], -1)
		y_hat_test = net(X_test)
		_, y_hat_test = t.max(y_hat_test, 1)
		acc = (y_hat_test == y_test).sum().item()
		print(acc)

	# Load test sets
	for tst in ["clean", "t1", "t2", "t3", "t4"]:
		data = np.load("./MNIST_test_variants/test_sets/" + tst + ".npy", allow_pickle=True).item()
		X_test, y_test = data['x'], data['y']
		X_test, y_test = t.from_numpy(X_test), t.from_numpy(y_test)
		X_test = X_test - t.mean(X_test)
		X_test = X_test / t.std(X_test)
		X_test = X_test.view(X_test.shape[0], -1)
		y_hat_test = net(X_test)
		_, y_hat_test = t.max(y_hat_test, 1)
		acc = (y_hat_test == y_test).sum().item()
		print(acc)

def train_all_data(argv):
	# Load t3 and t4
	t3 = np.load("./MNIST_test_variants/test_sets/" + 't3' + ".npy", allow_pickle=True).item()
	t3_X, t3_y = t3['x'], t3['y']
	t3_X_train, t3_y_train = t3_X[:8000], t3_y[:8000]
	t3_X_test, t3_y_test = t3_X[8000:], t3_y[8000:]
	t4 = np.load("./MNIST_test_variants/test_sets/" + 't4' + ".npy", allow_pickle=True).item()
	t4_X, t4_y = t4['x'], t4['y']
	t4_X_train, t4_y_train = t4_X[:8000], t4_y[:8000]
	t4_X_test, t4_y_test = t4_X[8000:], t4_y[8000:]
	variants_X_train = np.concatenate((t3_X_train, t4_X_train), axis=0)
	variants_y_train = np.concatenate((t3_y_train, t4_y_train), axis=0)
	variants_X_test = np.concatenate((t3_X_test, t4_X_test), axis=0)
	variants_y_test = np.concatenate((t3_y_test, t4_y_test), axis=0)

	variants_X_train = t.from_numpy(variants_X_train).float()
	variants_y_train = t.from_numpy(variants_y_train).float()
	variants_dataset_train = t.utils.data.TensorDataset(variants_X_train, variants_y_train)
	variants_dataset_train_loader = t.utils.data.DataLoader(variants_dataset_train, batch_size=20, shuffle=True)
	
	train_data_fetcher, _ = load_mnist_train_test_data()
	variants_data_fetcher = iter(variants_dataset_train_loader)
	net = MNISTClassifier()
	
	for epoch in range(FLAGS.epochs):
		for i, data in enumerate(train_data_fetcher, 0):
			X, y = data
			try: 
				X_v, y_v = variants_data_fetcher.next()
			except:
				variants_dataset_train_loader = t.utils.data.DataLoader(variants_dataset_train, batch_size=20, shuffle=True)
				variants_data_fetcher = iter(variants_dataset_train_loader)
				X_v, y_v = variants_data_fetcher.next()
			X_v = X_v - 0.1307
			X_v = X_v / 0.3081
			y_v = y_v.long()
			X = t.cat((X, X_v), 0)
			y = t.cat((y, y_v), 0)
			X = X.view(X.shape[0], -1)
			y_hat = net(X)
			net.optimizer.zero_grad()
			loss = net.loss(y_hat, y)
			loss.backward()
			net.optimizer.step()	
			print(loss)


	# Load test sets
	for tst in ["clean", "t1", "t2", "t3", "t4"]:
		data = np.load("./MNIST_test_variants/test_sets/" + tst + ".npy", allow_pickle=True).item()
		X_test, y_test = data['x'], data['y']
		X_test, y_test = t.from_numpy(X_test), t.from_numpy(y_test)
		X_test = X_test - t.mean(X_test)
		X_test = X_test / t.std(X_test)
		X_test = X_test.view(X_test.shape[0], -1)
		y_hat_test = net(X_test)
		_, y_hat_test = t.max(y_hat_test, 1)
		acc = (y_hat_test == y_test).sum().item()
		print(acc)

if __name__ == '__main__':
	app.run(train_all_data)