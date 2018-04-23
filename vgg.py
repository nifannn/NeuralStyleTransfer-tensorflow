import tensorflow as tf
import numpy as np
import scipy.io

class pretrainedVGG19(object):
	"""pretrained VGG 19 for Neural Style Transfer
	"""
	def __init__(self, height, width, channels):
		self.graph = {}
		self.graph['input'] = tf.Variable(np.zeros((1, height, width, channels)), dtype = 'float32')

	def _get_layer_weights(self, layer, layer_name):
		"""
		"""
		params = self.vgg_layers[0][layer][0][0][2][0]
		W = params[0]
		b = params[1]
		l_name = self.vgg_layers[0][layer][0][0][0][0]
		assert layer_name == l_name, "Input layer name doesn't match pretrained model's layer name"

		return W, b 
	
	def _conv2d(self, prev_layer, layer, layer_name):
		"""
		"""
		W, b = self._get_layer_weights(layer, layer_name)
		W = tf.constant(W)
		b = tf.constant(np.reshape(b, (b.size)))
		return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

	def _relu(self, layer):
		"""
		"""
		return tf.nn.relu(layer)
	
	def _conv2d_relu(self, prev_layer, layer, layer_name):
		"""
		"""
		return self._relu(self._conv2d(prev_layer, layer, layer_name))
	
	def _avgpool(self, layer):
		"""
		"""
		return tf.nn.avg_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
	def load_weights(self, path):
		"""
		"""
		self.vgg_layers = scipy.io.loadmat(path)['layers']
		self.graph['conv1_1'] = self._conv2d_relu(self.graph['input'], 0, 'conv1_1')
		self.graph['conv1_2'] = self._conv2d_relu(self.graph['conv1_1'], 2, 'conv1_2')
		self.graph['avgpool1'] = self._avgpool(self.graph['conv1_2'])
		self.graph['conv2_1'] = self._conv2d_relu(self.graph['avgpool1'], 5, 'conv2_1')
		self.graph['conv2_2'] = self._conv2d_relu(self.graph['conv2_1'], 7, 'conv2_2')
		self.graph['avgpool2'] = self._avgpool(self.graph['conv2_2'])
		self.graph['conv3_1'] = self._conv2d_relu(self.graph['avgpool2'], 10, 'conv3_1')
		self.graph['conv3_2'] = self._conv2d_relu(self.graph['conv3_1'], 12, 'conv3_2')
		self.graph['conv3_3'] = self._conv2d_relu(self.graph['conv3_2'], 14, 'conv3_3')
		self.graph['conv3_4'] = self._conv2d_relu(self.graph['conv3_3'], 16, 'conv3_4')
		self.graph['avgpool3'] = self._avgpool(self.graph['conv3_4'])
		self.graph['conv4_1'] = self._conv2d_relu(self.graph['avgpool3'], 19, 'conv4_1')
		self.graph['conv4_2'] = self._conv2d_relu(self.graph['conv4_1'], 21, 'conv4_2')
		self.graph['conv4_3'] = self._conv2d_relu(self.graph['conv4_2'], 23, 'conv4_3')
		self.graph['conv4_4'] = self._conv2d_relu(self.graph['conv4_3'], 25, 'conv4_4')
		self.graph['avgpool4'] = self._avgpool(self.graph['conv4_4'])
		self.graph['conv5_1'] = self._conv2d_relu(self.graph['avgpool4'], 28, 'conv5_1')
		self.graph['conv5_2'] = self._conv2d_relu(self.graph['conv5_1'], 30, 'conv5_2')
		self.graph['conv5_3'] = self._conv2d_relu(self.graph['conv5_2'], 32, 'conv5_3')
		self.graph['conv5_4'] = self._conv2d_relu(self.graph['conv5_3'], 34, 'conv5_4')
		self.graph['avgpool5'] = self._avgpool(self.graph['conv5_4'])

