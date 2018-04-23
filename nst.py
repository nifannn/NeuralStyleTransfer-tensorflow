import numpy as np
import tensorflow as tf
from tqdm import trange
import argparse
import skimage.io
import skimage.transform
from vgg import pretrainedVGG19
from default_config import *

class NeuralStyleTransfer(object):
	"""Neural Style Transfer with VGG-19.

	Args:

	Attributes:

	"""
	def __init__(self, content_img, style_img, model_graph, sess,
				 width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, channels=DEFAULT_CHANNELS,
				 alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
				 cl=DEFAULT_CONTENT_LAYER, sl=DEFAULT_STYLE_LAYERS, lr=DEFAULT_LR):
		self.width = width
		self.height = height
		self.channels = channels
		self.graph = model_graph
		self.sess = sess
		self.alpha = alpha
		self.beta = beta
		self.cl = cl
		self.sl = sl
		self.lr = lr
		self._set_content_img(content_img)
		self._set_style_img(style_img)
		self.generated_img = self._generate_noise_img()

	def _reshape_and_normalize_img(self, image):
		"""

		Args:

		Returns:

		"""
		image = skimage.transform.resize(image, (self.height, self.width, self.channels), mode='reflect', preserve_range=True)
		image = np.reshape(image, (1, self.height, self.width, self.channels))
		image = image - VGG_MEANS
		return image

	def _generate_noise_img(self):
		"""
		Returns:
		"""
		noise_img = np.random.uniform(-20, 20, (1, self.height, self.width, self.channels)).astype('float32')
		generated_img = noise_img * NOISE_RATIO + self._content * (1 - NOISE_RATIO)
		return generated_img

	def _set_content_img(self, content_img):
		"""
		"""
		content_image = skimage.io.imread(content_img)
		self._content = self._reshape_and_normalize_img(content_image)
		self._content_feature = self._get_conv_feature(self._content, self.cl)

	def _set_style_img(self, style_img):
		"""
		"""
		style_image = skimage.io.imread(style_img)
		self._style = self._reshape_and_normalize_img(style_image)
		self._style_features = {}
		for layer_name in self.sl.keys():
			self._style_features[layer_name] = self._get_conv_feature(self._style, layer_name)
	
	def _get_conv_feature(self, image, layer_name):
		"""

		"""
		self.sess.run(self.graph['input'].assign(image))
		features = self.sess.run(self.graph[layer_name])
		return features

	def _get_content_cost(self):
		"""
		"""
		m, H, W, C = self.graph[self.cl].get_shape().as_list()
		a_C_unrolled = tf.transpose(tf.reshape(self._content_feature, [m, H * W, C]), perm=[0,2,1])
		a_G_unrolled = tf.transpose(tf.reshape(self.graph[self.cl], [m, H * W, C]), perm=[0,2,1])
		content_cost = tf.reduce_sum(tf.square(tf.subtract(a_G_unrolled, a_C_unrolled))) / (4 * H * W * C)
		return content_cost

	def _get_gram_matrix(self, A):
		"""
		"""
		return tf.matmul(A, tf.transpose(A))

	def _get_layer_style_cost(self, layer_name):
		"""
		"""
		m, H, W, C = self.graph[layer_name].get_shape().as_list()
		a_S = tf.transpose(tf.reshape(self._style_features[layer_name], [H * W, C]))
		a_G = tf.transpose(tf.reshape(self.graph[layer_name], [H * W, C]))
		GS = self._get_gram_matrix(a_S)
		GG = self._get_gram_matrix(a_G)
		layer_style_cost = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4*(C**2)*((H*W)**2))
		return layer_style_cost

	def _get_style_cost(self):
		"""
		"""
		style_cost = 0
		for layer_name, coeff in self.sl.items():
			layer_style_cost = self._get_layer_style_cost(layer_name)
			style_cost += coeff * layer_style_cost
		return style_cost

	def _get_cost_op(self):
		"""
		"""
		content_cost = self._get_content_cost()
		style_cost = self._get_style_cost()
		cost = self.alpha * content_cost + self.beta * style_cost
		return cost, content_cost, style_cost

	def _get_train_op(self, cost):
		"""
		Get training op.

		Returns:
			train_op: op of training
		"""
		optimizer = tf.train.AdamOptimizer(self.lr)
		train_op = optimizer.minimize(cost)
		return train_op

	def prepare_training(self):
		"""
		Preparation work before training. Get important ops and initialize variables.
		"""
		self.cost, self.content_cost, self.style_cost = self._get_cost_op()
		self.train_op = self._get_train_op(self.cost)
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(self.graph['input'].assign(self.generated_img))

	def train_step(self):
		"""
		Train one step, updating generated image.

		Returns:
			t_cost: float, total cost
			c_cost: float, content cost
			s_cost: float, style cost
		"""
		self.sess.run(self.train_op)
		self.generated_img = self.sess.run(self.graph['input'])
		t_cost, c_cost, s_cost = self.sess.run([self.cost, self.content_cost, self.style_cost])
		return t_cost, c_cost, s_cost

	def save_generated_img(self, path):
		"""
		Save generated image.

		Args:
			path: str, path to save image
		"""
		image = self.generated_img + VGG_MEANS
		image = np.clip(image[0], 0, 255).astype('uint8')
		skimage.io.imsave(path, image)


def parse_parameter():
	"""
	Parse parameters from command line.

	Returns:
		dict, parsed parameters
	"""
	description = "Neural Style Transfer with VGG19."
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument('-c', '--content_img', required=True,
			   help='path to content image', type=str)
	parser.add_argument('-s', '--style_img', required=True,
			   help='path to style image', type=str)
	parser.add_argument('-o', '--output_folder',
						help='path to output folder', type=str, default=DEFAULT_OUTPUT_FOLDER)
	parser.add_argument('-n', '--n_iterations',
						help='number of iterations', type=int, default=DEFAULT_N_ITERATIONS)
	parser.add_argument('-e', '--save_every_n_iterations', type=int, default=DEFAULT_SAVE_EVERY_N_ITERATIONS,
	 help='every n iterations the model will save an output image')
	parser.add_argument('-f', '--output_name', type=str, default=DEFAULT_OUTPUT_NAME, help='output image name')
	parser.add_argument('-p', '--pretrained_model', type=str, default=DEFAULT_PRETRAINED_MODEL,
	 help='path to pretraned model')
	parser.add_argument('-a', '--alpha', type=float, default=DEFAULT_ALPHA, help='importance of content cost')
	parser.add_argument('-b', '--beta', type=float, default=DEFAULT_BETA, help='importance of style cost')
	parser.add_argument('-lr', '--learning_rate', type=float, default=DEFAULT_LR, help='learning rate')
	parser.add_argument('-ht', '--height', type=int, default=DEFAULT_HEIGHT, help='height of image')
	parser.add_argument('-w', '--width', type=int, default=DEFAULT_WIDTH, help='width of image')
	parser.add_argument('-ch', '--channels', type=int, default=DEFAULT_CHANNELS, help='channels of image')
	args = parser.parse_args()
	return vars(args)

def run_nst(content_img, style_img, output_folder=DEFAULT_OUTPUT_FOLDER,
		n_iterations=DEFAULT_N_ITERATIONS, save_every_n_iterations=DEFAULT_SAVE_EVERY_N_ITERATIONS,
		height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, channels=DEFAULT_CHANNELS,
		pretrained_model=DEFAULT_PRETRAINED_MODEL, learning_rate=DEFAULT_LR,
		output_name=DEFAULT_OUTPUT_NAME, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
		cl=DEFAULT_CONTENT_LAYER, sl=DEFAULT_STYLE_LAYERS):
	"""
	Run Neural Style Transfer based on VGG-19.

	Args:
		content_img: str, path to content image
		style_img: str, path to style image
		output_folder: str, path to output folder
		n_iterations: int, number of iterations
		save_every_n_iterations: int, number of iterations the model will save a output image
		height:
		width:
		channels:
		pretrained_model:
		learning_rate:
		output_name:
		alpha:
		beta:
		cl:
		sl:
	"""
	vgg = pretrainedVGG19(height, width, channels)
	vgg.load_weights(pretrained_model)
	with tf.Session() as sess:
		nst = NeuralStyleTransfer(content_img=content_img, style_img=style_img, model_graph=vgg.graph,
								  sess=sess, width=width, height=height, channels=channels,
								  alpha=alpha, beta=beta, cl=cl, sl=sl, lr=learning_rate)
		nst.prepare_training()
		with trange(n_iterations) as t:
			for it in t:
				t.set_description('NST')
				total_loss, content_loss, style_loss = nst.train_step()
				t.set_postfix(total=total_loss, content=content_loss, style=style_loss)
				if it % save_every_n_iterations == 0:
					img_path = output_folder+output_name+'_'+str(it)+'.png'
					nst.save_generated_img(img_path)
		img_path = output_folder+output_name+'.jpg'
		nst.save_generated_img(img_path)

if __name__ == '__main__':
	params = parse_parameter()
	run_nst(**params)
