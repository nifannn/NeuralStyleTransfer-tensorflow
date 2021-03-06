import numpy as np

VGG_MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
NOISE_RATIO = 0.6
DEFAULT_CONTENT_LAYER = 'conv4_2'
DEFAULT_STYLE_LAYERS = {
	'conv1_1': 0.2,
    'conv2_1': 0.2,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2
}
DEFAULT_OUTPUT_FOLDER = 'output/'
DEFAULT_N_ITERATIONS = 300
DEFAULT_SAVE_EVERY_N_ITERATIONS = 20
DEFAULT_OUTPUT_NAME = 'generated'
DEFAULT_PRETRAINED_MODEL = 'pretrained_model/imagenet-vgg-verydeep-19.mat'
DEFAULT_ALPHA = 10
DEFAULT_BETA = 40
DEFAULT_LR = 2.0
DEFAULT_HEIGHT = 300
DEFAULT_WIDTH = 400
DEFAULT_CHANNELS = 3