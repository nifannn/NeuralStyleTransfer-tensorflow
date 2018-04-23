# NeuralStyleTransfer-tensorflow

Implementation of Neural Style Transfer from the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) in Tensorflow

### Requirements

* tensorflow-gpu==1.4.0
* tqdm==4.23.0
* scikit-image==0.13.1

### Usage

* Create folder named pretrained_model
* Download [pretrained VGG 19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) and put it into pretrained_model
* Open terminal and type the command
* Command usage: 

```python
python  nst.py [-h] -c CONTENT_IMG -s STYLE_IMG [-o OUTPUT_FOLDER]
              [-n N_ITERATIONS] [-e SAVE_EVERY_N_ITERATIONS] [-f OUTPUT_NAME]
              [-p PRETRAINED_MODEL] [-a ALPHA] [-b BETA] [-lr LEARNING_RATE]
              [-ht HEIGHT] [-w WIDTH] [-ch CHANNELS]
```

### Example

### References

* [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
* [Convolutional Neural Networks by deeplearning.ai](https://www.coursera.org/learn/convolutional-neural-networks)

## Tensorflow神经风格迁移

用Tensorflow实现神经风格迁移， 论文：[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)

### 环境要求

* tensorflow-gpu==1.4.0
* tqdm==4.23.0
* scikit-image==0.13.1

### 使用方法

### 例子 

### 参考

* [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
* [deeplearning.ai 卷积神经网络](https://www.coursera.org/learn/convolutional-neural-networks)
