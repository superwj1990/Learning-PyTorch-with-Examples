

--------------------------------------------------------------------------------
# Learning-PyTorch-with-Examples
Learning PyTorch with Examples 中文版
本文通过PyTorch自带的examples来介绍它的一些基础概念。
PyTorch的核心是提供了两个主要的新特点：
 *一个n维的Tensor类，与numpy相似，但是可以在GPUs上运行
 *构建和训练神经网络时可以实现自动微分（Automatic differentiation）
我们使用一个全连接的ReLU网络作为我们的运行示例。这个网络包含一个隐含层，用梯度下降法来对其进行训练，并通过最小化网络输出和真实输出之间的欧几里得距离来拟合随机的数据。
目录
*Warm-up: numpy
*PyTorch: Tensors
*PyTorch: Variables and autograd
*PyTorch: Defining new autograd functions
*TensorFlow: Static Graphs
*PyTorch: nn
*PyTorch: optim
*PyTorch: Custom nn Modules
*PyTorch: Control Flow and Weight Sharing
