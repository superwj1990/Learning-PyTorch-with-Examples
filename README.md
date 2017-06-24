

--------------------------------------------------------------------------------
 
本文通过PyTorch自带的examples来介绍它的一些基础概念。<br>
PyTorch的核心是提供了两个主要的新特点：<br>
* 一个n维的Tensor类，与numpy相似，但是可以在GPUs上运行<br>
* 构建和训练神经网络时可以实现自动微分（Automatic differentiation）<br>

我们使用一个全连接的ReLU网络作为我们的运行示例。这个网络包含一个隐含层，用梯度下降法来对其进行训练，并通过最小化网络输出和真实输出之间的欧几里得距离来拟合随机的数据。<br>
'目录'<br>
* Warm-up: numpy<br>
* PyTorch: Tensors<br>
* PyTorch: Variables and autograd<br>
* PyTorch: Defining new autograd functions<br>
* TensorFlow: Static Graphs<br>
* PyTorch: nn<br>
* PyTorch: optim<br>
* PyTorch: Custom nn Modules<br>
* PyTorch: Control Flow and Weight Sharing<br>

##Warm-up:numpy<br>
在介绍PyTorch之前，我们首先用numpy来实现网络。<br>
Numpy提供了一个n维数组对象，以及许多操作这些数组的函数。Numpy是一个科学计算的通用框架；它与计算图、深度学习和梯度无关。然而我们可以通过使用numpy操作手动实现网络的forward和backward传播，使得一个两层的网络可以拟合随机的数据：<br>



