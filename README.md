

--------------------------------------------------------------------------------
 
本文通过PyTorch自带的examples来介绍它的一些基础概念。<br>
PyTorch的核心是提供了两个主要的新特点：<br>
* 一个n维的Tensor类，与numpy相似，但是可以在GPUs上运行<br>
* 构建和训练神经网络时可以实现自动微分（Automatic differentiation）<br>

我们使用一个全连接的ReLU网络作为我们的运行示例。这个网络包含一个隐含层，用梯度下降法来对其进行训练，并通过最小化网络输出和真实输出之间的欧几里得距离来拟合随机的数据。<br>

### 目录<br>
* Warm-up: numpy<br>
* PyTorch: Tensors<br>
* PyTorch: Variables and autograd<br>
* PyTorch: Defining new autograd functions<br>
* TensorFlow: Static Graphs<br>
* PyTorch: nn<br>
* PyTorch: optim<br>
* PyTorch: Custom nn Modules<br>
* PyTorch: Control Flow and Weight Sharing<br>

### Warm-up：numpy<br>
在介绍PyTorch之前，我们首先用numpy来实现网络。<br>
Numpy提供了一个n维数组对象，以及许多操作这些数组的函数。Numpy是一个科学计算的通用框架；它与计算图、深度学习和梯度无关。然而我们可以通过使用numpy操作手动实现网络的forward和backward传播，使得一个两层的网络可以拟合随机的数据：<br>

```two_layer_net_numpy.py
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
  # Forward pass: compute predicted y
  h = x.dot(w1)
  h_relu = np.maximum(h, 0)
  y_pred = h_relu.dot(w2)
  
  # Compute and print loss
  loss = np.square(y_pred - y).sum()
  print(t, loss)
  
  # Backprop to compute gradients of w1 and w2 with respect to loss
  grad_y_pred = 2.0 * (y_pred - y)
  grad_w2 = h_relu.T.dot(grad_y_pred)
  grad_h_relu = grad_y_pred.dot(w2.T)
  grad_h = grad_h_relu.copy()
  grad_h[h < 0] = 0
  grad_w1 = x.T.dot(grad_h)
 
  # Update weights
  w1 -= learning_rate * grad_w1
  w2 -= learning_rate * grad_w2
```




