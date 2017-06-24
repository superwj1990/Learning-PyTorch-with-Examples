

--------------------------------------------------------------------------------
 
本文通过PyTorch自带的examples来介绍它的一些基础概念。<br>
PyTorch的核心是提供了两个主要的新特点：<br>
* 一个n维的Tensor类，与numpy相似，但是可以在GPUs上运行<br>
* 构建和训练神经网络时可以实现自动微分（Automatic differentiation）<br>

我们使用一个全连接的ReLU网络作为我们的运行示例。这个网络包含一个隐含层，用梯度下降法来对其进行训练，并通过最小化网络输出和真实输出之间的欧几里得距离来拟合随机的数据。<br>

## 目录<br>
* Warm-up: numpy<br>
* PyTorch: Tensors<br>
* PyTorch: Variables and autograd<br>
* PyTorch: Defining new autograd functions<br>
* TensorFlow: Static Graphs<br>
* PyTorch: nn<br>
* PyTorch: optim<br>
* PyTorch: Custom nn Modules<br>
* PyTorch: Control Flow and Weight Sharing<br>

## Warm-up: numpy<br>
在介绍PyTorch之前，我们首先用numpy来实现网络。<br>
Numpy提供了一个n维数组对象，以及许多操作这些数组的函数。Numpy是一个科学计算的通用框架；它与计算图、深度学习和梯度无关。然而我们可以通过使用numpy操作手动实现网络的forward和backward传播，使得一个两层的网络可以拟合随机的数据：<br>

```two_layer_net_numpy.py
# Code in file tensor/two_layer_net_numpy.py
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
## PyTorch: Tensors<br>
Numpy是一个伟大的框架，但是它不能使用GPUs来加速它的数值运算。对于现在的深度神经网络，GPUs通常可以提供至少50倍的提速，因此numpy不能满足当前深度学习的需要。<br>

现在我们介绍PyTorch的最基本的概念：Tensor。PyTorch的Tensor在概念上与numpy的数组相似：一个Tensor是一个n维的数组，且PyTorch提供了许多函数来操作这些Tensors。与numpy类似，PyTorch的Tensors与深度学习，计算图和梯度无关；他们是科学计算的一个通用工具。<br>

与numpy不同的是，PyTorch的Tensors可以利用GPUs来加快他们的数值计算。如果想在GPU上运行PyTorch的Tensor，你只需要将它转换为一个新的数据类型。<br>

接下来我们使用PyTorch的Tensors来让一个两层的网络拟合随机数据。与上面的numpy示例相类似，我们需要手动的实现网络的forward和backward传播：<br>

```two_layer_net_tensor.py
# Code in file tensor/two_layer_net_tensor.py
import torch

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
for t in range(500):
  # Forward pass: compute predicted y
  h = x.mm(w1)
  h_relu = h.clamp(min=0)
  y_pred = h_relu.mm(w2)

  # Compute and print loss
  loss = (y_pred - y).pow(2).sum()
  print(t, loss)

  # Backprop to compute gradients of w1 and w2 with respect to loss
  grad_y_pred = 2.0 * (y_pred - y)
  grad_w2 = h_relu.t().mm(grad_y_pred)
  grad_h_relu = grad_y_pred.mm(w2.t())
  grad_h = grad_h_relu.clone()
  grad_h[h < 0] = 0
  grad_w1 = x.t().mm(grad_h)
  
  # Update weights using gradient descent
  w1 -= learning_rate * grad_w1
  w2 -= learning_rate * grad_w2
```
## PyTorch: Variables and autograd<br>
在上面的示例中，我们不得不为我们的神经网络手动实现forward和backward传播。对于一个两层的网络，手动实现backward传播非常简单，但是对于一个大型的复杂网络，该工作则变得十分麻烦。<br>

幸运的是，我们可以使用自动微分来实现神经网络的backward传播的自动计算。PyTorch中的**autograd**包正好提供了该功能。当时用autograd时，你的网络的forward传播会定义一个**计算图（computational graph）**；该图中的节点为Tensors类型，边是根据输入Tensors产生输出Tensors的函数。通过该图的后向传播可以很容易地进行梯度计算。<br>

这个听起来很复杂，操作起来却很容易。我们用**Variable**对象来封装PyTorch的Tensors；一个Variable表征计算图中的一个节点。如果x是一个Variable，则x.data是一个Tensor，且x.grad是一个保存x梯度（标量）的Variable。<br>

PyTorch的Variables有和Tensors一样的API：（几乎）任何可以在Tensos上执行的操作都可以在Variable上使用；不同的是，使用Variables定义一个计算图，可以自动计算梯度。<br>

接下来我们使用PyTorch的Variables和autograd来实现我们的两层网络；现在我们不再需要手动实现网络的backward传播了。<br>
```two_layer_net_autograd.py
# Code in file autograd/two_layer_net_autograd.py
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs, and wrap them in Variables.
# Setting requires_grad=False indicates that we do not need to compute 
# gradients
# with respect to these Variables during the backward pass.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
  # Forward pass: compute predicted y using operations on Variables; these
  # are exactly the same operations we used to compute the forward pass using
  # Tensors, but we do not need to keep references to intermediate values since
  # we are not implementing the backward pass by hand.
  y_pred = x.mm(w1).clamp(min=0).mm(w2)
  
  # Compute and print loss using operations on Variables.
  # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
  # (1,); loss.data[0] is a scalar value holding the loss.
  loss = (y_pred - y).pow(2).sum()
  print(t, loss.data[0])
  
  # Use autograd to compute the backward pass. This call will compute the
  # gradient of loss with respect to all Variables with requires_grad=True.
  # After this call w1.grad and w2.grad will be Variables holding the gradient
  # of the loss with respect to w1 and w2 respectively.
  loss.backward()

  # Update weights using gradient descent; w1.data and w2.data are Tensors,
  # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
  # Tensors.
  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data

  # Here is different from the source code. When we define the Variable x,
  # the x.grad is none, x.grad.data is non-existent. So we excute the
  # following two lines of code after backward().
  # Manually zero the gradients before running the backward pass
  w1.grad.data.zero_()
  w2.grad.data.zero_()
```
### PyTorch: Defining new autograd functions<br>


