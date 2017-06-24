

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
在后台，每一个原始的autograd操作实际上是在Tensors上执行的两个函数。Forward函数根据输入Tensors计算输出Tensors。Backward函数接收来自输出Tensors的梯度（标量），然后根据这些标量计算输入Tensors的梯度。<br>

在PyTorch中，我们可以很容易通过定义一个torch.autograd.Function的子类及实现它的forward和backward函数来定义我们自己的autograd操作。我们稍后可以给该子类构建一个实例，并像函数一样调用它，给它传递包含输入数据的Variables来实现我们自定义的autograd操作。<br>

在这个示例中，我们定义一个自定义的autograd函数来执行ReLU的非线性性，以及使用它来实现我们的两层网络：<br>
```two_layer_net_custom_function.py
# Code in file autograd/two_layer_net_custom_function.py
import torch
from torch.autograd import Variable

class MyReLU(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  def forward(self, input):
    """
    In the forward pass we receive a Tensor containing the input and return a
    Tensor containing the output. You can cache arbitrary Tensors for use in the
    backward pass using the save_for_backward method.
    """
    self.save_for_backward(input)
    return input.clamp(min=0)

  def backward(self, grad_output):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    input, = self.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input < 0] = 0
    return grad_input


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500) 
  # Construct an instance of our MyReLU class to use in our network
  # We need to construct the instance in each iteration, or when we trying
  # to backward through the graph second time, the buffers will have already
  # been freed. 
  relu = MyReLU()
  
  # Forward pass: compute predicted y using operations on Variables; we compute
  # ReLU using our custom autograd operation.
  y_pred = relu(x.mm(w1)).mm(w2)
  
  # Compute and print loss
  loss = (y_pred - y).pow(2).sum()
  print(t, loss.data[0])
  
  # Manually zero the gradients before running the backward pass
  w1.grad.data.zero_()
  w2.grad.data.zero_()

  # Use autograd to compute the backward pass.
  loss.backward()

  # Update weights using gradient descent
  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data
```
## TensorFlow: Static Graphs<br>
PyTorch的autograd与TensorFlow相似：两个框架中我们都定义了一个计算图，且使用自动微分来计算梯度。它们之间最大的不同是TensorFlow的计算图是静态的，而PyTorch的是动态的。<br>

在TensorFlow中，我们只定义计算图一次，然后反复调用该图，可能给该图输入不同的数据。在PyTorch中，每次forward传播定义一个新的计算图。<br>

静态图很好，因为你可以对它进行前期优化；例如，为了提高效率，一个框架会融合一些图操作，或者在很多GPUs或机器上进行图分布的策略。如果你重复使用同一张图，则这些潜在的前期优化的耗时会被分摊到一次又一次的重复运行中。<br>

静态和动态图的其中一个区别是控制流。对于一些模型，对于其中的每个数据点，我们可能会希望执行不同的计算；例如对于每个数据点，一个递归网络可以根据不同的时间步骤展开；这个展开可以通过一个循环来实现。对于一个静态图，循环结构是图的一部分；为此TensorFlow提供了诸如tf.scan等操作来将循环结构嵌入图中。对于动态图，这种情况就很简单了：由于我们在运行中为每个示例构件图，我们可以使用正常的命令流控制来对每个输入执行不同的计算。<br>

为了与上面的PyTorch的autograd示例进行对比，我们使用TensorFlow来拟合一个简单的两层网络：<br>
```tf_two_layer_net.py
# Code in file autograd/tf_two_layer_net.py
import tensorflow as tf
import numpy as np

# First we set up the computational graph:

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create placeholders for the input and target data; these will be filled
# with real data when we execute the graph.
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# Create Variables for the weights and initialize them with random data.
# A TensorFlow Variable persists its value across executions of the graph.
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# Forward pass: Compute the predicted y using operations on TensorFlow Tensors.
# Note that this code does not actually perform any numeric operations; it
# merely sets up the computational graph that we will later execute.
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# Compute loss using operations on TensorFlow Tensors
loss = tf.reduce_sum((y - y_pred) ** 2.0)

# Compute gradient of the loss with respect to w1 and w2.
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# Update the weights using gradient descent. To actually update the weights
# we need to evaluate new_w1 and new_w2 when executing the graph. Note that
# in TensorFlow the the act of updating the value of the weights is part of
# the computational graph; in PyTorch this happens outside the computational
# graph.
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# Now we have built our computational graph, so we enter a TensorFlow session to
# actually execute the graph.
with tf.Session() as sess:
  # Run the graph once to initialize the Variables w1 and w2.
  sess.run(tf.global_variables_initializer())

  # Create numpy arrays holding the actual data for the inputs x and targets y
  x_value = np.random.randn(N, D_in)
  y_value = np.random.randn(N, D_out)
  for _ in range(500):
    # Execute the graph many times. Each time it executes we want to bind
    # x_value to x and y_value to y, specified with the feed_dict argument.
    # Each time we execute the graph we want to compute the values for loss,
    # new_w1, and new_w2; the values of these Tensors are returned as numpy
    # arrays.
    loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                feed_dict={x: x_value, y: y_value})
    print(loss_value)
```
## PyTorch: nn<br>
计算图和autograd是定义复杂操作和自动处理衍生品的一种强大范式；然而对于大型的神经网络来说，原生的autograd有点太低级了。<br>

当构造神经网络时，我们经常考虑把计算放进层（layers）中，它们包含了可以学习的参数，能在学习中进行优化。<br>

在TensorFlow中，nn包可以实现该目的。nn包定义了一个Modules的集合，它们类似于神经网络的层。一个Module接收输入Variables并计算输出Variables，但也可以像Variables一样保持内部状态，包含可学习参数。nn包同样也定义了一个在训练神经网络时常用的损失函数的集合。<br>

在这个示例中，我们使用nn包来实现我们的两层网络：<br>
```two_layer_net_nn.py
# Code in file nn/two_layer_net_nn.py
import torch
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        )

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
  # Forward pass: compute predicted y by passing x to the model. Module objects
  # override the __call__ operator so you can call them like functions. When
  # doing so you pass a Variable of input data to the Module and it produces
  # a Variable of output data.
  y_pred = model(x)

  # Compute and print loss. We pass Variables containing the predicted and true
  # values of y, and the loss function returns a Variable containing the loss.
  loss = loss_fn(y_pred, y)
  print(t, loss.data[0])
  
  # Zero the gradients before running the backward pass.
  model.zero_grad()

  # Backward pass: compute gradient of the loss with respect to all the learnable
  # parameters of the model. Internally, the parameters of each Module are stored
  # in Variables with requires_grad=True, so this call will compute gradients for
  # all learnable parameters in the model.
  loss.backward()

  # Update the weights using gradient descent. Each parameter is a Variable, so
  # we can access its data and gradients like we did before.
  for param in model.parameters():
    param.data -= learning_rate * param.grad.data
```
## PyTorch: optim<br>
到目前为止，我们通过手动改变Variables中保存可学习参数的.data成员来更新我们的模型的权重。对于简单的优化算法来说，这不是巨大的负担，例如随机梯度下降，但是在实践中，我们经常使用更加复杂的优化器来训练神经网络，例如：AdaGrad，RMSProp，Adam等。<br>

PyTorch中的optim包借鉴了优化算法的思想并提供常用优化算法的实现。<br>

在本示例中，我们将像之前一样，使用nn包来定义我们的网络，但是我们将使用optim包中提供的Adam算法来优化我们的网络:<br>
```two_layer_net_optim.py
# Code in file nn/two_layer_net_optim.py
import torch
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        )
loss_fn = torch.nn.MSELoss(size_average=False)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Variables it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
  # Forward pass: compute predicted y by passing x to the model.
  y_pred = model(x)

  # Compute and print loss.
  loss = loss_fn(y_pred, y)
  print(t, loss.data[0])
  
  # Before the backward pass, use the optimizer object to zero all of the
  # gradients for the variables it will update (which are the learnable weights
  # of the model)
  optimizer.zero_grad()

  # Backward pass: compute gradient of the loss with respect to model parameters
  loss.backward()

  # Calling the step function on an Optimizer makes an update to its parameters
  optimizer.step()
```
## PyTorch: Custom nn Modules<br>
有时候你想要指定比现有模型序列更复杂的模型；在这种情况下，你可以通过子类nn.Module来定义自己的网络，同时使用其它模块或者其它在Variables上的autograd操作，来定义一个forward负责接收输入Variables和生成输出Variables。<br>

在本示例中，我们将我们的两层网络作为一个自定义模块的子类来实现：<br>
```two_layer_net_module.py
# Code in file nn/two_layer_net_module.py
import torch
from torch.autograd import Variable

class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear2 = torch.nn.Linear(H, D_out)

  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must return
    a Variable of output data. We can use Modules defined in the constructor as
    well as arbitrary operators on Variables.
    """
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
  # Forward pass: Compute predicted y by passing x to the model
  y_pred = model(x)

  # Compute and print loss
  loss = criterion(y_pred, y)
  print(t, loss.data[0])

  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
```
## PyTorch: Control Flow + Weight Sharing<br>
我们实现了一个非常奇特的模型，将其作为一个动态图和权值共享的示例：一个全连接ReLU网络，其中每次forward传播中选择1~4之间的一个随机数，且使随机数大小的隐含层，然后多次重复使用相同的权值来计算最内层的隐含层。<br>

对于该模型，我们使用标准的Python流控制来实现循环，然后在定义forward传播时，我们可以通过多次简单重复使用相同的模块，来实现最内层之间的权值共享。<br>

我们可以简单地将该模型作为一个模块的子类来实现:<br>
```dynamic_net.py
# Code in file nn/dynamic_net.py
import random
import torch
from torch.autograd import Variable

class DynamicNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    """
    In the constructor we construct three nn.Linear instances that we will use
    in the forward pass.
    """
    super(DynamicNet, self).__init__()
    self.input_linear = torch.nn.Linear(D_in, H)
    self.middle_linear = torch.nn.Linear(H, H)
    self.output_linear = torch.nn.Linear(H, D_out)

  def forward(self, x):
    """
    For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
    and reuse the middle_linear Module that many times to compute hidden layer
    representations.

    Since each forward pass builds a dynamic computation graph, we can use normal
    Python control-flow operators like loops or conditional statements when
    defining the forward pass of the model.

    Here we also see that it is perfectly safe to reuse the same Module many
    times when defining a computational graph. This is a big improvement from Lua
    Torch, where each Module could be used only once.
    """
    h_relu = self.input_linear(x).clamp(min=0)
    for _ in range(random.randint(0, 3)):
      h_relu = self.middle_linear(h_relu).clamp(min=0)
    y_pred = self.output_linear(h_relu)
    return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
  # Forward pass: Compute predicted y by passing x to the model
  y_pred = model(x)

  # Compute and print loss
  loss = criterion(y_pred, y)
  print(t, loss.data[0])

  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
```
