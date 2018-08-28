
# Quick 15 minute (ish) PyTorch Demo

By Joshua Mitchell

https://lelon.io

jlelonmitchell@gmail.com

https://www.linkedin.com/in/joshua-mitchell-17b94077/

## Some quick Jupyter Notebooks features


```python
# Install a conda package in the current Jupyter kernel

# import sys
# !conda install --yes --prefix {sys.prefix} <package name>
```


```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # makes jupyter's auto-print functionality work for everything in cell
```


```python
%load_ext autoreload
%autoreload 2

# e.g. : 

# from foo import some_function

# In [1]: some_function()
# Out[1]: 42

# In [2]: # open foo.py in an editor and change some_function to return 43

# In [3]: some_function()
# Out[3]: 43
```


```python
# make any plots you create with matplotlib be displayed here in your jupyter notebook
%matplotlib inline
```


```python
%autosave 30
```



    Autosaving every 30 seconds



```python
# Command palette (for those who think clicking is cumbersome, like me):
# CMD + Shift + P (might be CTRL + Shift + P or something on Windows)
```

#### Esc     -   command mode (navigation)
#### Enter  -   edit mode (typing)

#### D + D (press twice) to delete current cell

#### Shift + Tab   -   brings up the docstring for the object to the left of your cursor

#### Esc + O   -  toggle cell output

#### J, K   -   move up and down with vim commands

#### l (in command mode)    -   show line numbers

## Anyhow, back to PyTorch...


```python
# Let's make some random data

import matplotlib.pyplot as plt
import numpy as np

NUM_DATA_POINTS = 100

x = np.arange(NUM_DATA_POINTS) # create a an array: [0, 1, 2... NUM_DATA_POINTS]
noise = np.random.uniform(-5,5, size=(NUM_DATA_POINTS,)) # create some noise to add to a truly straight line
y = -0.5 * x + 2 + noise # create a straight line plus some noise

axes = plt.gca()
line, = axes.plot(list(x), list(y), 'bs')
```


![png](output_10_0.png)


## Awesome, we have some data. Looks like fair game for a linear regression model.


```python
# Data setup:

test_data_amount     = int(0.2 * NUM_DATA_POINTS)
training_data_amount = NUM_DATA_POINTS - test_data_amount

all_data_indices = [i for i in range(NUM_DATA_POINTS)]

x_test  = np.random.choice(all_data_indices, test_data_amount, replace=False)
# get the indices of 20% of the data for testing

x_train = list(set(all_data_indices).difference(set(x_test)))
# get the indices of the rest for training

y_test  = [y[i] for i in x_test]
y_train = [y[i] for i in x_train]

# Enter: PyTorch

import torch
from torch.autograd import Variable

x_train = torch.Tensor(x_train)
# an n-dimensional array : array (1d) --> matrix (2d) --> tensor (3d) --> tensor (4d) ...)

x_train_var = Variable(x_train) 
# wraps the tensor in a Variable class to add extra features we'll see later

x_test = torch.Tensor(x_test)
x_test_var = Variable(x_test)

# Let's check out our variables:

x_test_var
len(x_test_var)

x_train_var
len(x_train_var)
```




    tensor([27., 65., 79., 61., 75., 34.,  3., 40., 99., 52., 56., 48., 19., 82.,
            16., 60., 86., 92., 41., 30.])






    20






    tensor([ 0.,  1.,  2.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,
            15., 17., 18., 20., 21., 22., 23., 24., 25., 26., 28., 29., 31., 32.,
            33., 35., 36., 37., 38., 39., 42., 43., 44., 45., 46., 47., 49., 50.,
            51., 53., 54., 55., 57., 58., 59., 62., 63., 64., 66., 67., 68., 69.,
            70., 71., 72., 73., 74., 76., 77., 78., 80., 81., 83., 84., 85., 87.,
            88., 89., 90., 91., 93., 94., 95., 96., 97., 98.])






    80



## Sweet. Now let's actually make the linear model.

#### Quick linear regression review:

The line we're creating is:

y = mx + b

We have the data (the x's and the y's).

We want to know what the m and b are such that the line fits the data the best.

For now, let's just make an arbitrary line with PyTorch:


```python
import torch.nn as nn ## Neural Network package

line_model = nn.Linear(1, 1) # create a line with 1 slope (i.e. weight) and one intercept (i.e. bias)

# Done! We made a line. What's it look like?

line_model
```




    Linear(in_features=1, out_features=1, bias=True)



#### Notice: we're barely using any of the features here. 

We can actually make it with as many weights and as many outputs as we want. We'll do that later.

Speaking of weights, how do we find out what the m and b actually are?


```python
list(line_model.parameters())
```




    [Parameter containing:
     tensor([[-0.5528]], requires_grad=True), Parameter containing:
     tensor([0.2260], requires_grad=True)]



Note: These are randomly generated.

#### How do we make the line fit the data?


```python
test_input = Variable(torch.Tensor(x_test_var))
line_model(test_input)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-11-629c18b1ff76> in <module>()
          1 test_input = Variable(torch.Tensor(x_test_var))
    ----> 2 line_model(test_input)
    

    ~/anaconda3/envs/eopc/lib/python3.6/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        475             result = self._slow_forward(*input, **kwargs)
        476         else:
    --> 477             result = self.forward(*input, **kwargs)
        478         for hook in self._forward_hooks.values():
        479             hook_result = hook(self, input, result)


    ~/anaconda3/envs/eopc/lib/python3.6/site-packages/torch/nn/modules/linear.py in forward(self, input)
         53 
         54     def forward(self, input):
    ---> 55         return F.linear(input, self.weight, self.bias)
         56 
         57     def extra_repr(self):


    ~/anaconda3/envs/eopc/lib/python3.6/site-packages/torch/nn/functional.py in linear(input, weight, bias)
       1024         return torch.addmm(bias, input, weight.t())
       1025 
    -> 1026     output = input.matmul(weight.t())
       1027     if bias is not None:
       1028         output += bias


    RuntimeError: size mismatch, m1: [1 x 20], m2: [1 x 1] at /Users/soumith/miniconda2/conda-bld/pytorch_1532623076075/work/aten/src/TH/generic/THTensorMath.cpp:2070


## Tensors are tricky. What's going on here?

### What we wanted to do:

- Take the x values of our test data (which was literally the indices)
- Run all those values through our line, one at a time (i.e. put an x in, get a y)
- See the results

### What PyTorch thinks we want to do:

- Take all the x values of our test data
- Fit all the x's into the line at once, as though our line had 20 inputs

Our line only has one input. So what's the solution?

### Answer:

Reshape our tensor.


```python
test_input = Variable(torch.Tensor(x_test_var))
"BEFORE"
test_input
"AFTER"
test_input.view(-1, 1)
```




    'BEFORE'






    tensor([27., 65., 79., 61., 75., 34.,  3., 40., 99., 52., 56., 48., 19., 82.,
            16., 60., 86., 92., 41., 30.])






    'AFTER'






    tensor([[27.],
            [65.],
            [79.],
            [61.],
            [75.],
            [34.],
            [ 3.],
            [40.],
            [99.],
            [52.],
            [56.],
            [48.],
            [19.],
            [82.],
            [16.],
            [60.],
            [86.],
            [92.],
            [41.],
            [30.]])




```python
line_model(test_input.view(-1, 1)) 
```




    tensor([[-14.6997],
            [-35.7063],
            [-43.4456],
            [-33.4951],
            [-41.2344],
            [-18.5694],
            [ -1.4324],
            [-21.8862],
            [-54.5017],
            [-28.5199],
            [-30.7311],
            [-26.3087],
            [-10.2773],
            [-45.1040],
            [ -8.6189],
            [-32.9423],
            [-47.3153],
            [-50.6321],
            [-22.4390],
            [-16.3582]], grad_fn=<ThAddmmBackward>)



It works!

## Ugh, making lines is hard.

What are we doing again?


```python
axes = plt.gca()
line, = axes.plot(list(x_train), [-0.5 * x + 2 for x in list(x_train)], 'r-')
plt.show()
```


![png](output_24_0.png)


### Above: The line we're trying to recreate.

### Below: The data we're using to recreate the line above.


```python
axes = plt.gca()
line, = axes.plot(list(x_train), list(y_train), 'bs')
plt.show()
```


![png](output_26_0.png)


### Our line currently:


```python
axes = plt.gca()
line, = axes.plot(list(x_train), [-0.5 * x + 2 for x in list(x_train)], 'r-')
line, = axes.plot(list(x_train), list(line_model(x_train.view(-1, 1))), 'bs')
plt.show()
```


![png](output_28_0.png)


### Okay, so we're a little off. How do we fix it?

### Answer: Some kind of optimization on a loss function.


```python
# Now, let's bring in all the big guns from PyTorch.
# Fair warning, there's a lot of machine learning / deep learning vocabulary here. Can get kitchen sink-ey.

import torch.optim as optim # Optimization package

NUMBER_OF_EPOCHS = 30000       # Number of times to adjust our model
LEARNING_RATE = 1e-2           # How quickly our model "learns"
loss_function = nn.MSELoss()   # The loss function we use. In this case, Mean Squared Error (since it's Linear Regression)
optimizer = optim.Adagrad(line_model.parameters(), lr=LEARNING_RATE) # The optimizer: Stochastic Gradient Descent

y_train_var = Variable(torch.Tensor(y_train)).view(-1, 1) # making our labels into a form usable by PyTorch

for epoch in range(NUMBER_OF_EPOCHS):
    line_model.zero_grad()
    output = line_model(x_train_var.view(-1, 1)) # get our output from the model
    loss = loss_function(output, y_train_var)    # calculate the loss
    loss.backward()                              # calculate all the partial derivatives wrt to the loss function
    optimizer.step()                             # add or subtract a portion of the derivatives from each weight / bias


```


```python
list(line_model.parameters()) # should be -0.5 and 2 (ish)
```




    [Parameter containing:
     tensor([[-0.4958]], requires_grad=True), Parameter containing:
     tensor([2.0268], requires_grad=True)]




```python
axes = plt.gca()
line, = axes.plot(list(x_train), [-0.5 * x + 2 for x in list(x_train)], 'r-')
line, = axes.plot(list(x_train), list(line_model(x_train.view(-1, 1))), 'bs')
plt.show()
```


![png](output_32_0.png)


## Yeah, yeah, yeah, enough with the lines, I came here for the neural networks!

Ok fine. Here's a demo for that:


```python
x_train = Variable(torch.Tensor([[i] for i in range(100)]), requires_grad=False)
y_train = Variable(torch.Tensor([[i] for i in range(100)]), requires_grad=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

net = Net() # we made a blueprint above for our neural network, now we initialize one.

axes = plt.gca()
line, = axes.plot(list(x_train), list(net(x_train)), 'bs')
line, = axes.plot(list(x_train), list(x_train), 'r-')
plt.show()

# Red: The identity function: y = x (we're trying to approximate it)
# Blue: What our neural network does now
```


![png](output_34_0.png)



```python
%matplotlib inline

NUMBER_OF_EPOCHS = 10000
LEARNING_RATE = 0.1
loss_function = nn.MSELoss()
optimizer = optim.Adagrad(net.parameters(), lr=LEARNING_RATE) 

import pylab as pl
from IPython import display

for epoch in range(NUMBER_OF_EPOCHS):
    net.zero_grad()
    output = net(x_train)
    loss = loss_function(output, y_train)
    loss.backward()
    optimizer.step()
    
    display.clear_output(wait=True)
    axes = plt.gca()
    line, = axes.plot(list(x_train), list(net(x_train)), 'bs')
    line, = axes.plot(list(x_train), list(x_train), 'r-')
    plt.show()
```


![png](output_35_0.png)


It's (kind of) learning what y = x is!
