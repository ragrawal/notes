## Pytorch Tensor Types

| Data Type | dtype | Tensor Types
|------------|-------|------------|
|32 bit floating | torch.float32 or torch.float | torch.*.FloatTensor|
|64 bit floating | torch.float64 or torch.double | torch.*.DoubleTensor |
|16 bit floating | torch.float16 or torch.half | torch.*.HalfTensor |
|8 bit integer (unsigned) | torch.uint8 | torch.*.ByteTensor |
|8 bit integer (signed) | torch.int8 | torch.*.CharTensor |
|16 bit integer (signed) | torch.int16 or torch.short | torch.*.ShortTensor |
|32 bit integer (signed) | torch.int32 or torch.int | torch.*.IntTensor |
|64 bit integer (signed) | torch.int64 or torch.long | torch.*.LongTensor |


## Useful Commands
* `type`: Change data type : `a.type(torch.float)`
* `view`: Reshape Tensor: `a.view(-1, 5)` to convert to a matrix with 5 columns
* `from_numpy`: numpy -> pytorch
* `.numpy()`: Pytorch -> numpy
* `tolist`: return list 
* `item()`: return python data type
* `*`: element wise product of two tensors (also known as Hadamard Product)
* `+`: element wise sum of two tensors
* `torch.dot`: dot product of two vectors
* `torch.mm`: matrix multiplication of 2D tensors
* `linespace`: generates equally spaced elements between starting and endpoint. `torch.linespace(-2, 2, num=5)` will give [-2, -1, 0, 1, 2]
* `ndimension`: returns number of dimensions.
* `shape`: returns the size of each dimension
* `size`: returns number of elements
* `numel`: returns total number of elements




## Gradient Calculation 

```
# indicate the value around which 
# we will be calculating gradient
x = torch.tensor(2, requires_grad=True)

# define the curve for which we need 
# to calculate gradient
y = x**2  

# tell pytorch to calculate gradient for y
y.backward()

# calculate gradient value 
x.grad
```

## Partial Derivative
```
u=torch.tensor(1, requires_grad=True)
v=torch.tensor(2, requires_grad=True)
f = u*v + u**2
f.backward()

# calculates derivative of f wrt v
v.grad 

# calculates derivative of f wrt u
u.grad
```

## Detach Operation
Use `detach` operation to exclude further tracking of operations in the graph


## Dataset
1. Create a subclass of Dataset
2. Impelement `__len__(self)` method that returns number of samples 
3. Impelement `__getitem__(self, index)` method that returns a particular sample at a given index. The result is a tuple where the first element is related to X and second element is reated to Y. Note that if any transformations are defined they should be applied before returning the sample. 

```
from torch.utils.ata import Dataset

# Create a subclass of 
class ToySet(Dataset):
	
	def __init__(self, transform=None):
		self.transform = transform
		
	def __getitem__(self, index):
		sample = self.x[index], self.y[index]
		if self.transform:
			sample = self.transform(sample)
		return sample
	
	def __len__(self):
		return self.x.shape[0]
```

## transforms
Use `torchvision.transforms.Compose` to generate a sequence of transformers. 
Each transformer has to implement the following template

```
class MyTransformer(object):
	def __init__(self, parameters):
		pass
	
	def __call__(self, samples):
		# return a tuple of two elements (x, y)
		return samples
		
t = transforms.Compose([MyTransformer1(), MyTransformer2()])
```

## Linear Regression Prediction

Note: we don't need to call forward method explicitly. Like call method
we can just call it using parenthesis. 

```
import torch.nn as nn

class LR(nn.Module):
	
	def __init__(self, in_size, output_size):
		super(LR, self).__init__()
		# input size does not include bias term
		self.linear = nn.Linear(in_size, output_size)
	

	def forward(self, X):
		out = self.linear(X)
		return out
		
model=LR(1,1)
print(list(model.parameters())) # this forces gradient descent
yhat = model(torch.tensor([[1]])) # this will call the formward method
```

## Optimizer Tips
1. `optimizer.zero_grad()` -- set gradient to zero 
2. `optimizer.step()` -- updates the parameters. Note this should be called after running `backward` on loss criterion


## Gradient Descent 

```
import torch
lr = 0.1

def formward(x):
	y = w*x + b
	return y

def criterion(yhat, y):
	return torch.mean((yhat-y)**2)
	
for epoch in range(4)

	yhat = forward(X)
	loss = criterion(yhat, Y)  # in pytorch cost is referred as loss
	loss.backward() # run backward to compute gradient
	w.data = w.data - lr * w.grad.data # update paramters
	w.grad.data.zero_() # reset gradient parameters as gradient is computed in an iterative manner
	cost.append(loss.item())
		
```


## Module Abstractions (Sequential, nn.ModuleList)

Consider following network architecture example and how to leverage different abstractions

**Verbose**

```
class Net(nn.Module):
    
    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        self.l1 = nn.Linear(D_in, 50)
        self.l2 = nn.Linear(50, 40)
        self.l3 = nn.Linear(40, D_out)
    
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x
  
# using ModuleList()
class Net(nn.Module):
    def __init__(self,D_in, D_out):
        super(Net, self).__init__()
        
         # layers can an input argument 
         # for arbitrary number of layers
        layers = [D_in, 50, 40, D_out]        
        self.hidden = nn.ModuleList()
        
        # iterate over two elements 
        for inputSize, outSize in zip(layers, layers[1:]):
            self.hidden.append(nn.Linear(inputSize, outSize))
    
    def forward(self, x):
        l = len(self.hidden)
        for idx, layer in enumerate(self.hidden):
            if idx < l-1: # intermediate layers
                activation = torch.relu(layer(activation))
            else:
                activation = layer(activation)
        return activation
```

## Dropout
* use ``nn.Dropout(p=0.1)`` to implement dropout. P value indicates probability of zero. Thus, P=0.1 indicates that probability of zero is 10 percent. 
* Use low p value for layers with few neurons. use high P value for layers with lot of neurons.
* Call `model.train()` to enable dropout during training of the network
* Call `model.eval()` to disable dropout during evaluation

## Weight Initialization
**Xavier Method:**
```
linear = nn.Linear(input_size, output_size)
torch.nn.init.xavier_uniform_(linear.weight)
```

**He Method**:
```
linear = nn.Linear(input_size, output_size)
torch.nn.init.kaiming_uniform_(linear.weight)
```

## Batch Normalization
Advantages
* Reduces internal covariate shift
* remove dropout
* increases learning rate
* bias is not necessary

## Convolution
**size of output layer**
* Mx, My = Size of Original Matrix
* Kx, Ky = Size of Kernel
* Sx, Sy = Size of stride
* Px, Py = Padding Size
* $M'_x, M'_y$ = Size of output matrix

$$M'_x = (M_x - K_x + P_x)/S_x + 1$$
$$M'_y = (M_y - K_y + P_y)/S_y + 1$$

```
nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
```
Above `out_channels` refers to the number of convolution kernels to apply. The final output will be linear sum of kernel application on the input channel. `in_channel` refers to the number of input layers. For example an image has RGB color scheme and thus there are 3 input channels. 


