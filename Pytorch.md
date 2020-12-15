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

## define the curve for which we need 
# to calculate gradient
y = x**2  

## tell pytorch to calculate gradient for y
y.backward()

## calculate gradient value 
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

