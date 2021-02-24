import numpy as np
import Layer as l
from Tensor import Tensor, SGD




data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

model = l.Sequence([l.Linear(2,3), l.Tanh(), l.Linear(3, 1), l.Sigmoid()])
criterion = l.MSELoss()
optim = SGD(parameters=model.get_parameters(), alpha=1)

for i in range(10):
    pred = model.forward(data)
    
    loss = criterion.forward(pred, target)
    
    loss.backward(Tensor(np.ones_like(loss.data)))
    optim.step()
    print(loss)
    

"""
dim = 0
test = np.array([[1, 2, 3], [4, 5, 6]])
test2 = test.sum(dim)
print(test2)
copies = test.shape[dim]
sp = list(range(0, len(test.shape)))
sp.insert(dim, len(test.shape))
print("1", sp)
new_shape = list(test.shape) + [copies]
print("2", new_shape)
new_data = test.repeat(copies).reshape(new_shape)
print("3", new_data)
new_data = new_data.transpose(sp)
print("4", new_data)
"""