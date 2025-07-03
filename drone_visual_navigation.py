#https://neurohive.io/ru/tutorial/glubokoe-obuchenie-s-pytorch/
import torch
x = torch.ones(2, 2, requires_grad=True) * 2
print(x)