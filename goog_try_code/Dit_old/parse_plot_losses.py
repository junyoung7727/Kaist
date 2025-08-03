import torch
x = torch.tensor([1, 2, 3, 4])
print(x.unsqueeze(0))  # tensor([[1, 2, 3, 4]])
print(x.unsqueeze(1))  # tensor([[1], [2], [3], [4]])
print(x.unsqueeze(2)) 