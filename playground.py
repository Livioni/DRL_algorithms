import torch

t3=torch.tensor([[0.5317, 0.4683],
        [0.5355, 0.4645],
        [0.5545, 0.4455],
        [0.5294, 0.4706],
        [0.5418, 0.4582]])

result=t3.gather(1,torch.tensor([[0],[1],[1],[1],[0]]))
print(result)
