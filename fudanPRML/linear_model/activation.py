import torch
def softmax(X):
    x_max = torch.max(X, dim=1, keepdim=True).values
    x_exp = X - x_max
    x_exp = torch.exp(x_exp)
    partition = torch.sum(x_exp, dim=1, keepdim=True) 
    results=x_exp/partition
    return results