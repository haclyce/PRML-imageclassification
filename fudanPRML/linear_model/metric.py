import torch
def accuracy(preds, labels):

    preds = torch.argmax(preds,dim=1)
    return torch.sum(preds==labels)/preds.shape[0]
#计算准确率