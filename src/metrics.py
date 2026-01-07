import torch

def accuracy(y_true,y_pred):
    """
    Computes the accuracy between the prediction and true labels.
    """

    correct=(y_true==y_pred).sum().item()
    total=y_true.size(0)
    return correct/total

def precision(y_true,y_pred):
    """
    Computes the precision between the prediction and true labels.
    """

    correct=(y_true==y_pred).sum().item()
    total=(y_pred==1).sum().item()
    return correct/total

def mse(y_true,y_pred):
    return torch.cdist(y_true,y_pred,p=2)