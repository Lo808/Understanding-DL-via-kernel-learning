import torch

def gaussian_kernel(x,y,sigma=1.0):
    """
    Computes the Gaussian kernel between two inputs.
    sigma: Spread parameter
    """
    diff=torch.cdist(x,y,p=2)
    return torch.exp(-diff**2/sigma)


def laplacian_kernel(x,y,sigma):
    """
    Computes the Laplacian kernel between two inputs.
    
    sigma: Spread parameter"""

    diff=torch.cdist(x,y,p=2)
    return torch.exp(-diff/sigma)
