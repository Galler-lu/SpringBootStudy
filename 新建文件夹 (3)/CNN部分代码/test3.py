import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
if __name__ == '__main__':
    # x=[[[1,2,3],[2,3,4,6]]]
    # x=np.array(x)
    # print(x)
    x=[[1,2,3,4],[1,2,3,5]]
    y=[5,6,7]
    print(x+y)