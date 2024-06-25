
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# from tqdm import *
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torchvision.utils import save_image
from sklearn.cluster import KMeans
from random import sample


nmi = normalized_mutual_info_score
ari = adjusted_rand_score


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(21, 12),
            nn.Tanh(),
            nn.Linear(12, 6),
            nn.Tanh(),
            nn.Linear(6, 3))
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.Tanh(),
            nn.Linear(6, 12),
            nn.Tanh(),
            nn.Linear(12, 21))
        
        self.model = nn.Sequential(self.encoder, self.decoder)
        
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.model(x)
        return x


def add_noise(data):
    noise = (torch.randn(data.size()) * 0.0001)
    noisy_data = data + noise
    return noisy_data


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


def train(**kwargs):
    train_data = kwargs['train_data']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']    
    batch_size = kwargs['batch_size']
    checkpoint = kwargs['checkpoint']
    lr = kwargs['lr']
    file_path_prefix = kwargs['file_path_prefix']
    start_epoch = checkpoint['epoch']
    

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-5)

    data_loader = DataLoader(dataset=train_data,
                    batch_size=batch_size, 
                    shuffle=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss_list = []
    for epoch in range(start_epoch, num_epochs):
        loss = 0
        for data in data_loader:
            raw_data  = data.float()
            noisy_data = add_noise(raw_data)
            noisy_data = noisy_data.to(device)

            raw_data = raw_data.to(device)
            # ===================forward=====================
            output = model(noisy_data)
            
            output = output.squeeze(1)
            
            loss = nn.MSELoss()(output, raw_data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # ===============adjust_learning_rate================
        if lr > 1e-12:
            lr = lr * (0.1**(epoch // 200))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # =================log====================
        print('epoch [{}/{}], MSE_loss:{:.5f}'
          .format(epoch + 1, num_epochs, loss.item()))
        
        loss_list.append(loss.item())
        
        state = loss.item()
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best': state,
                        'epoch':epoch
                        }, savepath,
                        is_best)

    df = pd.DataFrame({"train":loss_list})
    df.to_excel(file_path_prefix+'Pretrain-KL-Loss.xlsx')
    
    fig = plt.figure()
    plt.plot(loss_list)
    fig.savefig(file_path_prefix+'Pretrain-KL-Loss.png')
    plt.close(fig)
    