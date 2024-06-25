
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from tqdm import *
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torchvision.utils import save_image
from sklearn.cluster import KMeans
from random import sample

# nmi = normalized_mutual_info_score
# ari = adjusted_rand_score


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(18, 24),
            nn.Tanh(),
            nn.Linear(24, 12),
            nn.Tanh(),
            nn.Linear(12, 6),
            nn.Tanh(),
            nn.Linear(6, 3))
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.Tanh(),
            nn.Linear(6, 12),
            nn.Tanh(),
            nn.Linear(12, 24),
            nn.Tanh(),
            nn.Linear(24, 18))
        
        self.model = nn.Sequential(self.encoder, self.decoder)
        
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.model(x)
        return x


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=5, hidden=3, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
            self.n_clusters,
            self.hidden,
            dtype=torch.float
            ).cuda()
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        
        self.cluster_centers = Parameter(initial_cluster_centers)
        
    def forward(self, x):        
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers)**2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t() #soft assignment using t-distribution
        return t_dist


class DEC(nn.Module):
    def __init__(self, n_clusters=5, autoencoder=None, hidden=3, cluster_centers=None, alpha=1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(self.n_clusters, self.hidden, self.cluster_centers, self.alpha)

    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        x = self.autoencoder.encode(x)         

        return self.clusteringlayer(x)

    def visualize(self, epoch, x, file_path_prefix):
        fig = plt.figure()
        ax = plt.subplot(111)
        
        x = self.autoencoder.encode(x).detach() 
        
        x = x.cpu().numpy()[:2000]
        x_embedded = TSNE(n_components=2, random_state=2, n_jobs=-1).fit_transform(x) # method='exact'
        plt.scatter(x_embedded[:,0], x_embedded[:,1])
        fig.savefig(file_path_prefix + '{}-epochs-{}-clusters.png'.format(epoch, self.n_clusters))
        plt.close(fig)


# denoising autoencoder[DAE]
def add_noise(data):
    noise = (torch.randn(data.size()) * 0.0001)
    noisy_data = data + noise
    return noisy_data


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        # print("=> Saving new checkpoint")
        torch.save(state, filename)


def pretrain(**kwargs):
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
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-4) #lr=1e-5, weight_decay=1e-4

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
        if lr > 1e-10:
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
    df.to_excel(file_path_prefix + 'Pretrain-KL-Loss.xlsx')
    
    fig = plt.figure()
    plt.plot(loss_list)
    fig.savefig(file_path_prefix + '/Pretrain-KL-Loss.png')
    plt.close(fig)
    

def train(**kwargs):
    data = kwargs['data']
    n_cluster = kwargs['n_cluster']
    draw_pic = kwargs['draw_pic']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    lr = kwargs['lr']
    savepath = kwargs['savepath']
    batch_size = kwargs['batch_size']
    checkpoint = kwargs['checkpoint']
    file_path_prefix = kwargs['file_path_prefix']
    start_epoch = checkpoint['epoch']
    
    features = []
    data_loader = DataLoader(dataset=data,
                            batch_size=batch_size, 
                            shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for i, batch in enumerate(data_loader):
        img = batch.float()
        img = img.to(device)
        features.append(model.autoencoder.encode(img).detach().cpu())
        
    features = torch.cat(features)
    
    # ============K-means=======================================
    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init=(50*n_cluster)).fit(features)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    
    # =========================================================
    loss_function = nn.KLDivLoss(reduction='sum')
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.95)
    print('Training')
    
    loss_list = []

    for epoch in range(start_epoch, num_epochs):
        batch = data
        img = batch.float()
        img = img.to(device)
        output = model(img)
        target = model.target_distribution(output).detach()
        
        if epoch % 5000 == 0 and draw_pic:
            print('plotting')
            model.visualize(epoch, img, file_path_prefix)
            
        loss = loss_function(output.log(), target) / output.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
        if epoch % 1000 == 0:
            print('Epochs: [{}/{}] Loss:{}'.format(epoch, num_epochs, loss))
        
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

    df = pd.DataFrame(loss_list)
    df.to_excel(file_path_prefix + '/KL-Loss-' + str(n_cluster) + '-.xlsx')
    
    fig = plt.figure()
    plt.plot(loss_list)
    fig.savefig(file_path_prefix + '/KL-Loss-{}-clusters.png'.format(n_cluster))
    plt.close(fig)
