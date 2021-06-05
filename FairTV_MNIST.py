from utils import *
from networks import *
import numpy as np
import math
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torchvision.utils import save_image
import argparse
import warnings
import logging 
warnings.filterwarnings('ignore')
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 

class FairTV(BaseModel):
    def __init__(self, n_chan, z_dim, s_dim, img_shape=(64, 64)):
        super().__init__()
        self.img_shape = img_shape
        self.n_chan = n_chan
        self.z_dim = z_dim
        self.s_dim = s_dim
        
        self.encoder = ImageEncoder(n_chan, 2 * z_dim)
        self.decoder = ImageDecoder(z_dim + s_dim, n_chan)
    
    def encode(self, x):
        # P(Z|X,S)
        h = self.encoder(x)
        self.mean = h[:,:self.z_dim]
        self.logvar = h[:,self.z_dim:]
        self.var = torch.exp(self.logvar)
        z = self.reparametrize()
        return z

    def decode(self, z, s):
        # P(X,S|Z,S)
        x_hat = self.decoder(torch.cat([z, F.one_hot(s, self.s_dim)],dim=1))
        return x_hat
    
    def calculate_re(self, x_hat, x):
        return F.binary_cross_entropy(x_hat, x, reduction='sum') / x.shape[0]
    
    def calculate_deltaE(self, s, batch_size):
        def func(index_i, index_j):
            var_i, var_j = self.var[index_i], self.var[index_j]
            mean_i, mean_j = self.mean[index_i], self.mean[index_j]
            item1 = var_i.unsqueeze(1) + var_j
            item2 = (mean_i.unsqueeze(1) - mean_j)**2 / item1
            item2 = torch.exp(-item2.sum(-1) / 2)
            item3 = torch.sqrt((2*math.pi*item1).prod(-1))
            return (item2 / item3).mean()
        
        ans = 0
        num = s.shape[0]
        index_j = s>-1
        item1 = func(index_j, index_j)
        for i in range(self.s_dim):
            index_i = s==i
            num_i = index_i.sum()
            if 0 < num_i < num:
                cur = item1 + func(index_i, index_i) - 2 * func(index_i, index_j)
                ans += (num_i / num)**2 * cur
        ans *= num / batch_size
        return ans
    
    def fit(self, train_data, epochs, lr, batch_size, verbose, beta, device, logger):
        assert beta >= 0
        self.to(device=device)
        self.train()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.parameters(), lr=lr)
        
        for epoch in range(1, epochs+1):
            train_re_loss = 0
            train_deltaE_loss = 0
            num = 0
            for x, s, _ in train_loader:
                x = x.to(device)
                s = s.to(device)
                z = self.encode(x)
                x_hat = self.decode(z, s)
                # loss
                re_loss = self.calculate_re(x_hat, x)
                deltaE_loss = self.calculate_deltaE(s, batch_size)
                kl_loss = self.calculate_kl()
                loss = re_loss + beta * deltaE_loss + kl_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # eval
                train_re_loss += re_loss.detach() * x.shape[0]
                train_deltaE_loss += deltaE_loss.detach() * x.shape[0]
                num += x.shape[0]
            
            train_re_loss = (train_re_loss / num).item()
            train_deltaE_loss = (train_deltaE_loss / num).item() * 1e5
            if verbose > 0 and epoch % verbose == 0:
                logger.info('Epoch {:04d}: train_re_loss={:.5f}, train_deltaE_loss={:.5f}'.format(epoch, train_re_loss, train_deltaE_loss))
        self.to('cpu')
        self.eval()

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--task', type=str)
args = parser.parse_args()

assert args.task in {'colorMNIST', 'MNIST'}
device = torch.device('cpu') if args.cuda < 0 else torch.device('cuda', args.cuda)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)

epochs = 1000
verbose = 200
batch_size = 2048
lr = 1e-3

z_dim = 10
beta = 1e7
save_path = './models'

setSeed(2021)
if args.task == 'colorMNIST':
    train_data, test_data = loadMnist(color=True)
    
    n_chan = 3
    s_dim = 3
    model = FairTV(n_chan=n_chan, z_dim=z_dim, s_dim=s_dim)
    if args.train:
        model.fit(train_data=train_data, epochs=epochs, lr=lr,  batch_size=batch_size, \
           verbose=verbose, beta=beta, device=device, logger=logger)
        if args.save:
            torch.save(model.state_dict(), '{}/FairTV_colorMNIST.pkl'.format(save_path))
    else:
        model.load_state_dict(torch.load('{}/FairTV_colorMNIST.pkl'.format(save_path)))
    
    imgs1 = test_data.X[:12*8]
    imgs2 = model.decode(model.encode(imgs1), torch.zeros(12*8).long())
    imgs3 = model.decode(model.encode(imgs1), torch.zeros(12*8).long()+1)
    imgs4 = model.decode(model.encode(imgs1), torch.zeros(12*8).long()+2)
    
    save_image(imgs1, './colorMNIST_1.pdf', format='pdf', nrow=12)
    save_image(imgs2, './colorMNIST_2.pdf', format='pdf', nrow=12)
    save_image(imgs3, './colorMNIST_3.pdf', format='pdf', nrow=12)
    save_image(imgs4, './colorMNIST_4.pdf', format='pdf', nrow=12)
else:
    train_data, test_data = loadMnist(color=False)
    
    n_chan = 1
    s_dim = 10
    model = FairTV(n_chan=n_chan, z_dim=z_dim, s_dim=s_dim)
    if args.train:
        model.fit(train_data=train_data, epochs=epochs, lr=lr,  batch_size=batch_size, \
            verbose=verbose, beta=beta, device=device, logger=logger)
        if args.save:
            torch.save(model.state_dict(), '{}/FairTV_MNIST.pkl'.format(save_path))
    else:
        model.load_state_dict(torch.load('{}/FairTV_MNIST.pkl'.format(save_path)))
    
    imgs1 = test_data.X[20:30]
    X = torch.cat([imgs1]*10, dim=0)
    X = X.view(10,10,1,64,64).transpose(0,1).contiguous().view(-1,1,64,64)
    S = torch.LongTensor([range(10)]*10).view(-1)
    imgs2 = model.decode(model.encode(X), S)
    
    save_image(imgs1, './MNIST_1.pdf', format='pdf', nrow=1)
    save_image(imgs2, './MNIST_2.pdf', format='pdf', nrow=10)

print('finish')
