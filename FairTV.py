from utils import *
from networks import *
import numpy as np
import math
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import argparse
import warnings
import logging 
warnings.filterwarnings('ignore')
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 

class FairTV(BaseModel):
    def __init__(self, x_dim, h_dim, z_dim, s_dim):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.encoder = VectorEncoder(x_dim+s_dim, h_dim, 2*z_dim)
        self.decoder = VectorDecoder(z_dim+s_dim, h_dim, x_dim)

    def encode(self, x, s):
        # P(Z|X,S)
        h = self.encoder(torch.cat([x, F.one_hot(s, self.s_dim)], dim=1))
        self.mean = h[:,:self.z_dim]
        self.logvar = h[:,self.z_dim:]
        self.var = torch.exp(self.logvar)
        z = self.reparametrize()
        return z
    
    def decode(self, z, s):
        # P(X|Z,S)
        x_hat = self.decoder(torch.cat([z, F.one_hot(s, self.s_dim)], dim=1))
        return x_hat
    
    def calculate_deltaE(self, s, batch_size):
        def func(index_i, index_j):
            var_i, var_j = self.var[index_i], self.var[index_j]
            mean_i, mean_j = self.mean[index_i], self.mean[index_j]
            item1 = var_i.unsqueeze(1) + var_j
            item2 = (mean_i.unsqueeze(1) - mean_j)**2 / item1
            item2 = torch.exp(-0.5 * item2.sum(-1))
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
            for x, t, s, _ in train_loader:
                x = x.to(device)
                t = t.to(device)
                s = s.to(device)
                z = self.encode(x, s)
                x_hat = self.decode(z, s)
                # loss
                re_loss = self.calculate_re(x_hat, t, train_data.D)
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

assert args.task in {'Adult', 'Compas', 'German', 'Health'}
device = torch.device('cpu') if args.cuda < 0 else torch.device('cuda', args.cuda)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)

setSeed(2021)
if args.task == 'Adult':
    train_data, test_data = loadAdult()
elif args.task == 'Compas':
    train_data, test_data = loadCompas()
elif args.task == 'German':
    train_data, test_data = loadGerman()
elif args.task == 'Health':
    train_data, test_data = loadHealth()

S_train, S_test = train_data.S.numpy(), test_data.S.numpy()
Y_train, Y_test = train_data.Y.numpy(), test_data.Y.numpy()

batch_size = 2048
epochs = 1000
verbose = 200
lr = 1e-3
x_dim = train_data.X.shape[1]
s_dim = train_data.S.max().item()+1
h_dim = 64
z_dim = 8

save_path = './models/{}'.format(args.task)
n_iter = 10

logs = []
for i in range(11):
    beta = 10**i
    model = FairTV(x_dim, h_dim, z_dim, s_dim)
    if args.train:
        model.fit(train_data=train_data, epochs=epochs, lr=lr, batch_size=batch_size, verbose=verbose, beta=beta, device=device, logger=logger)
        if args.save:
            torch.save(model.state_dict(), '{}/FairTV_{}.pkl'.format(save_path, i))
    else:
        model.load_state_dict(torch.load('{}/FairTV_{}.pkl'.format(save_path, i)))
    
    for _ in range(n_iter):
        with torch.no_grad():
            Z_train = model.encode(train_data.X, train_data.S).numpy()
            Z_test = model.encode(test_data.X, test_data.S).numpy()
            deltaE = model.calculate_deltaE(test_data.S, test_data.S.shape[0]).item() * 1e5
        
        eval_res, cols = evaluate(Z_train, Z_test, S_train, S_test, Y_train, Y_test)
        logs.append([i, deltaE] + eval_res)
    
    pd.DataFrame(logs, columns=['log_beta', 'deltaE']+cols)\
        .to_csv('./FairTV_{}.csv'.format(args.task), index=False)

print('finish')
