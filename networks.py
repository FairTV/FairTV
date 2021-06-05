import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

class VectorEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(VectorEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, X):
        return self.net(X)

class VectorDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(VectorDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, X):
        return self.net(X)

class ImageEncoder(nn.Module):
    def __init__(self, n_chan, out_dim, img_shape=(64, 64)):
        super(ImageEncoder, self).__init__()
        self.net = nn.Sequential(
                nn.Conv2d(n_chan, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),
                nn.ReLU(True),
                nn.Conv2d(256, out_dim, 1),
                nn.Flatten(),
            )
    
    def forward(self, X):
        return self.net(X)

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
        
    def forward(self, X):
        return X.unsqueeze(-1).unsqueeze(-1)

class ImageDecoder(nn.Module):
    def __init__(self, in_dim, n_chan, img_shape=(64, 64)):
        super(ImageDecoder, self).__init__()
        self.net = nn.Sequential(
            Reshape(),
            nn.Conv2d(in_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, n_chan, 4, 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, X):
        return self.net(X)

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.mean = None
        self.var = None
        self.logvar = None
    
    def reparametrize(self):
        gaussian_noise = torch.randn_like(self.mean, device=self.mean.device)
        Z = self.mean + gaussian_noise * torch.sqrt(self.var)
        return Z
    
    def calculate_kl(self):
        return -0.5 * (1 + self.logvar - self.mean**2 - self.var).sum() / self.mean.shape[0]
    
    def calculate_re(self, x_hat, T, D):
        ans = 0
        left, right = 0, 0
        for i, length in enumerate(D):
            right = left + length
            ans += F.cross_entropy(x_hat[:, left:right], T[:, i])
            left = right
        return ans

