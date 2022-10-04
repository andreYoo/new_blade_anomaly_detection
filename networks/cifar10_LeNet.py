import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
import numpy as np
import pdb


class CIFAR10_LeNet(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(15360, self.rep_dim, bias=False)

    def forward(self, x):
        #print(np.shape(x))
        x = x.view(-1, 1, 31, 320)
        x = self.conv1(x)
        #print(np.shape(x))
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        #print(np.shape(x))
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        #print(np.shape(x))
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        #print(np.shape(x))
        x = x.view(int(x.size(0)), -1)
        #print(np.shape(x))
        x = self.fc1(x)
        #print(np.shape(x))
        return x


class CIFAR10_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.fc1 = nn.Linear(self.rep_dim,15360, bias=False)
        self.deconv1 = nn.ConvTranspose2d(128, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=(1,2))
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 1, (6,5), bias=False, padding=(1,2))
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = F.leaky_relu(x)
        x = self.fc1(x)
        x = x.view(int(x.size(0)), 128, 3, 40)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        #print(np.shape(x))
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        #print(np.shape(x))
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        #print(np.shape(x))
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        #print(np.shape(x))
        x = torch.sigmoid(x)
        return x


class CIFAR10_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = CIFAR10_LeNet(rep_dim=rep_dim)
        self.decoder = CIFAR10_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x,get_latent=False):
        latent = self.encoder(x)
        out = self.decoder(latent)
        if get_latent==True:
            return out, latent
        else:
            return out

class Linear_BN_leakyReLU(nn.Module):
    """
    A nn.Module that consists of a Linear layer followed by BatchNorm1d and a leaky ReLu activation
    """

    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_leakyReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))
