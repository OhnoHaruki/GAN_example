import torch
import torch.nn as nn

class Discriminator(nn.Module): #識別モデル
    
    def __init__(self, image_size, hidden_size):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(image_size, hidden_size),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(hidden_size, 1),
                                 nn.Sigmoid())
    
    def forward(self, x):
        out = self.net(x)
        return out

class Generator(nn.Module): #生成モデル
    
    def __init__(self,image_size, latent_size, hidden_size):
        super(Generator, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(latent_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, image_size),
                                 nn.Tanh())
        
    def forward(self,x):
        out = self.net(x)
        return out