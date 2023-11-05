import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder 
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.SELU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=2),
            nn.BatchNorm2d(8),
            nn.SELU()
            )
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=True) #<<<<<< Bottleneck
        
        #decoder
        # Как работает Conv2dTranspose https://github.com/vdumoulin/conv_arithmetic

        self.unpool = nn.MaxUnpool2d(2, 2)
        
        self.conv1_t = nn.Sequential(
            nn.ConvTranspose2d(8, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.SELU()
            )
        self.conv2_t = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
            )     

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, indicies = self.pool(x) # ⟸ bottleneck
        return x, indicies

    def decode(self, x, indicies):
        x = self.unpool(x, indicies)
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        return x

    def forward(self, x):
        latent, indicies = self.encode(x)
        out = self.decode(latent, indicies)      
        return out