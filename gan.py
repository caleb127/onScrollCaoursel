import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import torchvision

import matplotlib.pyplot as plt
import numpy as np
learning_rate = 2e-4
noise_dim = 32
image_dim = 28 * 28 * 1 
batch_size = 32
num_epochs = 25
class Generator(nn.Module):
    def __init__(self, noise_dim, image_dim):
        super(Generator,self).__init__()
        self.linear1 = nn.Linear(noise_dim, 128)
        self.relu = nn.LeakyReLU(0.01)
        self.linear2 = nn.Linear(128, image_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.tanh(out)
        return out
class Discriminator(nn.Module):
    def __init__(self, in_image):
        super(Discriminator,self).__init__()
        self.linear1 = nn.Linear(in_image, 64)
        self.relu = nn.LeakyReLU(0.01)
        self.linear2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
discriminator = Discriminator(image_dim)
generator = Generator(noise_dim, image_dim)
noise = torch.randn((batch_size, noise_dim))
tf = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,)),]
)
ds = torchvision.datasets.MNIST(root="dataset/", transform=tf, download=True)
loader = data.DataLoader(ds, batch_size=batch_size, shuffle=True)
real_sample = iter(loader).next()[0]
img_grid_real = torchvision.utils.make_grid(real_sample, normalize=True)
npgrid = img_grid_real.cpu().numpy()
plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
plt.axis('off')
opt_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)
opt_generator = optim.Adam(generator.parameters(), lr=learning_rate)
criterion = nn.BCELoss()