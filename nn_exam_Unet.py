## The current version of the file is set up to load and train on CIFAR-10 with Unet.
## This code can be used for other datasets by appropriately modifying the code.

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os, random
import torch.nn.functional as F
from util import *
from SHNAG_optim import SHANG, ISHANG

# 中量 UNet
class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(base_ch*2, base_ch*4, 3, stride=2, padding=1), nn.ReLU(True))
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1), nn.ReLU(True))
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch,   4, stride=2, padding=1), nn.ReLU(True))
        self.dec1 = nn.Conv2d(base_ch*2, out_ch, 3, padding=1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        out = self.dec1(d2)
        return torch.sigmoid(out)

class LightweightUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(base_ch*2, base_ch*4, 3, stride=2, padding=1), nn.ReLU(True))

        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch*4 + base_ch*2, base_ch*2, 3, padding=1),
            nn.ReLU(True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2 + base_ch, base_ch, 3, padding=1),
            nn.ReLU(True)
        )
        self.final = nn.Conv2d(base_ch + in_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        u2 = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.final(torch.cat([d1, x], dim=1))
        return torch.sigmoid(out)


class Trainer:

    def __init__(self, model, opt_name):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.net = model.to(self.device)
        exec(f"self.optimizer = {opt_name}")
        self.train_losses = []
        self.test_losses = []
        self.start_epoch = 0

    def train(self, save_dir, num_epochs=100, batch_size=50,
              schedule_lr_epochs=0, lr_factor1=0.1, lr_factor2=2,
              test_each_epoch=True, verbose=False, manual_seed=False):
        print(f"Training on device: {self.device}")  # 应该看到 mps
        if manual_seed:
            torch.manual_seed(0)

        train_dataset = datasets.CIFAR10('data/cifar', train=True,
                                        download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss().to(self.device)

        if self.start_epoch == 0 and test_each_epoch:
            tl = self.test(batch_size)
            self.test_losses.append(tl)

        for epoch in range(self.start_epoch+1, num_epochs+1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            if verbose:
                print(f"Epoch {epoch}/{num_epochs}")

            if schedule_lr_epochs and epoch % schedule_lr_epochs == 0:
                for g in self.optimizer.param_groups:
                    # update the time scaling factor every schedule_lr_epochs epochs
                    if 'time_scale' in g:      g['time_scale'] *= lr_factor2
                   # If using an algorithm that includes learning rate or other similar parameter
                    if 'lr' in g:       g['lr'] *= lr_factor1
                    if 'correction' in g: g['correction'] *= lr_factor1


            self.net.train()
            for images, _ in train_loader:
                images = images.to(self.device)
                self.optimizer.zero_grad()
                recons = self.net(images)
                loss = criterion(recons, images)
                loss.backward()
                self.optimizer.step()
                self.train_losses.append(loss.item())

            # test
            if test_each_epoch:
                tl = self.test(batch_size)
                self.test_losses.append(tl)

            # checkpoint
            if epoch % 10 == 0:
                self.save_parameters(epoch, save_dir)

    def test(self, batch_size=250):
        self.net.eval()
        test_dataset = datasets.CIFAR10('data/cifar', train=False, download=True, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss().to(self.device)
        total_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                recons = self.net(images)
                total_loss += criterion(recons, images).item()
        return total_loss / len(test_loader)

    def save_parameters(self, epoch, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
        }, os.path.join(directory, f'checkpoint_{epoch}.pth'))


#number of times the experiment is repeated (for reporting average performance)
num_runs = 5
torch.use_deterministic_algorithms(True)
seeds = [23 + i for i in range(num_runs)]

for run, seed in enumerate(seeds):
    print("runs:", run+1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    opt_names = {
        'SHANG': 'SHANG(self.net.parameters(), alpha={} , time_scale={}, weight_decay={})'.format(0.5, 0.5, 1e-5),
        'ISHANG': 'ISHANG(self.net.parameters(), alpha={} , time_scale={}, rho = {}, weight_decay={})'.format(0.5, 0.5, 1.5, 1e-5),
        'AGNES': 'AGNES(self.net.parameters(), lr={} , momentum={} , correction={}, weight_decay = {})'.format(0.01, 0.99,  0.001, 1e-5),
        'NAG,': 'AGNES(self.net.parameters(), lr={} , momentum={} , correction={}, weight_decay={})'.format(1e-3, 0.99, 1e-3, 1e-5),
        'ADAM': 'torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)',
        'SHB': 'torch.optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.99, weight_decay=1e-5)',
        'SGD': 'torch.optim.SGD(self.net.parameters(), lr=1e-3, momentum=0, weight_decay=1e-5)',
        'SNAG': 'SNAG(self.net.parameters(), lr = {}, momentum = {}, weight_decay = {})'.format(0.05, 0.9, 1e-5)
    }

    for key, opt_name in opt_names.items():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = LightweightUNet(in_ch=3, out_ch=3).to(device)
        net = Trainer(model=model, opt_name=opt_name)
        net.train(save_dir = 'CIFAR10-UNet_batch5_for5runs'+key+'/'+str(run),
                  batch_size=5,
                  num_epochs = 50,
                  schedule_lr_epochs=25,
                  lr_factor1=0.1,
                  lr_factor2=2,
                  manual_seed = False,
                  verbose=False)
