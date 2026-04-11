## Adapted from the AGNES codebase by Gupta et al. (MIT License).
## Source: https://github.com/kanangupta/AGNES

## The current version of the file is set up to load and train on CIFAR-10.
## The corresponding lines for CIFAR-100 and MNIST have been commented out.
## This code can be used for other datasets by appropriately modifying the code defining
## train_dataset, train_transform, test_dataset, and test_transform.

import torch
import torch.nn as nn
import os, random
from torchvision import datasets, models, transforms
from nn_optim import *
from util import *

class Trainer:

    def __init__(self, model, opt_name):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.net = model
        self.optimizer = None
        self.train_accuracies = []
        self.train_losses =[]
        self.test_accuracies = []
        self.test_losses = []
        self.start_epoch = 0
        exec('self.optimizer ='+opt_name)

    def train(self, save_dir, num_epochs=100, batch_size=50, schedule_lr_epochs=0, lr_factor1=0.1, lr_factor2=2, test_each_epoch=True, verbose=False, manual_seed=False):
        """Trains the network.

        Parameters
        ----------
        save_dir : str
            The directory in which the parameters will be saved
        opt_name : str
            The name of the optimizer, can be one 'AGNES', 'ADAM', 'SGD0.99M', 'SGD', or 'SGD0.9M'
        num_epochs : int
            The number of epochs
        batch_size : int
            The batch size
        learning_rate : float
            The learning rate
        test_each_epoch : boolean
            True: Test the network after every training epoch, False: no testing
        verbose : boolean
            True: Print training progress to console, False: silent mode
        """
        print(f"Training on device: {self.device}")  # 应该看到 mps
        if manual_seed:
            torch.manual_seed(0)

        ### For CIFAR-10 change the last transform to transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ### For CIFAR-100 change the last transform to transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
        ### For MNIST, remove all transforms except transforms.ToTensor()
        ### Don't forget to make the same changes to test_transform as well
        train_transform = transforms.Compose([
            Cutout(num_cutouts=2, size=8, p=0.8),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        ### Only one of the following lines should be uncommented. Don't forget to use the same dataset
        ### for test_dataset as well.
        #train_dataset = datasets.CIFAR100('data/cifar', train=True, download=True, transform=train_transform)
        #train_dataset = datasets.MNIST('data/mnist', train=True, download=False, transform=transforms.ToTensor())
        train_dataset = datasets.CIFAR10('data/cifar', train=True, download=True, transform=train_transform)
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        progress_bar = ProgressBar()

        if self.start_epoch==0: #computing test loss and accuracy before the training starts
            test_loss, test_accuracy = self.test(batch_size=batch_size)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)

        for epoch in range(self.start_epoch + 1, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))

            if schedule_lr_epochs:
                if epoch%schedule_lr_epochs == 0:
                #update the time scaling factor every schedule_lr_epochs epochs
                    for g in self.optimizer.param_groups:
                        if 'time_scale' in g.keys():
                            g['time_scale'] *= lr_factor2
                        # If using an algorithm that includes learning rate or other similar parameter
                        if 'lr' in g.keys():
                            g['lr'] *= lr_factor1
                        if 'correction' in g.keys():
                            g['correction'] *= lr_factor1

            for i, data in enumerate(data_loader, 1):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net.forward(images)
                loss = criterion(outputs, labels.squeeze_())
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, dim=1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels.flatten()).sum().item()

                self.train_accuracies.append(batch_correct/batch_total)
                self.train_losses.append(loss.item())

                #epoch_total += batch_total
                #epoch_correct += batch_correct

                if verbose:
                    # Update progress bar in console
                    info_str = 'Last batch accuracy: {:.4f} - Running epoch accuracy {:.4f}'.\
                                format(batch_correct / batch_total)
                    progress_bar.update(max_value=len(data_loader), current_value=i, info=info_str)

            #self.train_accuracies.append(epoch_correct / epoch_total)
            if verbose:
                progress_bar.new_line()

            if test_each_epoch:
                test_loss, test_accuracy = self.test()
                self.test_losses.append(test_loss)
                self.test_accuracies.append(test_accuracy)
                if verbose:
                    print('Test accuracy: {}'.format(test_accuracy))

            # Save parameters after every 10 epochs
            if epoch%10==0:
                self.save_parameters(epoch, directory=save_dir)

    def test(self, batch_size=250):
        """Tests the network.

        """
        self.net.eval()

        ### For CIFAR-10 change the last transform to transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ### For CIFAR-100 change the last transform to transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
        ### For MNIST, remove all transforms except transforms.ToTensor()
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                             ])

        ### Only one of the following lines should be uncommented. Ideally, the same dataset that's used for training.
        #test_dataset = datasets.CIFAR100('data/cifar', train=False, download=True, transform=test_transform)
        #test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10('data/cifar', train=False, download=False, transform=test_transform)

        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)

                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels.flatten()).sum().item()

                loss += criterion(outputs, labels.squeeze_()).item() #loss is normalized by default, but the last batch here would be of a different size

        self.net.train()
        return (loss / (i+1), correct / total)

    def save_parameters(self, epoch, directory):
        """Saves the parameters of the network to the specified directory.

        Parameters
        ----------
        epoch : int
            The current epoch
        directory : str
            The directory to which the parameters will be saved
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            # 'opt_name': self.opt_name,
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracies': self.train_accuracies,
            'train_losses': self.train_losses,
            'test_accuracies': self.test_accuracies,
            'test_losses': self.test_losses
        }, os.path.join(directory, 'checkpoint_' + str(epoch) + '.pth'))

    def load_parameters(self, path):
        """Loads the given set of parameters.

        Parameters
        ----------
        path : str
            The file path pointing to the file containing the parameters
        """
        checkpoint = torch.load(path, map_location=self.device)

        # self.opt_name = checkpoint['opt_name']

        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_accuracies = checkpoint['train_accuracies']
        self.train_losses = checkpoint['train_losses']
        self.test_accuracies = checkpoint['test_accuracies']
        self.test_losses = checkpoint['test_losses']
        self.start_epoch = checkpoint['epoch']



num_runs = 5 #number of times the experiment is repeated (for reporting average performance)
torch.use_deterministic_algorithms(True)
seeds = [23 + i for i in range(num_runs)]
for run, seed in enumerate(seeds):
    print("runs:", run +1 )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    opt_names = {
        'SHANG': 'SHANG(self.net.parameters(), alpha={} , time_scale={}, weight_decay={})'.format(0.5, 10, 1e-5),
        'SHANG++': 'SHANGPlus(self.net.parameters(), alpha={} , time_scale={}, rho = {}, weight_decay={})'.format(0.5, 10, 1.5, 1e-5),
        'AGNES': 'AGNES(self.net.parameters(), lr={} , momentum={} , correction={}, weight_decay = {})'.format(0.01, 0.99,  0.001, 1e-5),
        'NAG': 'AGNES(self.net.parameters(), lr={} , momentum={} , correction={}, weight_decay={})'.format(1e-3, 0.99, 1e-3, 1e-5),
        'ADAM': 'torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)',
        'SHB': 'torch.optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.99, weight_decay=1e-5)',
        'SGD': 'torch.optim.SGD(self.net.parameters(), lr=1e-3, momentum=0, weight_decay=1e-5)',
        'SNAG': 'SNAG(self.net.parameters(), lr = {}, momentum = {}, weight_decay = {})'.format(0.05, 0.9, 1e-5)
    }

    for key, opt_name in opt_names.items():
        # LeNet-5
        #model = nn.Sequential(
        #    nn.Conv2d(1, 6, 5, padding=2),
        #    nn.Tanh(),
        #    nn.AvgPool2d(2, 2),
        #    nn.Conv2d(6, 16, 5),
        #    nn.Tanh(),
        #    nn.AvgPool2d(2, 2),
        #    nn.Flatten(),
        #    nn.Linear(5 * 5 * 16, 120),
        #    nn.Tanh(),
        #    nn.Linear(120, 84),
        #    nn.Tanh(),
        #    nn.Linear(84, 10)
        #)

       # model = models.resnet50(pretrained=True)
        #model.fc = nn.Linear(model.fc.in_features, 100, bias=True)

        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10, bias=True)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
        net = Trainer(model=model, opt_name=opt_name)

        net.train(save_dir = 'CIFAR10-ResNet34_batch50_for5runs'+key+'/'+str(run),
                  batch_size=50,
                  num_epochs = 50,
                  schedule_lr_epochs=25,
                  lr_factor1=0.1,
                  lr_factor2=2,
                  manual_seed=False,
                  verbose=False)


import torch
import matplotlib.pyplot as plt
import numpy as np


data = {}
runs = 5
# For MNIST, train_size = 60000
# For CIFAR10/100, train_size = 50000
train_size = 50000
batch = 50
epochs = 50
epoch_step = train_size / batch
total_steps = epoch_step * epochs
title = "CIFAR10-ResNet34_batch50_for5runs"

names = [
    'SGD',
    'SHB',
    'NAG',
    'ADAM',
    'SNAG',
    'AGNES',
    'SHANG',
    'SHANG++',
]

# 设置颜色和线型
colors = {}
linestyles = {}
style_list = ['-', '--', ':']

color_map = {
    'SHANG++': 'red',
    'SHANG': 'green',
    'AGNES': 'blue',
    'SNAG': 'orange',
    'SGD': 'gray',
    'SHB': 'black',
    'NAG': 'olive',
    'ADAM': 'yellow'
}

class_count = {key: 0 for key in color_map}

for name in names:
    for key in color_map:
        if name.startswith(key):
            colors[name] = color_map[key]
            linestyles[name] = style_list[class_count[key] % len(style_list)]
            class_count[key] += 1
            break
    else:
        colors[name] = 'black'
        linestyles[name] = '-'

metrics = ['Test Accuracy', 'Test Loss', 'Training Loss']
decay = 0.999

for name in names:
    data[name] = {'Test Loss': [], 'Training Loss': [], 'Test Accuracy': []}
    for i in range(runs):
        with open(title + name + '/' + str(i) + '/checkpoint_50.pth', 'rb') as file:
            temp = torch.load(file, map_location=torch.device('cpu'))
            data[name]['Test Loss'].append(temp['test_losses'])
            data[name]['Test Accuracy'].append(temp['test_accuracies'])
            running_averages = []
            last = temp['train_losses'][0]
            for num in temp['train_losses']:
                last = decay * last + (1 - decay) * num
                running_averages.append(last)
            data[name]['Training Loss'].append(running_averages)

for name in names:
    data[name]['Test Accuracy'] = 100 * np.array(data[name]['Test Accuracy'])
    for metric in metrics[1:]:
        data[name][metric] = np.array(data[name][metric])

# --- Test Accuracy ---
metric = metrics[0]
plt.figure()
for name in names:
    mean = np.mean(data[name][metric], axis=0)
    std = np.std(data[name][metric], axis=0)
    x_vals = np.arange(0, total_steps + 1, epoch_step)
    plt.plot(x_vals, mean, label=name, color=colors[name], linestyle=linestyles[name])
    plt.fill_between(x_vals, mean + std, mean - std, alpha=0.2, color=colors[name])
plt.ylim([95.5, 99.5])
plt.xlabel("Training steps")
plt.ylabel("Test Accuracy (%)")
#plt.title(title + metric)
plt.legend()
plt.savefig(title + metric)

# --- Test Loss ---
metric = metrics[1]
plt.figure()
for name in names:
    mean = np.clip(np.mean(data[name][metric], axis=0), 1e-8, None)
    std = np.std(data[name][metric], axis=0)
    x_vals = np.arange(0, total_steps + 1, epoch_step)
    plt.semilogy(x_vals, mean, label=name, color=colors[name], linestyle=linestyles[name])
    plt.fill_between(x_vals, mean + std, mean - std, alpha=0.2, color=colors[name])
plt.title(title + metric)
plt.xlabel("Training steps")
plt.ylabel("Loss (log scale)")
plt.legend()
plt.savefig(title + metric)
plt.show()

# --- Training Loss ---
metric = metrics[2]
plt.figure()
for name in names:
    mean = np.mean(data[name][metric], axis=0)
    std = np.std(data[name][metric], axis=0)
    x_vals = range(len(mean))
    plt.semilogy(x_vals, mean, label=name, color=colors[name], linestyle=linestyles[name])
    plt.fill_between(x_vals, mean + std, mean - std, alpha=0.2, color=colors[name])
plt.title(title + metric)
plt.xlabel("Training steps")
plt.ylabel("Loss (log scale)")
plt.legend()
plt.savefig(title + metric)
plt.show()
