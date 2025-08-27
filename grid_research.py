## Adapted from the AGNES codebase by Gupta et al. (MIT License).
## Source: https://github.com/kanangupta/AGNES

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
import pickle
from torchvision import datasets, models, transforms
from util import  *
from SHNAG_optim import SHNAG, ISHNAG
import pickle
import matplotlib.pyplot as plt



train_transform = transforms.Compose([
            Cutout(num_cutouts=2, size=8, p=0.8),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ])

#downloading the data
#train_dataset = torchvision.datasets.MNIST(root = 'data/mnist/', train = True, download=True, transform = ToTensor())
#test_dataset = torchvision.datasets.MNIST(root = 'data/mnist/', train = False, transform = ToTensor())

train_dataset = torchvision.datasets.CIFAR10('data/cifar/', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10('data/cifar/', train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 50, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 10, shuffle = True)


#defining data loader, optimizer, and the loss function
alphas = [1e-1, 5e-1, 1e-2, 5e-2, 5e-3]
gammas = [0.5, 1, 1.5, 2, 2.5, 5, 10, 15, 20, 30]
optim_names = [(i,j) for i in alphas for j in gammas]

# LeNet-5
#models = {name: nn.Sequential(
#	nn.Conv2d(1,6,5,padding=2),
#	nn.Tanh(),
#	nn.AvgPool2d(2,2),
#	nn.Conv2d(6,16,5),
#	nn.Tanh(),
#	nn.AvgPool2d(2,2),
#	nn.Flatten(),
#	nn.Linear(5*5*16,120),
#	nn.Tanh(),
#	nn.Linear(120,84),
#	nn.Tanh(),
#	nn.Linear(84,10)
#	) for name in optim_names}

models = {name: models.resnet34() for name in optim_names}

optimizers={name: ISHNAG(models[name].parameters(), alpha = name[0], time_scale = name[1], rho=1.5) for name in optim_names}
optim_name = "ISHNAG"
use_data = "CIFAR-10"
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Training on device: {device}")

for name in optim_names:
	models[name] = models[name].to(device)
	models[name].train()

loss_fn = nn.CrossEntropyLoss() #Note: If you use this loss function, do not use a softmax layer at the end of the neural network

no_of_epochs = 10
losses = {name:[] for name in optim_names}
for name in optim_names:
	for epoch in range(no_of_epochs):
		print("Currently on epoch",epoch,"for",name)
		avgloss = 0
		for datum, label in train_loader:
			datum = datum.to(device)
			label = label.to(device)
			optimizers[name].zero_grad()
			output = torch.squeeze(models[name](datum))
			loss=loss_fn(output, label)
			loss.backward()
			optimizers[name].step()
			avgloss += loss.item()
		losses[name].append(avgloss/len(train_loader))

with open('losses_from_grid_search', 'wb') as file:
	pickle.dump(losses, file)

optim_names = [(alpha, gamma) for alpha in alphas for gamma in gammas]

color_alpha = {
    1e-1: 'tab:red',
    5e-1: 'tab:blue',
    1e-2: 'tab:green',
    5e-2: 'tab:orange',
    1e-3: 'tab:pink',
    5e-3: 'tab:brown'
}

style_gamma = {
    0.5:  'solid',                     # ────
    1:    'dashed',                    # ─ ─ ─
    1.5:  'dotted',                    # · · ·
    2:    'dashdot',                   # ─ · ─ ·
    2.5:  (0, (5, 1)),                 # ─────·─────·
    5:    (0, (3, 2, 1, 2)),           # ───··─··
    10:   (0, (1, 1)),                 # ······
    15:   (0, (1, 3)),                 # ·   ·   ·
    20:   (0, (2, 2)),                 # ──  ──  ──
    30:   (0, (6, 2)),                 # ──────  ──────
}

with open('losses_from_grid_search', 'rb') as file:
    losses = pickle.load(file)


plot_lines = {}
plt.figure(figsize=(10, 6))

for alpha in alphas:
    for gamma in gammas:
        if (alpha, gamma) not in losses:
            continue
        line, = plt.semilogy(
            [1, 2, 3, 4, 5, 6,7,8,9,10],
            losses[(alpha, gamma)],
            color=color_alpha.get(alpha, 'black'),
            linestyle=style_gamma.get(gamma, 'solid'),
            linewidth=2
        )
        plot_lines[(alpha, gamma)] = line

legend1 = plt.legend(
    [plot_lines[(alpha, gammas[0])] for alpha in alphas if (alpha, gammas[0]) in plot_lines],
    [f"α={alpha}" for alpha in alphas if (alpha, gammas[0]) in plot_lines],
    loc='upper right', title="Alpha (Color)"
)
plt.gca().add_artist(legend1)

legend2 = plt.legend(
    [plot_lines[(alphas[0], gamma)] for gamma in gammas if (alphas[0], gamma) in plot_lines],
    [f"γ={gamma}" for gamma in gammas if (alphas[0], gamma) in plot_lines],
    loc='lower left', title="Gamma (Linestyle)"
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title(f"Grid Search over (α, γ) for {optim_name} on {use_data}")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(f"{optim_name}_{use_data}_grid_search_plot.pdf")
plt.show()

