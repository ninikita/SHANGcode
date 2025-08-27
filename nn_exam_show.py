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
title = "CIFAR100-ResNet50_batch50_for5runs"

names = [
    'SGD',
    'SHB',
    'NAG',
    'ADAM',
    'SNAG',
    'AGNES',
    'SHNAG,',
    'ISHNAG',
]

# 设置颜色和线型
colors = {}
linestyles = {}
style_list = ['-', '--', ':']

color_map = {
    'SHNAG': 'green',
    'ISHNAG': 'red',
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
#plt.ylim([95.5, 99.5])
plt.title(title + metric)
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
plt.legend()
plt.savefig(title + metric)
plt.show()