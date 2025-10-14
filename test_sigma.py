# Evaluate robustness to multiplicative noise σ on
# ResNet‑34 + CIFAR‑10 for SHANG / ISHANG / AGNES/ SNAG.
# ---------------------------------------------------------

import os, time, random, csv
import torch
import torch.nn as nn
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from SHNAG_optim import SHANG, ISHANG

if torch.backends.mps.is_available():
    DEVICE, PIN, NUM_WORKERS = torch.device('mps'), False, 0
elif torch.cuda.is_available():
    DEVICE, PIN, NUM_WORKERS = torch.device('cuda'), True, 4
else:
    DEVICE, PIN, NUM_WORKERS = torch.device('cpu'), False, 0

print(DEVICE)

NUM_CLASSES = 10
def get_loaders(batch_size: int = 50):
    mean, std = (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)
    train_tf = T.Compose([T.RandomCrop(32,4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean,std)])
    test_tf  = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
    train_set = torchvision.datasets.CIFAR10('data/cifar', train=True,
                                             download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10('data/cifar', train=False,
                                             download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size, True,
                              num_workers=NUM_WORKERS, pin_memory=PIN)
    val_loader   = DataLoader(test_set, 256, False,
                              num_workers=NUM_WORKERS, pin_memory=PIN)
    return train_loader, val_loader

def get_model():
    return torchvision.models.resnet34(num_classes=NUM_CLASSES).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer, sigma):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y); loss.backward()
        if sigma > 0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 + sigma * torch.randn_like(p.grad))
        optimizer.step()

@torch.inference_mode()
def evaluate(model, loader, criterion):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1); correct += (pred==y).sum().item()
        total += y.size(0)
    return 1 - correct/total   # Top‑1 error



BEST_CFGS = {
    'SHANG':  {'alpha':0.5, 'time_scale':10, 'weight_decay': 1e-5 },
    'ISHANG': {'alpha':0.5, 'time_scale':10, 'rho':1.5, 'weight_decay': 1e-5},
    'AGNES':  {'lr':0.01,'correction':0.001,'momentum':0.99, 'weight_decay': 1e-5},
    'SNAG': {'lr':0.05, 'momentum':0.9, 'weight_decay': 1e-5}
}


SIGMA_LIST = [0,0.05,0.1,0.2, 0.5]
SEEDS      = [23,24,25]
NUM_EPOCHS = 100
TARGET_ERR = 0.22          # 22% Top‑1 error
RESULT_CSV = 'sigma_results.csv'


def run():
    train_loader, val_loader = get_loaders()
    with open(RESULT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f,
            fieldnames=['sigma','algo','seed','final_err',
                        'epoch_to_target','gpu_h_to_target'])
        writer.writeheader()

        for sigma in SIGMA_LIST:
            for algo, cls in [('SHANG',SHANG), ('ISHANG',ISHANG), ('AGNES',AGNES), ('SNAG',SNAG)]:
                cfg = BEST_CFGS[algo]
                for seed in SEEDS:
                    torch.manual_seed(seed); random.seed(seed)
                    model = get_model()
                    opt = cls(model.parameters(), **cfg)
                    crit = nn.CrossEntropyLoss()

                    t0 = time.time(); hit_ep = None; hit_gpu = None
                    for ep in range(1, NUM_EPOCHS+1):
                        train_one_epoch(model, train_loader, crit, opt, sigma)
                        err = evaluate(model, val_loader, crit)
                        if hit_ep is None and err <= TARGET_ERR:
                            hit_ep  = ep
                            hit_gpu = (time.time()-t0)/3600.0
                    writer.writerow(dict(
                        sigma=sigma, algo=algo, seed=seed,
                        final_err=round(err,4),
                        epoch_to_target=hit_ep if hit_ep else 'n/a',
                        gpu_h_to_target=round(hit_gpu,2) if hit_gpu else 'n/a'))
                    f.flush()
                    print(f'σ={sigma} {algo} seed={seed} final_err={err:.4f}')


def analyze():
    df = pd.read_csv(RESULT_CSV)
    agg = df.groupby(['sigma','algo'])['final_err'].mean().reset_index()
    for algo in agg.algo.unique():
        part = agg[agg.algo==algo]
        plt.plot(part.sigma, part.final_err*100, '-o', label=algo)
    plt.xlabel('σ'); plt.ylabel('Top‑1 error (%)'); plt.legend()
    plt.tight_layout(); plt.savefig('sigma_curve.png', dpi=300); plt.close()
    print('Saved sigma_curve.png')

    base = agg[agg.sigma==0].set_index('algo')['final_err']
    rows = []
    for algo in base.index:
        base_err = base[algo]
        for s in [0.05, 0.1, 0.2, 0.5]:
            now_err = agg[(agg.algo==algo) & (agg.sigma==s)].final_err.iloc[0]
            rows.append({'algo':algo,'sigma':s,
                         'delta_%': round((now_err-base_err)/base_err*100,1)})
    pd.DataFrame(rows).to_csv('./sigma_delta.csv', index=False)
    print('Saved sigma_delta.csv')


if __name__ == '__main__':
    os.makedirs('.', exist_ok=True)
    run()
    analyze()
