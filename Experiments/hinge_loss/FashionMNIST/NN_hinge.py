from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST


import os 

# Directory where this script lives: .../cross_entropy/CIFAR10
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up TWO levels: CIFAR10 → cross_entropy → Experiments
EXPERIMENTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Now point to data/
DATA_DIR = os.path.join(EXPERIMENTS_DIR, "data")
DATA_DIR = os.path.abspath(DATA_DIR)

print("USING DATA_DIR =", DATA_DIR)

NUM_CLASSES = 10

import torch
import torch.nn as nn

class MultiClassHingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits, target):
        # logits: [N, C], target: [N]
        num_classes = logits.size(1)
        # Get the scores for the correct class
        correct_class_scores = logits[torch.arange(logits.size(0)), target].unsqueeze(1)
        # Compute margins: max(0, margin + other - correct)
        margins = (self.margin + logits - correct_class_scores).clamp(min=0)
        # Zero out the correct class terms
        margins[torch.arange(logits.size(0)), target] = 0
        return margins.sum(dim=1).mean()


class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name} {self.avg:.4f}"


def main(args):
    torch.manual_seed(args.seed)

    train_transform = T.Compose([
        T.RandomCrop(28, padding=3, padding_mode="reflect"),
        T.RandomHorizontalFlip(),   # keep if you’re okay with left/right flips
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,)),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,)),
    ])

    train_ds = FashionMNIST(DATA_DIR, train=True, transform=train_transform, download=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_ds = FashionMNIST(DATA_DIR, train=False, transform=test_transform, download=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


    # from the paper https://arxiv.org/abs/2002.08871:
    #
    # > Following Cuturi et al. (2019), we use a vanilla CNN (4 Conv2D with 2 maxpooling
    # > layers, ReLU activation, 2 fully connected layers with batch norm on each) ), the
    # > ADAM optimizer (Kingma & Ba, 2014) with a constant step size of 10−4, and set k = 1.
    #
    # there are no other details about the architecture in the paper. It reads they are
    # applying batch norm after the fully connected layers, but I think they meant on the
    # Conv2D.
    hidden = args.hidden_size

    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=hidden, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 14 -> 7
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(7 * 7 * hidden, 512),
        nn.ReLU(),
        nn.Linear(512, NUM_CLASSES),
    ).to(args.device)

    hinge_loss_fn = MultiClassHingeLoss()

    loss_fn = (
        F.cross_entropy
        if args.loss_fn == "cross_entropy"
        else hinge_loss_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    test_accs = []

    for epoch in range(args.epochs):
        train_loss = AverageMeter("train_loss")
        test_acc = AverageMeter("test_acc")

        # train step
        model.train()
        for (img, label) in train_dl:
            img, label = img.to(args.device), label.to(args.device)
            optimizer.zero_grad()

            pred = model(img)

            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=img.shape[0])

        # test step
        model.eval()
        with torch.no_grad():
            for (img, label) in test_dl:
                img, label = img.to(args.device), label.to(args.device)
                logit = model(img)
                test_acc.update(
                    (logit.argmax(-1) == label).float().mean(), img.shape[0]
                )

        print(epoch, test_acc, train_loss)
        test_accs.append(test_acc.avg)

    def smooth(xs, factor=0.9):
        out = [xs[0]]
        for x in xs[1:]:
            out.append(out[-1] * factor + x * (1 - factor))
        return out

    test_accs = torch.stack(test_accs).cpu().numpy()
    regularization = (
        f"_{args.regularization}_{args.regularization_strength}"
        if args.loss_fn == "cross_entropy"
        else ""
    )
    Path(f"seed_{args.seed}").mkdir(parents=True, exist_ok=True)
    np.save(f"seed_{args.seed}/{args.loss_fn}{regularization}_{args.batch_size}_acc_{args.seed}.npy", test_accs)


def plot():
    def smooth(xs, factor=0.9):
        out = [xs[0]]
        for x in xs[1:]:
            out.append(out[-1] * factor + x * (1 - factor))
        return out

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan"
    ]


    plt.figure(figsize=(5, 3))
    
    y_min, y_max = float("inf"), float("-inf")

    for i, file in enumerate(Path("./").glob("*.npy")):
        print(file)
        test_accs = np.load(file)
        plt.plot(test_accs, alpha=0.1, color=colors[i])
        plt.plot(smooth(test_accs), color=colors[i], label=file.stem)

        y_min = min(y_min, float(np.min(test_accs)))
        y_max = max(y_max, float(np.max(test_accs)))

    span = max(1e-6, y_max - y_min)
    pad = max(0.01, 0.05 * span)
    y_min = 0.5
    plt.ylim(y_min - pad, y_max + pad)

    plt.xlabel("Epochs")
    plt.ylabel("Test accuracy")
    plt.title("Fashion-MNIST")
    plt.legend()
    plt.savefig("fashion_mnist_test_accuracy.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--loss_fn", choices=["cross_entropy", "hinge"], default="hinge"
    )
    parser.add_argument("--regularization", default="kl")
    parser.add_argument("--regularization_strength", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device not specified, using', args.device)
        
    if args.plot:
        plot()
    else:
        main(args)
