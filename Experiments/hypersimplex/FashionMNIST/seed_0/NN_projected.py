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

import torchsort

NUM_CLASSES = 10


def get_hypersimplex_tensor(n,p):
    # Ensure p is between 0 and 1
    assert 0 <= p <= 1, "p must be between 0 and 1"
    # Calculate the number of 1's
    num_ones = int(n * p)
    # Create a tensor with n zeros
    tensor = torch.zeros(n)
    # Set the first num_ones elements to 1
    tensor[:num_ones] = 1
    return tensor

def hypersimplex_loss(input, target, regularization="l2", regularization_strength=1.0):
    n = input.shape[0]
    hypersimplex_basis = get_hypersimplex_tensor(n, 0.5).to(input.device)
    # computes projection into 10 hypersimplices
    hyper_simplex_projection = torchsort.conv_proj.apply(input.T, hypersimplex_basis, regularization, regularization_strength).T
    return F.mse_loss(hyper_simplex_projection, target).mean()


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
    torch.manual_seed(0)

    # Fashion-MNIST: 1x28x28 grayscale
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

    data_root = "/home/camilo1/research/multiclass/data"

    train_ds = FashionMNIST(data_root, train=True, transform=train_transform, download=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_ds = FashionMNIST(data_root, train=False, transform=test_transform, download=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

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

    loss_fn = (
        F.cross_entropy
        if args.loss_fn == "cross_entropy"
        else partial(
            hypersimplex_loss,
            regularization=args.regularization,
            regularization_strength=args.regularization_strength,
        )
    )


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    test_accs = []
    for epoch in range(args.epochs):
        train_loss = AverageMeter("train_loss")
        test_acc = AverageMeter("test_acc")

        model.train()
        for (img, label) in train_dl:
            img, label = img.to(args.device), label.to(args.device)
            optimizer.zero_grad()
            pred = model(img)

            # adding one-hot encoded target
            target = F.one_hot(label, num_classes=NUM_CLASSES).float()

            loss = loss_fn(pred, target)

            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=img.shape[0])

        # test step
        model.eval()
        if args.loss_fn == "cross_entropy":
            with torch.no_grad():
                for (img, label) in test_dl:
                    img, label = img.to(args.device), label.to(args.device)
                    logit = model(img)
                    test_acc.update(
                        (logit.argmax(-1) == label).float().mean(), img.shape[0]
                    )
        else:
            with torch.no_grad():
                for (img, label) in test_dl:
                    img, label = img.to(args.device), label.to(args.device)
                    logit = model(img)
                    pred = logit.argmax(-1)
                    test_acc.update((pred == label).float().mean(), img.shape[0])

        print(epoch, test_acc, train_loss)
        test_accs.append(test_acc.avg)

    test_accs = torch.stack(test_accs).cpu().numpy()
    regularization = (
        f"_{args.regularization}_{args.regularization_strength}"
        if args.loss_fn == "Hyper_simplex" else ""
    )
    np.save(f"{args.loss_fn}{regularization}_{args.batch_size}_acc.npy", test_accs)


def plot():
    def smooth(xs, factor=0.9):
        out = [xs[0]]
        for x in xs[1:]:
            out.append(out[-1] * factor + x * (1 - factor))
        return out

    colors = [
        "tab:blue","tab:orange","tab:green","tab:red","tab:purple",
        "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"
    ]

    plt.figure(figsize=(5, 3))
    y_min, y_max = float("inf"), float("-inf")

    files = sorted(Path("./").glob("*.npy"))
    for i, file in enumerate(files):
        print(file)
        test_accs = np.load(file)
        c = colors[i % len(colors)]
        plt.plot(test_accs, alpha=0.1, color=c)
        plt.plot(smooth(test_accs), color=c, label=file.stem)
        y_min = min(y_min, float(np.min(test_accs)))
        y_max = max(y_max, float(np.max(test_accs)))

    span = max(1e-6, y_max - y_min)
    pad = max(0.01, 0.05 * span)
    base = 0.8
    plt.ylim(base - pad, y_max + pad)

    plt.xlabel("Epochs")
    plt.ylabel("Test accuracy")
    plt.title("Fashion-MNIST")
    plt.legend()
    plt.savefig("fashion_mnist_test_accuracy.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--loss_fn", choices=["cross_entropy", "Hyper_simplex"], default="Hyper_simplex")
    parser.add_argument("--regularization", default="l2")
    parser.add_argument("--regularization_strength", type=float, default=1.5)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.plot:
        plot()
    else:
        main(args)