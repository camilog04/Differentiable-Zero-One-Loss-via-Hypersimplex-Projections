# sweep.py
import argparse, copy, torch
from NN_projected import main  # <-- import where your main(args) lives

def run_sweep():
    base = argparse.Namespace(
        batch_size=1024,
        loss_fn="Hyper_simplex",
        regularization="l2",
        regularization_strength=1.5,
        hidden_size=64,
        epochs=600,
        plot=False,
        device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    )
    for bs in [128, 256, 512, 1024, 2048, 4096, 8192]:
        args = copy.deepcopy(base)
        args.batch_size = bs
        print(f">>> Running batch_size={bs}")
        main(args)

if __name__ == "__main__":
    run_sweep()