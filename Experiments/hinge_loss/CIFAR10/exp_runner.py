# sweep.py
import subprocess as sp
SEEDS = [0, 42, 1337, 2025, 31415] # seed 0 already ran
BATCH_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]

# Common args you want to pass to NN_CE (tweak as needed)
COMMON = {
    "loss_fn": "hinge",
    "regularization": "kl", # not used for this experiment
    "regularization_strength": 1.0, # not used for this experiment
    "hidden_size": 64,
    "epochs": 600,
    "plot": False,
}

def build_args_str(extra: dict) -> str:
    parts = []
    for k, v in extra.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                parts.append(flag)
        else:
            parts.append(f"{flag} {v}")
    return " ".join(parts)

def run_seed(seed: int):
    print(f">>> Launching seed={seed} (7 jobs in parallel)")
    procs = []

    # First 4 batch sizes -> GPU 0
    for bs in BATCH_SIZES[:4]:
        args_str = build_args_str({
            **COMMON,
            "seed": seed,
            "batch_size": bs,
            "device": "cuda",  # NN_CE should map this to torch.device
        })
        cmd = f"CUDA_VISIBLE_DEVICES=0 python -m NN_hinge {args_str}"
        print("  ", cmd)
        procs.append(sp.Popen(cmd, shell=True))

    # Last 3 batch sizes -> GPU 1
    for bs in BATCH_SIZES[4:]:
        args_str = build_args_str({
            **COMMON,
            "seed": seed,
            "batch_size": bs,
            "device": "cuda",
        })
        cmd = f"CUDA_VISIBLE_DEVICES=1 python -m NN_hinge {args_str}"
        print("  ", cmd)
        procs.append(sp.Popen(cmd, shell=True))

    # Wait for all 7 to finish
    for p in procs:
        p.wait()
    print(f"<<< Done seed={seed}\n")

def main():
    for s in SEEDS:
        run_seed(s)

if __name__ == "__main__":
    main()