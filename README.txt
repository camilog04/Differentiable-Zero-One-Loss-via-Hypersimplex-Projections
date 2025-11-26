This REPO contains an anonymized implementation used for the experiments and JIT compilation CUDA code of soft binary argmax @k and the Hypersimplex Loss
No authors or institutions are referenced.

Instructions:
1. Install requirements listed in requirements.txt.
2. Replicate experiments:
    a) cd and run each (loss function, dataset) combination. For example, to run cross entropy with CIFAR10
    cd Experiments/cross_entropy/CIFAR10 && python exp_runner.py

    b) in general 
    cd Experiments/{loss_fn}/{dataset} && run python exp_runner.py
    loss_fn := cross_entropy, hinge_loss, hypersimplex (ours), MSE_multiclass
    dataset := CIFAR10, FashionMNIST

3. To replicate the statistical tests run error_bars_statistical_tets.ipynb
4. To replicate supplementary experiments see the supplementary directory
5. (Important) An illustration for the method can be found in SBAM_Illustration.ipynb