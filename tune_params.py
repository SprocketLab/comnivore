import os

batch_sizes = [32, 64, 128, 256]
l2s = [.1,]
lrs = [5e-4,1e-4,1e-3]
alphas = [2]
snorkel_eps = [100,500,1000]
snorkel_lrs = [1e-3,1e-4,5e-4]

for bs in batch_sizes:
    for l2 in l2s:
        for lr in lrs:
            # for alpha in alphas:
                # for ep in snorkel_eps:
                    # for s_lr in snorkel_lrs:
            os.system(f"python fuse_causal_estimates.py -c configs/CelebA.yaml -bs {bs} -lr {lr} -l2 {l2} -log tune_params")