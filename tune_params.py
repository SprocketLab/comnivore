import os

batch_sizes = [16,32,64,128]
l2s = [.1,]
lrs = [3e-5,1e-5,5e-6]
dropouts=[.1,.25,.5]

for bs in batch_sizes:
    # for l2 in l2s:
    for lr in lrs:
        for do in dropouts:
            os.system(f"python get_spurious_samples.py -c configs/spur_exp/CelebA.yaml -bs {bs} -lr {lr} -do {do} -log TUNE_PARAMS_SPURIOUS")