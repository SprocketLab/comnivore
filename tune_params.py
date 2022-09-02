import os
import numpy as np

batch_sizes = [16,32,64,128]
l2s = [.1,]
lrs = [3e-5,1e-5,5e-6]
dropouts=[.1,.25,.5]

ps = np.arange(0.05, 0.95, 0.05)
# for bs in batch_sizes:
    # for l2 in l2s:
    # for lr in lrs:
        # for do in dropouts:
            # os.system(f"python get_spurious_samples.py -c configs/spur_exp/CelebA.yaml -bs {bs} -lr {lr} -do {do} -log TUNE_PARAMS_SPURIOUS")
for p in ps:    
    os.system(f"python get_spurious_samples.py -c configs/spur_exp/Waterbirds.yaml -p_zero {p} -log ZERO_ONE_EXP")