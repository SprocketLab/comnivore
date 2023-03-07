import numpy as np 
import os
import matplotlib.pyplot as plt
import pandas as pd


log_dir_root = '../log/SPURIOUS_NEW/Synthetic_ColoredMNIST_Multi'

subdirs = os.listdir(log_dir_root)
subdirs = [os.path.join(log_dir_root, sub_) for sub_ in subdirs]
spur_p = np.arange(0.1,0.7,0.1)

def splitSerToArr(ser):
    return [ser.index, ser.to_numpy()]

def file_to_list(file_path):
    with open(file_path) as file:
        lines = [line.strip() for line in file]
    return lines

def get_info_from_log_lines(log_lines):
    '''
    1. get cb
    2. get corresponding test acc for the cbs
    '''
    active_cbs = np.arange(0.1,1.0,0.1)
    baseline_acc =np.empty((1, 9))
    baseline_acc[:] = np.NaN
    weighted_acc = np.empty((1, 9))
    weighted_acc[:] = np.NaN
    remove_feats_acc = np.empty((1, 9))
    remove_feats_acc[:] = np.NaN
    combined_acc = np.empty((1, 9))
    combined_acc[:] = np.NaN
    for i, line in enumerate(log_lines):
        if "######" in line and "N MASKED FEATURES" in log_lines[i+1]:
            cb = float(line[7:10])
            # print(cb, line[i+14])
            i_to_insert = np.argwhere(active_cbs == cb)
            baseline = float(log_lines[i+14][-5:])
            weighted = float(log_lines[i+39][-5:])
            removed = float(log_lines[i+54][-5:])
            combined = float(log_lines[i+68][-5:])
            # active_cbs.append(cb)
            baseline_acc[0, i_to_insert] = baseline
            weighted_acc[0, i_to_insert] = weighted
            remove_feats_acc[0, i_to_insert] = removed
            combined_acc[0, i_to_insert] = combined
            # .append(baseline)
            # weighted_acc.append(weighted)
            # remove_feats_acc.append(removed)
            # combined_acc.append(combined)
            i+=68
    return active_cbs, baseline_acc.flatten(), weighted_acc.flatten(), remove_feats_acc.flatten(), combined_acc.flatten()

fig, a =  plt.subplots(2,3, figsize=(20,8))
for i, sub_ in enumerate(subdirs):
    date_dirs = os.listdir(sub_)
    log_file_dirs = [os.path.join(sub_, d_, "log.txt") for d_ in date_dirs]
    # print(log_file_dirs)
    baseline_all = []
    weighted_all = []
    removed_all = []
    combined_all = []
    for file_ in log_file_dirs:
        log_lines = file_to_list(file_)
        active_cbs, baseline_acc, weighted_acc, remove_feats_acc, combined_acc = get_info_from_log_lines(log_lines)
        # baseline_acc
        baseline_all.append(baseline_acc)
        weighted_all.append(weighted_acc)
        removed_all.append(remove_feats_acc)
        combined_all.append(combined_acc)
    baseline_all = np.vstack(baseline_all)
    weighted_all = np.vstack(weighted_all)
    removed_all = np.vstack(removed_all)
    combined_all = np.vstack(combined_all)

    baseline_all = np.nanmean(baseline_all, axis=0).tolist()
    weighted_all = np.nanmean(weighted_all, axis=0).tolist()
    removed_all = np.nanmean(removed_all, axis=0).tolist()
    combined_all = np.nanmean(combined_all, axis=0).tolist()

    baseline_all = pd.Series(baseline_all, index=active_cbs)
    weighted_all = pd.Series(weighted_all, index=active_cbs)
    removed_all = pd.Series(removed_all, index=active_cbs)
    combined_all = pd.Series(combined_all, index=active_cbs)
    # print(len(active_cbs), len(np.nanmean(weighted_all, axis=0)))
    # print(i%3, i%2)
    a[i%2][i%3].plot(*splitSerToArr(baseline_all.dropna()), linestyle='-', marker='o', label='baseline')
    a[i%2][i%3].plot(*splitSerToArr(weighted_all.dropna()), linestyle='-', marker='o', label='weighted')
    a[i%2][i%3].plot(*splitSerToArr(removed_all.dropna()), linestyle='-', marker='o', label='remove causal')
    a[i%2][i%3].plot(*splitSerToArr(combined_all.dropna()), linestyle='-', marker='o', label='combined')
    # a[i%2][i%3].legend()
    a[i%2][i%3].set_title(f'Spurious %: {spur_p[i]}')
    a[i%2][i%3].set_xlabel("CB")
    a[i%2][i%3].set_ylabel("% Test ACC")
# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels)
fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)  # create some space below the plots by increasing the bottom-value
a.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)
# it would of course be better with a nicer handle to the middle-bottom axis object, but since I know it is the second last one in my 3 x 3 grid...
# plt.xlabel('CB')
# plt.ylabel('% Test ACC')
# fig.show()
plt.tight_layout()
plt.savefig(f'ALL.png')
plt.close()
    # break
    # print(log_file_dirs)