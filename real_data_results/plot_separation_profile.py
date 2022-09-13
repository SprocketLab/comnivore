import numpy as np
import matplotlib.pyplot as plt
import os

log_dir_root = '../log/SPURIOUS_NEW'
dataset_name = 'waterbirds'
dataset_log_dir = os.path.join(log_dir_root, dataset_name)
log_subdirs = [os.path.join(dataset_log_dir, dir_) for dir_ in os.listdir(dataset_log_dir)]
log_subdirs.sort(key=lambda x: os.path.getmtime(x))
latest_log_dir = log_subdirs[-1]

log_file = os.path.join(latest_log_dir, "log.txt")

def get_data_from_log(log_lines):
    active_cb = []
    unweighted = []
    weighted = []
    separation = []
    low_p = []
    high_p = []
    for line_idx, line_ in enumerate(log_lines):
        # print(line_)
        if "######" in line_:
            if "######" in log_lines[line_idx+1] or "BEST" in log_lines[line_idx+1]:
                continue
            active_cb.append(float(line_[7:10]))
            unweighted_test_acc = float(log_lines[line_idx+18][-5:])
            sep_ = float(log_lines[line_idx+23][-5:])
            weighted_test_acc = float(log_lines[line_idx+40][-5:])
            low_ = float(log_lines[line_idx+22][-5:])
            high_ = 1.-float(log_lines[line_idx+21][-5:])

            unweighted.append(unweighted_test_acc)
            weighted.append(weighted_test_acc)
            separation.append(sep_)
            high_p.append(high_)
            low_p.append(low_)

            line_idx+=40
    return active_cb, unweighted, weighted, separation, high_p, low_p

def file_to_list(file_path):
    with open(file_path) as file:
        lines = [line.strip() for line in file]
    return lines

file_lines = file_to_list(log_file)
active_cb, unweighted, weighted, separation, high_p, low_p = get_data_from_log(file_lines)
print(high_p, low_p)
# print(active_cb, unweighted, weighted, separation)
plt.plot(active_cb, unweighted, label="unweighted (test acc)", marker='X')
plt.plot(active_cb, weighted, label="weighted (test acc)", marker='X')

# plt.plot(active_cb, separation, label="separation")
plt.plot(active_cb, high_p, label="high weights acc", marker='v')
plt.plot(active_cb, low_p, label="low weights acc", marker='v')
plt.axhline(y=np.amax(weighted), xmin=0, xmax=len(active_cb), color='r', linestyle='--', alpha=0.15)
# plt.axhline(y=np.amax(high_p), xmin=0, xmax=len(active_cb), color='r', linestyle='--', alpha=0.15)
plt.axhline(y=np.amax(low_p), xmin=0, xmax=len(active_cb), color='r', linestyle='--', alpha=0.15)

plt.xticks(active_cb)
plt.xlabel("class balance")
plt.ylabel("%")
plt.tight_layout()
plt.legend()
plt.title(f"{dataset_name}")
plt.savefig(f"{dataset_name}_ORIG.png")

"""
What to read:
1. active cb: cb that was not skipped
2. unweighted perf
3. weighted perf
4. separation
"""
