import os
from glob import glob
import numpy as np
from tqdm import tqdm
'''
select model from logs in the same root dir
1. get root dir
2. scan all log.txt in the dir
3. select baseline and ours best on best val on each
    - look for the summary keyword
    - look for the val perf, 
    - select best val perf from all files
'''

def file_to_list(file_path):
    with open(file_path) as file:
        lines = [line.strip() for line in file]
    return lines

# def get_numbers(section_lines):

# def get_cb(cb_dict, val_avg, val_wg):
#     for cb in cb_dict:
#         dict_ = cb_dict[cb]
#         dict_vals = list(dict_.values())
#         # print("A", b_val_avg, b_val_wg, o_val_avg, o_val_wg)
#         # print("B", val_avg, val_wg)
#         if val_avg in dict_vals and val_wg in dict_vals:
#         # (val_avg == b_val_avg and val_wg == b_val_wg) or (val_avg == o_val_avg and val_wg == o_val_wg):
#             return cb

def scan_file(lines):
    cb_dict = {}
    for i, line in tqdm(enumerate(lines)):
        if 'HYPERPARAMS' in line:
            params_str = line
        # elif "######" in line:
        #     if "######" in lines[i+1]:
        #         continue
        #     cb = float(line[7:10])
        #     cb_dict[cb] = {}
        #     cb_dict[cb]['baseline_val_avg'] = float(lines[i+3][-5:])
        #     cb_dict[cb]['baseline_val_wg'] = float(lines[i+8][-5:])
        #     cb_dict[cb]['ours_val_avg'] = float(lines[i+21][-5:])
        #     cb_dict[cb]['ours_val_wg'] = float(lines[i+26][-5:])
        elif (line == 'baseline val') or (line == 'baseline test') \
            or (line == 'weighted sample val') or (line == 'weighted sample test'):
            avg_idx = i+1
            wg_idx = i
            while wg_idx < len(lines) and len(lines[wg_idx]) > 0:
                # print(wg_idx, len(lines))
                wg_idx += 1
            wg_idx -= 2
            # avg_idx = wg_idx+1
            # print("HIII", lines[avg_idx][-5:])
            # exit()
            avg = float(lines[avg_idx][-5:])
            wg = float(lines[wg_idx][-5:])
            if line == 'baseline val':
                baseline_val_avg = avg
                baseline_val_wg = wg
            elif line == 'baseline test':
                baseline_test_avg = avg
                baseline_test_wg = wg
            elif line == 'weighted sample val':
                ours_val_avg = avg
                ours_val_wg = wg
            elif line == 'weighted sample test':
                ours_test_avg = avg
                ours_test_wg = wg
            i += wg_idx
            continue
    
    # baseline_cb = get_cb(cb_dict, baseline_val_avg, baseline_val_wg)
    # ours_cb = get_cb(cb_dict, ours_val_avg, ours_val_wg)

    return baseline_val_avg, baseline_val_wg, baseline_test_avg, baseline_test_wg, \
            ours_val_avg, ours_val_wg, ours_test_avg, ours_test_wg,\
            params_str

def get_best_perf(val_list, test_list):
    best_val_idx = np.argmax(np.array(val_list))
    best_val_perf = val_list[best_val_idx]
    best_test_perf = test_list[best_val_idx]
    return best_val_perf, best_test_perf, best_val_idx


if __name__ == '__main__':
    dataset_name = 'celebA'
    log_dir = f'log/spurious_exp/{dataset_name}/TUNE_PARAMS_SPURIOUS'
    tune_by = 'acc_wg'
    
    log_files =  [y for x in os.walk(log_dir) for y in glob(os.path.join(x[0], '*.txt'))]

    baseline_val_avg = []
    baseline_val_wg = []
    baseline_test_avg = []
    baseline_test_wg = []
    ours_val_avg = []
    ours_val_wg = []
    ours_test_avg = []
    ours_test_wg = []
    params_str = []
    baseline_cb = []
    ours_cb = []

    for file_ in tqdm(log_files):
        lines = file_to_list(file_)
        if ('baseline val' not in lines) and ('baseline test' not in lines) and \
            ('weighted sample val' not in lines) and ('weighted sample test' not in lines):
            continue
        b_val_avg, b_val_wg, b_test_avg, b_test_wg, \
        o_val_avg, o_val_wg, o_test_avg, o_test_wg, \
        param_str_ = scan_file(lines)

        baseline_val_avg.append(b_val_avg)
        baseline_val_wg.append(b_val_wg)
        baseline_test_avg.append(b_test_avg)
        baseline_test_wg.append(b_test_wg)
        ours_val_avg.append(o_val_avg)
        ours_val_wg.append(o_val_wg)
        ours_test_avg.append(o_test_avg)
        ours_test_wg.append(o_test_wg)
        params_str.append(param_str_)

    if tune_by == 'acc_avg':
        best_baseline_val, best_baseline_test, best_baseline_idx = get_best_perf(baseline_val_avg, baseline_test_avg)

        best_ours_val, best_ours_test, best_ours_idx = get_best_perf(ours_val_avg, ours_test_avg)
        
        best_params = params_str[best_ours_idx]

        print(f"{best_params}")
        print("BASELINE")
        print(f"VAL ACC AVG: {best_baseline_val} | TEST ACC AVG: {best_baseline_test}")
        print("="*50)
        print("OURS")
        print(f"VAL ACC AVG: {best_ours_val} | TEST ACC AVG: {best_ours_test}")
    elif tune_by == 'acc_wg':
        best_baseline_val, best_baseline_test, best_baseline_idx = get_best_perf(baseline_val_wg, baseline_test_wg)
        best_ours_val, best_ours_test, best_ours_idx = get_best_perf(ours_val_wg, ours_test_wg)
        best_params = params_str[best_ours_idx]
    
        print(f"{best_params}")
        print("BASELINE")
        print(f"TEST ACC AVG: {baseline_test_avg[best_baseline_idx]} | TEST ACC WG: {best_baseline_test}")
        print("="*50)
        print("OURS")
        print(f"TEST ACC AVG: {ours_test_avg[best_ours_idx]} | TEST ACC WG: {best_ours_test}")
        
    print("FOR THE LOVE OF GOD")

    