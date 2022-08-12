CUDA_VISIBLE_DEVICES=$0
python generate_candidate_sets.py -c configs/ColoredMNIST.yaml 
python fuse_causal_estimates.py -c configs/ColoredMNIST.yaml