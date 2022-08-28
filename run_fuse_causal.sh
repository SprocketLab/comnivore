CUDA_VISIBLE_DEVICES=$1
python generate_candidate_sets.py -c configs/spur_exp/Synthetic_ColoredMNIST.yaml 
python get_spurious_samples.py -c configs/spur_exp/Synthetic_ColoredMNIST.yaml 
# python fuse_causal_estimates.py -c configs/irm_waterbirds.yaml